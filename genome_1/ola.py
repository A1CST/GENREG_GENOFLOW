# ola.py
from __future__ import annotations
import math, time, random
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OLAConfig:
    in_dim: int                 # compressed pattern dim
    out_dim: int                # flattened latent dim (prediction delta vector)
    state_dim: int = 128        # recurrent internal state size (per genome)
    pop_size: int = 64
    elite_frac: float = 0.15
    mutation_rate: float = 0.10
    structure_add_prob: float = 0.05  # probability to grow hidden width
    structure_max_dim: int = 512
    lr_state: float = 0.0       # optional learned state update (kept 0 for pure-evo)
    novelty_weight: float = 0.0 # set >0 later when you bring LSH novelty online
    device: str = "cuda"


class EvoCell(nn.Module):
    """
    Enhanced recurrent cell with rich temporal dynamics:
    - Delayed self-connections (past hidden states)
    - Memory gates (learned decay per neuron)
    - Persistent memory cells with slow decay
    Allows evolution to discover working memory modules.
    """
    def __init__(self, in_dim: int, out_dim: int, state_dim: int,
                 num_delay_taps: int = 3, memory_size: int = 64):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.state_dim = state_dim
        self.num_delay_taps = num_delay_taps  # How many past states to use
        self.memory_size = memory_size  # Size of persistent memory bank

        # Input fusion (current input + current hidden state + delayed states)
        self.in_proj = nn.Linear(in_dim + state_dim * (1 + num_delay_taps), state_dim)

        # Gated residual block
        self.h1 = nn.Linear(state_dim, state_dim)
        self.g1 = nn.Linear(state_dim, state_dim)

        # Output head (predict latent delta)
        self.out = nn.Linear(state_dim, out_dim)

        # State projection for next step
        self.next_state = nn.Linear(state_dim, state_dim)

        # Memory gate: learned decay coefficient per neuron [0, 1]
        # Higher = more persistence, lower = faster decay
        self.memory_gate = nn.Parameter(torch.sigmoid(torch.randn(state_dim)))

        # Persistent memory bank with learnable decay rates
        self.memory_bank = nn.Parameter(torch.zeros(memory_size))
        self.memory_decay = nn.Parameter(torch.sigmoid(torch.randn(memory_size)))  # [0, 1]

        # Memory read/write heads
        self.memory_write = nn.Linear(state_dim, memory_size)
        self.memory_read = nn.Linear(memory_size, state_dim)

        # Memory update gate (how much to write new values vs keep old)
        self.memory_write_gate = nn.Linear(state_dim, memory_size)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

        # Initialize memory gates to mid-range (0.5) for balanced decay
        nn.init.constant_(self.memory_gate, 0.0)  # sigmoid(0) = 0.5
        nn.init.constant_(self.memory_decay, 0.0)  # sigmoid(0) = 0.5

    @torch.no_grad()
    def mutate(self, p: float, grow_prob: float, max_dim: int):
        """
        Mutate weights, memory gates, and decay rates.
        Evolution can discover optimal temporal dynamics.
        """
        # Mutate linear layer weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if torch.rand(()) < p:
                    noise = 0.02 * torch.randn_like(m.weight)
                    m.weight.add_(noise)
                if torch.rand(()) < p:
                    m.bias.add_(0.02 * torch.randn_like(m.bias))

        # Mutate memory gates (decay coefficients)
        if torch.rand(()) < p:
            gate_noise = 0.01 * torch.randn_like(self.memory_gate)
            self.memory_gate.data.add_(gate_noise)
            self.memory_gate.data.clamp_(0.0, 1.0)

        # Mutate memory decay rates
        if torch.rand(()) < p:
            decay_noise = 0.01 * torch.randn_like(self.memory_decay)
            self.memory_decay.data.add_(decay_noise)
            self.memory_decay.data.clamp_(0.0, 1.0)

        # Mutate memory bank values (rarely, for exploration)
        if torch.rand(()) < p * 0.1:
            mem_noise = 0.005 * torch.randn_like(self.memory_bank)
            self.memory_bank.data.add_(mem_noise)

        # Optional structural growth: widen state_dim by 16 up to max_dim
        if self.state_dim < max_dim and torch.rand(()) < grow_prob:
            self._grow_hidden(16)

    def _grow_hidden(self, add: int):
        new_h = min(self.state_dim + add, self.out.in_features)
        if new_h == self.state_dim:
            return
        # Get device from existing parameters
        device = next(self.parameters()).device

        # recreate all state_dim-sized layers with padded weights
        def grow_linear(old: nn.Linear, in_features: int, out_features: int) -> nn.Linear:
            new = nn.Linear(in_features, out_features, bias=True).to(device)
            with torch.no_grad():
                new.weight.zero_(); new.bias.zero_()
                new.weight[:old.out_features, :old.in_features] = old.weight
                new.bias[:old.out_features] = old.bias
            return new

        old_state = self.state_dim
        self.state_dim = new_h

        # in_proj: in_dim + old_state*(1+delay_taps) -> new_h
        self.in_proj = grow_linear(self.in_proj,
                                   self.in_dim + old_state * (1 + self.num_delay_taps),
                                   new_h)
        # h1/g1: old_state -> new_h
        self.h1 = grow_linear(self.h1, old_state, new_h)
        self.g1 = grow_linear(self.g1, old_state, new_h)
        # next_state: old_state -> new_h
        self.next_state = grow_linear(self.next_state, old_state, new_h)
        # out: new_h -> out_dim
        self.out = nn.Linear(new_h, self.out.out_features, bias=True).to(device)

        # Grow memory gate
        old_gate = self.memory_gate.data
        self.memory_gate = nn.Parameter(torch.zeros(new_h, device=device))
        self.memory_gate.data[:old_state] = old_gate

        # Grow memory read/write heads
        self.memory_write = grow_linear(self.memory_write, old_state, self.memory_size)
        self.memory_read = grow_linear(self.memory_read, self.memory_size, new_h)
        self.memory_write_gate = grow_linear(self.memory_write_gate, old_state, self.memory_size)

    def forward(self, x: torch.Tensor, h_current: torch.Tensor,
                h_delayed: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhanced forward pass with temporal dynamics.

        Args:
            x: [B, in_dim] - current input
            h_current: [B, state_dim] - current hidden state
            h_delayed: List of [B, state_dim] - past hidden states (delay taps)

        Returns:
            delta: [B, out_dim] - predicted output
            h_next: [B, state_dim] - next hidden state
        """
        B = x.shape[0]

        # If no delayed states provided, use zeros
        if h_delayed is None or len(h_delayed) == 0:
            h_delayed = [torch.zeros_like(h_current) for _ in range(self.num_delay_taps)]

        # Pad delayed states if needed
        while len(h_delayed) < self.num_delay_taps:
            h_delayed.append(torch.zeros_like(h_current))

        # Concatenate current input, current state, and delayed states
        z = torch.cat([x, h_current] + h_delayed[:self.num_delay_taps], dim=-1)

        # Input projection with delayed temporal context
        s = torch.tanh(self.in_proj(z))

        # Read from persistent memory bank
        memory_content = self.memory_read(self.memory_bank.unsqueeze(0).expand(B, -1))
        s = s + memory_content  # Add memory content to state

        # Gated residual block
        r = torch.tanh(self.h1(s))
        g = torch.sigmoid(self.g1(s))
        s2 = s + g * r

        # Output prediction
        delta = self.out(s2)

        # Next state with memory gate (learned decay)
        h_raw = torch.tanh(self.next_state(s2))

        # Apply memory gate: h_next = gate * h_current + (1 - gate) * h_raw
        # This allows neurons to persist information across timesteps
        h_next = self.memory_gate * h_current + (1 - self.memory_gate) * h_raw

        # Update persistent memory bank
        # Write new values
        write_values = torch.tanh(self.memory_write(s2))
        write_gate = torch.sigmoid(self.memory_write_gate(s2))

        # Average write values across batch
        write_values_avg = write_values.mean(dim=0)
        write_gate_avg = write_gate.mean(dim=0)

        # Update memory: decay old + gate * new
        self.memory_bank.data = (
            self.memory_decay * self.memory_bank.data +
            write_gate_avg * write_values_avg
        )

        return delta, h_next


@dataclass
class Genome:
    cell: EvoCell
    score: float = float("-inf")
    id: int = 0

    def clone(self) -> "Genome":
        # Get device from source cell
        device = next(self.cell.parameters()).device
        g = Genome(EvoCell(self.cell.in_dim, self.cell.out_dim, self.cell.state_dim).to(device))
        g.cell.load_state_dict(self.cell.state_dict())
        g.score = self.score
        g.id = self.id
        return g

    @torch.no_grad()
    def mutate(self, cfg: OLAConfig):
        self.cell.mutate(cfg.mutation_rate, cfg.structure_add_prob, cfg.structure_max_dim)


class OrganicLogicAgent:
    """
    Population of small recurrent circuits that compete to minimize next-latent prediction error.
    Online evolutionary update each tick; keeps state per-genome.
    """
    def __init__(self, cfg: OLAConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.pop: List[Genome] = []
        self.states: Optional[torch.Tensor] = None  # [P, state_dim]
        self.step_idx = 0
        self.best: Optional[Genome] = None
        self._init_population()

    def _init_population(self):
        self.pop = []
        for i in range(self.cfg.pop_size):
            cell = EvoCell(self.cfg.in_dim, self.cfg.out_dim, self.cfg.state_dim).to(self.device)
            self.pop.append(Genome(cell=cell, id=i))
        self.states = torch.zeros(self.cfg.pop_size, self.cfg.state_dim, device=self.device)

    @torch.no_grad()
    def forward_population(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [1, in_dim] compressed pattern for this tick
        returns:
          delta_preds: [P, out_dim]
          next_states: [P, state_dim]
        """
        P = self.cfg.pop_size
        x_rep = x.repeat(P, 1)
        deltas = []
        next_states = []
        for i, g in enumerate(self.pop):
            d, h_next = g.cell(x_rep[i:i+1], self.states[i:i+1])
            deltas.append(d)
            next_states.append(h_next)
        delta_preds = torch.cat(deltas, dim=0)
        next_states = torch.cat(next_states, dim=0)
        return delta_preds, next_states

    @torch.no_grad()
    def step(self, compressed_t: torch.Tensor, current_latent_vec: torch.Tensor,
             prev_latent_vec: Optional[torch.Tensor]) -> Dict[str, float]:
        """
        One online step:
          1) Evaluate all genomes on predicting next-latent delta from compressed_t.
          2) Score = -MSE(pred_latent, true_latent) with small complexity penalty.
          3) Evolve population (elitism + mutation).
        Returns metrics dict.
        """
        self.step_idx += 1

        # If no prev latent, we can't form a target delta. Defer scoring until next tick.
        if prev_latent_vec is None:
            # push state forward with zero loss; mild exploration
            with torch.no_grad():
                _ = self.forward_population(compressed_t)
            return {"ola_loss": float("nan"), "ola_best": float("-inf"), "ola_state_dim": float(self.cfg.state_dim)}

        # Predict delta and latent
        deltas, next_states = self.forward_population(compressed_t)  # [P,D], [P,H]
        P = self.cfg.pop_size
        # delta -> latent prediction
        current_rep = current_latent_vec.repeat(P, 1)
        pred_latents = current_rep + deltas  # [P, D_out]

        # True next-latent is current_latent_vec (target is "now" vs prev)
        targets = current_latent_vec.repeat(P, 1)
        # MSE per genome
        mse = torch.mean((pred_latents - targets) ** 2, dim=1)  # [P]
        # small complexity penalty: favor smaller hidden (state) norms
        state_norm = torch.mean(next_states**2, dim=1)
        score = -mse - 1e-4 * state_norm  # maximize score

        # Update genome scores, keep best
        best_idx = int(torch.argmax(score).item())
        best_score = float(score[best_idx].item())
        for i, g in enumerate(self.pop):
            g.score = float(score[i].item())
        self.best = self.pop[best_idx].clone()

        # Evolve (elitism + mutation)
        self.states = next_states  # adopt next states
        self._evolve()

        return {
            "ola_loss": float(mse[best_idx].item()),
            "ola_best": best_score,
            "ola_state_dim": float(self.pop[0].cell.state_dim)
        }

    @torch.no_grad()
    def _evolve(self):
        P = self.cfg.pop_size
        k_elite = max(1, int(P * self.cfg.elite_frac))
        # select top-k
        ranked = sorted(self.pop, key=lambda g: g.score, reverse=True)
        elites = [g.clone() for g in ranked[:k_elite]]
        # refill with mutated copies of elites
        new_pop: List[Genome] = []
        for i in range(P):
            parent = elites[i % k_elite].clone()
            parent.mutate(self.cfg)
            parent.id = i
            new_pop.append(parent)
        self.pop = new_pop
        # reset states for new genomes where structure changed scale
        self.states = self.states[:P]
        if self.states.shape[1] != self.pop[0].cell.state_dim:
            self.states = torch.zeros(P, self.pop[0].cell.state_dim, device=self.device)

    def best_predict(self, compressed_t: torch.Tensor, current_latent_vec: torch.Tensor) -> torch.Tensor:
        """Use current best genome to produce the displayed next-latent prediction."""
        if self.best is None:
            # cold start with genome 0
            g = self.pop[0]
            delta, _ = g.cell(compressed_t, self.states[0:1])
            return current_latent_vec + delta
        else:
            # we evaluate with its own state slot (id may not match index in new_pop, so just use slot 0)
            delta, _ = self.best.cell(compressed_t, self.states[0:1])
            return current_latent_vec + delta

    def save_best_genome(self, path: str, metadata: Optional[Dict] = None) -> None:
        """
        Save the current best genome to a file for visualization and analysis.

        Args:
            path: File path to save the genome (e.g., 'genomes/best_genome.pt')
            metadata: Optional dictionary of additional metadata to save with the genome
        """
        if self.best is None:
            print(f"[OLA] Warning: No best genome available to save")
            return

        save_dict = {
            'champion_state_dict': self.best.cell.state_dict(),
            'config': {
                'in_dim': self.cfg.in_dim,
                'out_dim': self.cfg.out_dim,
                'state_dim': self.best.cell.state_dim,
                'pop_size': self.cfg.pop_size,
                'elite_frac': self.cfg.elite_frac,
                'mutation_rate': self.cfg.mutation_rate,
                'structure_add_prob': self.cfg.structure_add_prob,
                'structure_max_dim': self.cfg.structure_max_dim,
            },
            'genome_info': {
                'score': self.best.score,
                'id': self.best.id,
                'step_idx': self.step_idx,
                'state_dim': self.best.cell.state_dim,
            },
        }

        # Add custom metadata
        if metadata:
            save_dict['metadata'] = metadata

        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save(save_dict, path)
        print(f"[OLA] Saved best genome (score={self.best.score:.6f}) to {path}")
