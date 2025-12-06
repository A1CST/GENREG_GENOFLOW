"""
Simplified OLA Genome for isolated evolution visualization
Stripped of game-specific logic, focused on continuous mutation and trust dynamics
"""
from __future__ import annotations
import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import Dict
import time
import numpy as np


@dataclass
class GenomeStats:
    """Statistics for a single genome"""
    genome_id: int
    total_ticks: int = 0
    trust_score: float = 1.0
    trust_raw: float = .5
    creation_time: float = 0.0
    last_used_time: float = 0.0
    activations: int = 0
    mutation_count: int = 0
    consistency_score: float = 0.0


class EvoCell(nn.Module):
    """
    Simple recurrent cell that can mutate
    """
    def __init__(self, in_dim: int, out_dim: int, state_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.state_dim = state_dim

        # Input fusion
        self.in_proj = nn.Linear(in_dim + state_dim, state_dim)
        # Gated residual block
        self.h1 = nn.Linear(state_dim, state_dim)
        self.g1 = nn.Linear(state_dim, state_dim)
        # Output head
        self.out = nn.Linear(state_dim, out_dim)
        # State projection for next step
        self.next_state = nn.Linear(state_dim, state_dim)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    @torch.no_grad()
    def mutate(self, mutation_rate: float, grow_prob: float, max_state_dim: int):
        """Mutate weights with noise"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if torch.rand(()) < mutation_rate:
                    noise = 0.02 * torch.randn_like(m.weight)
                    m.weight.add_(noise)
                if torch.rand(()) < mutation_rate:
                    m.bias.add_(0.02 * torch.randn_like(m.bias))

        # Optional structural growth
        if self.state_dim < max_state_dim and torch.rand(()) < grow_prob:
            self._grow_hidden(16)

    def _grow_hidden(self, add: int):
        new_h = min(self.state_dim + add, 512)
        if new_h == self.state_dim:
            return

        device = next(self.parameters()).device

        def grow_linear(old: nn.Linear, in_features: int, out_features: int) -> nn.Linear:
            new = nn.Linear(in_features, out_features, bias=True).to(device)
            with torch.no_grad():
                new.weight.zero_()
                new.bias.zero_()
                new.weight[:old.out_features, :old.in_features] = old.weight
                new.bias[:old.out_features] = old.bias
            return new

        old_state = self.state_dim
        self.state_dim = new_h

        self.in_proj = grow_linear(self.in_proj, self.in_dim + new_h, new_h)
        self.h1 = grow_linear(self.h1, new_h, new_h)
        self.g1 = grow_linear(self.g1, new_h, new_h)
        self.next_state = grow_linear(self.next_state, new_h, new_h)
        self.out = nn.Linear(new_h, self.out.out_features, bias=True).to(device)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        x: [B, in_dim]
        h: [B, state_dim]
        returns: (output: [B, out_dim], h_next: [B, state_dim])
        """
        z = torch.cat([x, h], dim=-1)
        s = torch.tanh(self.in_proj(z))
        # Gated residual
        r = torch.tanh(self.h1(s))
        g = torch.sigmoid(self.g1(s))
        s2 = s + g * r
        output = self.out(s2)
        h_next = torch.tanh(self.next_state(s2))
        return output, h_next


class OLAGenome:
    """
    Single OLA genome with trust tracking
    Simplified for continuous evolution visualization
    """

    def __init__(self, genome_id: int, in_dim: int, out_dim: int, state_dim: int,
                 trust_decay: float = 0.995, device: str = "cpu"):
        """
        Initialize OLA genome

        Args:
            genome_id: Unique identifier
            in_dim: Input dimension (from PatternLSTM)
            out_dim: Output dimension
            state_dim: Recurrent state dimension
            trust_decay: Trust decay factor per tick
            device: Device to run on
        """
        self.genome_id = genome_id
        self.device = torch.device(device)

        # Create the EvoCell
        self.cell = EvoCell(in_dim, out_dim, state_dim).to(self.device)

        # Genome statistics
        self.stats = GenomeStats(
            genome_id=genome_id,
            creation_time=time.time()
        )

        # Trust system
        self.trust_decay = trust_decay
        self.trust_raw = .25
        self.stats.trust_raw = self.trust_raw
        self.stats.trust_score = self._compress_trust(self.trust_raw)

        # Probation window for dampened early boosts
        self.probation_ticks = 100

        # Internal state
        self.state_dim = state_dim
        self.h = torch.zeros(1, state_dim, device=self.device)

        # External cap for trust boost (set by controller)
        self.external_boost_cap: float = 1.0
        # Optional base boost override (set by controller)
        self.boost_base_override = None

        # Output history for consistency tracking
        self.output_history = []
        self.max_history = 64

    def _compress_trust(self, trust_raw: float) -> float:
        """Compress trust to [0, 1] range"""
        return 0.8 / (1.0 + math.exp(-5.0 * (trust_raw - 0.5)))

    def add_trust_offset(self, offset: float):
        """Add offset to trust score"""
        self.trust_raw = max(0.0, self.trust_raw + offset)
        self.stats.trust_raw = self.trust_raw
        self.stats.trust_score = self._compress_trust(self.trust_raw)

    def reset_state(self):
        """Reset recurrent state"""
        self.h = torch.zeros(1, self.state_dim, device=self.device)
        self.output_history = []

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through genome

        Args:
            x: Input tensor [1, in_dim]

        Returns:
            output: Output tensor [1, out_dim]
            h_next: Next recurrent state [1, state_dim]
        """
        with torch.no_grad():
            output, h_next = self.cell(x, self.h)
            self.h = h_next

            # Track output for consistency
            self.output_history.append(output.cpu().numpy().copy())
            if len(self.output_history) > self.max_history:
                self.output_history.pop(0)

        return output, h_next

    def update_trust(self):
        """
        Update trust based on output consistency
        Trust increases when outputs are consistent across timesteps
        """
        # Decay trust
        self.stats.trust_score *= self.trust_decay

        # Calculate consistency
        if len(self.output_history) >= 3:
            outputs = np.array(self.output_history[-5:])
            # Calculate variance across recent outputs
            variance = np.var(outputs)
            consistency = 1.0 / (1.0 + variance)

            self.stats.consistency_score = float(consistency)

            # Boost trust if consistent, with early dampening and external ceiling
            boost_base = 0.0 if self.stats.total_ticks < self.probation_ticks else 0.002
            if getattr(self, 'boost_base_override', None) is not None:
                try:
                    boost_base = float(self.boost_base_override)
                except Exception:
                    pass
            cap = float(getattr(self, 'external_boost_cap', 1.0))
            boost = min(boost_base, max(0.0, cap))
            if consistency > 0.5:
                self.stats.trust_score = min(self.stats.trust_score + boost * consistency, 1.0)

        self.stats.total_ticks += 1

    def mutate(self, mutation_rate: float = 0.1, grow_prob: float = 0.05,
               max_state_dim: int = 512):
        """
        Mutate genome in-place

        Args:
            mutation_rate: Probability of mutating each parameter
            grow_prob: Probability of growing hidden dimension
            max_state_dim: Maximum state dimension
        """
        self.cell.mutate(mutation_rate, grow_prob, max_state_dim)
        self.stats.mutation_count += 1

    def reset_trust(self, initial_trust: float = 1.0):
        """Reset trust score (used after mutation)"""
        self.trust_raw = float(initial_trust)
        self.stats.trust_raw = self.trust_raw
        self.stats.trust_score = self._compress_trust(self.trust_raw)

    def get_state_dict(self) -> Dict:
        """Get complete genome state for checkpointing"""
        return {
            'genome_id': self.genome_id,
            'cell_state': self.cell.state_dict(),
            'stats': {
                'genome_id': self.stats.genome_id,
                'total_ticks': self.stats.total_ticks,
                'trust_score': self.stats.trust_score,
                'trust_raw': self.trust_raw,
                'creation_time': self.stats.creation_time,
                'last_used_time': self.stats.last_used_time,
                'activations': self.stats.activations,
                'mutation_count': self.stats.mutation_count,
                'consistency_score': self.stats.consistency_score
            },
            'state_dim': self.state_dim,
            'trust_decay': self.trust_decay
        }

    @staticmethod
    def from_state_dict(state_dict: Dict, in_dim: int, out_dim: int,
                       device: str = "cpu") -> OLAGenome:
        """Restore genome from state dict"""
        genome_id = state_dict['genome_id']
        state_dim = state_dict['state_dim']
        trust_decay = state_dict['trust_decay']

        genome = OLAGenome(
            genome_id=genome_id,
            in_dim=in_dim,
            out_dim=out_dim,
            state_dim=state_dim,
            trust_decay=trust_decay,
            device=device
        )

        genome.cell.load_state_dict(state_dict['cell_state'])

        stats_dict = state_dict['stats']
        genome.stats = GenomeStats(
            genome_id=stats_dict['genome_id'],
            total_ticks=stats_dict['total_ticks'],
            trust_score=stats_dict['trust_score'],
            trust_raw=stats_dict.get('trust_raw', stats_dict['trust_score']),
            creation_time=stats_dict['creation_time'],
            last_used_time=stats_dict['last_used_time'],
            activations=stats_dict['activations'],
            mutation_count=stats_dict.get('mutation_count', 0),
            consistency_score=stats_dict.get('consistency_score', 0.0)
        )
        genome.trust_raw = genome.stats.trust_raw
        genome.stats.trust_score = genome._compress_trust(genome.trust_raw)

        return genome

    def __repr__(self) -> str:
        return (f"OLAGenome(id={self.genome_id}, trust={self.stats.trust_score:.3f}, "
                f"ticks={self.stats.total_ticks}, mutations={self.stats.mutation_count}, "
                f"consistency={self.stats.consistency_score:.3f})")
