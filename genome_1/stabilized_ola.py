# stabilized_ola.py
from __future__ import annotations
import time
import copy
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ola import EvoCell  # reuse existing cell implementation


@dataclass
class StabilizedOLAConfig:
    in_dim: int
    out_dim: int
    state_dim: int = 128
    mutation_rate: float = 0.25
    mutation_decay: float = 0.999
    mutation_floor: float = 1e-5
    stability_factor: float = 0.98  # EMA damping (0.95-0.995)
    rollback_threshold: float = 5.0  # revert if loss > threshold * ema_loss
    grow_prob: float = 0.0
    max_state_dim: int = 512
    cosine_weight: float = 0.2
    reg_weight: float = 1e-4
    device: str = "cuda"

    # Champion replay
    num_champions: int = 3

    # Adaptive mutation
    variance_window: int = 10
    variance_threshold: float = 0.01
    mutation_decrease: float = 0.9
    mutation_increase: float = 1.05

    # Temporal regularization
    temporal_beta: float = 0.02

    # Multi-step temporal consistency
    multistep_weight: float = 0.3  # λ weight for t+2 prediction loss
    multistep_horizon: int = 2  # How many steps ahead to predict (2 = t+1 and t+2)

    # Age-based reset
    max_no_improve_age: int = 20


class StabilizedOLA:
    """
    Stabilized single-tick evolution with EMA smoothing and adaptive mutation decay.
    - Each tick: single evolution step with fitness = -(loss / time_delta)
    - EMA on loss and score to avoid oscillation
    - Gradual mutation decay for continuous learning
    - Rollback protection against catastrophic drift
    """
    def __init__(self, cfg: StabilizedOLAConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Champion genome
        self.champion = EvoCell(cfg.in_dim, cfg.out_dim, cfg.state_dim).to(self.device)
        self.h_champ = torch.zeros(1, cfg.state_dim, device=self.device)

        # Delayed state history for temporal dynamics (3 delay taps)
        self.h_delay_buffer = []  # Circular buffer of past hidden states
        self.max_delay_taps = 3

        # Previous champion for rollback
        self.prev_champion = None
        self.prev_h_champ = None
        self.prev_h_delay_buffer = None

        # Champion replay: keep top-N champions
        self.champion_hall = []  # List of (EvoCell, score) tuples

        # Stability tracking
        self.ema_loss = None
        self.ema_score = None
        self.prev_mse = None  # for normalized loss
        self.prev_latent = None  # for temporal regularization
        self.last_update_time = time.time()
        self.mutation_rate = cfg.mutation_rate
        self.mutation_decay = cfg.mutation_decay
        self.stability_factor = cfg.stability_factor

        # Multi-step frame history for temporal consistency
        self.frame_history = []  # Store last N frames as (x_t, z_t) tuples
        self.max_frame_history = cfg.multistep_horizon + 1

        # Adaptive mutation tracking
        self.loss_history = []  # rolling window for variance
        self.loss_variance = 0.0

        # Age tracking
        self.no_improve_age = 0
        self.best_fitness_ever = float("-inf")

        # Metrics
        self.step_idx = 0
        self.champion_score = float("-inf")
        self._last_mse = 0.0
        self._last_cos = 0.0
        self._last_time_delta = 0.0
        self._rollback_count = 0
        self._temporal_penalty = 0.0
        self._relative_loss = 0.0
        self._multistep_loss = 0.0

    @torch.no_grad()
    def _evaluate(self, cell: EvoCell, x: torch.Tensor, z_cur: torch.Tensor,
                  z_prev: torch.Tensor, h: torch.Tensor,
                  h_delayed: Optional[list] = None) -> Tuple[float, torch.Tensor]:
        """
        Evaluate a genome and return (loss, next_state).
        """
        # Forward pass with delayed states
        delta, h_next = cell(x, h, h_delayed)

        # Target delta
        target_delta = z_cur - z_prev

        # MSE loss
        mse = torch.mean((delta - target_delta) ** 2)

        # Cosine loss on deltas
        n1 = F.normalize(delta, dim=-1, eps=1e-8)
        n2 = F.normalize(target_delta, dim=-1, eps=1e-8)
        cos = 1.0 - torch.sum(n1 * n2, dim=-1).mean()

        # State regularization
        reg = h_next.pow(2).mean()

        # Combined loss
        loss = mse + self.cfg.cosine_weight * cos + self.cfg.reg_weight * reg

        # Store for logging
        self._last_mse = float(mse.item())
        self._last_cos = float(cos.item())

        return float(loss.item()), h_next

    @torch.no_grad()
    def _evaluate_multistep(self, cell: EvoCell, h: torch.Tensor) -> float:
        """
        Evaluate temporal consistency by simulating multi-step prediction.
        Returns the multi-step loss component.

        Given frame history [(x_0, z_0), (x_1, z_1), ..., (x_t, z_t)],
        predict z_1, z_2, ..., z_t by rolling out from (x_0, z_0).
        """
        if len(self.frame_history) < self.cfg.multistep_horizon:
            return 0.0  # Not enough history yet

        total_loss = 0.0
        num_steps = 0

        # Start from the oldest frame in history
        for start_idx in range(len(self.frame_history) - 1):
            x_start, z_start = self.frame_history[start_idx]

            # Predict forward as many steps as possible
            z_pred = z_start.clone()
            h_sim = h.clone()
            h_sim_delay_buffer = []  # Maintain delay buffer for simulation

            for step in range(1, min(self.cfg.multistep_horizon, len(self.frame_history) - start_idx)):
                # Predict next frame with delayed states
                delta, h_sim_next = cell(x_start, h_sim, h_sim_delay_buffer)
                z_pred = z_pred + delta

                # Update delay buffer
                h_sim_delay_buffer.insert(0, h_sim.clone())
                if len(h_sim_delay_buffer) > self.max_delay_taps:
                    h_sim_delay_buffer.pop()

                h_sim = h_sim_next

                # Get ground truth
                _, z_target = self.frame_history[start_idx + step]

                # Compute loss for this prediction
                step_mse = torch.mean((z_pred - z_target) ** 2)
                total_loss += step_mse.item()
                num_steps += 1

                # Update x for next iteration (use current frame's input)
                x_start, _ = self.frame_history[start_idx + step]

        if num_steps == 0:
            return 0.0

        return total_loss / num_steps

    def _update_champion_hall(self, cell: EvoCell, score: float):
        """Maintain top-N champion genomes for replay"""
        # Clone the cell
        champion_clone = EvoCell(cell.in_dim, cell.out_dim, cell.state_dim).to(self.device)
        champion_clone.load_state_dict(cell.state_dict())

        self.champion_hall.append((champion_clone, score))
        # Sort by score (descending) and keep top N
        self.champion_hall.sort(key=lambda x: x[1], reverse=True)
        self.champion_hall = self.champion_hall[:self.cfg.num_champions]

    def _get_champion_baseline(self) -> Optional[EvoCell]:
        """Get a champion from hall for mutation baseline"""
        if not self.champion_hall:
            return None
        # Return best champion from hall
        return self.champion_hall[0][0]

    def _update_mutation_rate(self, loss: float):
        """Adaptive mutation based on loss variance"""
        # Update loss history
        self.loss_history.append(loss)
        if len(self.loss_history) > self.cfg.variance_window:
            self.loss_history.pop(0)

        # Calculate variance
        if len(self.loss_history) >= self.cfg.variance_window:
            import numpy as np
            self.loss_variance = float(np.var(self.loss_history))

            # Adjust mutation rate
            if self.loss_variance < self.cfg.variance_threshold:
                # Low variance - reduce exploration
                self.mutation_rate *= self.cfg.mutation_decrease
            else:
                # High variance - increase exploration
                self.mutation_rate *= self.cfg.mutation_increase

            # Clamp
            self.mutation_rate = max(self.cfg.mutation_floor, min(self.mutation_rate, 0.5))

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, z_t: torch.Tensor,
             z_tm1: Optional[torch.Tensor]) -> Dict[str, float]:
        """
        Stabilized single-tick evolution with champion replay, adaptive mutation,
        temporal regularization, and normalized loss.
        """
        self.step_idx += 1
        current_time = time.time()
        time_delta = current_time - self.last_update_time
        self._last_time_delta = time_delta

        # Add current frame to history
        self.frame_history.append((x_t.detach().clone(), z_t.detach().clone()))
        if len(self.frame_history) > self.max_frame_history:
            self.frame_history.pop(0)

        if z_tm1 is None:
            # Warmup: just advance state
            _, h_next = self._evaluate(self.champion, x_t, z_t, z_t, self.h_champ, self.h_delay_buffer)
            # Update delay buffer
            self.h_delay_buffer.insert(0, self.h_champ.clone())
            if len(self.h_delay_buffer) > self.max_delay_taps:
                self.h_delay_buffer.pop()
            self.h_champ = h_next
            self.prev_latent = z_t.detach().clone()
            self.last_update_time = current_time
            return {
                "ola_mode": "StabilizedTick",
                "ola_loss": float("nan"),
                "ema_loss": self.ema_loss if self.ema_loss else 0.0,
                "ema_score": self.ema_score if self.ema_score else 0.0,
                "mutation_rate": self.mutation_rate,
                "mutation_decay": self.mutation_decay,
                "frame_time": time_delta,
                "champion_score": self.champion_score,
                "state_dim": float(self.champion.state_dim),
                "rollback_count": self._rollback_count,
                "loss_variance": 0.0,
                "no_improve_age": 0,
                "temporal_penalty": 0.0,
                "relative_loss": 0.0,
            }

        # Backup previous champion for rollback
        if self.prev_champion is None:
            self.prev_champion = EvoCell(self.champion.in_dim, self.champion.out_dim,
                                         self.champion.state_dim).to(self.device)
        self.prev_champion.load_state_dict(self.champion.state_dict())
        self.prev_h_champ = self.h_champ.clone()
        self.prev_h_delay_buffer = [h.clone() for h in self.h_delay_buffer]

        # Evaluate current champion (single-step loss) with delayed states
        loss, h_next = self._evaluate(self.champion, x_t, z_t, z_tm1, self.h_champ, self.h_delay_buffer)

        # Multi-step temporal consistency loss
        self._multistep_loss = self._evaluate_multistep(self.champion, self.h_champ)

        # Combined loss: L = MSE(t, t+1) + λ * MSE_multistep
        total_loss = loss + self.cfg.multistep_weight * self._multistep_loss

        # Temporal regularization: penalize large changes in latent space
        if self.prev_latent is not None:
            delta_z_pred, _ = self.champion(x_t, self.h_champ, self.h_delay_buffer)
            pred_latent = z_tm1 + delta_z_pred
            temporal_diff = torch.mean(torch.abs(pred_latent - self.prev_latent))
            self._temporal_penalty = self.cfg.temporal_beta * float(temporal_diff.item())
        else:
            self._temporal_penalty = 0.0

        # Normalized loss: relative improvement over previous MSE
        if self.prev_mse is not None:
            self._relative_loss = (self.prev_mse - self._last_mse) / (self.prev_mse + 1e-8)
        else:
            self._relative_loss = 0.0

        # Compute fitness: relative improvement - temporal penalty - multistep penalty
        fitness = self._relative_loss - self._temporal_penalty - self.cfg.multistep_weight * self._multistep_loss

        # Update adaptive mutation rate
        self._update_mutation_rate(loss)

        # Update EMA (use total_loss which includes multi-step)
        if self.ema_loss is None:
            self.ema_loss = total_loss
            self.ema_score = fitness
            self.prev_mse = self._last_mse
        else:
            self.ema_loss = self.stability_factor * self.ema_loss + (1 - self.stability_factor) * total_loss
            self.ema_score = self.stability_factor * self.ema_score + (1 - self.stability_factor) * fitness

        # Rollback protection: revert if loss explodes
        if self.ema_loss is not None and total_loss > self.cfg.rollback_threshold * self.ema_loss:
            # Catastrophic drift detected - rollback
            self.champion.load_state_dict(self.prev_champion.state_dict())
            self.h_champ = self.prev_h_champ.clone()
            self.h_delay_buffer = [h.clone() for h in self.prev_h_delay_buffer]
            # Increase exploration temporarily
            self.mutation_rate *= 1.1
            self.mutation_rate = min(self.mutation_rate, 0.5)  # cap at 0.5
            self._rollback_count += 1
            print(f"[StabilizedOLA] Rollback at step {self.step_idx}: total_loss={total_loss:.4f} > {self.cfg.rollback_threshold}×ema={self.ema_loss:.4f}")
        else:
            # Accept new state and update delay buffer
            self.h_delay_buffer.insert(0, self.h_champ.clone())
            if len(self.h_delay_buffer) > self.max_delay_taps:
                self.h_delay_buffer.pop()
            self.h_champ = h_next

            # Track age and check for reset
            if fitness > self.best_fitness_ever:
                self.best_fitness_ever = fitness
                self.no_improve_age = 0
                # Add to champion hall
                self._update_champion_hall(self.champion, fitness)
            else:
                self.no_improve_age += 1

            # Age-based reset: if no improvement for max_no_improve_age ticks, reset genome
            if self.no_improve_age >= self.cfg.max_no_improve_age:
                # Try to use a champion from hall as baseline
                baseline = self._get_champion_baseline()
                if baseline is not None:
                    self.champion.load_state_dict(baseline.state_dict())
                    print(f"[StabilizedOLA] Age reset at step {self.step_idx}: using champion baseline")
                else:
                    # Full reset
                    self.champion = EvoCell(self.cfg.in_dim, self.cfg.out_dim,
                                           self.cfg.state_dim).to(self.device)
                    print(f"[StabilizedOLA] Age reset at step {self.step_idx}: fresh genome")
                self.no_improve_age = 0
                self.mutation_rate = self.cfg.mutation_rate  # reset mutation rate

            # Check if we should evolve (fitness improvement)
            if fitness > self.ema_score or self.champion_score == float("-inf"):
                # Create candidate by mutating champion (or champion hall baseline)
                mutation_baseline = self._get_champion_baseline() if len(self.champion_hall) > 0 else self.champion

                candidate = EvoCell(mutation_baseline.in_dim, mutation_baseline.out_dim,
                                   mutation_baseline.state_dim).to(self.device)
                candidate.load_state_dict(mutation_baseline.state_dict())
                candidate.mutate(self.mutation_rate, self.cfg.grow_prob, self.cfg.max_state_dim)

                # Evaluate candidate (single-step) with delayed states
                cand_loss, cand_h = self._evaluate(candidate, x_t, z_t, z_tm1, self.h_champ, self.h_delay_buffer)

                # Evaluate candidate (multi-step)
                cand_multistep_loss = self._evaluate_multistep(candidate, self.h_champ)

                # Combined candidate loss
                cand_total_loss = cand_loss + self.cfg.multistep_weight * cand_multistep_loss

                # Compute candidate fitness with temporal reg and multistep
                cand_relative_loss = (self.prev_mse - self._last_mse) / (self.prev_mse + 1e-8) if self.prev_mse else 0.0
                cand_fitness = cand_relative_loss - self._temporal_penalty - self.cfg.multistep_weight * cand_multistep_loss

                # Replace if better fitness
                if cand_fitness > fitness:
                    self.champion = candidate
                    self.h_champ = cand_h
                    self.champion_score = cand_fitness
                    fitness = cand_fitness
                    loss = cand_loss
                    total_loss = cand_total_loss
                    self._multistep_loss = cand_multistep_loss

        # Update for next tick
        self.prev_mse = self._last_mse
        self.prev_latent = z_t.detach().clone()
        self.last_update_time = current_time

        return {
            "ola_mode": "StabilizedTick",
            "ola_loss": loss,
            "total_loss": total_loss,
            "multistep_loss": self._multistep_loss,
            "ema_loss": self.ema_loss,
            "ema_score": self.ema_score,
            "mutation_rate": self.mutation_rate,
            "mutation_decay": self.mutation_decay,
            "frame_time": time_delta,
            "champion_score": self.champion_score,
            "state_dim": float(self.champion.state_dim),
            "mse_best": self._last_mse,
            "cos_best": self._last_cos,
            "fitness": fitness,
            "rollback_count": self._rollback_count,
            "loss_variance": self.loss_variance,
            "no_improve_age": self.no_improve_age,
            "temporal_penalty": self._temporal_penalty,
            "relative_loss": self._relative_loss,
            "num_champions": len(self.champion_hall),
        }

    @torch.no_grad()
    def predict(self, x_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """Predict next latent using champion with temporal context"""
        delta, _ = self.champion(x_t, self.h_champ, self.h_delay_buffer)
        return z_t + delta

    def save_best_genome(self, path: str, metadata: Optional[Dict] = None) -> None:
        """
        Save the current champion genome to a file for visualization and analysis.

        Args:
            path: File path to save the genome (e.g., 'genomes/best_genome.pt')
            metadata: Optional dictionary of additional metadata to save with the genome
        """
        import os
        from typing import Dict, Optional

        save_dict = {
            'champion_state_dict': self.champion.state_dict(),
            'config': {
                'in_dim': self.cfg.in_dim,
                'out_dim': self.cfg.out_dim,
                'state_dim': self.champion.state_dim,
                'mutation_rate': self.mutation_rate,
                'mutation_decay': self.cfg.mutation_decay,
                'mutation_floor': self.cfg.mutation_floor,
                'stability_factor': self.cfg.stability_factor,
                'rollback_threshold': self.cfg.rollback_threshold,
                'num_champions': self.cfg.num_champions,
            },
            'genome_info': {
                'score': self.champion_score,
                'step_idx': self.step_idx,
                'state_dim': self.champion.state_dim,
                'ema_loss': self.ema_loss if self.ema_loss else 0.0,
                'ema_score': self.ema_score if self.ema_score else 0.0,
                'mutation_rate': self.mutation_rate,
            },
        }

        # Add custom metadata
        if metadata:
            save_dict['metadata'] = metadata

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save(save_dict, path)
        print(f"[StabilizedOLA] Saved champion genome (score={self.champion_score:.6f}) to {path}")

    def _init_population(self):
        """Reset to fresh genome (for compatibility with reset_state)"""
        self.champion = EvoCell(self.cfg.in_dim, self.cfg.out_dim,
                               self.cfg.state_dim).to(self.device)
        self.h_champ = torch.zeros(1, self.cfg.state_dim, device=self.device)
        self.prev_champion = None
        self.prev_h_champ = None
        self.ema_loss = None
        self.ema_score = None
        self.last_update_time = time.time()
        self.mutation_rate = self.cfg.mutation_rate
        self.step_idx = 0
        self.champion_score = float("-inf")
        self._rollback_count = 0
