"""
Action OLA: Modified OLA specifically for observation-to-action mapping

Unlike standard OLA which predicts latent space deltas, Action OLA learns
to map observations directly to action vectors for control tasks.
"""
import torch
import torch.nn.functional as F
from typing import Dict
from dataclasses import dataclass

from ola import OrganicLogicAgent, OLAConfig, EvoCell


@dataclass
class ActionOLAConfig(OLAConfig):
    """Configuration for action-prediction OLA"""
    # Fitness shaping
    reward_scale: float = 10.0  # Scale environment rewards for fitness
    action_penalty: float = 0.001  # Penalty for large actions


class ActionOLA:
    """
    OLA variant that learns to predict actions from observations.

    Key differences from standard OLA:
    - Takes observations as input
    - Outputs action vectors (not latent deltas)
    - Fitness based on environment rewards (not prediction error)
    - Simplified training: no temporal consistency constraints
    """

    def __init__(self, cfg: ActionOLAConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Use standard OLA but interpret differently
        self.ola = OrganicLogicAgent(cfg)

        # Performance tracking
        self.total_reward = 0.0
        self.step_count = 0

    @torch.no_grad()
    def predict_action(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Predict action from observation using best genome.

        Args:
            observation: [1, obs_dim] observation tensor

        Returns:
            action: [1, action_dim] predicted action tensor
        """
        # Use best genome directly
        if self.ola.best is None:
            # Cold start: use first genome
            g = self.ola.pop[0]
            state = self.ola.states[0:1]
        else:
            g = self.ola.best
            state = self.ola.states[0:1]  # Use first state slot

        # Forward pass to get action (delta output)
        delta, _ = g.cell(observation, state)

        # Apply tanh to bound actions to [-1, 1]
        action = torch.tanh(delta)

        return action

    def step(self, observation: torch.Tensor, reward: float) -> Dict:
        """
        Evolutionary step with environment feedback.

        Args:
            observation: [1, obs_dim] current observation
            reward: scalar reward from environment

        Returns:
            metrics: dict of training metrics
        """
        self.step_count += 1
        self.total_reward += reward

        # Predict action
        action = self.predict_action(observation)

        # Fitness is based on reward, not prediction error
        # We want to maximize reward, so fitness = reward
        fitness = reward * self.cfg.reward_scale

        # Add small penalty for large actions (energy cost)
        action_magnitude = torch.mean(action ** 2).item()
        fitness -= action_magnitude * self.cfg.action_penalty

        # Update OLA scores manually (bypass normal prediction error scoring)
        # Set all genome scores to fitness (they all produce same action for this obs)
        for genome in self.ola.pop:
            genome.score = fitness

        # Evolve population
        self.ola._evolve()

        # Update best
        best_idx = max(range(len(self.ola.pop)), key=lambda i: self.ola.pop[i].score)
        self.ola.best = self.ola.pop[best_idx].clone()

        return {
            'ola_loss': -fitness,  # Negative fitness for consistency with loss
            'ola_best': fitness,
            'ola_state_dim': float(self.ola.pop[0].cell.state_dim),
            'reward': reward,
            'total_reward': self.total_reward,
            'action_magnitude': action_magnitude,
            'fitness': fitness,
        }

    def save_best_genome(self, path: str, metadata: Dict = None):
        """Save the best genome"""
        self.ola.save_best_genome(path, metadata)
