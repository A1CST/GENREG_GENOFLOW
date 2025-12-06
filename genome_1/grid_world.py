"""
GridWorld: Competitive Territory Control Environment for Dual OLAs
Each OLA controls a region and competes for grid coverage through continuous actions.
"""
import numpy as np
import torch
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class GridWorldConfig:
    """Configuration for the competitive grid environment"""
    grid_width: int = 64
    grid_height: int = 64
    action_dim: int = 6  # [action_x, action_y, speed, wait, expand, contract]
    observation_dim: int = 128  # Compressed observation size
    max_speed: float = 2.0
    max_radius: float = 5.0
    min_radius: float = 1.0
    expansion_rate: float = 0.1
    decay_rate: float = 0.001  # Small decay for entropy pressure
    collision_bonus: float = 0.5  # Reward for successful invasion
    device: str = "cuda"


class GridWorld:
    """
    Competitive grid environment where two OLAs compete for territory.

    Key mechanics:
    - Continuous action space: [dx, dy, speed, wait, expand, contract]
    - Territory ownership tracked per-pixel
    - No death mechanic - continuous evolution
    - Fitness = occupied tiles + expansion rate + conflict resolution
    """

    def __init__(self, config: GridWorldConfig):
        self.cfg = config
        self.device = torch.device(config.device)

        # Grid state: 0 = empty, 1 = agent 0, 2 = agent 1
        self.grid = np.zeros((config.grid_height, config.grid_width), dtype=np.int32)

        # Agent states
        self.agent_positions = np.array([
            [config.grid_width * 0.25, config.grid_height * 0.5],  # Agent 0 left side
            [config.grid_width * 0.75, config.grid_height * 0.5],  # Agent 1 right side
        ], dtype=np.float32)

        self.agent_radii = np.array([2.5, 2.5], dtype=np.float32)
        self.agent_velocities = np.zeros((2, 2), dtype=np.float32)

        # Territory tracking
        self.territory_counts = np.zeros(2, dtype=np.int32)  # Tiles owned per agent
        self.territory_history = [[], []]  # Track expansion rate
        self.collision_events = np.zeros(2, dtype=np.int32)  # Successful invasions

        # Performance metrics
        self.tick = 0
        self.prev_territory_counts = np.zeros(2, dtype=np.int32)

        # Initialize starting territories
        self._update_territories()

    def reset(self):
        """Reset environment to initial state"""
        self.grid.fill(0)
        self.agent_positions = np.array([
            [self.cfg.grid_width * 0.25, self.cfg.grid_height * 0.5],
            [self.cfg.grid_width * 0.75, self.cfg.grid_height * 0.5],
        ], dtype=np.float32)
        self.agent_radii = np.array([2.5, 2.5], dtype=np.float32)
        self.agent_velocities = np.zeros((2, 2), dtype=np.float32)
        self.territory_counts = np.zeros(2, dtype=np.int32)
        self.territory_history = [[], []]
        self.collision_events = np.zeros(2, dtype=np.int32)
        self.tick = 0
        self.prev_territory_counts = np.zeros(2, dtype=np.int32)
        self._update_territories()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Execute one environment step with actions from both agents.

        Args:
            actions: [2, action_dim] continuous action vectors for each agent
                    [action_x, action_y, speed, wait, expand, contract]

        Returns:
            observations: [2, obs_dim] compressed observations for each agent
            rewards: [2] fitness scores for each agent
            info: dict with diagnostic information
        """
        self.tick += 1

        # Parse actions into interpretable behaviors
        actions = np.tanh(actions)  # Normalize to [-1, 1]

        for agent_id in range(2):
            action = actions[agent_id]

            # Decompose action vector
            action_x = action[0]
            action_y = action[1]
            speed = (action[2] + 1) * 0.5 * self.cfg.max_speed  # [0, max_speed]
            wait = action[3]  # Negative = active, positive = wait
            expand = action[4]
            contract = action[5]

            # Apply wait logic: if wait > 0, reduce movement (planning phase)
            if wait > 0:
                speed *= (1.0 - wait * 0.5)  # Reduce speed by up to 50%

            # Update velocity with directional force
            direction = np.array([action_x, action_y])
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                direction = direction / direction_norm

            self.agent_velocities[agent_id] = direction * speed

            # Update position
            self.agent_positions[agent_id] += self.agent_velocities[agent_id]

            # Boundary wrapping (toroidal world)
            self.agent_positions[agent_id, 0] %= self.cfg.grid_width
            self.agent_positions[agent_id, 1] %= self.cfg.grid_height

            # Update radius (expand/contract)
            radius_delta = (expand - contract) * self.cfg.expansion_rate
            self.agent_radii[agent_id] += radius_delta
            self.agent_radii[agent_id] = np.clip(
                self.agent_radii[agent_id],
                self.cfg.min_radius,
                self.cfg.max_radius
            )

        # Update territory ownership
        self._update_territories()

        # Calculate fitness (rewards)
        rewards = self._calculate_fitness()

        # Generate observations
        observations = self._generate_observations()

        # Diagnostic info
        info = {
            'tick': self.tick,
            'territory_counts': self.territory_counts.copy(),
            'agent_positions': self.agent_positions.copy(),
            'agent_radii': self.agent_radii.copy(),
            'collision_events': self.collision_events.copy(),
            'expansion_rates': self._get_expansion_rates(),
        }

        return observations, rewards, info

    def _update_territories(self):
        """
        Update grid ownership based on agent positions and radii.

        PAINTING MECHANIC:
        - Agents permanently paint cells as they pass over them
        - Once painted, cells stay that color (can be overwritten by opponent)
        - Goal: paint as much of the board as possible in your color
        """
        prev_counts = self.territory_counts.copy()

        # Don't clear grid - territories are PERSISTENT!
        # Grid persists between frames, creating a painting trail

        # Draw each agent's territory as a circle (painting new cells)
        y_coords, x_coords = np.ogrid[0:self.cfg.grid_height, 0:self.cfg.grid_width]

        for agent_id in range(2):
            pos = self.agent_positions[agent_id]
            radius = self.agent_radii[agent_id]

            # Calculate distance from agent center (with toroidal wrapping)
            dx = np.minimum(
                np.abs(x_coords - pos[0]),
                self.cfg.grid_width - np.abs(x_coords - pos[0])
            )
            dy = np.minimum(
                np.abs(y_coords - pos[1]),
                self.cfg.grid_height - np.abs(y_coords - pos[1])
            )
            dist = np.sqrt(dx**2 + dy**2)

            # Paint territory within radius
            mask = dist <= radius

            # Count cells being overwritten (invasion of opponent's territory)
            opponent_id = 2 - agent_id  # 0->2, 1->1 (wait this needs fix)
            opponent_value = (1 - agent_id) + 1  # Agent 0 -> check for 2, Agent 1 -> check for 1
            overlap_mask = mask & (self.grid == opponent_value)
            if overlap_mask.any():
                # Count successful invasions (painting over opponent)
                invaded_tiles = overlap_mask.sum()
                self.collision_events[agent_id] += invaded_tiles

            # Paint cells (overwrite anything, including opponent's color)
            self.grid[mask] = agent_id + 1

        # Update territory counts (total painted area)
        self.territory_counts[0] = (self.grid == 1).sum()
        self.territory_counts[1] = (self.grid == 2).sum()

        # Calculate territory gained this frame
        territory_gained = self.territory_counts - prev_counts

        # Track history for expansion rate calculation
        for agent_id in range(2):
            self.territory_history[agent_id].append(self.territory_counts[agent_id])
            if len(self.territory_history[agent_id]) > 20:
                self.territory_history[agent_id].pop(0)

        self.prev_territory_counts = prev_counts

    def _calculate_fitness(self) -> np.ndarray:
        """
        Calculate fitness for each agent based on:
        1. Total painted territory (primary - maximize coverage)
        2. Painting new cells (bonus for exploration)
        3. Painting over opponent's cells (invasion bonus)
        4. Penalty for not moving (encourage exploration)

        Goal: Paint as much of the board as possible!
        """
        rewards = np.zeros(2, dtype=np.float32)

        total_tiles = self.cfg.grid_width * self.cfg.grid_height

        for agent_id in range(2):
            # Primary: territory coverage percentage (0-100)
            # This is the main objective: maximize painted area
            coverage = self.territory_counts[agent_id] / total_tiles
            coverage_score = coverage * 100.0  # Scale to 0-100

            # Secondary: reward painting NEW cells (exploration bonus)
            # Encourage agents to seek unpainted areas
            territory_gained = self.territory_counts[agent_id] - self.prev_territory_counts[agent_id]
            exploration_bonus = max(0, territory_gained) * 0.5

            # Tertiary: bonus for painting over opponent (aggressive play)
            invasion_bonus = self.collision_events[agent_id] * self.cfg.collision_bonus

            # Movement penalty: small penalty if velocity is too low
            # Encourages constant movement and exploration
            velocity = np.linalg.norm(self.agent_velocities[agent_id])
            movement_penalty = -0.01 if velocity < 0.1 else 0.0

            # Combined fitness
            rewards[agent_id] = (
                coverage_score +           # 0-100 based on % painted
                exploration_bonus +        # Bonus for finding new territory
                invasion_bonus * 0.1 +     # Small bonus for conquering opponent
                movement_penalty           # Penalty for standing still
            )

        # Reset collision events after scoring
        self.collision_events.fill(0)

        return rewards

    def _get_expansion_rate(self, agent_id: int) -> float:
        """Calculate rate of territory expansion over recent history"""
        history = self.territory_history[agent_id]
        if len(history) < 2:
            return 0.0

        # Linear regression over last 10 ticks
        recent = history[-10:]
        if len(recent) < 2:
            return 0.0

        # Simple slope calculation
        slope = (recent[-1] - recent[0]) / len(recent)
        return float(slope)

    def _get_expansion_rates(self) -> np.ndarray:
        """Get expansion rates for both agents"""
        return np.array([
            self._get_expansion_rate(0),
            self._get_expansion_rate(1)
        ])

    def _generate_observations(self) -> np.ndarray:
        """
        Generate compressed observations for each agent.

        Observation includes:
        - Local grid view (compressed)
        - Own position, radius, velocity
        - Opponent position, radius, velocity
        - Territory statistics
        """
        observations = np.zeros((2, self.cfg.observation_dim), dtype=np.float32)

        for agent_id in range(2):
            opponent_id = 1 - agent_id

            # Normalize values to [-1, 1] range
            pos_norm = self.agent_positions[agent_id] / np.array([
                self.cfg.grid_width, self.cfg.grid_height
            ]) * 2 - 1

            opp_pos_norm = self.agent_positions[opponent_id] / np.array([
                self.cfg.grid_width, self.cfg.grid_height
            ]) * 2 - 1

            radius_norm = (self.agent_radii[agent_id] - self.cfg.min_radius) / (
                self.cfg.max_radius - self.cfg.min_radius
            ) * 2 - 1

            opp_radius_norm = (self.agent_radii[opponent_id] - self.cfg.min_radius) / (
                self.cfg.max_radius - self.cfg.min_radius
            ) * 2 - 1

            vel_norm = self.agent_velocities[agent_id] / self.cfg.max_speed
            opp_vel_norm = self.agent_velocities[opponent_id] / self.cfg.max_speed

            # Territory statistics
            total_tiles = self.cfg.grid_width * self.cfg.grid_height
            territory_ratio = (self.territory_counts[agent_id] -
                             self.territory_counts[opponent_id]) / total_tiles

            expansion_rate = self._get_expansion_rate(agent_id)

            # Extract local grid view (downsampled)
            local_view = self._get_local_view(agent_id, view_size=16)

            # Compose observation vector
            obs = np.concatenate([
                pos_norm,                    # 2
                [radius_norm],              # 1
                vel_norm,                   # 2
                opp_pos_norm,               # 2
                [opp_radius_norm],          # 1
                opp_vel_norm,               # 2
                [territory_ratio],          # 1
                [expansion_rate],           # 1
                local_view.flatten(),       # 16*16 = 256 -> compressed to remaining
            ])

            # Pad or truncate to observation_dim
            if len(obs) > self.cfg.observation_dim:
                # Compress local view further
                obs = obs[:self.cfg.observation_dim]
            else:
                obs = np.pad(obs, (0, self.cfg.observation_dim - len(obs)))

            observations[agent_id] = obs

        return observations

    def _get_local_view(self, agent_id: int, view_size: int = 16) -> np.ndarray:
        """Extract and downsample local grid view centered on agent"""
        pos = self.agent_positions[agent_id].astype(int)
        half_view = view_size // 2

        # Extract local region with wrapping
        local_view = np.zeros((view_size, view_size), dtype=np.float32)

        for i in range(view_size):
            for j in range(view_size):
                x = (pos[0] - half_view + j) % self.cfg.grid_width
                y = (pos[1] - half_view + i) % self.cfg.grid_height

                # Encode: -1 = opponent, 0 = empty, 1 = self
                cell_value = self.grid[y, x]
                if cell_value == 0:
                    local_view[i, j] = 0.0
                elif cell_value == agent_id + 1:
                    local_view[i, j] = 1.0
                else:
                    local_view[i, j] = -1.0

        return local_view

    def get_grid_state(self) -> np.ndarray:
        """Return current grid state for visualization"""
        return self.grid.copy()

    def get_agent_states(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return agent positions, radii, and velocities for visualization"""
        return (
            self.agent_positions.copy(),
            self.agent_radii.copy(),
            self.agent_velocities.copy()
        )
