"""
Competitive OLA: Wrapper for dual OLA agents competing in GridWorld

Integrates stabilized_ola.py with grid_world.py to create co-evolutionary pressure.
Each agent is its own OLA with independent evolution competing for dominance.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

from action_ola import ActionOLA, ActionOLAConfig
from grid_world import GridWorld, GridWorldConfig


@dataclass
class CompetitiveOLAConfig:
    """Configuration for dual competitive OLAs"""
    # GridWorld settings
    grid_width: int = 64
    grid_height: int = 64

    # OLA architecture settings
    state_dim: int = 128
    action_dim: int = 6  # [action_x, action_y, speed, wait, expand, contract]
    observation_dim: int = 128

    # Evolution settings
    pop_size: int = 32
    elite_frac: float = 0.15
    mutation_rate: float = 0.1
    structure_add_prob: float = 0.01
    structure_max_dim: int = 512
    grow_prob: float = 0.005

    # Fitness shaping
    reward_scale: float = 10.0
    action_penalty: float = 0.001
    competitive_bonus: float = 0.5  # Bonus for outperforming opponent

    device: str = "cuda"


class CompetitiveOLA:
    """
    Manages two competing OLA agents in a shared GridWorld environment.

    Each agent:
    - Has its own independent OLA with separate evolution
    - Receives observations from the environment
    - Outputs continuous actions to control movement and territory
    - Evolves based on territory control fitness

    The key innovation: each agent becomes the other's environment,
    creating co-evolutionary pressure for intelligence.
    """

    def __init__(self, config: CompetitiveOLAConfig):
        self.cfg = config
        self.device = torch.device(config.device)

        # Create GridWorld environment
        grid_cfg = GridWorldConfig(
            grid_width=config.grid_width,
            grid_height=config.grid_height,
            action_dim=config.action_dim,
            observation_dim=config.observation_dim,
            device=config.device
        )
        self.env = GridWorld(grid_cfg)

        # Create two independent ActionOLA agents
        self.agents = []
        for agent_id in range(2):
            ola_cfg = ActionOLAConfig(
                in_dim=config.observation_dim,
                out_dim=config.action_dim,
                state_dim=config.state_dim,
                pop_size=config.pop_size,
                elite_frac=config.elite_frac,
                mutation_rate=config.mutation_rate,
                structure_add_prob=config.structure_add_prob,
                structure_max_dim=config.structure_max_dim,
                reward_scale=config.reward_scale,
                action_penalty=config.action_penalty,
                device=config.device
            )
            agent = ActionOLA(ola_cfg)
            self.agents.append(agent)

        # State tracking
        self.prev_observations = [None, None]
        self.step_count = 0

        # Performance tracking
        self.cumulative_rewards = np.zeros(2, dtype=np.float32)
        self.win_counts = np.zeros(2, dtype=np.int32)
        self.avg_territory = [[], []]

        # Co-evolution metrics
        self.dominance_history = []  # Track which agent is dominating over time
        self.strategy_diversity = []  # Track action diversity

    def reset(self):
        """Reset environment and agent states"""
        self.env.reset()
        self.prev_observations = [None, None]
        self.step_count = 0
        self.cumulative_rewards = np.zeros(2, dtype=np.float32)

    def step(self) -> Dict:
        """
        Execute one competitive step:
        1. Get observations from environment
        2. Each OLA predicts actions
        3. Execute actions in environment
        4. Calculate fitness with competitive bonus
        5. Update both OLAs independently
        """
        self.step_count += 1

        # Generate actions from both agents
        with torch.no_grad():
            observations_np, env_rewards, info = self._get_current_state()

            # Convert observations to tensors
            observations = [
                torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
                for obs in observations_np
            ]

            # Get actions from each OLA
            actions = []
            for agent_id, (agent, obs) in enumerate(zip(self.agents, observations)):
                # Use ActionOLA to predict action from observation
                action_pred = agent.predict_action(obs)
                actions.append(action_pred.cpu().numpy().squeeze())

        # Execute actions in environment
        observations_np, env_rewards, info = self.env.step(np.array(actions))

        # Calculate competitive fitness
        fitness_scores = self._calculate_competitive_fitness(env_rewards, info)

        # Update both ActionOLAs with environment feedback
        agent_metrics = []
        for agent_id, agent in enumerate(self.agents):
            obs = observations[agent_id]

            # ActionOLA step: evolution based on environment reward
            metrics = agent.step(obs, fitness_scores[agent_id])

            # Augment metrics with environment info
            metrics['env_reward'] = float(env_rewards[agent_id])
            metrics['competitive_fitness'] = float(fitness_scores[agent_id])
            metrics['territory_count'] = int(info['territory_counts'][agent_id])
            metrics['expansion_rate'] = float(info['expansion_rates'][agent_id])
            metrics['state_dim'] = metrics['ola_state_dim']
            metrics['mutation_rate'] = agent.cfg.mutation_rate
            metrics['rollback_count'] = 0  # ActionOLA doesn't have rollbacks

            agent_metrics.append(metrics)

        # Update tracking
        self.cumulative_rewards += env_rewards
        self.prev_observations = observations

        for agent_id in range(2):
            self.avg_territory[agent_id].append(info['territory_counts'][agent_id])
            if len(self.avg_territory[agent_id]) > 100:
                self.avg_territory[agent_id].pop(0)

        # Track dominance (who's winning)
        if info['territory_counts'][0] > info['territory_counts'][1]:
            dominance = 0
        elif info['territory_counts'][1] > info['territory_counts'][0]:
            dominance = 1
        else:
            dominance = -1  # Tie
        self.dominance_history.append(dominance)
        if len(self.dominance_history) > 200:
            self.dominance_history.pop(0)

        # Track action diversity (exploration metric)
        action_diversity = np.std(actions, axis=1).mean()
        self.strategy_diversity.append(action_diversity)
        if len(self.strategy_diversity) > 100:
            self.strategy_diversity.pop(0)

        # Determine winner based on current territory
        if info['territory_counts'][0] > info['territory_counts'][1]:
            self.win_counts[0] += 1
        elif info['territory_counts'][1] > info['territory_counts'][0]:
            self.win_counts[1] += 1

        # Compile comprehensive metrics
        metrics = {
            'step': self.step_count,
            'agent_0': agent_metrics[0],
            'agent_1': agent_metrics[1],
            'env_info': info,
            'cumulative_rewards': self.cumulative_rewards.copy(),
            'win_counts': self.win_counts.copy(),
            'dominance_balance': self._calculate_dominance_balance(),
            'strategy_diversity': action_diversity,
            'avg_diversity': np.mean(self.strategy_diversity) if self.strategy_diversity else 0,
        }

        return metrics

    def _get_current_state(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Get current observations and rewards from environment"""
        # Generate observations (without stepping)
        observations = self.env._generate_observations()

        # Get current rewards (without stepping)
        rewards = self.env._calculate_fitness()

        # Get current info
        info = {
            'tick': self.env.tick,
            'territory_counts': self.env.territory_counts.copy(),
            'agent_positions': self.env.agent_positions.copy(),
            'agent_radii': self.env.agent_radii.copy(),
            'collision_events': self.env.collision_events.copy(),
            'expansion_rates': self.env._get_expansion_rates(),
        }

        return observations, rewards, info

    def _calculate_competitive_fitness(self, env_rewards: np.ndarray, info: Dict) -> np.ndarray:
        """
        Calculate fitness with competitive bonus.

        Fitness components:
        1. Base environment reward (territory, expansion)
        2. Competitive bonus for outperforming opponent
        3. Relative territory advantage
        """
        fitness = env_rewards.copy()

        # Add competitive bonus for having more territory
        territory_diff = info['territory_counts'][0] - info['territory_counts'][1]
        if territory_diff > 0:
            fitness[0] += self.cfg.competitive_bonus
            fitness[1] -= self.cfg.competitive_bonus * 0.5
        elif territory_diff < 0:
            fitness[1] += self.cfg.competitive_bonus
            fitness[0] -= self.cfg.competitive_bonus * 0.5

        # Bonus for expansion rate advantage
        expansion_diff = info['expansion_rates'][0] - info['expansion_rates'][1]
        fitness[0] += expansion_diff * 0.1
        fitness[1] -= expansion_diff * 0.1

        return fitness

    def _calculate_dominance_balance(self) -> float:
        """
        Calculate how balanced the competition is.

        Returns value in [-1, 1]:
        -1 = Agent 1 completely dominates
        0 = Perfectly balanced
        1 = Agent 0 completely dominates
        """
        if not self.dominance_history:
            return 0.0

        agent_0_wins = sum(1 for d in self.dominance_history if d == 0)
        agent_1_wins = sum(1 for d in self.dominance_history if d == 1)
        total = agent_0_wins + agent_1_wins

        if total == 0:
            return 0.0

        balance = (agent_0_wins - agent_1_wins) / total
        return balance

    def get_grid_visualization_data(self) -> Dict:
        """Get data needed for visualization"""
        grid = self.env.get_grid_state()
        positions, radii, velocities = self.env.get_agent_states()

        return {
            'grid': grid,
            'positions': positions,
            'radii': radii,
            'velocities': velocities,
            'territory_counts': self.env.territory_counts.copy(),
            'step': self.step_count,
        }

    def save_champions(self, path_prefix: str, metadata: Optional[Dict] = None):
        """Save both champion genomes"""
        for agent_id, agent in enumerate(self.agents):
            path = f"{path_prefix}_agent{agent_id}.pt"

            agent_metadata = {
                'agent_id': agent_id,
                'step_count': self.step_count,
                'cumulative_reward': float(self.cumulative_rewards[agent_id]),
                'win_count': int(self.win_counts[agent_id]),
                'avg_territory': float(np.mean(self.avg_territory[agent_id])) if self.avg_territory[agent_id] else 0,
                'dominance_balance': self._calculate_dominance_balance(),
            }

            if metadata:
                agent_metadata.update(metadata)

            agent.save_best_genome(path, agent_metadata)

        print(f"[CompetitiveOLA] Saved champions: {path_prefix}_agent*.pt")
