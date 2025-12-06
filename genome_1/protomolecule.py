# protomolecule.py
"""
ProtoMolecule agent with EvoCell genome.
Tick logic: observe local environment, compute replication readiness, attempt replication or move.
Phase 0: replication only (no mutation on spawn).
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from ola import EvoCell


class ProtoMolecule:
    """
    A single proto-molecule agent.
    State: pos (grid coords), energy, genome (EvoCell), hidden state h.
    """
    def __init__(self, genome: EvoCell, pos: Tuple[int, int], energy: float,
                 device: torch.device, lineage_depth: int = 0,
                 parent_pos: Optional[Tuple[int, int]] = None,
                 birth_tick: int = 0, rng=None, lineage_id: int = 0):
        self.genome = genome
        self.pos = pos
        self.energy = energy
        self.device = device

        # Lineage tracking
        self.lineage_depth = lineage_depth
        self.parent_pos = parent_pos
        self.birth_tick = birth_tick
        self.lineage_id = lineage_id  # Founder ID for diversity tracking

        # Hidden state for genome
        self.h = torch.zeros(1, genome.state_dim, device=device)

        # Replication head: tiny linear layer to compute replication score from hidden state
        self.rep_head = nn.Linear(genome.state_dim, 1).to(device)

        # Energy and resource constants
        self.E_MAX = 1.0
        self.energy_move = 0.002
        self.energy_replication = 0.02
        self.R_replicate_cost = 0.12

        # Phase 0 throttles: threshold jitter and per-parent cap
        import random
        if rng is None:
            rng = random.Random()
        self.s_threshold = 0.6 + rng.uniform(-0.02, 0.02)  # ±0.02 jitter
        self.births_this_tick = 0

    def observe_and_step(self, env, agents_list) -> Tuple[float, bool]:
        """
        Build observation, run genome forward, return (replication_score, is_dead).
        """
        # Reset per-tick birth counter
        self.births_this_tick = 0

        # Get observation from environment
        obs_np = env.get_observation_stencil(self.pos[0], self.pos[1], agents_list)
        x_t = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)

        # Forward pass through genome
        delta, h_next = self.genome(x_t, self.h, h_delayed=None)
        self.h = h_next

        # Compute replication score
        with torch.no_grad():
            s = torch.sigmoid(self.rep_head(h_next)).item()

        # Check if agent is dead (energy <= 0)
        is_dead = self.energy <= 0.0

        return s, is_dead

    def attempt_replication(self, env, agents_list, current_tick: int) -> Optional['ProtoMolecule']:
        """
        Try to replicate if conditions are met:
        - Per-parent cap: max 1 child per tick
        - Tile not on cooldown
        - R_center >= R_replicate_cost
        - replication score s >= s_threshold
        Returns new child ProtoMolecule or None.
        """
        gx, gy = self.pos

        # Phase 0 throttles
        if self.births_this_tick >= 1:
            return None  # Already birthed this tick

        if not env.can_birth_at(gx, gy):
            return None  # Tile on cooldown

        # Check resource availability
        if not env.consume_resource(gx, gy, self.R_replicate_cost):
            return None

        # Find a free adjacent water tile
        neighbors = env.get_neighbors(gx, gy)
        water_neighbors = [(x, y) for x, y in neighbors if env.is_water(x, y)]

        # Filter out occupied tiles
        occupied = {agent.pos for agent in agents_list}
        free_tiles = [pos for pos in water_neighbors if pos not in occupied]

        if len(free_tiles) == 0:
            # No free tile, refund resource
            env.add_resource(gx, gy, self.R_replicate_cost)
            return None

        # Choose random free tile
        import random
        child_pos = random.choice(free_tiles)

        # Clone genome and maybe mutate (Phase 1: p=0.05)
        child_genome = EvoCell(self.genome.in_dim, self.genome.out_dim, self.genome.state_dim).to(self.device)
        child_genome.load_state_dict(self.genome.state_dict())

        # Phase 1 child mutation: p=0.05, σ=0.01
        if hasattr(self, 'phase') and self.phase == 1:
            import random
            if random.random() < 0.05:
                # Mutate with small sigma=0.01
                child_genome.mutate(p=0.01, grow_prob=0.0, max_dim=256)

        # Create child
        child = ProtoMolecule(
            genome=child_genome,
            pos=child_pos,
            energy=max(0.0, self.energy - self.energy_replication),
            device=self.device,
            lineage_depth=self.lineage_depth + 1,
            parent_pos=self.pos,
            birth_tick=current_tick,
            lineage_id=self.lineage_id  # Inherit parent's lineage ID
        )

        # Parent pays energy cost and increments birth counter
        self.energy = max(0.0, self.energy - self.energy_replication)
        self.births_this_tick += 1

        # Set tile cooldown (8-12 ticks)
        env.set_birth_cooldown(gx, gy, ticks=10)

        return child

    def attempt_move(self, env, agents_list):
        """
        Attempt a small random move to an adjacent water tile.
        Costs trivial energy.
        """
        gx, gy = self.pos
        neighbors = env.get_neighbors(gx, gy)
        water_neighbors = [(x, y) for x, y in neighbors if env.is_water(x, y)]

        # Filter out occupied tiles
        occupied = {agent.pos for agent in agents_list}
        free_tiles = [pos for pos in water_neighbors if pos not in occupied]

        if len(free_tiles) > 0:
            import random
            new_pos = random.choice(free_tiles)
            self.pos = new_pos
            self.energy = max(0.0, self.energy - self.energy_move)

    def decompose(self, env) -> float:
        """
        Return 100% of remaining energy to neighboring tiles as resource.
        Phase 0: perfect recycling, no entropy leak.
        Returns total resource returned for mass conservation tracking.
        """
        gx, gy = self.pos
        decomp_yield = max(0.0, self.energy)  # 100% return in Phase 0
        neighbors = env.get_neighbors(gx, gy)

        if len(neighbors) > 0:
            per_neighbor = decomp_yield / len(neighbors)
            for nx, ny in neighbors:
                if env.is_water(nx, ny):
                    env.add_resource(nx, ny, per_neighbor)

        return decomp_yield
