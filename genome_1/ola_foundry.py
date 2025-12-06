# ola_foundry.py
"""
Thin wrapper around StabilizedOLA for genome generation.
Maintains a single champion EvoCell and produces candidate genomes via mutation.
Phase 0: mutation only at foundry level, not on replication.
Phase 1: RoR fitness tracking, founder diversity, immigrant injection.
"""
import torch
import numpy as np
from typing import Dict, List
from stabilized_ola import StabilizedOLA, StabilizedOLAConfig
from ola import EvoCell


class OLAFoundry:
    """
    Genome foundry using StabilizedOLA.
    - Maintains a champion genome
    - Emits genomes (clones or mutated candidates)
    - Tracks internal fitness via stable observations
    - Phase 1: RoR fitness tracking per lineage
    """
    def __init__(self, cfg: StabilizedOLAConfig):
        self.ola = StabilizedOLA(cfg)
        self.emit_counter = 0
        self.cfg = cfg

        # Phase 1: RoR fitness tracking per lineage
        self.lineage_births = {}  # lineage_id -> birth count
        self.lineage_R_spent = {}  # lineage_id -> total R spent
        self.lineage_RoR = {}  # lineage_id -> RoR (births / R_spent)
        self.RoR_ema = 0.0  # EMA of best RoR
        self.RoR_ema_alpha = 0.01

        # Founder pool for diversity (Phase 1)
        self.founder_pool = []  # List[EvoCell]
        self.next_lineage_id = 0

    def emit_genome(self, every_k: int = 5) -> EvoCell:
        """
        Emit a genome for spawning.
        Most calls return a clone of current champion.
        Every K-th call returns a mutated candidate.
        """
        self.emit_counter += 1

        if self.emit_counter % every_k == 0:
            # Mutate and return candidate
            candidate = EvoCell(self.ola.champion.in_dim,
                               self.ola.champion.out_dim,
                               self.ola.champion.state_dim).to(self.ola.device)
            candidate.load_state_dict(self.ola.champion.state_dict())
            candidate.mutate(self.ola.mutation_rate, self.cfg.grow_prob, self.cfg.max_state_dim)
            return candidate
        else:
            # Clone champion
            clone = EvoCell(self.ola.champion.in_dim,
                           self.ola.champion.out_dim,
                           self.ola.champion.state_dim).to(self.ola.device)
            clone.load_state_dict(self.ola.champion.state_dict())
            return clone

    def tick(self, pooled_resource: float, live_count: int):
        """
        Feed a stable observation to OLA to keep its mutation logic alive.
        We use pooled resource stats and live count to build a dummy vector.
        """
        # Build stable x and z vectors (same dim as OLA in/out)
        # Use simple stats as dummy inputs
        x_t = torch.zeros(1, self.cfg.in_dim, device=self.ola.device)
        x_t[0, 0] = pooled_resource
        x_t[0, 1] = live_count / 1000.0  # normalize

        z_t = torch.zeros(1, self.cfg.out_dim, device=self.ola.device)
        z_t[0, 0] = pooled_resource
        z_t[0, 1] = live_count / 1000.0

        # Previous z (just use same for stability)
        z_tm1 = z_t.clone() if self.ola.prev_latent is not None else None

        # Step OLA
        metrics = self.ola.step(x_t, z_t, z_tm1)
        return metrics

    def get_metrics(self) -> Dict[str, float]:
        """Get OLA metrics for HUD"""
        return {
            "mutation_rate": self.ola.mutation_rate,
            "ema_loss": self.ola.ema_loss if self.ola.ema_loss else 0.0,
            "state_dim": float(self.ola.champion.state_dim)
        }

    def save_best_genome(self, path: str):
        """Save current champion genome"""
        self.ola.save_best_genome(path)

    def init_founder_pool(self, num_founders: int = 8, phase: int = 0):
        """
        Initialize founder pool with diverse genomes using farthest-point sampling.
        Phase 1 feature: ensures initial diversity.
        """
        if phase == 0:
            # Phase 0: just add champion as single founder
            self.founder_pool = [self._clone_genome(self.ola.champion)]
            return

        # Phase 1: farthest-point sampling for diversity
        self.founder_pool = []
        candidates = []

        # Generate N candidate genomes via mutation
        for _ in range(num_founders * 4):
            candidate = self._clone_genome(self.ola.champion)
            candidate.mutate(self.ola.mutation_rate, self.cfg.grow_prob, self.cfg.max_state_dim)
            candidates.append(candidate)

        # Farthest-point sampling: pick diverse subset
        # Start with a random candidate
        import random
        selected_indices = [random.randint(0, len(candidates) - 1)]
        self.founder_pool.append(candidates[selected_indices[0]])

        # Iteratively pick the candidate farthest from selected set
        for _ in range(num_founders - 1):
            max_min_dist = -1
            best_idx = 0

            for i, candidate in enumerate(candidates):
                if i in selected_indices:
                    continue

                # Compute min distance to selected set
                min_dist = min(
                    self._genome_distance(candidate, self.founder_pool[j])
                    for j in range(len(self.founder_pool))
                )

                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i

            selected_indices.append(best_idx)
            self.founder_pool.append(candidates[best_idx])

        print(f"[Foundry] Initialized {len(self.founder_pool)} diverse founders")

    def _clone_genome(self, source: EvoCell) -> EvoCell:
        """Clone a genome"""
        clone = EvoCell(source.in_dim, source.out_dim, source.state_dim).to(self.ola.device)
        clone.load_state_dict(source.state_dict())
        return clone

    def _genome_distance(self, g1: EvoCell, g2: EvoCell) -> float:
        """
        Compute L2 distance between two genomes in weight space.
        Simple metric for farthest-point sampling.
        """
        dist = 0.0
        with torch.no_grad():
            for p1, p2 in zip(g1.parameters(), g2.parameters()):
                dist += torch.sum((p1 - p2) ** 2).item()
        return np.sqrt(dist)

    def emit_immigrant(self) -> EvoCell:
        """
        Emit a new immigrant genome for injection.
        Phase 1: mutated founder from pool.
        """
        if len(self.founder_pool) == 0:
            # Fallback: mutate champion
            immigrant = self._clone_genome(self.ola.champion)
            immigrant.mutate(self.ola.mutation_rate * 2.0, self.cfg.grow_prob, self.cfg.max_state_dim)
            return immigrant

        # Pick random founder and mutate heavily
        import random
        founder = random.choice(self.founder_pool)
        immigrant = self._clone_genome(founder)
        immigrant.mutate(self.ola.mutation_rate * 2.0, self.cfg.grow_prob, self.cfg.max_state_dim)
        return immigrant

    def update_lineage_stats(self, lineage_id: int, births: int, R_spent: float):
        """
        Update lineage stats for RoR fitness tracking.
        Phase 1: tracks births and resource spent per lineage.
        """
        if lineage_id not in self.lineage_births:
            self.lineage_births[lineage_id] = 0
            self.lineage_R_spent[lineage_id] = 0.0
            self.lineage_RoR[lineage_id] = 0.0

        self.lineage_births[lineage_id] += births
        self.lineage_R_spent[lineage_id] += R_spent

        # Compute RoR = births / R_spent
        if self.lineage_R_spent[lineage_id] > 0:
            self.lineage_RoR[lineage_id] = self.lineage_births[lineage_id] / self.lineage_R_spent[lineage_id]

        # Update RoR EMA (best RoR)
        if len(self.lineage_RoR) > 0:
            best_RoR = max(self.lineage_RoR.values())
            if self.RoR_ema == 0.0:
                self.RoR_ema = best_RoR
            else:
                self.RoR_ema = (1 - self.RoR_ema_alpha) * self.RoR_ema + self.RoR_ema_alpha * best_RoR

    def compute_simpson_diversity(self, lineage_counts: Dict[int, int]) -> float:
        """
        Compute Simpson diversity index: 1 - Σ(p_i^2)
        where p_i is the proportion of lineage i.
        Returns value in [0, 1], higher = more diverse.
        """
        if len(lineage_counts) == 0:
            return 0.0

        total = sum(lineage_counts.values())
        if total == 0:
            return 0.0

        simpson_sum = sum((count / total) ** 2 for count in lineage_counts.values())
        return 1.0 - simpson_sum

    def anneal_mutation_rate(self, diversity: float, min_rate: float = 0.01, max_rate: float = 0.2):
        """
        Anneal mutation rate based on diversity.
        Low diversity -> high mutation (exploration)
        High diversity -> low mutation (exploitation)
        """
        # Inverse relationship: low diversity -> high mutation
        target_rate = max_rate - (max_rate - min_rate) * diversity
        target_rate = np.clip(target_rate, min_rate, max_rate)

        # Smooth transition with EMA
        alpha = 0.05
        self.ola.mutation_rate = (1 - alpha) * self.ola.mutation_rate + alpha * target_rate
