"""
Simplified genome library for isolated evolution visualization
Manages genome population with trust-weighted mutation
"""
from __future__ import annotations
import torch
from typing import List, Optional, Dict
import numpy as np

from ola_genome import OLAGenome


class GenomeLibrary:
    """
    Dynamic library of OLA genomes with automatic mutation based on trust
    """

    def __init__(self, in_dim: int, out_dim: int, state_dim: int = 128,
                 initial_genomes: int = 8, max_genomes: int = 16,
                 trust_decay: float = 0.991, blacklist_threshold: float = 0.68,
                 mutation_rate: float = 0.20, device: str = "cpu"):
        """
        Initialize genome library

        Args:
            in_dim: Input dimension for genomes
            out_dim: Output dimension for genomes
            state_dim: Recurrent state dimension
            initial_genomes: Number of genomes to start with
            max_genomes: Maximum number of genomes to maintain
            trust_decay: Trust decay factor per tick
            blacklist_threshold: Trust threshold below which genome is mutated
            mutation_rate: Mutation rate for blacklisted genomes
            device: Device to run on
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.state_dim = state_dim
        self.max_genomes = max_genomes
        self.trust_decay = trust_decay
        self.blacklist_threshold = blacklist_threshold
        self.mutation_rate = mutation_rate
        self.device = device

        # Genome pool
        self.genomes: List[OLAGenome] = []
        self.next_genome_id = 0

        # Mutation control (hysteresis, cooldown, throughput cap)
        self.mutation_low_threshold = 0.55  # mutate if below
        self.mutation_safe_threshold = 0.72  # considered safe only if >=
        self.max_mutations_per_check = 2  # cap per check

        # Statistics
        self.total_mutations = 0
        self.total_genomes_created = 0
        self.total_pruned_genomes = 0
        self.prune_cooldown_ticks = 200
        self.last_prune_tick = -10**9

        # Initialize with starting genomes
        for _ in range(initial_genomes):
            self._create_genome()

    def _create_genome(self) -> OLAGenome:
        """Create a new genome and add to library"""
        genome = OLAGenome(
            genome_id=self.next_genome_id,
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            state_dim=self.state_dim,
            trust_decay=self.trust_decay,
            device=self.device
        )
        self.next_genome_id += 1
        self.genomes.append(genome)
        self.total_genomes_created += 1
        return genome

    def add_genome_if_needed(self, force: bool = False) -> Optional[OLAGenome]:
        """
        Add a new genome if library is not at capacity

        Args:
            force: If True, add even if at max capacity (replace lowest trust)

        Returns:
            The new genome, or None if at capacity and not forced
        """
        if len(self.genomes) < self.max_genomes:
            return self._create_genome()
        elif force:
            # Replace genome with lowest trust
            lowest_trust_genome = min(self.genomes, key=lambda g: g.stats.trust_score)
            idx = self.genomes.index(lowest_trust_genome)
            new_genome = self._create_genome()
            self.genomes[idx] = new_genome
            return new_genome
        return None

    def get_genome_by_id(self, genome_id: int) -> Optional[OLAGenome]:
        """Get genome by ID"""
        for genome in self.genomes:
            if genome.genome_id == genome_id:
                return genome
        return None

    def get_highest_trust_genome(self) -> Optional[OLAGenome]:
        """Get genome with highest trust score"""
        if not self.genomes:
            return None
        return max(self.genomes, key=lambda g: g.stats.trust_score)

    def update_all_genomes(self, pattern_vector: torch.Tensor, current_tick: Optional[int] = None,
                           allow_prune: bool = True):
        """
        Update all genomes with the current pattern vector
        This allows each genome to process the same input and update trust

        Args:
            pattern_vector: Pattern embedding from PatternLSTM [1, in_dim]
        """
        for genome in self.genomes:
            # Forward pass
            genome.forward(pattern_vector)
            # Track previous trust
            prev_trust = float(genome.stats.trust_score)
            # Update trust based on consistency
            genome.update_trust()
            # Mark activity when progress/very consistent
            if current_tick is not None:
                if genome.stats.trust_score > prev_trust or genome.stats.consistency_score > 0.9:
                    setattr(genome, 'last_active_tick', int(current_tick))

            # Trust drift: small random decay each tick (~0.0005 ± 0.0001)
            try:
                drift = float(np.random.uniform(0.0003, 0.0004))
            except Exception:
                drift = 0.00035
            genome.stats.trust_score = max(0.0, float(genome.stats.trust_score) - drift)

        # After updates, optionally prune underperformers
        if current_tick is not None and allow_prune:
            self.prune_inactive_genomes(current_tick)

    def check_and_mutate_blacklisted(self):
        """
        Check all genomes for trust < threshold and mutate in-place
        Mutation variance increases with trust deficit

        Returns:
            List of genome IDs that were mutated
        """
        mutated_ids = []

        # Adaptive mutation cap based on current average trust
        trusts_for_cap = [float(g.stats.trust_score) for g in self.genomes]
        avg_trust_for_cap = (sum(trusts_for_cap) / len(trusts_for_cap)) if trusts_for_cap else 0.0
        max_mutation_rate = 0.05
        trust_scaler = max(0.1, 1.0 - float(avg_trust_for_cap))
        base_mutation_rate = float(self.mutation_rate)
        dynamic_mutation_rate = min(base_mutation_rate * trust_scaler, max_mutation_rate)

        # iterate over a snapshot to avoid issues when replacing entries
        for genome in list(self.genomes):
            if len(mutated_ids) >= self.max_mutations_per_check:
                break

            # Cooldown: skip if not yet allowed
            cooldown_until = getattr(genome, 'mutation_cooldown_until', 0)
            if genome.stats.total_ticks < cooldown_until:
                continue

            trust = float(genome.stats.trust_score)

            # Hysteresis: only mutate if below low threshold; safe zone is >= safe threshold
            if trust < self.mutation_low_threshold:
                prev_trust = trust
                trust_deficit = max(0.0, self.mutation_safe_threshold - prev_trust)
                adaptive_rate = min(0.8, dynamic_mutation_rate * (1.5 + trust_deficit * 2.0))
                grow_prob = min(0.3, 0.05 + trust_deficit * 0.5)
                mutation_passes = 2 if trust_deficit > 0.15 else 1

                # If at capacity, replace the global lowest-trust genome for mixing
                if len(self.genomes) >= self.max_genomes:
                    lowest = min(self.genomes, key=lambda g: g.stats.trust_score)
                    idx = self.genomes.index(lowest)
                    new_genome = self._create_genome()
                    self.genomes[idx] = new_genome
                    # Count as a single mutation action
                    self.total_mutations += 1
                    mutated_ids.append(new_genome.genome_id)
                else:
                    for _ in range(mutation_passes):
                        genome.mutate(
                            mutation_rate=adaptive_rate,
                            grow_prob=grow_prob,
                            max_state_dim=self.state_dim
                        )

                    # Set cooldown: 600 ticks from now (per-genome ticks)
                    genome.mutation_cooldown_until = genome.stats.total_ticks + 600

                    genome.reset_trust(initial_trust=0.8)

                    self.total_mutations += mutation_passes
                    mutated_ids.append(genome.genome_id)

        return mutated_ids

    def reset_all_states(self):
        """Reset recurrent states for all genomes"""
        for genome in self.genomes:
            genome.reset_state()

    def get_library_stats(self) -> Dict:
        """Get summary statistics for the entire library"""
        if not self.genomes:
            return {
                'total_genomes': 0,
                'avg_trust': 0.0,
                'min_trust': 0.0,
                'max_trust': 0.0,
                'total_mutations': self.total_mutations,
                'total_genomes_created': self.total_genomes_created,
                'avg_mutation_count': 0.0,
                'avg_consistency': 0.0
            }

        trust_scores = [g.stats.trust_score for g in self.genomes]
        total_ticks = sum(g.stats.total_ticks for g in self.genomes)
        mutation_counts = [g.stats.mutation_count for g in self.genomes]
        consistency_scores = [g.stats.consistency_score for g in self.genomes]

        return {
            'total_genomes': len(self.genomes),
            'avg_trust': sum(trust_scores) / len(trust_scores),
            'min_trust': min(trust_scores),
            'max_trust': max(trust_scores),
            'total_ticks': total_ticks,
            'total_mutations': self.total_mutations,
            'total_genomes_created': self.total_genomes_created,
            'total_pruned': self.total_pruned_genomes,
            'avg_mutation_count': sum(mutation_counts) / len(mutation_counts),
            'avg_consistency': sum(consistency_scores) / len(consistency_scores)
        }

    def get_top_genomes(self, k: int = 5) -> List[OLAGenome]:
        """Get top K genomes by trust score"""
        sorted_genomes = sorted(self.genomes, key=lambda g: g.stats.trust_score, reverse=True)
        return sorted_genomes[:k]

    def prune_inactive_genomes(self, current_tick: int, k: float = 1.2,
                                consistency_threshold: float = 0.8,
                                inactivity_limit: int = 800):
        """
        Selective adaptive pruning:
        A genome is removed if 2 or more of the following are true:
          1. trust < (avg_trust - k * trust_std)
          2. consistency < consistency_threshold
          3. inactive for > inactivity_limit ticks
        """
        if not self.genomes:
            return

        # Pruning cooldown
        if int(current_tick) - int(self.last_prune_tick) < int(self.prune_cooldown_ticks):
            return

        trusts = [float(g.stats.trust_score) for g in self.genomes]
        avg_trust = float(np.mean(trusts)) if trusts else 0.0
        trust_std = float(np.std(trusts)) if trusts else 0.0

        to_prune: List[OLAGenome] = []
        for g in self.genomes:
            inactivity = int(current_tick - int(getattr(g, 'last_active_tick', 0)))
            failures = 0
            if float(g.stats.trust_score) < (avg_trust - k * trust_std):
                failures += 1
            if float(g.stats.consistency_score) < consistency_threshold:
                failures += 1
            if inactivity > inactivity_limit:
                failures += 1
            if failures >= 2:
                to_prune.append(g)

        if to_prune:
            for g in to_prune:
                try:
                    self.genomes.remove(g)
                except ValueError:
                    pass
            self.total_pruned_genomes += len(to_prune)
            self.last_prune_tick = int(current_tick)
            print(f"[Pruning] Removed {len(to_prune)} genomes for underperformance.")

    def save_checkpoint(self, path: str):
        """Save genome library to checkpoint"""
        checkpoint = {
            'in_dim': self.in_dim,
            'out_dim': self.out_dim,
            'state_dim': self.state_dim,
            'max_genomes': self.max_genomes,
            'trust_decay': self.trust_decay,
            'blacklist_threshold': self.blacklist_threshold,
            'mutation_rate': self.mutation_rate,
            'next_genome_id': self.next_genome_id,
            'total_mutations': self.total_mutations,
            'total_genomes_created': self.total_genomes_created,
            'genomes': [g.get_state_dict() for g in self.genomes]
        }
        torch.save(checkpoint, path)
        print(f"[GenomeLibrary] Saved {len(self.genomes)} genomes to {path}")

    @staticmethod
    def load_checkpoint(path: str, device: str = "cpu") -> GenomeLibrary:
        """Load genome library from checkpoint"""
        checkpoint = torch.load(path, map_location=device)

        library = GenomeLibrary(
            in_dim=checkpoint['in_dim'],
            out_dim=checkpoint['out_dim'],
            state_dim=checkpoint['state_dim'],
            initial_genomes=0,
            max_genomes=checkpoint['max_genomes'],
            trust_decay=checkpoint['trust_decay'],
            blacklist_threshold=checkpoint['blacklist_threshold'],
            mutation_rate=checkpoint['mutation_rate'],
            device=device
        )

        library.next_genome_id = checkpoint['next_genome_id']
        library.total_mutations = checkpoint['total_mutations']
        library.total_genomes_created = checkpoint['total_genomes_created']

        library.genomes = []
        for genome_dict in checkpoint['genomes']:
            genome = OLAGenome.from_state_dict(
                genome_dict,
                in_dim=checkpoint['in_dim'],
                out_dim=checkpoint['out_dim'],
                device=device
            )
            library.genomes.append(genome)

        print(f"[GenomeLibrary] Loaded {len(library.genomes)} genomes from {path}")
        return library

    def __repr__(self) -> str:
        stats = self.get_library_stats()
        return (f"GenomeLibrary(genomes={stats['total_genomes']}/{self.max_genomes}, "
                f"avg_trust={stats['avg_trust']:.3f}, "
                f"mutations={self.total_mutations})")
