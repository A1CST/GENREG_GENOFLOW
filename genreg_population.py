# ================================================================
# GENREG Population - Evolutionary Algorithm
# Payton Miller — 2025
# ================================================================

import random
from genreg_genome import Genome


class Population:
    """
    Manages a population of genomes and evolution.
    """

    def __init__(self, size=50):
        self.size = size
        self.genomes = [Genome() for _ in range(size)]
        self.generation = 0

        # Evolution parameters
        self.survival_rate = 0.2  # Top 20% survive
        self.mutation_rate = 0.1
        self.mutation_scale = 0.3
        self.trust_inheritance = 0.5  # Children inherit 50% of parent trust

        # Stats
        self.best_trust = 0.0
        self.avg_trust = 0.0
        self.best_genome_idx = 0

    def evaluate(self, env, steps_per_life=200):
        """
        Evaluate all genomes in the environment.

        Args:
            env: SnakeEnvironment instance
            steps_per_life: max steps per genome evaluation
        """
        for i, genome in enumerate(self.genomes):
            genome.reset()
            signals = env.reset()

            for _ in range(steps_per_life):
                action, _ = genome.step(signals)
                signals, done = env.step(action)
                genome.food_eaten = env.food_eaten

                if done:
                    break

        self._update_stats()

    def evolve(self):
        """
        Run one generation of evolution:
        1. Sort by trust
        2. Select survivors
        3. Reproduce with mutation
        4. Trust inheritance
        """
        # Sort by trust (descending)
        self.genomes.sort(key=lambda g: g.trust, reverse=True)

        # Select survivors
        n_survivors = max(1, int(self.size * self.survival_rate))
        survivors = self.genomes[:n_survivors]

        # Create new population
        new_genomes = []

        # Keep best unchanged (elitism)
        new_genomes.append(survivors[0].clone())

        # Fill rest with mutated offspring
        while len(new_genomes) < self.size:
            parent = random.choice(survivors)
            child = parent.clone()
            child.mutate(self.mutation_rate, self.mutation_scale)

            # Trust inheritance
            child.trust = parent.trust * self.trust_inheritance
            new_genomes.append(child)

        self.genomes = new_genomes
        self.generation += 1
        self._update_stats()

        return self.get_stats()

    def _update_stats(self):
        """Update population statistics."""
        trusts = [g.trust for g in self.genomes]
        self.best_trust = max(trusts)
        self.avg_trust = sum(trusts) / len(trusts)
        self.best_genome_idx = trusts.index(self.best_trust)

    def get_stats(self):
        """Get current population stats."""
        foods = [g.food_eaten for g in self.genomes]
        return {
            "generation": self.generation,
            "best_trust": self.best_trust,
            "avg_trust": self.avg_trust,
            "best_food": max(foods) if foods else 0,
            "avg_food": sum(foods) / len(foods) if foods else 0,
            "population_size": len(self.genomes)
        }

    def get_best(self):
        """Get the best genome."""
        return self.genomes[self.best_genome_idx]

    def get_genome(self, idx):
        """Get genome by index."""
        if 0 <= idx < len(self.genomes):
            return self.genomes[idx]
        return None

    def set_genome(self, idx, genome):
        """Set genome at index."""
        if 0 <= idx < len(self.genomes):
            self.genomes[idx] = genome

    def to_dict(self):
        """Serialize population."""
        return {
            "generation": self.generation,
            "size": self.size,
            "survival_rate": self.survival_rate,
            "mutation_rate": self.mutation_rate,
            "mutation_scale": self.mutation_scale,
            "trust_inheritance": self.trust_inheritance,
            "genomes": [g.to_dict() for g in self.genomes]
        }

    @classmethod
    def from_dict(cls, d):
        """Deserialize population."""
        pop = cls(size=d["size"])
        pop.generation = d["generation"]
        pop.survival_rate = d.get("survival_rate", 0.2)
        pop.mutation_rate = d.get("mutation_rate", 0.1)
        pop.mutation_scale = d.get("mutation_scale", 0.3)
        pop.trust_inheritance = d.get("trust_inheritance", 0.5)
        pop.genomes = [Genome.from_dict(gd) for gd in d["genomes"]]
        pop._update_stats()
        return pop
