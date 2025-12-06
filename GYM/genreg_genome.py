
# ================================================================
# GENREG v2.1 — Genome (FIX: Immutable Laws)
# ================================================================

import random
import copy
from genreg_proteins import run_protein_cascade, TrustModifierProtein # Import Type
from genreg_controller import GENREGController

# ================================================================
# GENOME
# ================================================================
class GENREGGenome:
    def __init__(self, proteins, controller):
        self.proteins = proteins
        self.controller = controller
        self.trust = 0.0
        self.id = random.randint(1000, 9999)

        # Stats for logging
        self.metrics = {"distance": 0.0}

    def clone(self):
        """Deep copy of genetics (params), reset biological state."""
        new_genome = GENREGGenome(
            proteins=[copy.deepcopy(p) for p in self.proteins],
            controller=self.controller.clone()
        )
        # Reset protein biological memory
        for p in new_genome.proteins:
            p.state = {}
            if p.type == "sensor": p.state["running_max"] = 1.0
            if p.type == "integrator": p.state["accum"] = 0.0
            if p.type == "trend": 
                p.state["velocity"] = 0.0
                p.state["last"] = None
            if p.type == "trust_modifier": p.state["running"] = 0.0

        return new_genome

    def mutate(self, rate=0.1):
        # 1. Mutate Proteins (BUT NOT THE TRUST MODIFIERS)
        for p in self.proteins:
            # CRITICAL FIX: The "Law" cannot mutate. 
            # If agents can change what "Good" means, they will just 
            # make "doing nothing" equal "Good".
            if isinstance(p, TrustModifierProtein):
                continue 

            for k in p.params:
                if random.random() < rate:
                    p.mutate_param(k, scale=0.2)
        
        # 2. Mutate Brain
        self.controller.mutate(rate=rate, scale=0.1)
        return self

    def forward(self, signals):
        _, trust_delta = run_protein_cascade(self.proteins, signals)
        self.trust += trust_delta
        return trust_delta

# ================================================================
# POPULATION
# ================================================================
class GENREGPopulation:
    def __init__(self, template_proteins, input_size=17, hidden_size=64, output_size=6, size=20):
        self.size = size
        self.genomes = []
        
        for _ in range(size):
            controller = GENREGController(input_size, hidden_size, output_size)
            # Create fresh protein instances from template
            prots = [copy.deepcopy(p) for p in template_proteins]
            self.genomes.append(GENREGGenome(prots, controller))

        self.active_idx = 0

    def get_active(self):
        return self.genomes[self.active_idx]

    def next_genome(self):
        self.active_idx = (self.active_idx + 1) % self.size
        return self.get_active()

    def evolve(self):
        # 1. Sort by Trust
        self.genomes.sort(key=lambda g: g.trust, reverse=True)
        
        best = self.genomes[0]
        print(f"  > Evolution: Best Trust={best.trust:.2f} | Dist={best.metrics.get('distance',0):.2f}")

        # 2. Keep Top 20%
        cutoff = max(1, int(self.size * 0.2))
        survivors = self.genomes[:cutoff]

        # 3. Replenish Population
        new_pop = []
        for i in range(self.size):
            # Select parent (weighted by rank)
            parent = random.choice(survivors)
            
            # Clone & Mutate
            child = parent.clone()
            
            # Inherit 10% of parent's trust (Estate Tax)
            child.trust = parent.trust * 0.1
            
            # Mutate
            child.mutate(rate=0.1)
            new_pop.append(child)

        self.genomes = new_pop
        self.active_idx = 0