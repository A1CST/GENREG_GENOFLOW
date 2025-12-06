# ================================================================
# GENREG - Embedding Stability Protein
# ================================================================
# Protects learned embedding structure when words are not actively
# being trained (e.g., during phase restarts or lower tier training)
#
# The Problem:
#   When phase 2 restarts at tier 1, words 21-1200 sit idle.
#   Random mutations destroy their learned structure.
#   By the time tier 10 resumes, embeddings are garbage.
#
# The Solution:
#   Track which embeddings are "active" (in current tier vocab).
#   Freeze or heavily suppress mutation on inactive embeddings.
#   Only allow mutation on embeddings currently under selection pressure.
# ================================================================

import torch
import random
from proteins import Protein


class EmbeddingStabilityProtein(Protein):
    """
    Regulatory protein that controls embedding mutation rates.
    
    Tracks which word IDs are currently "active" (in training vocab).
    Suppresses mutation on inactive embeddings to preserve structure.
    
    Signals:
        - current_tier: The active training tier
        - active_word_ids: Set of word IDs currently in use
    
    Output:
        - mutation_mask: Tensor of per-embedding mutation multipliers
          1.0 = normal mutation, 0.0 = frozen
    """
    
    def __init__(self, name="embedding_stability"):
        super().__init__(name, "stability")
        
        # How much to suppress mutation on inactive embeddings
        # 0.0 = completely frozen
        # 0.1 = 10% of normal mutation rate
        self.params["inactive_mutation_rate"] = 0.0
        
        # How much to suppress mutation on recently-learned embeddings
        # Even active embeddings might benefit from some stability
        self.params["active_mutation_rate"] = 1.0
        
        # Generations of "cooldown" after an embedding was last active
        # During cooldown, use a gradual unfreezing
        self.params["cooldown_generations"] = 10
        
        # Track when each embedding was last active
        self.state["last_active_gen"] = {}
        self.state["current_gen"] = 0
        
        # Cache the mutation mask
        self.state["mutation_mask"] = None
        self.state["active_ids"] = set()
    
    def set_active_vocabulary(self, word_ids, generation):
        """
        Update which embeddings are currently active.
        Call this at the start of each generation.
        
        Args:
            word_ids: Set or list of word IDs in current tier vocab
            generation: Current generation number
        """
        self.state["current_gen"] = generation
        self.state["active_ids"] = set(word_ids)
        
        # Update last-active tracking
        for wid in word_ids:
            self.state["last_active_gen"][wid] = generation
        
        # Invalidate cached mask
        self.state["mutation_mask"] = None
    
    def get_mutation_mask(self, vocab_size, device):
        """
        Get per-embedding mutation multipliers.
        
        Returns:
            Tensor of shape [vocab_size] with values 0.0-1.0
        """
        if self.state["mutation_mask"] is not None:
            if self.state["mutation_mask"].shape[0] == vocab_size:
                return self.state["mutation_mask"]
        
        mask = torch.zeros(vocab_size, device=device)
        current_gen = self.state["current_gen"]
        active_ids = self.state["active_ids"]
        inactive_rate = self.params["inactive_mutation_rate"]
        active_rate = self.params["active_mutation_rate"]
        cooldown = self.params["cooldown_generations"]
        
        for wid in range(vocab_size):
            if wid in active_ids:
                # Currently active - use active rate
                mask[wid] = active_rate
            else:
                # Not active - check cooldown
                last_active = self.state["last_active_gen"].get(wid, -999)
                gens_since_active = current_gen - last_active
                
                if gens_since_active <= cooldown:
                    # In cooldown - gradual transition from active to inactive
                    progress = gens_since_active / cooldown
                    mask[wid] = active_rate * (1 - progress) + inactive_rate * progress
                else:
                    # Fully inactive
                    mask[wid] = inactive_rate
        
        self.state["mutation_mask"] = mask
        return mask
    
    def forward(self, signals, protein_outputs):
        """
        Standard protein forward pass.
        
        Expects signals:
            - vocab_size: int
            - device: torch device
        
        Returns the fraction of embeddings currently protected.
        """
        vocab_size = signals.get("vocab_size", 0)
        active_count = len(self.state["active_ids"])
        
        if vocab_size > 0:
            protected_fraction = 1.0 - (active_count / vocab_size)
            self.output = protected_fraction
        else:
            self.output = 0.0
        
        return self.output


# ================================================================
# Integration with TieredGenome mutation
# ================================================================

def apply_stability_to_genome_mutation(genome, stability_protein):
    """
    Apply stability mask during genome mutation.
    Call this INSTEAD of genome.mutate() when stability is active.
    
    Args:
        genome: TieredGenome instance
        stability_protein: EmbeddingStabilityProtein instance
    """
    config = genome.config
    rate = config.get("standard_mutation_rate", 0.1)
    guided_strength = config.get("guided_mutation_strength", 0.0)
    mutation_scale = config["mutation_scale"]
    
    # Mutate controller (unchanged - controller is always mutated normally)
    genome.controller.mutate(rate, scale=mutation_scale)
    
    # Guided mutation (unchanged)
    if guided_strength > 0 and len(genome.error_history) > 0:
        with torch.no_grad():
            avg_error = torch.stack([e['error'] for e in genome.error_history]).mean(dim=0)
            guided_nudge = avg_error * guided_strength * mutation_scale
            for i in range(min(len(genome.controller.fc2.bias), len(guided_nudge))):
                if random.random() < rate:
                    genome.controller.fc2.bias.data[i] += guided_nudge[i].item()
    
    # Mutate private embeddings WITH STABILITY MASK
    if not genome.use_shared or genome.shared_mix_ratio < 1.0:
        # Get mutation mask from stability protein
        mask = stability_protein.get_mutation_mask(genome.vocab_size, genome.device)
        
        emb_mut_prob = config["embedding_mutation_prob"]
        emb_mut_scale = config["embedding_mutation_scale"]
        
        with torch.no_grad():
            num_mutations = max(1, int(genome.vocab_size * rate))
            indices = torch.randint(0, genome.vocab_size, (num_mutations,), device=genome.device)
            
            for idx in indices:
                # APPLY STABILITY MASK
                idx_mask_value = mask[idx].item()
                if idx_mask_value < 0.01:
                    continue  # Skip frozen embeddings
                
                effective_scale = emb_mut_scale * idx_mask_value
                mut_mask = torch.rand(genome.embedding_dim, device=genome.device) < emb_mut_prob
                noise = torch.randn(genome.embedding_dim, device=genome.device) * effective_scale
                genome.embeddings[idx] += noise * mut_mask.float()
            
            emb_min = config.get("embedding_min_value", -1.0)
            emb_max = config.get("embedding_max_value", 1.0)
            genome.embeddings.clamp_(emb_min, emb_max)
    
    # Optionally mutate mix ratio
    mix_mut_rate = config.get("mix_ratio_mutation_rate", 0.05)
    if random.random() < mix_mut_rate:
        mix_mut_scale = config.get("mix_ratio_mutation_scale", 0.05)
        genome.shared_mix_ratio += random.gauss(0, mix_mut_scale)
        mix_min = config.get("mix_ratio_min", 0.0)
        mix_max = config.get("mix_ratio_max", 1.0)
        genome.shared_mix_ratio = max(mix_min, min(mix_max, genome.shared_mix_ratio))
    
    genome.error_history = []


def apply_stability_to_shared_embedding_evolution(shared_layer, genomes, stability_protein, top_percent=0.1):
    """
    Evolve shared embeddings with stability mask protection.
    Call this INSTEAD of shared_layer.evolve_from_population() when stability is active.
    
    Args:
        shared_layer: SharedEmbeddingLayer instance
        genomes: List of genomes (sorted by trust, best first)
        stability_protein: EmbeddingStabilityProtein instance
        top_percent: Top percent of genomes to use for contributions
    """
    # Get mutation mask
    mask = stability_protein.get_mutation_mask(shared_layer.vocab_size, shared_layer.device)
    
    contribution_rate = shared_layer.config.get("shared_embedding_contribution_rate", 0.01)
    
    top_n = max(1, int(len(genomes) * top_percent))
    top_genomes = genomes[:top_n]
    
    with torch.no_grad():
        # Guided contributions - ONLY for active embeddings
        for genome in top_genomes:
            for record in genome.error_history:
                target_id = record['target_id']
                if record['magnitude'] < shared_layer.config.get("vector_hit_threshold", 3.5):
                    if target_id < shared_layer.vocab_size:
                        # Check if this embedding is active
                        mask_value = mask[target_id].item()
                        mask_threshold = shared_layer.config.get("stability_mask_threshold", 0.01)
                        if mask_value > mask_threshold:
                            error_mult = shared_layer.config.get("error_contribution_multiplier", 0.5)
                            shared_layer.contribute(
                                target_id,
                                -record['error'] * error_mult,
                                strength=contribution_rate * mask_value
                            )
        
        # Random mutation - RESPECT THE MASK
        mutation_rate = shared_layer.config.get("shared_embedding_mutation_rate", 0.01)
        num_mutations = max(1, int(shared_layer.vocab_size * mutation_rate))
        indices = torch.randint(0, shared_layer.vocab_size, (num_mutations,), device=shared_layer.device)
        
        mutation_scale = shared_layer.config.get("shared_embedding_mutation_scale", 0.1)
        mask_threshold = shared_layer.config.get("stability_mask_threshold", 0.01)
        emb_min = shared_layer.config.get("embedding_min_value", -1.0)
        emb_max = shared_layer.config.get("embedding_max_value", 1.0)
        
        for idx in indices:
            idx_mask_value = mask[idx].item()
            if idx_mask_value < mask_threshold:
                continue  # Skip frozen embeddings
            
            effective_scale = mutation_scale * idx_mask_value
            noise = torch.randn(shared_layer.embedding_dim, device=shared_layer.device) * effective_scale
            shared_layer.embeddings[idx] += noise
        
        shared_layer.embeddings.clamp_(emb_min, emb_max)


def get_active_word_ids_for_tier(vocab_tier):
    """
    Get all word IDs that are active for the given tier.
    
    Returns:
        set: Word IDs for PAD, BLANK, and all words up to current tier
    """
    from config import TIER_VOCAB
    
    active_ids = {0, 1}  # PAD and BLANK are always "active" (but rarely mutated anyway)
    current_id = 2
    
    for t in range(1, vocab_tier + 1):
        if t in TIER_VOCAB:
            for _ in TIER_VOCAB[t]:
                active_ids.add(current_id)
                current_id += 1
    
    return active_ids

