# ================================================================
# GENREG Tiered Learning - Genome with Tiered Shared Embeddings
# ================================================================
# Genome system with vocabulary that grows with tiers
# ================================================================

import random
import copy
import math
import torch
import torch.nn.functional as F

from config import CONFIG, DEVICE
from geometric_predictor import GeometricPredictor
from vocabulary import TieredVocabulary
from proteins import run_protein_cascade
from language_proteins import create_language_protein_network, reset_protein_network


class SharedEmbeddingLayer:
    """
    Shared embedding matrix that grows with vocabulary tiers.
    All genomes read from the same embeddings.
    Top performers contribute to updating shared embeddings.
    """
    
    def __init__(self, vocab_size, embedding_dim, config, device=None):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.config = config
        self.device = device if device else DEVICE
        
        # Shared embeddings - initialized for current vocab size
        self.embeddings = torch.empty(vocab_size, embedding_dim, device=self.device)
        emb_min = config.get("embedding_min_value", -1.0)
        emb_max = config.get("embedding_max_value", 1.0)
        torch.nn.init.uniform_(self.embeddings, emb_min, emb_max)
        
        # Track which embeddings have been trained
        self.trained_mask = torch.zeros(vocab_size, dtype=torch.bool, device=self.device)
        
        # Track contributions
        self.contribution_counts = torch.zeros(vocab_size, device=self.device)
    
    def get(self, token_id):
        """Get embedding for a token."""
        if token_id < self.vocab_size:
            return self.embeddings[token_id]
        else:
            # Return zero embedding for out-of-vocab
            return torch.zeros(self.embedding_dim, device=self.device)
    
    def expand_vocabulary(self, new_vocab_size):
        """Expand embeddings for larger vocabulary (tier advancement)."""
        if new_vocab_size <= self.vocab_size:
            return
        
        # Create new larger embedding matrix
        new_embeddings = torch.empty(new_vocab_size, self.embedding_dim, device=self.device)
        emb_min = self.config.get("embedding_min_value", -1.0)
        emb_max = self.config.get("embedding_max_value", 1.0)
        torch.nn.init.uniform_(new_embeddings, emb_min, emb_max)
        
        # Copy existing embeddings
        new_embeddings[:self.vocab_size] = self.embeddings
        
        # Update
        self.embeddings = new_embeddings
        
        # Expand tracking tensors
        new_trained = torch.zeros(new_vocab_size, dtype=torch.bool, device=self.device)
        new_trained[:self.vocab_size] = self.trained_mask
        self.trained_mask = new_trained
        
        new_counts = torch.zeros(new_vocab_size, device=self.device)
        new_counts[:self.vocab_size] = self.contribution_counts
        self.contribution_counts = new_counts
        
        old_size = self.vocab_size
        self.vocab_size = new_vocab_size
        print(f"[SharedEmb] Expanded vocabulary: {old_size} -> {new_vocab_size}")
    
    def contribute(self, token_id, direction, strength=None):
        if strength is None:
            strength = self.config.get("shared_embedding_contribution_rate", 0.01)
        """Top genomes contribute their learned directions."""
        if token_id >= self.vocab_size:
            return
        
        with torch.no_grad():
            self.embeddings[token_id] += direction * strength
            emb_min = self.config.get("embedding_min_value", -1.0)
            emb_max = self.config.get("embedding_max_value", 1.0)
            self.embeddings[token_id].clamp_(emb_min, emb_max)
            self.contribution_counts[token_id] += 1
            self.trained_mask[token_id] = True
    
    def evolve_from_population(self, genomes, top_percent=0.1, stability_protein=None):
        """Called each generation - top performers update shared embeddings."""
        contribution_rate = self.config.get("shared_embedding_contribution_rate", 0.01)
        
        # Get stability mask if protein is provided
        stability_mask = None
        if stability_protein is not None:
            stability_mask = stability_protein.get_mutation_mask(self.vocab_size, self.device)
        
        top_n = max(1, int(len(genomes) * top_percent))
        top_genomes = genomes[:top_n]
        
        with torch.no_grad():
            for genome in top_genomes:
                for record in genome.error_history:
                    if record['magnitude'] < self.config.get("vector_hit_threshold", 3.5):
                        target_id = record['target_id']
                        if target_id < self.vocab_size:
                            # Apply stability mask to contribution
                            if stability_mask is not None:
                                mask_value = stability_mask[target_id].item()
                                if mask_value < 0.01:
                                    continue  # Skip frozen embeddings
                                effective_strength = contribution_rate * mask_value
                            else:
                                effective_strength = contribution_rate
                            
                            error_mult = self.config.get("error_contribution_multiplier", 0.5)
                            self.contribute(
                                target_id,
                                -record['error'] * error_mult,
                                strength=effective_strength
                            )
            
            # Random mutation for exploration (with stability protection)
            # Select mutations only from ACTIVE indices (not frozen ones)
            mutation_rate = self.config.get("shared_embedding_mutation_rate", 0.01)
            # Fetch scale from config, default to larger step size for phase 2+
            mutation_scale = self.config.get("shared_embedding_mutation_scale", 0.1)
            
            if stability_mask is not None:
                mask_threshold = self.config.get("stability_mask_threshold", 0.01)
                active_indices = (stability_mask > mask_threshold).nonzero().squeeze(-1)
                num_active = len(active_indices) if active_indices.dim() > 0 else 0
                if num_active > 0:
                    num_mutations = max(1, int(num_active * mutation_rate))
                    selected = torch.randint(0, num_active, (num_mutations,), device=self.device)
                    indices = active_indices[selected]
                else:
                    indices = torch.tensor([], device=self.device, dtype=torch.long)
            else:
                num_mutations = max(1, int(self.vocab_size * mutation_rate))
                indices = torch.randint(0, self.vocab_size, (num_mutations,), device=self.device)
            
            for idx in indices:
                # Get effective scale from stability mask
                if stability_mask is not None:
                    effective_scale = mutation_scale * stability_mask[idx].item()
                else:
                    effective_scale = mutation_scale
                
                noise = torch.randn(self.embedding_dim, device=self.device) * effective_scale
                self.embeddings[idx] += noise
            
            emb_min = self.config.get("embedding_min_value", -1.0)
            emb_max = self.config.get("embedding_max_value", 1.0)
            self.embeddings.clamp_(emb_min, emb_max)
    
    def state_dict(self):
        """For checkpointing."""
        return {
            'embeddings': self.embeddings.cpu(),
            'vocab_size': self.vocab_size,
            'trained_mask': self.trained_mask.cpu(),
            'contribution_counts': self.contribution_counts.cpu(),
        }
    
    def load_state_dict(self, state):
        """Load from checkpoint."""
        self.vocab_size = state['vocab_size']
        self.embeddings = state['embeddings'].to(self.device)
        self.trained_mask = state['trained_mask'].to(self.device)
        self.contribution_counts = state['contribution_counts'].to(self.device)


class TieredGenome:
    """
    Genome for tiered fill-in-the-blank learning.
    Uses bidirectional controller and shared embeddings.
    Includes protein network for shaped trust signals.
    """
    
    def __init__(self, controller, vocab_size, embedding_dim, config, 
                 device=None, shared_embeddings=None):
        self.controller = controller
        self.config = config
        self.device = device if device else DEVICE
        
        # Reference to shared embeddings
        self.shared_embeddings = shared_embeddings
        self.use_shared = shared_embeddings is not None and config.get("shared_embeddings", True)
        
        # Private embeddings (hybrid mode)
        self.embeddings = torch.empty(vocab_size, embedding_dim, device=self.device)
        emb_min = config.get("embedding_min_value", -1.0)
        emb_max = config.get("embedding_max_value", 1.0)
        torch.nn.init.uniform_(self.embeddings, emb_min, emb_max)
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.shared_mix_ratio = config.get("shared_embedding_mix_ratio", 0.7)
        
        # State
        self.trust = 0.0
        self.accuracy = 0.0
        self.id = random.randint(10000, 99999)
        
        # Error history for guided mutation
        self.error_history = []
        
        # Context-aware mutation: track words seen in this generation
        self.active_words_this_gen = set()
        
        # Triplet-guided mutation: track failed triplet nudges
        self.triplet_errors = []  # List of (word_id, nudge_tensor) tuples
        
        # Protein network for shaped trust signals
        self.proteins = create_language_protein_network(config)
    
    def reset_error_history(self):
        """Clear error history and reset proteins for new episode."""
        self.error_history = []
        self.active_words_this_gen = set()  # Reset active words tracking
        self.triplet_errors = []  # Reset triplet errors
        reset_protein_network(self.proteins)
    
    def run_proteins(self, signals):
        """
        Run the protein network with given signals.
        
        Args:
            signals: Dict with keys like 'prediction_distance', 'token_hit', 'category_match'
        
        Returns:
            float: Trust delta from all proteins
        """
        outputs, trust_delta = run_protein_cascade(self.proteins, signals)
        return trust_delta
    
    def record_error(self, predicted_tensor, target_tensor, target_id):
        """Record prediction error for guided mutation."""
        error = (target_tensor - predicted_tensor).detach()
        self.error_history.append({
            'error': error,
            'target_id': target_id,
            'magnitude': torch.norm(error).item()
        })
    
    def record_cooccurrence(self, word_ids):
        """Track words that appeared together in a sentence."""
        # Filter out PAD tokens (id 0)
        valid_ids = [wid for wid in word_ids if wid > 0]
        for wid in valid_ids:
            self.active_words_this_gen.add(wid)
    
    def record_triplet_error(self, word_id, nudge):
        """
        Record a triplet failure nudge direction.
        
        Args:
            word_id: The anchor word ID that needs to be nudged
            nudge: Tensor direction to nudge (toward positive example)
        """
        if word_id >= 0 and word_id < self.vocab_size:
            self.triplet_errors.append((word_id, nudge.detach().clone()))
    
    def get_embedding_tensor(self, token_id):
        """Get embedding using shared/private mix."""
        if self.use_shared and self.shared_embeddings is not None:
            shared_emb = self.shared_embeddings.get(token_id)
            
            if self.shared_mix_ratio < 1.0 and token_id < self.vocab_size:
                private_emb = self.embeddings[token_id]
                return (self.shared_mix_ratio * shared_emb + 
                        (1 - self.shared_mix_ratio) * private_emb)
            else:
                return shared_emb
        else:
            if token_id < self.vocab_size:
                return self.embeddings[token_id]
            else:
                return torch.zeros(self.embedding_dim, device=self.device)
    
    def expand_vocabulary(self, new_vocab_size):
        """Expand private embeddings for tier advancement."""
        if new_vocab_size <= self.vocab_size:
            return
        
        new_embeddings = torch.empty(new_vocab_size, self.embedding_dim, device=self.device)
        emb_min = self.config.get("embedding_min_value", -1.0)
        emb_max = self.config.get("embedding_max_value", 1.0)
        torch.nn.init.uniform_(new_embeddings, emb_min, emb_max)
        new_embeddings[:self.vocab_size] = self.embeddings
        
        self.embeddings = new_embeddings
        self.vocab_size = new_vocab_size
    
    def clone(self):
        """Deep copy genome."""
        new_controller = self.controller.clone()
        
        new_genome = TieredGenome(
            new_controller,
            self.vocab_size,
            self.embedding_dim,
            self.config,
            self.device,
            self.shared_embeddings
        )
        
        new_genome.embeddings = self.embeddings.clone()
        new_genome.shared_mix_ratio = self.shared_mix_ratio
        new_genome.trust = 0.0
        new_genome.error_history = self.error_history.copy()
        new_genome.active_words_this_gen = self.active_words_this_gen.copy()
        return new_genome
    
    def mutate(self, rate=0.02, stability_protein=None):
        """Mutate genome with optional error guidance and stability protection."""
        guided_strength = self.config.get("guided_mutation_strength", 0.0)
        mutation_scale = self.config["mutation_scale"]
        
        # Mutate controller (always - controller is not protected)
        self.controller.mutate(rate, scale=mutation_scale)
        
        # Guided mutation using error history
        # For GeometricPredictor, guide the dim_weights based on error patterns
        if guided_strength > 0 and len(self.error_history) > 0:
            with torch.no_grad():
                avg_error = torch.stack([e['error'] for e in self.error_history]).mean(dim=0)
                
                # Guided mutation affects dim_weights - which dimensions matter more
                guided_nudge = avg_error.abs().mean()  # Scalar guidance
                if random.random() < rate:
                    guided_mult = self.config.get("guided_mutation_scale_multiplier", 0.1)
                    self.controller.dim_weights += guided_nudge * mutation_scale * guided_mult
                    dim_min = self.config.get("predictor_dim_weight_min", 0.1)
                    dim_max = self.config.get("predictor_dim_weight_max", 3.0)
                    self.controller.dim_weights.clamp_(dim_min, dim_max)
        
        # Mutate private embeddings (with optional stability protection)
        if not self.use_shared or self.shared_mix_ratio < 1.0:
            emb_mut_prob = self.config["embedding_mutation_prob"]
            emb_mut_scale = self.config["embedding_mutation_scale"]
            
            # Get stability mask if protein is provided
            stability_mask = None
            if stability_protein is not None:
                stability_mask = stability_protein.get_mutation_mask(self.vocab_size, self.device)
            
            with torch.no_grad():
                # Context-aware mutation: select from words seen in this generation
                # Intersect with stability mask if available
                candidate_indices = None
                
                if self.active_words_this_gen:
                    # Use context-aware active words
                    candidate_list = list(self.active_words_this_gen)
                    # Filter to valid vocab indices
                    candidate_list = [wid for wid in candidate_list if wid < self.vocab_size]
                    
                    if candidate_list:
                        candidate_indices = torch.tensor(candidate_list, device=self.device, dtype=torch.long)
                else:
                    # Fallback: if no active words tracked, use full vocab
                    candidate_indices = torch.arange(self.vocab_size, device=self.device)
                
                # Apply stability mask if available
                if stability_mask is not None and candidate_indices is not None:
                    # Filter candidate_indices to only include stable (active) ones
                    mask_values = stability_mask[candidate_indices]
                    stable_mask = mask_values > 0.01
                    stable_indices = candidate_indices[stable_mask]
                    if len(stable_indices) > 0:
                        candidate_indices = stable_indices
                    else:
                        candidate_indices = torch.tensor([], device=self.device, dtype=torch.long)
                
                # Select mutations from candidate indices
                if candidate_indices is not None and len(candidate_indices) > 0:
                    num_candidates = len(candidate_indices)
                    num_mutations = max(1, int(num_candidates * rate))
                    selected = torch.randint(0, num_candidates, (num_mutations,), device=self.device)
                    indices = candidate_indices[selected]
                else:
                    indices = torch.tensor([], device=self.device, dtype=torch.long)
                
                for idx in indices:
                    # Get effective scale from stability mask
                    if stability_mask is not None:
                        effective_scale = emb_mut_scale * stability_mask[idx].item()
                    else:
                        effective_scale = emb_mut_scale
                    
                    mask = torch.rand(self.embedding_dim, device=self.device) < emb_mut_prob
                    noise = torch.randn(self.embedding_dim, device=self.device) * effective_scale
                    self.embeddings[idx] += noise * mask.float()
                
                emb_min = self.config.get("embedding_min_value", -1.0)
            emb_max = self.config.get("embedding_max_value", 1.0)
            
            # Apply triplet-guided nudges (push anchors toward positives)
            if self.triplet_errors:
                triplet_mutation_scale = self.config.get("triplet_mutation_scale", 0.1)
                with torch.no_grad():
                    for word_id, nudge in self.triplet_errors:
                        if word_id < self.vocab_size:
                            # Normalize nudge to prevent large jumps
                            nudge_normalized = nudge / (torch.norm(nudge) + 1e-8)
                            self.embeddings[word_id] += nudge_normalized * triplet_mutation_scale
            
            self.embeddings.clamp_(emb_min, emb_max)
        
        # Optionally mutate mix ratio
        mix_mut_rate = self.config.get("mix_ratio_mutation_rate", 0.05)
        if random.random() < mix_mut_rate:
            mix_mut_scale = self.config.get("mix_ratio_mutation_scale", 0.05)
            self.shared_mix_ratio += random.gauss(0, mix_mut_scale)
            mix_min = self.config.get("mix_ratio_min", 0.0)
            mix_max = self.config.get("mix_ratio_max", 1.0)
            self.shared_mix_ratio = max(mix_min, min(mix_max, self.shared_mix_ratio))
        
        self.error_history = []
        self.active_words_this_gen = set()  # Reset for next generation
        self.triplet_errors = []  # Reset triplet errors
        return self


class TieredPopulation:
    """Population manager for tiered learning."""
    
    def __init__(self, vocab, config, device=None):
        self.config = config
        self.vocab = vocab
        self.size = config["population_size"]
        self.generation = 0
        self.device = device if device else DEVICE
        
        embedding_dim = config["embedding_dim"]
        context_window = config["context_window"]
        hidden_size = config["controller_hidden_size"]
        
        # Create shared embeddings
        self.shared_embedding_layer = SharedEmbeddingLayer(
            vocab.size, embedding_dim, config, self.device
        )
        print(f"[Pop] Shared Embeddings: ENABLED (mix ratio: {config.get('shared_embedding_mix_ratio', 0.7):.0%})")
        print(f"[Pop] Using GeometricPredictor (no neural controller)")
        print(f"[Pop] Init: Vocab={vocab.size}, EmbDim={embedding_dim}, ContextWindow={context_window}")
        
        # Create genomes
        self.genomes = []
        for _ in range(self.size):
            controller = GeometricPredictor(embedding_dim, context_window, config)
            g = TieredGenome(
                controller=controller,
                vocab_size=vocab.size,
                embedding_dim=embedding_dim,
                config=config,
                device=self.device,
                shared_embeddings=self.shared_embedding_layer
            )
            self.genomes.append(g)
        
        self.active_index = 0
        self.last_crossover_count = 0
    
    def get_active(self):
        return self.genomes[self.active_index]
    
    def next_genome(self):
        self.active_index = (self.active_index + 1) % self.size
        return self.get_active()
    
    def advance_tier(self, new_vocab):
        """Advance all genomes to new tier vocabulary."""
        new_size = new_vocab.size
        
        # Expand shared embeddings
        self.shared_embedding_layer.expand_vocabulary(new_size)
        
        # Expand each genome's private embeddings
        for g in self.genomes:
            g.expand_vocabulary(new_size)
        
        self.vocab = new_vocab
        print(f"[Pop] Advanced to tier {new_vocab.current_tier}")
    
    def resize_population(self, new_size):
        """Resize population to new_size, keeping best genomes."""
        old_size = self.size
        if new_size == old_size:
            return 0
        
        # Sort by trust (best first)
        self.genomes.sort(key=lambda g: getattr(g, 'trust', 0), reverse=True)
        
        if new_size < old_size:
            # Keep top N
            self.genomes = self.genomes[:new_size]
            self.size = new_size
            return -(old_size - new_size)  # Return negative for removed count
        else:
            # Add new random genomes
            added = new_size - old_size
            embedding_dim = self.config["embedding_dim"]
            context_window = self.config["context_window"]
            vocab_size = self.genomes[0].vocab_size if self.genomes else self.vocab.size
            
            for _ in range(added):
                controller = GeometricPredictor(embedding_dim, context_window, self.config)
                new_genome = TieredGenome(
                    controller=controller,
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    config=self.config,
                    device=self.device,
                    shared_embeddings=self.shared_embedding_layer
                )
                self.genomes.append(new_genome)
            
            self.size = new_size
            return added  # Return positive for added count
    
    def evolve(self, stability_protein=None):
        """Evolve population with optional embedding stability protection."""
        self.generation += 1
        
        # Get config values
        trust_scale_trigger = self.config["trust_scale_trigger"]
        trust_scale_target = self.config["trust_scale_target"]
        survival_rate = self.config["survival_rate"]
        elite_mutation_rate = self.config["elite_mutation_rate"]
        standard_mutation_rate = self.config["standard_mutation_rate"]
        hyper_mutation_rate = self.config["hyper_mutation_rate"]
        hyper_mutation_start = self.config["hyper_mutation_start"]
        crossover_rate = self.config.get("crossover_rate", 0.3)
        
        # Trust scaling
        max_abs_trust = max(abs(g.trust) for g in self.genomes)
        if max_abs_trust > trust_scale_trigger:
            scale_factor = trust_scale_target / max_abs_trust
            for g in self.genomes:
                g.trust *= scale_factor
        
        # Sort by trust
        self.genomes.sort(key=lambda g: g.trust, reverse=True)
        
        highest_trust = self.genomes[0].trust
        median_trust = self.genomes[self.size // 2].trust
        lowest_trust = self.genomes[-1].trust
        
        # Evolve shared embeddings (with stability protection)
        self.shared_embedding_layer.evolve_from_population(
            self.genomes,
            top_percent=self.config.get("shared_embedding_top_percent", 0.1),
            stability_protein=stability_protein
        )
        
        # Survival
        cutoff = max(1, int(self.size * survival_rate))
        survivors = self.genomes[:cutoff]
        
        # Selection weights
        log_scaled_trusts = [math.log(max(0.0, g.trust) + 1.0) for g in survivors]
        min_log_trust = min(log_scaled_trusts)
        weights = [lt - min_log_trust + 1.0 for lt in log_scaled_trusts]
        
        new_population = []
        hyper_mutate_start_index = int(self.size * hyper_mutation_start)
        crossover_count = 0
        
        for i in range(self.size):
            # Elite always clones
            if i == 0:
                child = survivors[0].clone()
                child.trust = 0.0
                child.mutate(rate=elite_mutation_rate, stability_protein=stability_protein)
                new_population.append(child)
                continue
            
            # Crossover or clone
            use_crossover = random.random() < crossover_rate and len(survivors) >= 2
            
            if use_crossover:
                parent_a = random.choices(survivors, weights=weights, k=1)[0]
                parent_b = random.choices(survivors, weights=weights, k=1)[0]
                if parent_a != parent_b:
                    child = self._crossover(parent_a, parent_b)
                    crossover_count += 1
                else:
                    child = parent_a.clone()
            else:
                parent = random.choices(survivors, weights=weights, k=1)[0]
                child = parent.clone()
            
            child.trust = 0.0
            
            # Determine mutation rate
            if i >= hyper_mutate_start_index:
                current_mutation_rate = hyper_mutation_rate
            else:
                current_mutation_rate = standard_mutation_rate
            
            child.mutate(rate=current_mutation_rate, stability_protein=stability_protein)
            new_population.append(child)
        
        self.last_crossover_count = crossover_count
        self.genomes = new_population
        self.active_index = 0
        
        return highest_trust, median_trust, lowest_trust
    
    def inject_random_genomes(self, num_random=None, percent=0.1):
        """
        Replace the worst genomes with fresh random ones.
        Call this after evolve() to inject diversity.
        
        Args:
            num_random: Exact number to replace (overrides percent). If 0, no replacement.
            percent: Fraction of population to replace (default 10%). If 0.0, no replacement.
        """
        if num_random is None:
            num_random = int(self.size * percent)
        
        # Early return if disabled
        if num_random <= 0:
            return 0
        
        # Get current config values
        embedding_dim = self.config["embedding_dim"]
        context_window = self.config["context_window"]
        vocab_size = self.genomes[0].vocab_size
        
        # Replace the worst genomes (at the end after sorting)
        for i in range(num_random):
            idx = self.size - 1 - i  # Replace from worst
            
            # Create fresh random genome
            controller = GeometricPredictor(embedding_dim, context_window, self.config)
            new_genome = TieredGenome(
                controller=controller,
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                config=self.config,
                device=self.device,
                shared_embeddings=self.shared_embedding_layer
            )
            
            self.genomes[idx] = new_genome
        
        return num_random
    
    def _crossover(self, parent_a, parent_b):
        """Blend two parent genomes."""
        # Determine stronger parent
        if parent_a.trust >= parent_b.trust:
            blend_ratio = self.config.get("crossover_blend_ratio_strong", 0.7)
        else:
            blend_ratio = self.config.get("crossover_blend_ratio_weak", 0.3)
        
        child = parent_a.clone()
        
        with torch.no_grad():
            # Blend geometric predictor parameters
            child.controller.position_weights = (
                blend_ratio * parent_a.controller.position_weights +
                (1 - blend_ratio) * parent_b.controller.position_weights
            )
            child.controller.dim_weights = (
                blend_ratio * parent_a.controller.dim_weights +
                (1 - blend_ratio) * parent_b.controller.dim_weights
            )
            
            # Blend private embeddings
            child.embeddings = (
                blend_ratio * parent_a.embeddings +
                (1 - blend_ratio) * parent_b.embeddings
            )
        
        return child
    
    def state_dict(self):
        """For checkpointing."""
        return {
            'genomes': [(g.controller.state_dict(), g.embeddings.cpu()) for g in self.genomes],
            'shared_embeddings': self.shared_embedding_layer.state_dict(),
            'generation': self.generation,
        }
    
    def scale_architecture(self, scale_factor=1.5):
        """
        Scale up embedding and controller dimensions when plateau detected.
        
        This is a major operation that creates new, larger genomes and
        transfers knowledge from the old ones.
        """
        from config import AUTO_SCALE_CONFIG
        
        old_emb_dim = self.config["embedding_dim"]
        old_hidden = self.config["controller_hidden_size"]
        
        # Calculate new dimensions
        new_emb_dim = min(
            int(old_emb_dim * scale_factor),
            AUTO_SCALE_CONFIG.get("max_embedding_dim", 128)
        )
        new_hidden = min(
            int(old_hidden * scale_factor),
            AUTO_SCALE_CONFIG.get("max_hidden_size", 256)
        )
        
        # Don't scale if already at max
        if new_emb_dim == old_emb_dim and new_hidden == old_hidden:
            print(f"[AutoScale] Already at max dimensions, skipping")
            return False
        
        print(f"\n[AutoScale] SCALING ARCHITECTURE")
        print(f"  Embedding: {old_emb_dim} -> {new_emb_dim}")
        print(f"  Hidden: {old_hidden} -> {new_hidden}")
        
        # Update config
        self.config["embedding_dim"] = new_emb_dim
        self.config["controller_hidden_size"] = new_hidden
        
        # Scale shared embeddings
        self._scale_shared_embeddings(old_emb_dim, new_emb_dim)
        
        # Scale each genome
        for g in self.genomes:
            self._scale_genome(g, old_emb_dim, new_emb_dim, old_hidden, new_hidden)
        
        print(f"[AutoScale] Complete")
        return True
    
    def _scale_shared_embeddings(self, old_dim, new_dim):
        """Scale shared embeddings to new dimension."""
        old_embeddings = self.shared_embedding_layer.embeddings
        vocab_size = old_embeddings.size(0)
        
        # Create new larger embeddings
        new_embeddings = torch.zeros(vocab_size, new_dim, device=self.device)
        
        # Copy old values and initialize new dimensions randomly
        new_embeddings[:, :old_dim] = old_embeddings
        init_std = self.config.get("architecture_scale_init_std", 0.1)
        new_embeddings[:, old_dim:] = torch.randn(vocab_size, new_dim - old_dim, device=self.device) * init_std
        
        self.shared_embedding_layer.embeddings = new_embeddings
        self.shared_embedding_layer.embedding_dim = new_dim
    
    def _scale_genome(self, genome, old_emb_dim, new_emb_dim, old_hidden, new_hidden):
        """Scale a single genome to new dimensions."""
        # Scale private embeddings
        old_embeddings = genome.embeddings
        vocab_size = old_embeddings.size(0)
        
        new_embeddings = torch.zeros(vocab_size, new_emb_dim, device=self.device)
        new_embeddings[:, :old_emb_dim] = old_embeddings
        init_std = self.config.get("architecture_scale_init_std", 0.1)
        new_embeddings[:, old_emb_dim:] = torch.randn(vocab_size, new_emb_dim - old_emb_dim, device=self.device) * init_std
        genome.embeddings = new_embeddings
        genome.embedding_dim = new_emb_dim
        
        # Create new larger GeometricPredictor
        context_window = self.config["context_window"]
        new_controller = GeometricPredictor(new_emb_dim, context_window, self.config)
        
        # Transfer evolvable parameters
        with torch.no_grad():
            old_controller = genome.controller
            
            # Position weights stay the same (context window unchanged)
            new_controller.position_weights = old_controller.position_weights.clone()
            new_controller.mode = old_controller.mode
            
            # Dimension weights: copy old, initialize new dimensions
            new_dim_weights = torch.ones(new_emb_dim, device=self.device)
            new_dim_weights[:old_emb_dim] = old_controller.dim_weights
            new_controller.dim_weights = new_dim_weights
        
        genome.controller = new_controller


if __name__ == "__main__":
    # Test
    vocab = TieredVocabulary()
    pop = TieredPopulation(vocab, CONFIG, DEVICE)
    
    print(f"\nPopulation created with {pop.size} genomes")
    print(f"Vocab size: {vocab.size}")
    
    # Test genome
    genome = pop.get_active()
    print(f"\nGenome ID: {genome.id}")
    print(f"Controller params: {genome.controller.get_param_count()}")
    
    # Test embedding
    emb = genome.get_embedding_tensor(2)  # "cat"
    print(f"Embedding shape: {emb.shape}")
    
    # Test tier advancement
    vocab.advance_tier()
    pop.advance_tier(vocab)
    print(f"\nAfter tier advance:")
    print(f"Vocab size: {vocab.size}")
    print(f"Genome vocab size: {genome.vocab_size}")

