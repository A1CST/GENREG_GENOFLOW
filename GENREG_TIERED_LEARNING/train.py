# ================================================================
# GENREG Tiered Learning - Main Training Loop v2
# ================================================================
# Child-like vocabulary acquisition with fill-in-the-blank learning
# Auto-scaling architecture when plateau detected
# Threshold tightening on tier advancement
# ================================================================

import os
import sys
import time
import pickle
import re
import glob
import torch
from datetime import datetime

from config import CONFIG, DEVICE, CHECKPOINT_DIR, TIER_VOCAB, AUTO_SCALE_CONFIG, THRESHOLD_PHASES
from vocabulary import TieredVocabulary, reset_vocabulary
from genome import TieredPopulation
from environment import FillInBlankEnv, TieredCurriculum
from embedding_stability_protein import (
    EmbeddingStabilityProtein,
    get_active_word_ids_for_tier
)
from language_proteins import (
    check_category_match,
    find_nearest_word,
    get_word_category
)
from triplet_protein import TripletProtein


def find_best_checkpoint(checkpoint_dir=CHECKPOINT_DIR):
    """
    Find the most advanced checkpoint to resume from.
    
    Priority: Higher threshold phase index > Higher tier > Higher generation
    
    Returns:
        tuple: (filepath, phase_idx, threshold, tier, generation) or None if no checkpoints
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Pattern: checkpoint_th{X}_{Y}_tier{Z}_gen{N}_{timestamp}.pkl
    pattern = os.path.join(checkpoint_dir, "checkpoint_th*.pkl")
    checkpoint_files = glob.glob(pattern)
    
    if not checkpoint_files:
        return None
    
    best_checkpoint = None
    best_score = (-1, -1, -1)  # (phase_idx, tier, generation)
    
    for filepath in checkpoint_files:
        filename = os.path.basename(filepath)
        
        # Parse filename: checkpoint_th2_5_tier10_gen150_20251130_123456.pkl
        match = re.match(r'checkpoint_th(\d+)_(\d+)_tier(\d+)_gen(\d+)_', filename)
        if match:
            th_major = int(match.group(1))
            th_minor = int(match.group(2))
            threshold = float(f"{th_major}.{th_minor}")
            tier = int(match.group(3))
            generation = int(match.group(4))
            
            # Find phase index for this threshold
            try:
                phase_idx = THRESHOLD_PHASES.index(threshold)
            except ValueError:
                continue  # Unknown threshold, skip
            
            score = (phase_idx, tier, generation)
            if score > best_score:
                best_score = score
                best_checkpoint = (filepath, phase_idx, threshold, tier, generation)
    
    return best_checkpoint


def save_checkpoint(pop, vocab, generation, tier, config, threshold_phase=None, checkpoint_dir=CHECKPOINT_DIR):
    """Save full checkpoint to disk."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if threshold_phase is not None:
        # Include threshold in filename (e.g., th2.5)
        threshold_str = f"th{threshold_phase:.1f}".replace(".", "_")
        filename = f"checkpoint_{threshold_str}_tier{tier}_gen{generation}_{timestamp}.pkl"
    else:
        filename = f"checkpoint_gen{generation}_tier{tier}_{timestamp}.pkl"
    filepath = os.path.join(checkpoint_dir, filename)
    
    # Move tensors to CPU for saving
    for g in pop.genomes:
        g.embeddings = g.embeddings.cpu()
        g.controller.cpu()
    
    # Save shared embeddings
    shared_state = pop.shared_embedding_layer.state_dict()
    
    checkpoint_data = {
        'generation': generation,
        'tier': tier,
        'vocab_tier': vocab.current_tier,
        'population': pop,
        'shared_embeddings': shared_state,
        'config': config,
        'threshold_phase': threshold_phase,
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    # Move tensors back to GPU
    for g in pop.genomes:
        g.embeddings = g.embeddings.to(DEVICE)
        g.controller.to(DEVICE)
    
    print(f"[Checkpoint] Saved to {filepath}")
    return filepath


def load_checkpoint(filepath):
    """Load checkpoint from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Move tensors to GPU
    pop = checkpoint['population']
    for g in pop.genomes:
        g.device = DEVICE
        g.embeddings = g.embeddings.to(DEVICE)
        g.controller.to(DEVICE)
    pop.device = DEVICE
    
    # Restore shared embeddings
    if pop.shared_embedding_layer is not None:
        pop.shared_embedding_layer.device = DEVICE
        pop.shared_embedding_layer.load_state_dict(checkpoint['shared_embeddings'])
        pop.shared_embedding_layer.embeddings = pop.shared_embedding_layer.embeddings.to(DEVICE)
        
        for g in pop.genomes:
            g.shared_embeddings = pop.shared_embedding_layer
    
    # FORCE UPDATE: Apply current config's shared embedding settings to all genomes
    # (checkpoint may have been saved with different settings)
    from config import CONFIG
    current_use_shared = CONFIG.get("shared_embeddings", True)
    current_mix_ratio = CONFIG.get("shared_embedding_mix_ratio", 0.7)
    for g in pop.genomes:
        g.use_shared = current_use_shared
        g.shared_mix_ratio = current_mix_ratio
    print(f"[Checkpoint] Applied current config: use_shared={current_use_shared}, mix_ratio={current_mix_ratio}")
    
    # Resize population if config changed
    target_size = CONFIG["population_size"]
    if pop.size != target_size:
        delta = pop.resize_population(target_size)
        if delta > 0:
            print(f"[Checkpoint] Expanded population: {pop.size - delta} -> {pop.size} (added {delta} new genomes)")
        else:
            print(f"[Checkpoint] Reduced population: {pop.size + delta} -> {pop.size} (kept top {pop.size})")
    
    print(f"[Checkpoint] Loaded from {filepath}")
    print(f"  Generation: {checkpoint['generation']}")
    print(f"  Tier: {checkpoint['tier']}")
    
    return checkpoint


def get_context_embeddings(genome, context_ids):
    """Get concatenated embeddings for context word IDs."""
    embeddings = []
    for word_id in context_ids:
        if word_id is not None:
            emb = genome.get_embedding_tensor(word_id)
        else:
            emb = torch.zeros(genome.embedding_dim, device=DEVICE)
        embeddings.append(emb)
    return torch.cat(embeddings)


class PlateauDetector:
    """Detects when training is stuck at a plateau."""
    
    def __init__(self, config):
        self.threshold = config.get("plateau_threshold", 0.85)
        self.required_gens = config.get("plateau_generations", 50)
        self.enabled = config.get("enabled", True)
        
        self.plateau_count = 0
        self.last_accuracy = 0.0
    
    def check(self, avg_accuracy, mastery_threshold):
        """Check if we're at a plateau. Returns True if auto-scale should trigger."""
        if not self.enabled:
            return False
        
        # Plateau = stuck between threshold and mastery for too long
        if self.threshold <= avg_accuracy < mastery_threshold:
            self.plateau_count += 1
        else:
            self.plateau_count = 0
        
        self.last_accuracy = avg_accuracy
        
        if self.plateau_count >= self.required_gens:
            self.plateau_count = 0  # Reset after triggering
            return True
        
        return False
    
    def reset(self):
        """Reset counter (call after tier advancement)."""
        self.plateau_count = 0


def train():
    """Main training loop with multi-phase threshold tightening.
    
    Runs through ALL tiers at each threshold level before tightening:
    Phase 1: All tiers at 2.5 threshold
    Phase 2: All tiers at 1.7 threshold  
    Phase 3: All tiers at 1.0 threshold
    
    Automatically resumes from checkpoint if one exists.
    """
    print("=" * 60)
    print("GENREG TIERED LEARNING v2 - Multi-Phase Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Tiers defined: {len(TIER_VOCAB)}")
    print(f"Threshold phases: {THRESHOLD_PHASES}")
    print(f"Auto-scaling: {'ENABLED' if AUTO_SCALE_CONFIG.get('enabled', True) else 'DISABLED'}")
    print()
    
    # ================================================================
    # CHECK FOR EXISTING CHECKPOINT TO RESUME FROM
    # ================================================================
    resume_checkpoint = find_best_checkpoint()
    
    if resume_checkpoint:
        filepath, resume_phase_idx, resume_threshold, resume_tier, resume_gen = resume_checkpoint
        print(f"[Resume] Found checkpoint: {os.path.basename(filepath)}")
        print(f"[Resume] Phase {resume_phase_idx + 1} (threshold {resume_threshold}), Tier {resume_tier}, Gen {resume_gen}")
        print()
        
        # Load checkpoint
        checkpoint = load_checkpoint(filepath)
        pop = checkpoint['population']
        
        # Set up vocabulary at the checkpoint tier
        vocab = reset_vocabulary()
        vocab.set_tier(resume_tier)
        pop.vocab = vocab
        
        # Population is fully expanded if we completed phase 1 or are past tier 1 in any phase
        population_fully_expanded = (resume_phase_idx > 0) or (resume_tier == max(TIER_VOCAB.keys()))
        
        # Start from the checkpoint's phase
        start_phase_idx = resume_phase_idx
        start_tier = resume_tier
        start_gen = resume_gen
        
        print(f"[Resume] Resuming from Phase {start_phase_idx + 1}, Tier {start_tier}")
        print()
    else:
        print("[Init] No checkpoint found - starting fresh training")
        print()
        
        # Initialize vocabulary
        vocab = reset_vocabulary()
        print(f"[Init] Tier 1 vocabulary: {vocab.size} words")
        
        # Initialize population
        pop = TieredPopulation(vocab, CONFIG, DEVICE)
        print(f"[Init] Population: {pop.size} genomes")
        
        # Track if population is already fully expanded (from previous phase)
        population_fully_expanded = False
        
        start_phase_idx = 0
        start_tier = 1
        start_gen = 0
    
    # Initialize embedding stability protein
    # This protects learned embeddings when restarting phases at lower tiers
    stability_protein = EmbeddingStabilityProtein()
    stability_protein.params["inactive_mutation_rate"] = CONFIG.get("stability_inactive_mutation_rate", 0.0)
    stability_protein.params["active_mutation_rate"] = CONFIG.get("stability_active_mutation_rate", 1.0)
    stability_protein.params["cooldown_generations"] = CONFIG.get("stability_cooldown_generations", 5)
    print(f"[Stability] Embedding stability protein: ENABLED (frozen when inactive)")

    # Initialize triplet protein for semantic clustering evaluation
    triplet_protein = TripletProtein(CONFIG)
    
    # Initialize plateau detector
    plateau_detector = PlateauDetector(AUTO_SCALE_CONFIG)
    
    max_tier = max(TIER_VOCAB.keys())
    scale_events = 0
    total_gen = 0  # Overall generation counter across all phases
    
    # ================================================================
    # MULTI-PHASE TRAINING: Run all tiers at each threshold
    # ================================================================
    for phase_idx, threshold in enumerate(THRESHOLD_PHASES):
        # Skip phases before our resume point
        if phase_idx < start_phase_idx:
            print(f"[Skip] Phase {phase_idx + 1} (threshold {threshold}) - already completed")
            population_fully_expanded = True  # Previous phases expanded the population
            continue
        print()
        print("=" * 60)
        print(f"PHASE {phase_idx + 1}/{len(THRESHOLD_PHASES)}: Threshold = {threshold}")
        print("=" * 60)
        
        # Set threshold for this phase
        CONFIG["vector_hit_threshold"] = threshold
        
        # Phase 3 specific settings: lower mastery threshold, no autoscaling
        is_phase_3 = (phase_idx == 2)  # 0-indexed, so phase 3 = index 2
        if is_phase_3:
            phase_mastery_threshold = CONFIG.get("phase_3_mastery_threshold", 0.85)
            print(f"[Phase 3] Using relaxed mastery threshold: {phase_mastery_threshold:.0%}")
            print(f"[Phase 3] Auto-scaling DISABLED for this phase")
        else:
            phase_mastery_threshold = CONFIG["mastery_threshold"]  # Default 90%
        
        # Determine starting tier for this phase
        if phase_idx == start_phase_idx and start_tier > 1:
            # Resuming mid-phase: start at checkpoint tier
            current_tier = start_tier
            phase_gen = start_gen
            
            # Set up vocabulary and curriculum at resume tier
            vocab = reset_vocabulary()
            vocab.set_tier(current_tier)
            curriculum = TieredCurriculum(vocab, DEVICE)
            curriculum.current_tier = current_tier
            curriculum.vocab = vocab
            curriculum.env = FillInBlankEnv(vocab, tier=current_tier, device=DEVICE)
            env = curriculum.get_env()
            pop.vocab = vocab
            
            print(f"[Phase {phase_idx + 1}] RESUMING at Tier {current_tier}, Gen {phase_gen}")
            
            # Set phase-specific mastery threshold on curriculum
            curriculum.mastery_threshold = phase_mastery_threshold
            
            # Clear resume state so next phase starts fresh
            start_tier = 1
            start_gen = 0
        else:
            # Fresh start for this phase: begin at tier 1
            current_tier = 1
            phase_gen = 0
            
            vocab = reset_vocabulary()
            curriculum = TieredCurriculum(vocab, DEVICE)
            env = curriculum.get_env()
            pop.vocab = vocab
            
            # Set phase-specific mastery threshold on curriculum
            curriculum.mastery_threshold = phase_mastery_threshold
            
            print(f"[Phase {phase_idx + 1}] Starting at Tier 1 with threshold {threshold}")
        
        # If this is phase 2+, population is already fully expanded from previous phase
        if population_fully_expanded:
            print(f"[Phase {phase_idx + 1}] Population already expanded from previous phase")
        
        print(f"[Phase {phase_idx + 1}] Population embedding size: {pop.genomes[0].vocab_size}")
        print()
        
        # Train through all tiers at this threshold
        while current_tier <= max_tier:
            gen_start_time = time.time()
            
            # Update stability protein with current active vocabulary
            active_ids = get_active_word_ids_for_tier(current_tier)
            stability_protein.set_active_vocabulary(active_ids, total_gen)
            
            # Reset environment for this generation
            env.reset()
            
            # Evaluate each genome
            accuracies = []
            triplet_scores = []
            
            for genome_idx in range(pop.size):
                genome = pop.genomes[genome_idx]
                genome.reset_error_history()
                genome.trust = 0.0
                
                hits = 0
                total = 0
                
                # Build id_to_word mapping for category detection
                id_to_word = {0: "<PAD>", 1: "<BLANK>"}
                current_id = 2
                for t in range(1, current_tier + 1):
                    if t in TIER_VOCAB:
                        for w in TIER_VOCAB[t]:
                            if w not in id_to_word.values():
                                id_to_word[current_id] = w
                                current_id += 1
                
                # Run through all sentences
                task = env.get_current_task()
                while task is not None:
                    # DEBUG: Check if embeddings are different between genomes
                    # if total == 0 and genome_idx in [0, 1, 50]:  # Check genomes 0, 1, 50
                    #     emb_sample = genome.get_embedding_tensor(2)  # word ID 2
                    #     private_emb = genome.embeddings[2] if 2 < genome.embeddings.shape[0] else None
                    #     print(f"[DEBUG] Gen {phase_gen}, Genome {genome_idx}, Emb[2] (mixed): {[round(x, 4) for x in emb_sample[:5].tolist()]}")
                    #     if private_emb is not None:
                    #         print(f"[DEBUG] Gen {phase_gen}, Genome {genome_idx}, Private[2]: {[round(x, 4) for x in private_emb[:5].tolist()]}")
                    
                    # Get context embeddings
                    left_emb = get_context_embeddings(genome, task["left_context"])
                    right_emb = get_context_embeddings(genome, task["right_context"])
                    
                    # Predict
                    predicted = genome.controller(
                        left_emb.unsqueeze(0),
                        right_emb.unsqueeze(0)
                    ).squeeze(0)
                    
                    # Evaluate
                    result = env.evaluate_prediction(
                        predicted,
                        task["target_id"],
                        genome
                    )
                    
                    # Record error for guided mutation
                    genome.record_error(
                        predicted,
                        result["target_embedding"],
                        task["target_id"]
                    )
                    
                    # === PROTEIN-BASED TRUST SIGNALS ===
                    # Find nearest word to prediction for category matching
                    embeddings = genome.shared_embeddings.embeddings if genome.use_shared else genome.embeddings
                    pred_word, pred_id, pred_dist = find_nearest_word(
                        predicted, embeddings, id_to_word
                    )
                    target_word = task["answer"]
                    
                    # Compute category match signal
                    category_match = check_category_match(pred_word, target_word)
                    
                    # Build signals for protein network
                    signals = {
                        "prediction_distance": result["distance"],
                        "token_hit": 1.0 if result["hit"] else 0.0,
                        "category_match": category_match,
                    }
                    
                    # Run proteins to get trust delta
                    trust_delta = genome.run_proteins(signals)
                    genome.trust += trust_delta
                    
                    # Update stats
                    total += 1
                    if result["hit"]:
                        hits += 1
                    
                    # Track co-occurrence for context-aware mutation
                    sentence_word_ids = task["left_context"] + [task["target_id"]] + task["right_context"]
                    genome.record_cooccurrence(sentence_word_ids)
                    
                    # Next sentence
                    task = env.step()
                
                # Calculate accuracy (for logging/mastery detection)
                accuracy = hits / total if total > 0 else 0.0
                accuracies.append(accuracy)
                genome.accuracy = accuracy

                # === Triplet-based semantic clustering score ===
                triplet_score = triplet_protein.evaluate(genome, vocab)
                triplet_scores.append(triplet_score)

                # === Multiplicative trust calculation (CRITICAL) ===
                base_trust = CONFIG.get("triplet_base_trust", 1000000.0)
                genome.trust = base_trust * accuracy * triplet_score
                
                # Reset env for next genome
                env.reset()
            
            # Calculate stats
            avg_acc = sum(accuracies) / len(accuracies)
            min_acc = min(accuracies)
            max_acc = max(accuracies)
            
            avg_triplet = sum(triplet_scores) / len(triplet_scores) if triplet_scores else 0.0
            min_triplet = min(triplet_scores) if triplet_scores else 0.0
            max_triplet = max(triplet_scores) if triplet_scores else 0.0
            
            # Evolve population (with embedding stability protection)
            highest_trust, median_trust, lowest_trust = pop.evolve(stability_protein=stability_protein)
            
            # PHASE 2 ONLY: Inject random genomes every 3 generations to maintain diversity
            inject_info = ""
            if phase_idx == 1:  # Phase 2 (0-indexed)
                inject_percent = CONFIG.get("random_injection_percent", 0.1)
                if inject_percent > 0.0 and phase_gen % 3 == 0:  # Every 3 generations
                    num_injected = pop.inject_random_genomes(percent=inject_percent)
                    inject_info = f" | R:{num_injected}"
                else:
                    inject_info = ""  # Disabled or not this generation
            
            # Log progress
            gen_time = time.time() - gen_start_time
            crossover_info = f" | X:{pop.last_crossover_count}" if pop.last_crossover_count > 0 else ""
            emb_dim = CONFIG["embedding_dim"]
            hidden = CONFIG["controller_hidden_size"]
            print(f"[P{phase_idx + 1} Gen {phase_gen}] T{current_tier} | Acc: {avg_acc:.1%}/{max_acc:.1%}/{min_acc:.1%} | Triplet: {avg_triplet:.2%}/{max_triplet:.2%}/{min_triplet:.2%} | Th:{threshold:.1f} | Trust: {highest_trust:.0f}/{median_trust:.0f}/{lowest_trust:.0f}{crossover_info}{inject_info} | E{emb_dim}H{hidden} | {gen_time:.1f}s")
            
            # Check for plateau and auto-scale (disabled in phase 3)
            if not is_phase_3 and plateau_detector.check(avg_acc, phase_mastery_threshold):
                print(f"\n>>> PLATEAU DETECTED at {avg_acc:.1%} for {AUTO_SCALE_CONFIG['plateau_generations']} generations")
                if pop.scale_architecture(AUTO_SCALE_CONFIG.get("scale_factor", 1.5)):
                    scale_events += 1
                    save_checkpoint(pop, vocab, phase_gen, current_tier, CONFIG, threshold)
                    plateau_detector.reset()
            
            # Check for tier mastery
            if curriculum.check_mastery(avg_acc):
                # Advanced to a new tier
                current_tier = curriculum.current_tier
                plateau_detector.reset()
                
                # Save checkpoint at tier transition
                save_checkpoint(pop, vocab, phase_gen, current_tier - 1, CONFIG, threshold)
                
                # Advance to next tier
                # Only expand population if not already fully expanded from previous phase
                if not population_fully_expanded:
                    curriculum.advance_population(pop)
                else:
                    # Just update the population's vocab reference without expanding
                    pop.vocab = vocab
                env = curriculum.get_env()
                print(f">>> TIER ADVANCE: Now training Tier {current_tier} at threshold {threshold}")
            
            # Check if we've MASTERED the MAX tier (check_mastery returns False when no next tier)
            elif current_tier == max_tier and avg_acc >= phase_mastery_threshold:
                print(f"\n>>> FINAL TIER MASTERY! Accuracy: {avg_acc:.1%}")
                print(f">>> PHASE {phase_idx + 1} COMPLETE: All tiers mastered at threshold {threshold}")
                plateau_detector.reset()
                save_checkpoint(pop, vocab, phase_gen, current_tier, CONFIG, threshold)
                population_fully_expanded = True  # Mark that embeddings are now at full size
                break
            
            # Periodic checkpoint
            if phase_gen > 0 and phase_gen % CONFIG["checkpoint_interval"] == 0:
                save_checkpoint(pop, vocab, phase_gen, current_tier, CONFIG, threshold)
            
            phase_gen += 1
            total_gen += 1
            
            # Safety: max generations per phase
            if phase_gen >= CONFIG["generations"]:
                print(f"\n>>> Max generations ({CONFIG['generations']}) reached in phase {phase_idx + 1}")
                break
        
        # Note: checkpoint already saved when max tier was mastered
        print(f"\n[Phase {phase_idx + 1}] Phase complete. Continuing to next phase...\n")
    
    # ================================================================
    # TRAINING COMPLETE
    # ================================================================
    print()
    print("=" * 60)
    print("ALL PHASES COMPLETE!")
    print(f"Threshold phases completed: {THRESHOLD_PHASES}")
    print(f"Total generations: {total_gen}")
    print(f"Scale events: {scale_events}")
    print("=" * 60)
    
    # Final save
    save_checkpoint(pop, vocab, total_gen, current_tier, CONFIG, THRESHOLD_PHASES[-1])
    print("\n[Done] Training complete!")


if __name__ == "__main__":
    train()
