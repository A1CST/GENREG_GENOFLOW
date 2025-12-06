# ================================================================
# GENREG Phase 4 - Next-Word Prediction Training
# ================================================================
# Trains CausalController for autoregressive generation
# Uses embeddings from Phase 3 (threshold 1.0) checkpoint
# ================================================================

import os
import sys
import time
import json
import pickle
import random
import argparse
import glob
import re
import torch
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path

from config import (
    CONFIG, DEVICE, CHECKPOINT_DIR, TIER_VOCAB, DATA_DIR,
    PREDICTOR_CONFIG, PREDICTOR_CHECKPOINT_DIR
)
from predictor import CausalController
from vocabulary import TieredVocabulary


def list_available_checkpoints(checkpoint_dir=CHECKPOINT_DIR):
    """List all available embedding checkpoints."""
    if not os.path.exists(checkpoint_dir):
        print(f"[Error] Checkpoint directory not found: {checkpoint_dir}")
        return []
    
    pattern = os.path.join(checkpoint_dir, "checkpoint_th*.pkl")
    checkpoint_files = glob.glob(pattern)
    
    if not checkpoint_files:
        print(f"[Error] No checkpoints found in {checkpoint_dir}")
        return []
    
    # Parse and sort checkpoints
    checkpoints = []
    format_pattern = re.compile(r'checkpoint_th(\d+)_(\d+)_tier(\d+)_gen(\d+)_')
    
    for filepath in checkpoint_files:
        filename = os.path.basename(filepath)
        match = format_pattern.match(filename)
        if match:
            th_major = int(match.group(1))
            th_minor = int(match.group(2))
            threshold = float(f"{th_major}.{th_minor}")
            tier = int(match.group(3))
            gen = int(match.group(4))
            checkpoints.append((filepath, threshold, tier, gen, filename))
    
    # Sort by threshold (asc), tier (desc), gen (desc)
    checkpoints.sort(key=lambda x: (x[1], -x[2], -x[3]))
    
    print("\nAvailable checkpoints:")
    print("-" * 70)
    for i, (fp, th, tier, gen, fname) in enumerate(checkpoints):
        print(f"  [{i+1}] th={th}, tier={tier}, gen={gen}")
        print(f"      {fname}")
    print("-" * 70)
    
    return checkpoints


def find_best_embedding_checkpoint(checkpoint_dir=CHECKPOINT_DIR):
    """
    Find the best checkpoint from fill-in-blank training.
    Priority: Lowest threshold (1.0 best) > Highest tier > Highest generation
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    pattern = os.path.join(checkpoint_dir, "checkpoint_th*.pkl")
    checkpoint_files = glob.glob(pattern)
    
    if not checkpoint_files:
        return None
    
    # Pattern: checkpoint_th{X}_{Y}_tier{Z}_gen{N}_{timestamp}.pkl
    new_format = re.compile(r'checkpoint_th(\d+)_(\d+)_tier(\d+)_gen(\d+)_')
    
    best_checkpoint = None
    best_score = (float('inf'), -1, -1)
    
    for filepath in checkpoint_files:
        filename = os.path.basename(filepath)
        match = new_format.match(filename)
        if match:
            th_major = int(match.group(1))
            th_minor = int(match.group(2))
            threshold = float(f"{th_major}.{th_minor}")
            tier = int(match.group(3))
            gen = int(match.group(4))
            
            score = (threshold, -tier, -gen)
            if score < best_score:
                best_score = score
                best_checkpoint = (filepath, threshold, tier, gen)
    
    return best_checkpoint


def load_embeddings_from_checkpoint(filepath):
    """Load shared embeddings and vocabulary info from checkpoint."""
    print(f"[Load] Loading embeddings from: {os.path.basename(filepath)}")
    
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    
    shared_state = checkpoint['shared_embeddings']
    embeddings = shared_state['embeddings'].to(DEVICE)
    vocab_size = shared_state['vocab_size']
    embedding_dim = embeddings.shape[1]
    
    print(f"[Load] Vocab size: {vocab_size}, Embedding dim: {embedding_dim}")
    
    return embeddings, vocab_size, embedding_dim


def build_vocabulary(max_tier=10):
    """Build full vocabulary mapping."""
    word_to_id = {"<PAD>": 0, "<BLANK>": 1}
    id_to_word = {0: "<PAD>", 1: "<BLANK>"}
    
    current_id = 2
    for tier in range(1, max_tier + 1):
        if tier in TIER_VOCAB:
            for word in TIER_VOCAB[tier]:
                if word not in word_to_id:
                    word_to_id[word] = current_id
                    id_to_word[current_id] = word
                    current_id += 1
    
    return word_to_id, id_to_word


def load_all_sentences(data_dir=DATA_DIR, max_tier=10):
    """Load sentences from all tier data files."""
    all_sentences = []
    
    for tier in range(1, max_tier + 1):
        filepath = os.path.join(data_dir, f"tier{tier}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                sentences = data.get('sentences', [])
                all_sentences.extend(sentences)
                print(f"[Data] Loaded {len(sentences)} sentences from tier {tier}")
    
    print(f"[Data] Total sentences: {len(all_sentences)}")
    return all_sentences


class PredictorGenome:
    """
    A genome for next-word prediction.
    Contains a CausalController and uses shared (frozen) embeddings.
    """
    
    def __init__(self, embedding_dim, context_length, hidden_size, shared_embeddings):
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.controller = CausalController(embedding_dim, context_length, hidden_size)
        self.shared_embeddings = shared_embeddings  # Frozen embeddings from Phase 3
        
        self.trust = 0.0
        self.accuracy = 0.0
    
    def get_embedding(self, word_id):
        """Get embedding for a word ID."""
        if word_id is not None and word_id < self.shared_embeddings.shape[0]:
            return self.shared_embeddings[word_id]
        return torch.zeros(self.embedding_dim, device=DEVICE)
    
    def predict(self, context_ids):
        """
        Predict next word embedding given context word IDs.
        
        Args:
            context_ids: List of word IDs (previous words)
        
        Returns:
            Predicted embedding for next word
        """
        # Get embeddings for context
        context_embs = [self.get_embedding(wid) for wid in context_ids]
        
        # Run through controller
        return self.controller.forward_single(context_embs)
    
    def clone(self):
        """Create a deep copy."""
        new_genome = PredictorGenome(
            self.embedding_dim,
            self.context_length,
            self.controller.hidden_size,
            self.shared_embeddings
        )
        new_genome.controller = self.controller.clone()
        return new_genome
    
    def mutate(self, rate=0.05, scale=0.07):
        """Mutate controller weights."""
        self.controller.mutate(rate, scale)


class PredictorPopulation:
    """Population of PredictorGenomes for evolutionary training."""
    
    def __init__(self, size, embedding_dim, context_length, hidden_size, shared_embeddings, config):
        self.size = size
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.hidden_size = hidden_size
        self.shared_embeddings = shared_embeddings
        self.config = config
        
        # Create initial population
        self.genomes = [
            PredictorGenome(embedding_dim, context_length, hidden_size, shared_embeddings)
            for _ in range(size)
        ]
        
        self.generation = 0
    
    def evolve(self):
        """Evolve population based on trust scores."""
        self.generation += 1
        
        # Sort by trust
        self.genomes.sort(key=lambda g: g.trust, reverse=True)
        
        highest_trust = self.genomes[0].trust
        median_trust = self.genomes[self.size // 2].trust
        lowest_trust = self.genomes[-1].trust
        
        # Survival selection
        survival_count = max(2, int(self.size * self.config.get("survival_rate", 0.4)))
        survivors = self.genomes[:survival_count]
        
        # Create new generation
        new_genomes = []
        
        # Elite (best genome with minimal mutation)
        elite = survivors[0].clone()
        elite.mutate(
            rate=self.config.get("elite_mutation_rate", 0.005),
            scale=self.config.get("mutation_scale", 0.1)
        )
        new_genomes.append(elite)
        
        # Fill rest of population
        while len(new_genomes) < self.size:
            parent = random.choice(survivors)
            child = parent.clone()
            
            # Mutation rate based on position
            if len(new_genomes) < self.size * 0.5:
                # Standard mutation
                child.mutate(
                    rate=self.config.get("standard_mutation_rate", 0.1),
                    scale=self.config.get("mutation_scale", 0.1)
                )
            else:
                # Hyper mutation for exploration
                child.mutate(
                    rate=self.config.get("hyper_mutation_rate", 0.2),
                    scale=self.config.get("mutation_scale", 0.1) * 2
                )
            
            new_genomes.append(child)
        
        self.genomes = new_genomes
        return highest_trust, median_trust, lowest_trust


def save_predictor_checkpoint(population, embeddings, word_to_id, generation, accuracy, config):
    """Save predictor checkpoint."""
    os.makedirs(PREDICTOR_CHECKPOINT_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"predictor_gen{generation}_acc{accuracy:.0%}_{timestamp}.pkl"
    filepath = os.path.join(PREDICTOR_CHECKPOINT_DIR, filename)
    
    # Get best genome
    best_genome = population.genomes[0]
    
    # Move to CPU for saving
    best_genome.controller.cpu()
    
    checkpoint_data = {
        'generation': generation,
        'accuracy': accuracy,
        'controller_state': best_genome.controller.state_dict(),
        'embeddings': embeddings.cpu(),
        'word_to_id': word_to_id,
        'config': config,
        'embedding_dim': best_genome.embedding_dim,
        'context_length': best_genome.context_length,
        'hidden_size': best_genome.controller.hidden_size,
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    # Move back to GPU
    best_genome.controller.to(DEVICE)
    
    print(f"[Checkpoint] Saved to {filepath}")
    return filepath


def train_predictor(checkpoint_path=None):
    """
    Main training loop for next-word prediction.
    
    Args:
        checkpoint_path: Optional path to specific checkpoint file.
                        If None, automatically finds the best checkpoint.
    """
    print("=" * 60)
    print("GENREG PHASE 4 - Next-Word Prediction Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print()
    
    # ================================================================
    # LOAD EMBEDDINGS FROM PHASE 3
    # ================================================================
    if checkpoint_path:
        # Use specified checkpoint
        if not os.path.exists(checkpoint_path):
            print(f"[Error] Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        filepath = checkpoint_path
        print(f"[Load] Using specified checkpoint: {os.path.basename(filepath)}")
    else:
        # Auto-find best checkpoint
        checkpoint_info = find_best_embedding_checkpoint()
        
        if checkpoint_info is None:
            print("[Error] No embedding checkpoint found!")
            print("Please complete Phase 3 (fill-in-blank training) first.")
            print("Or specify a checkpoint with: python train_predictor.py <path>")
            sys.exit(1)
        
        filepath, threshold, tier, gen = checkpoint_info
        print(f"[Load] Auto-selected best checkpoint: threshold={threshold}, tier={tier}, gen={gen}")
    
    embeddings, vocab_size, embedding_dim = load_embeddings_from_checkpoint(filepath)
    
    # Build vocabulary
    word_to_id, id_to_word = build_vocabulary()
    print(f"[Vocab] Built vocabulary: {len(word_to_id)} words")
    
    # Load training sentences
    sentences = load_all_sentences()
    random.shuffle(sentences)
    
    # ================================================================
    # INITIALIZE POPULATION
    # ================================================================
    context_length = PREDICTOR_CONFIG["context_length"]
    hidden_size = PREDICTOR_CONFIG["predictor_hidden_size"]
    population_size = CONFIG["population_size"]
    
    population = PredictorPopulation(
        size=population_size,
        embedding_dim=embedding_dim,
        context_length=context_length,
        hidden_size=hidden_size,
        shared_embeddings=embeddings,
        config=PREDICTOR_CONFIG
    )
    
    print(f"[Init] Population: {population_size} genomes")
    print(f"[Init] Context length: {context_length}")
    print(f"[Init] Hidden size: {hidden_size}")
    print()
    
    # ================================================================
    # TRAINING LOOP
    # ================================================================
    max_generations = PREDICTOR_CONFIG["predictor_generations"]
    mastery_threshold = PREDICTOR_CONFIG["mastery_threshold"]
    checkpoint_interval = PREDICTOR_CONFIG["checkpoint_interval"]
    hit_threshold = CONFIG["vector_hit_threshold"]  # Use same threshold as fill-in-blank
    
    best_accuracy = 0.0
    
    for gen in range(max_generations):
        gen_start = time.time()
        
        # Shuffle sentences each generation
        random.shuffle(sentences)
        
        # Evaluate each genome
        accuracies = []
        
        for genome in population.genomes:
            hits = 0
            total = 0
            
            # Sample sentences for this generation (use subset for speed)
            sample_size = min(500, len(sentences))
            sample_sentences = sentences[:sample_size]
            
            for sentence in sample_sentences:
                words = sentence.get('words', [])
                if len(words) < 3:
                    continue
                
                # Convert to IDs
                word_ids = [word_to_id.get(w.lower()) for w in words]
                word_ids = [wid for wid in word_ids if wid is not None]
                
                if len(word_ids) < 3:
                    continue
                
                # Create prediction tasks: predict each word from its context
                for i in range(1, len(word_ids)):
                    context_ids = word_ids[max(0, i - context_length):i]
                    target_id = word_ids[i]
                    
                    # Predict
                    predicted_emb = genome.predict(context_ids)
                    
                    # Get target embedding
                    target_emb = embeddings[target_id]
                    
                    # Check if prediction is close enough
                    distance = torch.norm(predicted_emb - target_emb).item()
                    
                    total += 1
                    if distance < hit_threshold:
                        hits += 1
            
            # Calculate accuracy
            accuracy = hits / total if total > 0 else 0.0
            accuracies.append(accuracy)
            
            # Set trust based on accuracy
            genome.trust = accuracy * 10000
            genome.accuracy = accuracy
        
        # Calculate stats
        avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
        max_acc = max(accuracies) if accuracies else 0.0
        min_acc = min(accuracies) if accuracies else 0.0
        
        # Track best
        if max_acc > best_accuracy:
            best_accuracy = max_acc
        
        # Evolve
        highest_trust, median_trust, lowest_trust = population.evolve()
        
        # Log progress
        gen_time = time.time() - gen_start
        print(f"[Gen {gen}] Avg: {avg_acc:.1%} | Max: {max_acc:.1%} | Min: {min_acc:.1%} | Best: {best_accuracy:.1%} | {gen_time:.1f}s")
        
        # Check mastery
        if avg_acc >= mastery_threshold:
            print(f"\n>>> MASTERY ACHIEVED! Accuracy: {avg_acc:.1%}")
            save_predictor_checkpoint(
                population, embeddings, word_to_id, gen, avg_acc, PREDICTOR_CONFIG
            )
            break
        
        # Periodic checkpoint
        if gen > 0 and gen % checkpoint_interval == 0:
            save_predictor_checkpoint(
                population, embeddings, word_to_id, gen, max_acc, PREDICTOR_CONFIG
            )
    
    # Final save
    save_predictor_checkpoint(
        population, embeddings, word_to_id, gen, max_acc, PREDICTOR_CONFIG
    )
    
    print()
    print("=" * 60)
    print("PHASE 4 TRAINING COMPLETE")
    print(f"Best accuracy: {best_accuracy:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CausalController for next-word prediction (Phase 4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_predictor.py                     # Auto-select best checkpoint
  python train_predictor.py --list              # List available checkpoints
  python train_predictor.py checkpoints/checkpoint_th1_0_tier10_gen500_*.pkl
        """
    )
    
    parser.add_argument(
        "checkpoint",
        type=str,
        nargs='?',
        default=None,
        help="Path to embedding checkpoint file (.pkl)"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available checkpoints and exit"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_checkpoints()
        sys.exit(0)
    
    train_predictor(checkpoint_path=args.checkpoint)

