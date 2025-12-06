# ================================================================
# GENREG Generalization Test - FIXED VERSION
# ================================================================
# Critical fix: Uses genome.get_embedding_tensor() which applies
# the shared/private embedding mix ratio, instead of directly
# accessing shared embeddings only.
# ================================================================

import torch
import random
import pickle
import sys
import re
import json
import os
from pathlib import Path

from config import TIER_VOCAB, CHECKPOINT_DIR, CONFIG, DATA_DIR, DEVICE

# Import sentence generation logic
from dataset_generator import get_cumulative_vocab, categorize_words, generate_sentence_pattern

# ==============================================================================
# CHECKPOINT LOADING
# ==============================================================================

def find_best_checkpoint(checkpoint_dir="checkpoints"):
    """
    Find the best checkpoint: lowest threshold (most advanced phase) with highest tier.
    """
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint directory '{checkpoint_dir}' not found.")
        return None
    
    checkpoints = list(checkpoint_path.glob("*.pkl"))
    
    if not checkpoints:
        print(f"Error: No checkpoint files found in '{checkpoint_dir}'")
        return None
    
    new_format = re.compile(r'checkpoint_th(\d+)_(\d+)_tier(\d+)_gen(\d+)_')
    legacy_format = re.compile(r'checkpoint_gen(\d+)_tier(\d+)_')
    
    best_checkpoint = None
    best_score = (float('inf'), -1, -1)
    
    for cp in checkpoints:
        filename = cp.name
        
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
                best_checkpoint = (cp, threshold, tier, gen)
            continue
        
        match = legacy_format.match(filename)
        if match:
            gen = int(match.group(1))
            tier = int(match.group(2))
            threshold = 999.0
            
            score = (threshold, -tier, -gen)
            
            if score < best_score:
                best_score = score
                best_checkpoint = (cp, threshold, tier, gen)
    
    if best_checkpoint:
        cp, threshold, tier, gen = best_checkpoint
        if threshold < 999:
            print(f"Found best checkpoint: {cp.name}")
            print(f"  Threshold: {threshold}, Tier: {tier}, Generation: {gen}")
        else:
            print(f"Found legacy checkpoint: {cp.name}")
            print(f"  Tier: {tier}, Generation: {gen}")
        return cp, threshold, tier, gen
    
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    print(f"No parseable checkpoint found, using most recent: {checkpoints[0].name}")
    return checkpoints[0], 1.7, 1, 0


# ==============================================================================
# FIXED: Use genome's embedding method instead of raw shared embeddings
# ==============================================================================

def get_context_embeddings_from_genome(genome, context_ids, embedding_dim, device):
    """
    Get concatenated embeddings for context word IDs using the genome's
    get_embedding_tensor method, which properly applies shared/private mix.
    """
    emb_list = []
    for word_id in context_ids:
        if word_id is not None:
            # THIS IS THE FIX: Use genome's method which applies mix ratio
            emb = genome.get_embedding_tensor(word_id)
        else:
            emb = torch.zeros(embedding_dim, device=device)
        emb_list.append(emb)
    return torch.cat(emb_list)


def evaluate_prediction_from_genome(predicted, target_id, genome, threshold):
    """
    Evaluate if prediction hits target embedding.
    Uses genome's get_embedding_tensor for consistency.
    """
    # THIS IS THE FIX: Get target embedding the same way training does
    target_embedding = genome.get_embedding_tensor(target_id)
    distance = torch.sqrt(torch.sum((predicted - target_embedding) ** 2)).item()
    hit = distance < threshold
    return hit, distance


# ==============================================================================
# NOVEL SENTENCE GENERATION
# ==============================================================================

def generate_novel_sentences(tier, count=500, seed=99999):
    """
    Generate novel sentences that were NOT in the training set.
    Uses a different random seed to ensure different selections.
    """
    random.seed(seed)
    
    vocab = get_cumulative_vocab(tier)
    cats = categorize_words(vocab)
    
    from config import TIER_SENTENCE_LENGTHS
    min_len, max_len = TIER_SENTENCE_LENGTHS.get(tier, (3, 5))
    
    sentences = []
    seen = set()
    attempts = 0
    max_attempts = count * 20
    
    while len(sentences) < count and attempts < max_attempts:
        attempts += 1
        
        words, blank_pos, answer = generate_sentence_pattern(cats, min_len, max_len, tier)
        
        if words is None:
            continue
        
        key = (tuple(words), blank_pos)
        if key in seen:
            continue
        seen.add(key)
        
        if not all(w in vocab or w == "is" for w in words):
            continue
        
        sentences.append({
            "words": words,
            "blank_pos": blank_pos,
            "answer": answer
        })
    
    random.seed()
    
    return sentences


def load_training_sentences(tier):
    """Load the original training sentences."""
    data_path = os.path.join(DATA_DIR, f"tier{tier}.json")
    
    if not os.path.exists(data_path):
        print(f"Warning: Training data not found at {data_path}")
        return []
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    return data.get("sentences", [])


# ==============================================================================
# MAIN EVALUATION - FIXED
# ==============================================================================

def prepare_task(sentence, vocab, context_window):
    """Convert a sentence dict to evaluation-ready format using vocabulary object."""
    words = sentence["words"]
    blank_pos = sentence["blank_pos"]
    answer = sentence["answer"]
    
    # Get context word IDs using vocabulary
    left_words = words[:blank_pos]
    right_words = words[blank_pos + 1:]
    
    # Convert to IDs using vocabulary's get_id method
    left_ids = [vocab.get_id(w) for w in left_words]
    right_ids = [vocab.get_id(w) for w in right_words]
    
    # Pad left context (pad at beginning)
    while len(left_ids) < context_window:
        left_ids.insert(0, vocab.pad_id)
    
    # Pad right context (pad at end)
    while len(right_ids) < context_window:
        right_ids.append(vocab.pad_id)
    
    # Take only context_window items
    left_ids = left_ids[-context_window:]
    right_ids = right_ids[:context_window]
    
    target_id = vocab.get_id(answer)
    
    return {
        "left_context": left_ids,
        "right_context": right_ids,
        "target_id": target_id,
        "answer": answer,
        "sentence": " ".join(words),
        "blank_pos": blank_pos
    }


def evaluate_on_sentences(sentences, genome, vocab, context_window, embedding_dim, threshold, device):
    """
    Evaluate model on a set of sentences.
    
    FIXED: Uses genome's get_embedding_tensor for both context and target.
    """
    hits = 0
    total = 0
    distances = []
    
    for sentence in sentences:
        task = prepare_task(sentence, vocab, context_window)
        
        if task["target_id"] is None:
            continue  # Skip if answer word not in vocab
        
        # Get context embeddings USING GENOME'S METHOD
        left_emb = get_context_embeddings_from_genome(
            genome, task["left_context"], embedding_dim, device
        )
        right_emb = get_context_embeddings_from_genome(
            genome, task["right_context"], embedding_dim, device
        )
        
        # Predict using genome's controller
        with torch.no_grad():
            predicted = genome.controller(
                left_emb.unsqueeze(0),
                right_emb.unsqueeze(0)
            ).squeeze(0)
        
        # Evaluate USING GENOME'S METHOD for target embedding
        hit, distance = evaluate_prediction_from_genome(
            predicted, task["target_id"], genome, threshold
        )
        
        distances.append(distance)
        total += 1
        if hit:
            hits += 1
    
    accuracy = hits / total if total > 0 else 0.0
    avg_distance = sum(distances) / len(distances) if distances else 0.0
    
    return hits, total, accuracy, avg_distance


def check_sentence_overlap(training, novel):
    """Check how many novel sentences appear in training."""
    training_keys = set()
    for s in training:
        key = (tuple(s["words"]), s["blank_pos"])
        training_keys.add(key)
    
    overlap = 0
    for s in novel:
        key = (tuple(s["words"]), s["blank_pos"])
        if key in training_keys:
            overlap += 1
    
    return overlap


def run_generalization_test(checkpoint_path, novel_seed=99999, novel_count=500):
    """
    Run the generalization test.
    
    FIXED: Properly loads and uses the genome with its hybrid embeddings.
    """
    print(f"\n{'='*70}")
    print("GENERALIZATION vs MEMORIZATION TEST (FIXED)")
    print(f"{'='*70}")
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    
    # Get the population and best genome
    population = data['population']
    
    # Move population to device and restore shared embeddings
    for g in population.genomes:
        g.device = DEVICE
        g.embeddings = g.embeddings.to(DEVICE)
        g.controller.to(DEVICE)
    
    # Restore shared embeddings properly
    if population.shared_embedding_layer is not None:
        population.shared_embedding_layer.device = DEVICE
        population.shared_embedding_layer.load_state_dict(data['shared_embeddings'])
        population.shared_embedding_layer.embeddings = population.shared_embedding_layer.embeddings.to(DEVICE)
        
        # CRITICAL: Reconnect genomes to shared embeddings
        for g in population.genomes:
            g.shared_embeddings = population.shared_embedding_layer
    
    # Find best genome
    best_genome = max(population.genomes, key=lambda g: getattr(g, 'accuracy', 0))
    best_genome.controller.eval()
    
    # Get checkpoint metadata
    generation = data.get('generation', 'unknown')
    vocab_tier = data.get('vocab_tier', data.get('tier', 1))
    threshold = data.get('threshold_phase', CONFIG.get('vector_hit_threshold', 1.7))
    
    # Get embedding info
    embedding_dim = best_genome.embedding_dim
    vocab_size = best_genome.vocab_size
    shared_mix = getattr(best_genome, 'shared_mix_ratio', 0.7)
    use_shared = getattr(best_genome, 'use_shared', True)
    
    print(f"\nCheckpoint Info:")
    print(f"  Generation: {generation}")
    print(f"  Tier: {vocab_tier}")
    print(f"  Threshold: {threshold}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Best genome accuracy: {getattr(best_genome, 'accuracy', 'N/A')}")
    print(f"  Shared embedding mix ratio: {shared_mix}")
    print(f"  Using shared embeddings: {use_shared}")
    
    # Rebuild vocabulary using the SAME method as training
    from vocabulary import TieredVocabulary
    vocab = TieredVocabulary()
    vocab.set_tier(vocab_tier)
    
    context_window = CONFIG.get("context_window", 4)
    
    # Verify vocabulary matches
    print(f"\n  Vocabulary rebuilt: {len(vocab)} words")
    if len(vocab) != vocab_size:
        print(f"  WARNING: Vocab size mismatch! Checkpoint: {vocab_size}, Rebuilt: {len(vocab)}")
    
    # Load training sentences
    print(f"\n{'─'*70}")
    print("Loading datasets...")
    training_sentences = load_training_sentences(vocab_tier)
    print(f"  Training sentences: {len(training_sentences)}")
    
    # Generate novel sentences
    print(f"  Generating {novel_count} novel sentences (seed={novel_seed})...")
    novel_sentences = generate_novel_sentences(vocab_tier, novel_count, novel_seed)
    print(f"  Novel sentences generated: {len(novel_sentences)}")
    
    # Check for overlap
    overlap = check_sentence_overlap(training_sentences, novel_sentences)
    if overlap > 0:
        print(f"  WARNING: {overlap} novel sentences overlap with training!")
        training_keys = set((tuple(s["words"]), s["blank_pos"]) for s in training_sentences)
        novel_sentences = [s for s in novel_sentences if (tuple(s["words"]), s["blank_pos"]) not in training_keys]
        print(f"  Filtered to {len(novel_sentences)} truly novel sentences")
    
    # Evaluate on training data
    print(f"\n{'─'*70}")
    print("Evaluating on TRAINING sentences...")
    train_hits, train_total, train_acc, train_avg_dist = evaluate_on_sentences(
        training_sentences, best_genome, vocab,
        context_window, embedding_dim, threshold, DEVICE
    )
    print(f"  Training Accuracy: {train_acc:.1%} ({train_hits}/{train_total})")
    print(f"  Average distance: {train_avg_dist:.3f}")
    
    # Evaluate on novel data
    print(f"\nEvaluating on NOVEL sentences...")
    novel_hits, novel_total, novel_acc, novel_avg_dist = evaluate_on_sentences(
        novel_sentences, best_genome, vocab,
        context_window, embedding_dim, threshold, DEVICE
    )
    print(f"  Novel Accuracy: {novel_acc:.1%} ({novel_hits}/{novel_total})")
    print(f"  Average distance: {novel_avg_dist:.3f}")
    
    # Calculate drop
    accuracy_drop = train_acc - novel_acc
    relative_drop = (accuracy_drop / train_acc * 100) if train_acc > 0 else 0
    
    # Results and Diagnosis
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Training Set Accuracy:    {train_acc:>6.1%}")
    print(f"  Novel Sentences Accuracy: {novel_acc:>6.1%}")
    print(f"  Accuracy Drop:            {accuracy_drop:>6.1%} ({relative_drop:.0f}% relative)")
    print(f"  Avg Distance (train):     {train_avg_dist:.3f}")
    print(f"  Avg Distance (novel):     {novel_avg_dist:.3f}")
    
    print(f"\n{'─'*70}")
    print("DIAGNOSIS")
    print(f"{'─'*70}")
    
    # Random baseline
    vocab_words = len(vocab) - 2  # Exclude PAD and BLANK
    random_baseline = 1 / vocab_words if vocab_words > 0 else 0
    
    # Interpretation
    if train_acc < 0.05:
        print("✗ EVALUATION ERROR")
        print("  Training accuracy is near zero - something is still wrong.")
        print("  Check that vocabulary IDs match between training and evaluation.")
    elif accuracy_drop < 0.05:
        print("✓ GENERALIZES WELL")
        print("  The model learned patterns, not specific sentences.")
        print("  Novel accuracy matches training accuracy.")
    elif accuracy_drop < 0.20:
        print("⚠ PARTIAL GENERALIZATION")
        print("  Some learning, some memorization.")
        print("  The model learned some patterns but also memorized specifics.")
    elif accuracy_drop < 0.40:
        print("⚠ SIGNIFICANT MEMORIZATION")
        print("  The model memorized many training sentences.")
        print("  Limited ability to generalize to new contexts.")
    else:
        print("✗ SEVERE MEMORIZATION")
        print("  The model mostly memorized training sentences.")
        print("  Very poor generalization to novel sentences.")
        print("  This is classic overfitting.")
    
    print(f"\n  Random guessing baseline: {random_baseline:.1%} (1/{vocab_words} words)")
    
    if novel_acc < random_baseline * 2:
        print("  ⚠ Novel accuracy is near random chance!")
    elif novel_acc > random_baseline * 10:
        print("  ✓ Novel accuracy is significantly above random chance!")
    
    print(f"{'='*70}")
    
    return train_acc, novel_acc, accuracy_drop


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test if model memorized or learned generalizable patterns (FIXED)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark3_fixed.py                              # Auto-find best checkpoint
  python benchmark3_fixed.py checkpoints/checkpoint.pkl  # Use specific checkpoint
  python benchmark3_fixed.py --novel-count 1000          # Generate 1000 novel sentences
  python benchmark3_fixed.py --seed 42                   # Use specific random seed
        """
    )
    
    parser.add_argument(
        "checkpoint",
        type=str,
        nargs='?',
        help="Path to checkpoint file (.pkl)"
    )
    
    parser.add_argument(
        "--novel-count", "-n",
        type=int,
        default=500,
        help="Number of novel sentences to generate (default: 500)"
    )
    
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=99999,
        help="Random seed for novel sentence generation (default: 99999)"
    )
    
    args = parser.parse_args()
    
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not Path(checkpoint_path).exists():
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
    else:
        print("No checkpoint specified. Finding best checkpoint...")
        result = find_best_checkpoint(CHECKPOINT_DIR)
        
        if result is None:
            print("\nUsage: python benchmark3_fixed.py [checkpoint_path]")
            sys.exit(1)
        
        checkpoint_path = result[0]
    
    run_generalization_test(
        checkpoint_path,
        novel_seed=args.seed,
        novel_count=args.novel_count
    )
