# ================================================================
# GENREG Proximity Benchmark
# ================================================================
# Tests word embeddings using semantic proximity
# Question: Is "hot" closer to "cold" than it is to "cat"?
# Automatically loads the best checkpoint (lowest threshold, highest tier)
# ================================================================

import torch
import torch.nn.functional as F
import pickle
import sys
import re
from pathlib import Path

from config import TIER_VOCAB, CHECKPOINT_DIR

# ==============================================================================
# PROXIMITY TEST SUITE
# ==============================================================================
# Format: (word, should_be_close, should_be_far)
# Test: Is 'word' closer to 'should_be_close' than to 'should_be_far'?
# ==============================================================================

TEST_SUITE = [
    # --- TIER 1 TESTS (Basic vocabulary) ---
    
    # Animals should cluster together
    ("cat", "dog", "red"),           # cat closer to dog than to red
    ("dog", "bird", "big"),          # dog closer to bird than to big
    ("bird", "fish", "the"),         # bird closer to fish than to the
    ("fish", "cat", "runs"),         # fish closer to cat than to runs
    
    # Colors should cluster together
    ("red", "blue", "cat"),          # red closer to blue than to cat
    ("blue", "green", "dog"),        # blue closer to green than to dog
    ("green", "red", "tree"),        # green closer to red than to tree
    
    # Size adjectives should cluster
    ("big", "small", "cat"),         # big closer to small than to cat
    ("small", "big", "sun"),         # small closer to big than to sun
    
    # Verbs should cluster together
    ("runs", "sits", "cat"),         # runs closer to sits than to cat
    ("sits", "eats", "red"),         # sits closer to eats than to red
    ("eats", "runs", "tree"),        # eats closer to runs than to tree
    
    # Nature words
    ("tree", "grass", "runs"),       # tree closer to grass than to runs
    ("sun", "sky", "cat"),           # sun closer to sky than to cat
    ("sky", "sun", "dog"),           # sky closer to sun than to dog
    
    # --- TIER 2 TESTS (Extended vocabulary) ---
    
    # More animals/nature
    ("house", "car", "jumps"),       # objects together
    ("ball", "book", "swims"),       # objects together
    ("water", "fire", "tall"),       # elements together
    
    # More verbs
    ("jumps", "walks", "house"),     # verbs together
    ("flies", "swims", "rock"),      # verbs together
    ("grows", "walks", "leaf"),      # verbs together
    
    # Temperature adjectives
    ("hot", "cold", "cat"),          # temperature together
    ("cold", "hot", "ball"),         # temperature together
    
    # Speed adjectives  
    ("fast", "slow", "water"),       # speed together
    ("slow", "fast", "fire"),        # speed together
    
    # More colors
    ("yellow", "black", "jumps"),    # colors together
    ("black", "yellow", "grows"),    # colors together
    
    # --- TIER 3+ TESTS ---
    
    # Prepositions
    ("in", "on", "man"),             # prepositions together
    ("at", "to", "woman"),           # prepositions together
    ("from", "with", "boy"),         # prepositions together
    
    # Pronouns
    ("he", "she", "door"),           # pronouns together
    ("they", "we", "window"),        # pronouns together
    
    # Body parts
    ("hand", "foot", "in"),          # body parts together
    ("head", "eye", "on"),           # body parts together
    
    # People
    ("man", "woman", "door"),        # people together
    ("boy", "girl", "floor"),        # people together
    
    # More verbs
    ("sees", "hears", "man"),        # perception verbs together
    ("goes", "comes", "hand"),       # movement verbs together
    
    # --- TIER 4+ TESTS ---
    
    # Places
    ("room", "kitchen", "horse"),    # places together
    ("garden", "park", "cow"),       # outdoor places together
    ("school", "store", "pig"),      # buildings together
    
    # Farm animals
    ("horse", "cow", "room"),        # farm animals together
    ("pig", "sheep", "kitchen"),     # farm animals together
    ("chicken", "duck", "garden"),   # birds together
    
    # Time words
    ("morning", "night", "arm"),     # time together
    ("today", "now", "leg"),         # time together
    
    # --- TIER 5+ TESTS ---
    
    # Emotions
    ("happy", "sad", "table"),       # emotions together
    ("angry", "scared", "chair"),    # emotions together
    ("tired", "hungry", "bed"),      # states together
    
    # Food
    ("bread", "milk", "happy"),      # food together
    ("apple", "orange", "sad"),      # fruits together
    ("meat", "egg", "angry"),        # protein together
    
    # Furniture
    ("table", "chair", "bread"),     # furniture together
    ("bed", "desk", "milk"),         # furniture together
    
    # Weather
    ("rain", "snow", "table"),       # weather together
    ("wind", "storm", "chair"),      # weather together
    
    # Adverbs
    ("quickly", "slowly", "rain"),   # adverbs together
    ("always", "never", "snow"),     # frequency together
]


def find_best_checkpoint(checkpoint_dir="checkpoints"):
    """
    Find the best checkpoint: lowest threshold (most advanced phase) with highest tier.
    
    Priority: Lower threshold > Higher tier > Higher generation
    
    Returns the path to the checkpoint file.
    """
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint directory '{checkpoint_dir}' not found.")
        return None
    
    # Find all .pkl files
    checkpoints = list(checkpoint_path.glob("*.pkl"))
    
    if not checkpoints:
        print(f"Error: No checkpoint files found in '{checkpoint_dir}'")
        return None
    
    # Parse from new format: checkpoint_th{X}_{Y}_tier{Z}_gen{N}_{timestamp}.pkl
    new_format = re.compile(r'checkpoint_th(\d+)_(\d+)_tier(\d+)_gen(\d+)_')
    # Legacy format: checkpoint_gen{N}_tier{Z}_{timestamp}.pkl
    legacy_format = re.compile(r'checkpoint_gen(\d+)_tier(\d+)_')
    
    best_checkpoint = None
    best_score = (float('inf'), -1, -1)  # (threshold, -tier, -gen) - lower threshold is better
    
    for cp in checkpoints:
        filename = cp.name
        
        # Try new format first
        match = new_format.match(filename)
        if match:
            th_major = int(match.group(1))
            th_minor = int(match.group(2))
            threshold = float(f"{th_major}.{th_minor}")
            tier = int(match.group(3))
            gen = int(match.group(4))
            
            # Score: lower threshold better, then higher tier, then higher gen
            # Use negative tier/gen so lower tuple = better
            score = (threshold, -tier, -gen)
            
            if score < best_score:
                best_score = score
                best_checkpoint = (cp, threshold, tier, gen)
            continue
        
        # Try legacy format
        match = legacy_format.match(filename)
        if match:
            gen = int(match.group(1))
            tier = int(match.group(2))
            threshold = 999.0  # Legacy checkpoints treated as lowest priority
            
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
        return cp
    
    # Fallback: return the most recently modified checkpoint
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    print(f"No parseable checkpoint found, using most recent: {checkpoints[0].name}")
    return checkpoints[0]


def cosine_distance(v1, v2):
    """Compute cosine distance (1 - cosine_similarity). Lower = closer."""
    sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
    return (1 - sim).item()


def euclidean_distance(v1, v2):
    """Compute Euclidean distance. Lower = closer."""
    return torch.norm(v1 - v2).item()


def run_benchmark(checkpoint_path, use_cosine=True):
    print(f"\n{'='*70}")
    print(f"LOADING CHECKPOINT: {checkpoint_path}")
    print(f"{'='*70}")
    
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract the embeddings
    shared_state = data['shared_embeddings']
    embeddings = shared_state['embeddings']
    vocab_size = shared_state['vocab_size']
    generation = data.get('generation', 'unknown')
    
    print(f"Generation: {generation}")
    print(f"Vocab size: {vocab_size}")
    print(f"Embedding dim: {embeddings.shape[1]}")
    
    # Determine tier from vocab size
    vocab_tier = data.get('vocab_tier', None)
    
    if vocab_tier is None:
        cumulative = 2  # PAD + BLANK
        for t in range(1, 11):
            if t in TIER_VOCAB:
                cumulative += len(TIER_VOCAB[t])
                if cumulative >= vocab_size:
                    vocab_tier = t
                    break
        if vocab_tier is None:
            vocab_tier = 10
    
    print(f"Tier: {vocab_tier}")
    
    # Rebuild word-to-id mapping (only up to actual vocab_size)
    word_to_id = {}
    current_id = 2  # Skip PAD (0) and BLANK (1)
    
    for t in range(1, vocab_tier + 1):
        if t in TIER_VOCAB:
            for w in TIER_VOCAB[t]:
                if w not in word_to_id and current_id < vocab_size:
                    word_to_id[w] = current_id
                    current_id += 1
    
    print(f"Vocabulary rebuilt: {len(word_to_id)} words (vocab_size: {vocab_size})")
    
    distance_fn = cosine_distance if use_cosine else euclidean_distance
    distance_name = "Cosine" if use_cosine else "Euclidean"
    
    # RUN TESTS
    print(f"\n{'='*70}")
    print(f"GENREG PROXIMITY BENCHMARK ({distance_name} Distance)")
    print(f"{'='*70}")
    print(f"Test: Is WORD closer to CLOSE than to FAR?")
    print(f"{'='*70}")
    print(f"{'WORD':<10} | {'CLOSE':<10} | {'FAR':<10} | {'D(CLOSE)':<10} | {'D(FAR)':<10} | {'RESULT'}")
    print("-" * 75)
    
    score = 0
    total = 0
    skipped = 0
    
    # Track category performance
    results_by_margin = []
    
    for word, close, far in TEST_SUITE:
        # Check if words exist in current vocab
        if not all(w in word_to_id for w in [word, close, far]):
            skipped += 1
            continue
        
        # Verify all IDs are within embedding bounds
        ids = [word_to_id[word], word_to_id[close], word_to_id[far]]
        if any(idx >= embeddings.shape[0] for idx in ids):
            skipped += 1
            continue
        
        total += 1
        
        # Get vectors
        v_word = embeddings[word_to_id[word]]
        v_close = embeddings[word_to_id[close]]
        v_far = embeddings[word_to_id[far]]
        
        # Calculate distances
        dist_close = distance_fn(v_word, v_close)
        dist_far = distance_fn(v_word, v_far)
        
        # Test: Is word closer to 'close' than to 'far'?
        is_correct = dist_close < dist_far
        margin = dist_far - dist_close  # Positive = correct
        
        if is_correct:
            score += 1
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        
        results_by_margin.append((word, close, far, margin, is_correct))
        
        print(f"{word:<10} | {close:<10} | {far:<10} | {dist_close:<10.4f} | {dist_far:<10.4f} | {status}")

    print("=" * 75)
    
    if total > 0:
        percentage = score / total * 100
        print(f"\nFINAL SCORE: {score}/{total} ({percentage:.1f}%)")
        
        # Random baseline is 50%
        print(f"(Random baseline: 50%)")
        
        # Interpretation
        if percentage >= 90:
            print("\nResult: EXCELLENT - Strong semantic clustering!")
        elif percentage >= 75:
            print("\nResult: GOOD - Clear semantic structure.")
        elif percentage >= 60:
            print("\nResult: MODERATE - Some semantic understanding.")
        elif percentage > 50:
            print("\nResult: WEAK - Slightly better than random.")
        else:
            print("\nResult: POOR - No better than random guessing.")
        
        # Show worst failures (largest negative margins)
        failures = [(w, c, f, m) for w, c, f, m, correct in results_by_margin if not correct]
        if failures:
            failures.sort(key=lambda x: x[3])  # Sort by margin (most negative first)
            print(f"\nWorst failures (word should be closer to CLOSE than FAR):")
            for w, c, f, m in failures[:5]:
                print(f"  • {w} should be closer to {c} than {f} (margin: {m:.4f})")
        
        # Show best successes (largest positive margins)
        successes = [(w, c, f, m) for w, c, f, m, correct in results_by_margin if correct]
        if successes:
            successes.sort(key=lambda x: x[3], reverse=True)
            print(f"\nBest successes:")
            for w, c, f, m in successes[:5]:
                print(f"  • {w} is closer to {c} than {f} (margin: +{m:.4f})")
    else:
        print(f"\nNo valid tests found for tier {vocab_tier}.")
    
    if skipped > 0:
        print(f"\n(Skipped {skipped} tests - words not yet in vocabulary)")
    
    print()
    return score, total


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test embedding proximity: Is A closer to B than to C?",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py                              # Auto-find best checkpoint (lowest threshold, highest tier)
  python benchmark.py checkpoints/checkpoint.pkl  # Use specific checkpoint
  python benchmark.py --euclidean                 # Use Euclidean distance
        """
    )
    
    parser.add_argument(
        "checkpoint",
        type=str,
        nargs='?',
        help="Path to checkpoint file (.pkl)"
    )
    
    parser.add_argument(
        "--euclidean", "-e",
        action="store_true",
        help="Use Euclidean distance instead of Cosine"
    )
    
    args = parser.parse_args()
    
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not Path(checkpoint_path).exists():
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
    else:
        print("No checkpoint specified. Finding best checkpoint (lowest threshold, highest tier)...")
        checkpoint_path = find_best_checkpoint(CHECKPOINT_DIR)
        
        if checkpoint_path is None:
            print("\nUsage: python benchmark.py [checkpoint_path]")
            sys.exit(1)
    
    run_benchmark(checkpoint_path, use_cosine=not args.euclidean)
