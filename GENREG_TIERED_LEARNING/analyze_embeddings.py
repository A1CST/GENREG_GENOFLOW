# ================================================================
# Embedding Quality Analyzer
# ================================================================
# Loads a checkpoint and analyzes whether embeddings are meaningful
# or collapsed (all words having the same vector).
# ================================================================

import argparse
import pickle
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

from config import TIER_VOCAB, DEVICE
from vocabulary import TieredVocabulary


def load_checkpoint(checkpoint_path):
    """Load a checkpoint file."""
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_embeddings_from_checkpoint(checkpoint_data):
    """Extract shared embeddings from checkpoint."""
    if 'shared_embeddings' in checkpoint_data:
        emb_state = checkpoint_data['shared_embeddings']
        embeddings = emb_state['embeddings']
        vocab_size = emb_state['vocab_size']
        return embeddings, vocab_size
    else:
        raise ValueError("No shared_embeddings found in checkpoint")


def cosine_distance(v1, v2):
    """Compute cosine distance (1 - cosine_similarity)."""
    v1 = v1.float()
    v2 = v2.float()
    cos_sim = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
    return (1 - cos_sim).item()


def euclidean_distance(v1, v2):
    """Compute Euclidean distance."""
    return torch.norm(v1.float() - v2.float()).item()


def analyze_embeddings(checkpoint_path, verbose=True):
    """
    Main analysis function.
    
    Returns a dict with analysis results.
    """
    print("=" * 60)
    print("EMBEDDING QUALITY ANALYZER")
    print("=" * 60)
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    embeddings, vocab_size = get_embeddings_from_checkpoint(checkpoint)
    generation = checkpoint.get('generation', 'unknown')
    
    print(f"Generation: {generation}")
    print(f"Vocab size: {vocab_size}")
    print(f"Embedding dim: {embeddings.shape[1]}")
    
    # Determine tier from vocab size and build vocabulary
    vocab = TieredVocabulary()
    for tier in range(1, 11):
        vocab.set_tier(tier)
        if vocab.size >= vocab_size:
            break
    
    print(f"Tier: {vocab.current_tier}")
    print()
    
    # Define word groups for comparison
    # Similar words (should have small but non-zero distance)
    similar_pairs = [
        ("cat", "dog"),      # Both animals
        ("bird", "fish"),    # Both animals
        ("red", "blue"),     # Both colors
        ("big", "small"),    # Both size adjectives
        ("runs", "sits"),    # Both verbs
    ]
    
    # Different words (should have larger distance)
    different_pairs = [
        ("cat", "red"),      # Animal vs color
        ("dog", "is"),       # Animal vs verb
        ("tree", "runs"),    # Object vs action
        ("sun", "small"),    # Object vs adjective
        ("the", "cat"),      # Article vs noun
    ]
    
    # Filter to only include words in vocabulary
    def filter_pairs(pairs):
        valid = []
        for w1, w2 in pairs:
            if w1 in vocab and w2 in vocab:
                valid.append((w1, w2))
        return valid
    
    similar_pairs = filter_pairs(similar_pairs)
    different_pairs = filter_pairs(different_pairs)
    
    print("-" * 60)
    print("SIMILAR WORD PAIRS (should be close but not identical)")
    print("-" * 60)
    
    similar_distances = []
    for w1, w2 in similar_pairs:
        id1 = vocab.get_id(w1)
        id2 = vocab.get_id(w2)
        emb1 = embeddings[id1]
        emb2 = embeddings[id2]
        
        euc_dist = euclidean_distance(emb1, emb2)
        cos_dist = cosine_distance(emb1, emb2)
        
        similar_distances.append(euc_dist)
        
        if verbose:
            print(f"  {w1:12s} <-> {w2:12s}  |  Euclidean: {euc_dist:.6f}  |  Cosine: {cos_dist:.6f}")
    
    print()
    print("-" * 60)
    print("DIFFERENT WORD PAIRS (should be farther apart)")
    print("-" * 60)
    
    different_distances = []
    for w1, w2 in different_pairs:
        id1 = vocab.get_id(w1)
        id2 = vocab.get_id(w2)
        emb1 = embeddings[id1]
        emb2 = embeddings[id2]
        
        euc_dist = euclidean_distance(emb1, emb2)
        cos_dist = cosine_distance(emb1, emb2)
        
        different_distances.append(euc_dist)
        
        if verbose:
            print(f"  {w1:12s} <-> {w2:12s}  |  Euclidean: {euc_dist:.6f}  |  Cosine: {cos_dist:.6f}")
    
    # Compute all pairwise distances for a random sample
    print()
    print("-" * 60)
    print("GLOBAL STATISTICS")
    print("-" * 60)
    
    # Sample words for global stats
    all_words = vocab.get_all_words()
    sample_size = min(50, len(all_words))
    sample_words = all_words[:sample_size]
    
    all_distances = []
    for i, w1 in enumerate(sample_words):
        for w2 in sample_words[i+1:]:
            id1 = vocab.get_id(w1)
            id2 = vocab.get_id(w2)
            if id1 is not None and id2 is not None and id1 < vocab_size and id2 < vocab_size:
                dist = euclidean_distance(embeddings[id1], embeddings[id2])
                all_distances.append(dist)
    
    if all_distances:
        all_distances = np.array(all_distances)
        print(f"  Sampled {len(all_distances)} word pairs from {sample_size} words")
        print(f"  Min distance:    {all_distances.min():.6f}")
        print(f"  Max distance:    {all_distances.max():.6f}")
        print(f"  Mean distance:   {all_distances.mean():.6f}")
        print(f"  Std distance:    {all_distances.std():.6f}")
        print(f"  Median distance: {np.median(all_distances):.6f}")
    
    # Check for identical embeddings
    print()
    print("-" * 60)
    print("COLLAPSE CHECK")
    print("-" * 60)
    
    near_zero_threshold = 0.0001
    near_zero_count = np.sum(all_distances < near_zero_threshold) if len(all_distances) > 0 else 0
    collapse_ratio = near_zero_count / len(all_distances) if len(all_distances) > 0 else 0
    
    print(f"  Pairs with distance < {near_zero_threshold}: {near_zero_count} ({collapse_ratio*100:.2f}%)")
    
    # Final verdict
    print()
    print("=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    
    avg_similar = np.mean(similar_distances) if similar_distances else 0
    avg_different = np.mean(different_distances) if different_distances else 0
    global_mean = all_distances.mean() if len(all_distances) > 0 else 0
    global_std = all_distances.std() if len(all_distances) > 0 else 0
    
    verdict = ""
    details = []
    
    # Check for total collapse
    if global_std < 0.01:
        verdict = "TERRIBLE"
        details.append("All embeddings are nearly identical (std < 0.01)")
        details.append("The model has collapsed - all words map to the same vector")
    
    # Check for partial collapse
    elif collapse_ratio > 0.5:
        verdict = "BAD"
        details.append(f"Over {collapse_ratio*100:.0f}% of word pairs are nearly identical")
        details.append("Significant embedding collapse detected")
    
    # Check if similar words are too similar
    elif avg_similar < 0.01:
        verdict = "BAD"
        details.append("Similar words have near-zero distance")
        details.append("Words that should be similar are essentially identical")
    
    # Check if there's no differentiation
    elif abs(avg_similar - avg_different) < 0.05:
        verdict = "MEDIOCRE"
        details.append("Similar and different word pairs have similar distances")
        details.append("The model hasn't learned meaningful semantic relationships")
    
    # Check for good separation with similarity
    elif avg_similar < avg_different and avg_similar > 0.01:
        verdict = "GOOD"
        details.append("Similar words are closer than different words")
        details.append(f"Similar pairs avg: {avg_similar:.4f}, Different pairs avg: {avg_different:.4f}")
        details.append("Embeddings show semantic structure!")
    
    else:
        verdict = "UNCLEAR"
        details.append("Results don't fit expected patterns")
        details.append("Manual inspection recommended")
    
    # Print verdict
    color_map = {
        "GOOD": "\033[92m",      # Green
        "MEDIOCRE": "\033[93m",  # Yellow
        "BAD": "\033[91m",       # Red
        "TERRIBLE": "\033[91m",  # Red
        "UNCLEAR": "\033[94m",   # Blue
    }
    reset = "\033[0m"
    
    color = color_map.get(verdict, "")
    print(f"\n  Result: {color}{verdict}{reset}\n")
    
    for detail in details:
        print(f"  • {detail}")
    
    print()
    print("=" * 60)
    
    # Return structured results
    return {
        "checkpoint": str(checkpoint_path),
        "generation": generation,
        "vocab_size": vocab_size,
        "embedding_dim": embeddings.shape[1],
        "tier": vocab.current_tier,
        "verdict": verdict,
        "avg_similar_distance": avg_similar,
        "avg_different_distance": avg_different,
        "global_mean_distance": global_mean,
        "global_std_distance": global_std,
        "collapse_ratio": collapse_ratio,
    }


def compare_word_pairs(checkpoint_path, word_pairs):
    """
    Compare specific word pairs provided by user.
    
    word_pairs: list of tuples [(word1, word2), ...]
    """
    checkpoint = load_checkpoint(checkpoint_path)
    embeddings, vocab_size = get_embeddings_from_checkpoint(checkpoint)
    
    vocab = TieredVocabulary()
    for tier in range(1, 11):
        vocab.set_tier(tier)
        if vocab.size >= vocab_size:
            break
    
    print("\nCustom Word Pair Comparison:")
    print("-" * 60)
    
    for w1, w2 in word_pairs:
        id1 = vocab.get_id(w1)
        id2 = vocab.get_id(w2)
        
        if id1 is None:
            print(f"  '{w1}' not in vocabulary")
            continue
        if id2 is None:
            print(f"  '{w2}' not in vocabulary")
            continue
        
        emb1 = embeddings[id1]
        emb2 = embeddings[id2]
        
        euc_dist = euclidean_distance(emb1, emb2)
        cos_dist = cosine_distance(emb1, emb2)
        
        print(f"  {w1:15s} <-> {w2:15s}  |  Euclidean: {euc_dist:.6f}  |  Cosine: {cos_dist:.6f}")


def show_embedding_stats(checkpoint_path, words=None):
    """Show raw embedding values for specific words."""
    checkpoint = load_checkpoint(checkpoint_path)
    embeddings, vocab_size = get_embeddings_from_checkpoint(checkpoint)
    
    vocab = TieredVocabulary()
    for tier in range(1, 11):
        vocab.set_tier(tier)
        if vocab.size >= vocab_size:
            break
    
    if words is None:
        words = ["cat", "dog", "the", "is", "red"]
    
    print("\nRaw Embedding Values:")
    print("-" * 60)
    
    for word in words:
        word_id = vocab.get_id(word)
        if word_id is None or word_id >= vocab_size:
            print(f"  '{word}' not in vocabulary")
            continue
        
        emb = embeddings[word_id]
        norm = torch.norm(emb).item()
        
        print(f"\n  {word} (id={word_id}, norm={norm:.4f}):")
        print(f"    {emb.numpy()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze embedding quality from a GENREG checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_embeddings.py checkpoints/checkpoint_gen50_tier2_20251130.pkl
  python analyze_embeddings.py checkpoints/latest.pkl --words cat dog bird fish
  python analyze_embeddings.py checkpoints/latest.pkl --compare cat,dog bird,fish
        """
    )
    
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to checkpoint file (.pkl)"
    )
    
    parser.add_argument(
        "--words", "-w",
        nargs="+",
        help="Show raw embeddings for specific words"
    )
    
    parser.add_argument(
        "--compare", "-c",
        nargs="+",
        help="Compare word pairs (format: word1,word2)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Less verbose output"
    )
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        exit(1)
    
    # Run main analysis
    results = analyze_embeddings(args.checkpoint, verbose=not args.quiet)
    
    # Show raw embeddings if requested
    if args.words:
        show_embedding_stats(args.checkpoint, args.words)
    
    # Compare specific pairs if requested
    if args.compare:
        pairs = []
        for pair_str in args.compare:
            parts = pair_str.split(",")
            if len(parts) == 2:
                pairs.append((parts[0].strip(), parts[1].strip()))
            else:
                print(f"Invalid pair format: {pair_str} (use word1,word2)")
        
        if pairs:
            compare_word_pairs(args.checkpoint, pairs)




