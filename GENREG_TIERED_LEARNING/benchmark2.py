# ================================================================
# GENREG Neighborhood Topology Benchmark
# ================================================================
# Tests word embeddings by examining nearest neighbors
# Critical test: What are the actual nearest neighbors for each word?
# This reveals true semantic clustering vs coincidental proximity
# ================================================================

import torch
import torch.nn.functional as F
import pickle
import sys
import re
from pathlib import Path

from config import TIER_VOCAB, CHECKPOINT_DIR

# ==============================================================================
# TEST WORDS - Words to examine neighborhoods for
# ==============================================================================

TEST_WORDS = [
    # Animals
    "cat", "dog", "bird", "fish",
    # Colors  
    "red", "blue", "green", "yellow",
    # Verbs
    "runs", "sits", "eats", "walks",
    # Adjectives
    "big", "small", "hot", "cold",
    # Nature
    "tree", "sun", "sky", "water",
    # Abstract (higher tiers)
    "happy", "sad", "fast", "slow",
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


def cosine_similarity_batch(query_vec, all_vecs):
    """Compute cosine similarity between query vector and all vectors."""
    # Normalize vectors
    query_norm = F.normalize(query_vec.unsqueeze(0), dim=1)
    all_norm = F.normalize(all_vecs, dim=1)
    
    # Compute similarities
    similarities = torch.mm(query_norm, all_norm.t()).squeeze(0)
    return similarities


def get_nearest_neighbors(word, word_to_id, id_to_word, embeddings, k=10):
    """
    Find the k nearest neighbors for a word in embedding space.
    
    Returns list of (word, similarity) tuples, sorted by similarity descending.
    """
    if word not in word_to_id:
        return None
    
    word_id = word_to_id[word]
    word_vec = embeddings[word_id]
    
    # Compute similarities to all words
    similarities = cosine_similarity_batch(word_vec, embeddings)
    
    # Get top k+1 (including the word itself)
    top_k_values, top_k_indices = torch.topk(similarities, min(k + 1, len(similarities)))
    
    neighbors = []
    for sim, idx in zip(top_k_values.tolist(), top_k_indices.tolist()):
        neighbor_word = id_to_word.get(idx, f"<id:{idx}>")
        # Skip the word itself
        if neighbor_word != word:
            neighbors.append((neighbor_word, sim))
    
    return neighbors[:k]


def categorize_word(word):
    """Categorize a word by its semantic type."""
    categories = {
        "animal": ["cat", "dog", "bird", "fish", "horse", "cow", "pig", "sheep", 
                   "chicken", "duck", "rabbit", "mouse"],
        "color": ["red", "blue", "green", "yellow", "black", "white", "brown", "orange"],
        "verb": ["runs", "sits", "eats", "walks", "jumps", "flies", "swims", "grows",
                 "sees", "hears", "goes", "comes", "takes", "gives", "makes", "puts",
                 "opens", "closes", "holds", "drops", "pulls", "pushes", "throws",
                 "loves", "likes", "wants", "needs", "knows", "thinks", "feels", "says"],
        "adjective": ["big", "small", "hot", "cold", "tall", "fast", "slow", 
                      "old", "new", "long", "short", "heavy", "soft", "hard",
                      "happy", "sad", "angry", "scared", "tired", "hungry"],
        "nature": ["tree", "grass", "sun", "sky", "water", "fire", "rock", "leaf",
                   "flower", "plant", "forest", "lake", "sea", "beach", "hill"],
        "body": ["hand", "foot", "head", "eye", "arm", "leg", "nose", "mouth", "ear"],
        "food": ["bread", "milk", "egg", "meat", "fruit", "apple", "orange", "rice"],
        "article": ["the", "a"],
        "preposition": ["in", "on", "at", "to", "from", "with", "by", "for",
                        "under", "over", "behind", "near", "between", "into", "out"],
        "pronoun": ["he", "she", "it", "they", "we", "you", "this", "that"],
    }
    
    for category, words in categories.items():
        if word in words:
            return category
    return "other"


def analyze_neighborhood_quality(word, neighbors, word_to_id):
    """Analyze how many neighbors are in the same semantic category."""
    word_cat = categorize_word(word)
    same_category = 0
    
    for neighbor_word, sim in neighbors:
        neighbor_cat = categorize_word(neighbor_word)
        if neighbor_cat == word_cat and word_cat != "other":
            same_category += 1
    
    return same_category, word_cat


def run_neighborhood_benchmark(checkpoint_path, k=10):
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
    id_to_word = {0: "<PAD>", 1: "<BLANK>"}
    current_id = 2  # Skip PAD (0) and BLANK (1)
    
    for t in range(1, vocab_tier + 1):
        if t in TIER_VOCAB:
            for w in TIER_VOCAB[t]:
                if w not in word_to_id and current_id < vocab_size:
                    word_to_id[w] = current_id
                    id_to_word[current_id] = w
                    current_id += 1
    
    print(f"Vocabulary rebuilt: {len(word_to_id)} words (vocab_size: {vocab_size})")
    
    # RUN NEIGHBORHOOD ANALYSIS
    print(f"\n{'='*70}")
    print(f"NEIGHBORHOOD TOPOLOGY ANALYSIS (k={k} nearest neighbors)")
    print(f"{'='*70}")
    
    total_same_category = 0
    total_neighbors = 0
    words_analyzed = 0
    skipped_words = []
    
    for word in TEST_WORDS:
        if word not in word_to_id:
            skipped_words.append(word)
            continue
        
        neighbors = get_nearest_neighbors(word, word_to_id, id_to_word, embeddings, k)
        
        if neighbors is None:
            skipped_words.append(word)
            continue
        
        words_analyzed += 1
        same_cat, word_cat = analyze_neighborhood_quality(word, neighbors, word_to_id)
        total_same_category += same_cat
        total_neighbors += len(neighbors)
        
        # Print results
        print(f"\n┌─ {word.upper()} (category: {word_cat})")
        print(f"│  Neighbors in same category: {same_cat}/{len(neighbors)}")
        print("│")
        
        for i, (neighbor, sim) in enumerate(neighbors, 1):
            neighbor_cat = categorize_word(neighbor)
            cat_match = "✓" if neighbor_cat == word_cat and word_cat != "other" else " "
            print(f"│  {i:2d}. {neighbor:<12} (sim: {sim:.4f}) [{neighbor_cat}] {cat_match}")
        
        print("└" + "─" * 50)
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    if words_analyzed > 0:
        avg_same_cat = total_same_category / words_analyzed
        pct_same_cat = (total_same_category / total_neighbors * 100) if total_neighbors > 0 else 0
        
        print(f"Words analyzed: {words_analyzed}")
        print(f"Total neighbors examined: {total_neighbors}")
        print(f"Neighbors in same category: {total_same_category} ({pct_same_cat:.1f}%)")
        print(f"Average same-category neighbors per word: {avg_same_cat:.1f}")
        
        # Interpretation
        print(f"\n{'─'*70}")
        if pct_same_cat >= 40:
            print("Result: STRONG semantic clustering - words cluster by meaning!")
        elif pct_same_cat >= 25:
            print("Result: MODERATE clustering - some semantic structure present.")
        elif pct_same_cat >= 15:
            print("Result: WEAK clustering - minimal semantic organization.")
        else:
            print("Result: NO clustering - embeddings appear random.")
        print(f"{'─'*70}")
        
        # Random baseline calculation
        # If vocabulary has N words evenly distributed across C categories,
        # random chance of same category = 1/C
        # With ~10 categories, random baseline is ~10%
        print(f"\n(Random baseline with ~10 categories: ~10%)")
    
    if skipped_words:
        print(f"\nSkipped words (not in vocabulary): {', '.join(skipped_words)}")
    
    print()
    return total_same_category, total_neighbors


def show_full_vocabulary_clusters(checkpoint_path, k=5):
    """Show clusters for all words in vocabulary."""
    print(f"\n{'='*70}")
    print(f"FULL VOCABULARY CLUSTER ANALYSIS (top {k} neighbors each)")
    print(f"{'='*70}")
    
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    
    shared_state = data['shared_embeddings']
    embeddings = shared_state['embeddings']
    vocab_size = shared_state['vocab_size']
    vocab_tier = data.get('vocab_tier', 1)
    
    # Rebuild mappings (only up to actual vocab_size)
    word_to_id = {}
    id_to_word = {0: "<PAD>", 1: "<BLANK>"}
    current_id = 2
    
    for t in range(1, vocab_tier + 1):
        if t in TIER_VOCAB:
            for w in TIER_VOCAB[t]:
                if w not in word_to_id and current_id < vocab_size:
                    word_to_id[w] = current_id
                    id_to_word[current_id] = w
                    current_id += 1
    
    # Group words by category and show their clusters
    categories = {}
    for word in word_to_id.keys():
        cat = categorize_word(word)
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(word)
    
    for cat_name, words in sorted(categories.items()):
        print(f"\n{'─'*70}")
        print(f"CATEGORY: {cat_name.upper()} ({len(words)} words)")
        print(f"{'─'*70}")
        
        for word in sorted(words)[:10]:  # Limit to 10 words per category
            neighbors = get_nearest_neighbors(word, word_to_id, id_to_word, embeddings, k)
            if neighbors:
                neighbor_str = ", ".join([f"{n}({s:.2f})" for n, s in neighbors[:5]])
                print(f"  {word:<12} → {neighbor_str}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze neighborhood topology of word embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark2.py                              # Auto-find best checkpoint
  python benchmark2.py checkpoints/checkpoint.pkl  # Use specific checkpoint
  python benchmark2.py -k 15                        # Show 15 nearest neighbors
  python benchmark2.py --full                       # Show all vocabulary clusters
        """
    )
    
    parser.add_argument(
        "checkpoint",
        type=str,
        nargs='?',
        help="Path to checkpoint file (.pkl)"
    )
    
    parser.add_argument(
        "-k", "--neighbors",
        type=int,
        default=10,
        help="Number of nearest neighbors to show (default: 10)"
    )
    
    parser.add_argument(
        "--full",
        action="store_true",
        help="Show clusters for entire vocabulary"
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
            print("\nUsage: python benchmark2.py [checkpoint_path]")
            sys.exit(1)
    
    run_neighborhood_benchmark(checkpoint_path, k=args.neighbors)
    
    if args.full:
        show_full_vocabulary_clusters(checkpoint_path)

