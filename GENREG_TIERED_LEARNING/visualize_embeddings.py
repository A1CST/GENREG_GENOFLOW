# ================================================================
# Embedding 3D Visualizer
# ================================================================
# Loads a checkpoint and visualizes embeddings in 3D space
# using dimensionality reduction (PCA or t-SNE).
# ================================================================

import argparse
import pickle
import torch
import numpy as np
from pathlib import Path

from config import TIER_VOCAB, DEVICE
from vocabulary import TieredVocabulary

# Try to import visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# Word categories for coloring
WORD_CATEGORIES = {
    "article": ["the", "a"],
    "noun": [
        "cat", "dog", "bird", "fish", "tree", "grass", "sun", "sky",
        "house", "car", "ball", "book", "water", "fire", "rock", "leaf",
        "man", "woman", "boy", "girl", "baby", "hand", "foot", "head",
        "eye", "door", "window", "floor", "room", "kitchen", "garden",
        "street", "park", "school", "store", "road", "river", "mountain",
        "horse", "cow", "pig", "sheep", "chicken", "duck", "rabbit", "mouse",
        "arm", "leg", "nose", "mouth", "ear", "hair",
    ],
    "verb": [
        "is", "has", "runs", "sits", "eats", "jumps", "walks", "flies",
        "swims", "grows", "sees", "hears", "goes", "comes", "takes", "gives",
        "makes", "puts", "opens", "closes", "holds", "drops", "pulls", "pushes",
        "throws", "catches", "kicks", "hits", "cuts", "breaks", "fixes", "builds",
        "cleans", "loves", "likes", "wants", "needs", "knows", "thinks", "feels",
        "says", "tells", "asks", "answers", "helps", "finds", "loses", "keeps",
        "leaves", "starts", "stops",
    ],
    "adjective": [
        "red", "blue", "big", "small", "green", "tall", "fast", "slow",
        "hot", "cold", "yellow", "black", "white", "brown", "old", "new",
        "long", "short", "dark", "light", "heavy", "soft", "hard", "wet",
        "dry", "clean", "dirty", "quiet", "happy", "sad", "angry", "scared",
        "tired", "hungry", "thirsty", "sick", "pretty", "ugly", "nice", "bad",
        "good", "strong", "weak", "young", "full", "empty",
    ],
    "preposition": [
        "in", "on", "at", "to", "from", "with", "by", "for",
        "under", "over", "behind", "near", "between", "into", "out",
    ],
    "pronoun": ["he", "she", "it", "they", "we", "you"],
    "adverb": [
        "very", "really", "always", "never", "often", "sometimes", "quickly",
        "slowly", "well", "badly", "here", "there", "up", "down", "away",
        "now", "then", "today",
    ],
    "number": [
        "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "many", "few", "some", "all", "no", "more", "less", "most",
    ],
    "conjunction": ["and", "but", "or", "so", "because", "if", "before", "after"],
    "question": ["what", "where", "when", "who", "why", "how", "which"],
    "special": ["<PAD>", "<BLANK>"],
}

# Color palette for categories
CATEGORY_COLORS = {
    "article": "#FF6B6B",      # Coral red
    "noun": "#4ECDC4",         # Teal
    "verb": "#45B7D1",         # Sky blue
    "adjective": "#96CEB4",    # Sage green
    "preposition": "#FFEAA7",  # Pale yellow
    "pronoun": "#DDA0DD",      # Plum
    "adverb": "#98D8C8",       # Mint
    "number": "#F7DC6F",       # Gold
    "conjunction": "#BB8FCE",  # Light purple
    "question": "#F8B500",     # Orange
    "special": "#808080",      # Gray
    "other": "#C0C0C0",        # Silver
}


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


def get_word_category(word):
    """Get category for a word."""
    for category, words in WORD_CATEGORIES.items():
        if word.lower() in [w.lower() for w in words]:
            return category
    return "other"


def reduce_dimensions(embeddings, method='pca', n_components=3, perplexity=30):
    """Reduce embedding dimensions to 3D."""
    if not HAS_SKLEARN:
        raise ImportError("sklearn is required for dimensionality reduction. Install with: pip install scikit-learn")
    
    embeddings_np = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings
    
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components)
        reduced = reducer.fit_transform(embeddings_np)
        explained_var = reducer.explained_variance_ratio_
        print(f"  PCA explained variance: {explained_var}")
        print(f"  Total variance captured: {sum(explained_var)*100:.1f}%")
    elif method.lower() == 'tsne':
        # t-SNE needs perplexity < n_samples
        perplexity = min(perplexity, len(embeddings_np) - 1)
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        reduced = reducer.fit_transform(embeddings_np)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca' or 'tsne'")
    
    return reduced


def visualize_plotly(coords, words, categories, title="Embedding Space"):
    """Create interactive 3D visualization with Plotly."""
    if not HAS_PLOTLY:
        raise ImportError("Plotly is required. Install with: pip install plotly")
    
    # Create color list
    colors = [CATEGORY_COLORS.get(cat, "#C0C0C0") for cat in categories]
    
    # Create figure
    fig = go.Figure()
    
    # Add points for each category (for legend)
    for category in set(categories):
        mask = [c == category for c in categories]
        cat_coords = coords[mask]
        cat_words = [w for w, m in zip(words, mask) if m]
        
        fig.add_trace(go.Scatter3d(
            x=cat_coords[:, 0],
            y=cat_coords[:, 1],
            z=cat_coords[:, 2],
            mode='markers+text',
            marker=dict(
                size=8,
                color=CATEGORY_COLORS.get(category, "#C0C0C0"),
                opacity=0.8,
                line=dict(width=1, color='white')
            ),
            text=cat_words,
            textposition='top center',
            textfont=dict(size=9),
            name=category,
            hovertemplate='<b>%{text}</b><br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3',
            bgcolor='rgb(20, 20, 30)',
            xaxis=dict(gridcolor='rgb(50, 50, 60)', zerolinecolor='rgb(80, 80, 90)'),
            yaxis=dict(gridcolor='rgb(50, 50, 60)', zerolinecolor='rgb(80, 80, 90)'),
            zaxis=dict(gridcolor='rgb(50, 50, 60)', zerolinecolor='rgb(80, 80, 90)'),
        ),
        paper_bgcolor='rgb(30, 30, 40)',
        font=dict(color='white'),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(50, 50, 60, 0.8)'
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    
    return fig


def visualize_matplotlib(coords, words, categories, title="Embedding Space"):
    """Create 3D visualization with Matplotlib."""
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib is required. Install with: pip install matplotlib")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each category
    for category in set(categories):
        mask = [c == category for c in categories]
        cat_coords = coords[mask]
        cat_words = [w for w, m in zip(words, mask) if m]
        color = CATEGORY_COLORS.get(category, "#C0C0C0")
        
        ax.scatter(
            cat_coords[:, 0],
            cat_coords[:, 1],
            cat_coords[:, 2],
            c=color,
            label=category,
            s=60,
            alpha=0.8,
            edgecolors='white',
            linewidths=0.5
        )
        
        # Add labels for words
        for i, word in enumerate(cat_words):
            ax.text(
                cat_coords[i, 0],
                cat_coords[i, 1],
                cat_coords[i, 2],
                word,
                fontsize=7,
                alpha=0.9
            )
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left', fontsize=8)
    
    # Dark background
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.tick_params(colors='white')
    ax.title.set_color('white')
    
    return fig


def visualize_embeddings(checkpoint_path, method='pca', backend='auto', 
                         show_labels=True, perplexity=30, save_path=None):
    """
    Main visualization function.
    """
    print("=" * 60)
    print("EMBEDDING 3D VISUALIZER")
    print("=" * 60)
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    embeddings, vocab_size = get_embeddings_from_checkpoint(checkpoint)
    generation = checkpoint.get('generation', 'unknown')
    
    print(f"Generation: {generation}")
    print(f"Vocab size: {vocab_size}")
    print(f"Embedding dim: {embeddings.shape[1]}")
    
    # Build vocabulary
    vocab = TieredVocabulary()
    for tier in range(1, 11):
        vocab.set_tier(tier)
        if vocab.size >= vocab_size:
            break
    
    print(f"Tier: {vocab.current_tier}")
    
    # Get word list and categories
    words = []
    categories = []
    valid_indices = []
    
    for word_id in range(vocab_size):
        word = vocab.get_word(word_id)
        if word != "<UNK>":
            words.append(word)
            categories.append(get_word_category(word))
            valid_indices.append(word_id)
    
    # Filter embeddings to valid words
    valid_embeddings = embeddings[valid_indices]
    
    print(f"\nReducing {valid_embeddings.shape[1]}D -> 3D using {method.upper()}...")
    
    # Reduce dimensions
    coords = reduce_dimensions(valid_embeddings, method=method, perplexity=perplexity)
    
    # Choose backend
    if backend == 'auto':
        backend = 'plotly' if HAS_PLOTLY else 'matplotlib'
    
    title = f"Embedding Space (Gen {generation}, Tier {vocab.current_tier}, {method.upper()})"
    
    print(f"\nCreating visualization with {backend}...")
    
    if backend == 'plotly':
        fig = visualize_plotly(coords, words, categories, title)
        
        if save_path:
            fig.write_html(save_path)
            print(f"Saved to: {save_path}")
        else:
            fig.show()
            
    elif backend == 'matplotlib':
        fig = visualize_matplotlib(coords, words, categories, title)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
            print(f"Saved to: {save_path}")
        else:
            plt.show()
    
    print("\nDone!")
    
    return coords, words, categories


def check_dependencies():
    """Check and report on available dependencies."""
    print("Dependency Check:")
    print("-" * 40)
    print(f"  scikit-learn: {'✓ Available' if HAS_SKLEARN else '✗ Missing (pip install scikit-learn)'}")
    print(f"  plotly:       {'✓ Available' if HAS_PLOTLY else '✗ Missing (pip install plotly)'}")
    print(f"  matplotlib:   {'✓ Available' if HAS_MATPLOTLIB else '✗ Missing (pip install matplotlib)'}")
    print()
    
    if not HAS_SKLEARN:
        print("ERROR: scikit-learn is required for dimensionality reduction.")
        return False
    
    if not HAS_PLOTLY and not HAS_MATPLOTLIB:
        print("ERROR: Either plotly or matplotlib is required for visualization.")
        return False
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize embeddings from a GENREG checkpoint in 3D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_embeddings.py checkpoints/checkpoint_gen50_tier2.pkl
  python visualize_embeddings.py checkpoints/latest.pkl --method tsne
  python visualize_embeddings.py checkpoints/latest.pkl --backend matplotlib --save plot.png
  python visualize_embeddings.py checkpoints/latest.pkl --save embeddings.html

Requirements:
  pip install scikit-learn plotly
  (or: pip install scikit-learn matplotlib)
        """
    )
    
    parser.add_argument(
        "checkpoint",
        type=str,
        nargs='?',
        help="Path to checkpoint file (.pkl)"
    )
    
    parser.add_argument(
        "--method", "-m",
        type=str,
        choices=['pca', 'tsne'],
        default='pca',
        help="Dimensionality reduction method (default: pca)"
    )
    
    parser.add_argument(
        "--backend", "-b",
        type=str,
        choices=['plotly', 'matplotlib', 'auto'],
        default='auto',
        help="Visualization backend (default: auto)"
    )
    
    parser.add_argument(
        "--perplexity", "-p",
        type=int,
        default=30,
        help="Perplexity for t-SNE (default: 30)"
    )
    
    parser.add_argument(
        "--save", "-s",
        type=str,
        help="Save visualization to file (.html for plotly, .png for matplotlib)"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check dependencies and exit"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if args.check_deps or args.checkpoint is None:
        deps_ok = check_dependencies()
        if args.checkpoint is None and deps_ok:
            print("Usage: python visualize_embeddings.py <checkpoint_path>")
        exit(0 if deps_ok else 1)
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        exit(1)
    
    # Run visualization
    visualize_embeddings(
        args.checkpoint,
        method=args.method,
        backend=args.backend,
        perplexity=args.perplexity,
        save_path=args.save
    )




