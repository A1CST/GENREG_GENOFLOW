# ================================================================
# GENREG Phase 5 - Autoregressive Text Generation
# ================================================================
# Generates text using trained CausalController and embeddings
# ================================================================

import os
import sys
import pickle
import random
import glob
import re
import torch
import torch.nn.functional as F
from pathlib import Path

from config import DEVICE, PREDICTOR_CONFIG, PREDICTOR_CHECKPOINT_DIR
from predictor import CausalController


def find_best_predictor_checkpoint(checkpoint_dir=PREDICTOR_CHECKPOINT_DIR):
    """
    Find the best predictor checkpoint.
    Priority: Highest accuracy > Highest generation
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    pattern = os.path.join(checkpoint_dir, "predictor_*.pkl")
    checkpoint_files = glob.glob(pattern)
    
    if not checkpoint_files:
        return None
    
    # Pattern: predictor_gen{N}_acc{X}%_{timestamp}.pkl
    format_pattern = re.compile(r'predictor_gen(\d+)_acc(\d+)%_')
    
    best_checkpoint = None
    best_score = (-1, -1)  # (accuracy, generation)
    
    for filepath in checkpoint_files:
        filename = os.path.basename(filepath)
        match = format_pattern.match(filename)
        if match:
            gen = int(match.group(1))
            acc = int(match.group(2))
            
            score = (acc, gen)
            if score > best_score:
                best_score = score
                best_checkpoint = (filepath, acc, gen)
    
    return best_checkpoint


def load_predictor(filepath):
    """Load predictor from checkpoint."""
    print(f"[Load] Loading predictor from: {os.path.basename(filepath)}")
    
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    
    embedding_dim = checkpoint['embedding_dim']
    context_length = checkpoint['context_length']
    hidden_size = checkpoint['hidden_size']
    
    # Create controller and load state
    controller = CausalController(embedding_dim, context_length, hidden_size)
    controller.load_state_dict(checkpoint['controller_state'])
    controller.to(DEVICE)
    controller.eval()
    
    # Load embeddings
    embeddings = checkpoint['embeddings'].to(DEVICE)
    
    # Load vocabulary
    word_to_id = checkpoint['word_to_id']
    id_to_word = {v: k for k, v in word_to_id.items()}
    
    print(f"[Load] Loaded: dim={embedding_dim}, context={context_length}, vocab={len(word_to_id)}")
    
    return controller, embeddings, word_to_id, id_to_word


class TextGenerator:
    """
    Autoregressive text generator using CausalController.
    """
    
    def __init__(self, controller, embeddings, word_to_id, id_to_word, config=None):
        self.controller = controller
        self.embeddings = embeddings
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.config = config or PREDICTOR_CONFIG
        
        self.embedding_dim = embeddings.shape[1]
        self.vocab_size = embeddings.shape[0]
        self.context_length = controller.context_length
        
        # Precompute normalized embeddings for cosine similarity
        self.normalized_embeddings = F.normalize(embeddings, dim=1)
    
    def get_embedding(self, word_id):
        """Get embedding for a word ID."""
        if word_id is not None and word_id < self.vocab_size:
            return self.embeddings[word_id]
        return torch.zeros(self.embedding_dim, device=DEVICE)
    
    def find_nearest_words(self, predicted_embedding, top_k=5):
        """
        Find the K nearest words to a predicted embedding.
        
        Returns:
            List of (word, word_id, similarity_score) tuples
        """
        # Normalize predicted embedding
        pred_norm = F.normalize(predicted_embedding.unsqueeze(0), dim=1)
        
        # Compute cosine similarities with all word embeddings
        similarities = torch.mm(pred_norm, self.normalized_embeddings.t()).squeeze(0)
        
        # Get top K
        top_similarities, top_indices = torch.topk(similarities, min(top_k, self.vocab_size))
        
        results = []
        for sim, idx in zip(top_similarities.tolist(), top_indices.tolist()):
            word = self.id_to_word.get(idx, "<UNK>")
            results.append((word, idx, sim))
        
        return results
    
    def sample_word(self, predicted_embedding, temperature=1.0, top_k=5):
        """
        Sample a word from the predicted embedding.
        
        Args:
            predicted_embedding: The predicted embedding vector
            temperature: Sampling temperature (1.0 = greedy, higher = more random)
            top_k: Number of candidates to consider
        
        Returns:
            (word, word_id, confidence) or None if below confidence threshold
        """
        candidates = self.find_nearest_words(predicted_embedding, top_k)
        
        if not candidates:
            return None
        
        # Check minimum confidence
        min_confidence = self.config.get("min_confidence", 0.3)
        if candidates[0][2] < min_confidence:
            return None
        
        if temperature <= 0.01:
            # Greedy: return best match
            return candidates[0]
        
        # Temperature sampling
        similarities = torch.tensor([c[2] for c in candidates], device=DEVICE)
        
        # Apply temperature (higher temp = more uniform distribution)
        logits = similarities / temperature
        probs = F.softmax(logits, dim=0)
        
        # Sample
        idx = torch.multinomial(probs, 1).item()
        return candidates[idx]
    
    def tokenize(self, text):
        """Convert text to word IDs."""
        words = text.lower().split()
        word_ids = []
        unknown_words = []
        
        for word in words:
            # Clean punctuation
            word = word.strip('.,!?;:"\'-')
            if word in self.word_to_id:
                word_ids.append(self.word_to_id[word])
            else:
                unknown_words.append(word)
        
        return word_ids, unknown_words
    
    def generate(self, prompt, max_tokens=None, temperature=None, top_k=None, verbose=False):
        """
        Generate text autoregressively from a prompt.
        
        Args:
            prompt: Starting text (string or list of words)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Number of candidates for sampling
            verbose: Print each generated word
        
        Returns:
            Generated text (string)
        """
        max_tokens = max_tokens or self.config.get("max_generation_length", 20)
        temperature = temperature if temperature is not None else self.config.get("temperature", 1.0)
        top_k = top_k or self.config.get("top_k", 5)
        
        # Tokenize prompt
        if isinstance(prompt, str):
            context_ids, unknown = self.tokenize(prompt)
            if unknown and verbose:
                print(f"[Warning] Unknown words skipped: {unknown}")
        else:
            context_ids = prompt
        
        if not context_ids:
            if verbose:
                print("[Warning] Empty prompt after tokenization")
            return ""
        
        # Generate tokens
        generated_words = []
        
        for _ in range(max_tokens):
            # Get context (last N words)
            context = context_ids[-self.context_length:]
            
            # Get embeddings
            context_embs = [self.get_embedding(wid) for wid in context]
            
            # Predict next embedding
            with torch.no_grad():
                predicted = self.controller.forward_single(context_embs)
            
            # Sample word
            result = self.sample_word(predicted, temperature, top_k)
            
            if result is None:
                if verbose:
                    print("[Stop] Low confidence")
                break
            
            word, word_id, confidence = result
            
            # Skip special tokens
            if word in ["<PAD>", "<BLANK>"]:
                continue
            
            generated_words.append(word)
            context_ids.append(word_id)
            
            if verbose:
                print(f"  -> {word} ({confidence:.2f})")
            
            # Simple stop conditions
            # Could be extended with learned stop token
            if len(generated_words) >= 3 and word in [".", "!", "?"]:
                break
        
        return " ".join(generated_words)
    
    def complete(self, prompt, max_tokens=10, temperature=1.0, top_k=5):
        """
        Complete a sentence starting with the prompt.
        Returns prompt + generated text.
        """
        generated = self.generate(prompt, max_tokens, temperature, top_k)
        
        if isinstance(prompt, str):
            return f"{prompt} {generated}".strip()
        else:
            prompt_words = [self.id_to_word.get(wid, "?") for wid in prompt]
            return f"{' '.join(prompt_words)} {generated}".strip()


def interactive_generate():
    """Interactive generation mode."""
    print("=" * 60)
    print("GENREG Text Generator - Interactive Mode")
    print("=" * 60)
    
    # Find and load checkpoint
    checkpoint_info = find_best_predictor_checkpoint()
    
    if checkpoint_info is None:
        print("[Error] No predictor checkpoint found!")
        print("Please run train_predictor.py first.")
        sys.exit(1)
    
    filepath, acc, gen = checkpoint_info
    print(f"[Load] Best checkpoint: accuracy={acc}%, generation={gen}")
    
    controller, embeddings, word_to_id, id_to_word = load_predictor(filepath)
    
    generator = TextGenerator(controller, embeddings, word_to_id, id_to_word)
    
    print()
    print("Commands:")
    print("  Type a prompt to generate text")
    print("  'temp X' to set temperature (e.g., 'temp 0.5')")
    print("  'topk X' to set top-k (e.g., 'topk 10')")
    print("  'max X' to set max tokens (e.g., 'max 15')")
    print("  'vocab' to see sample vocabulary words")
    print("  'quit' to exit")
    print()
    
    temperature = 1.0
    top_k = 5
    max_tokens = 15
    
    while True:
        try:
            user_input = input("Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if user_input.lower().startswith('temp '):
            try:
                temperature = float(user_input.split()[1])
                print(f"Temperature set to {temperature}")
            except:
                print("Usage: temp 0.5")
            continue
        
        if user_input.lower().startswith('topk '):
            try:
                top_k = int(user_input.split()[1])
                print(f"Top-K set to {top_k}")
            except:
                print("Usage: topk 10")
            continue
        
        if user_input.lower().startswith('max '):
            try:
                max_tokens = int(user_input.split()[1])
                print(f"Max tokens set to {max_tokens}")
            except:
                print("Usage: max 15")
            continue
        
        if user_input.lower() == 'vocab':
            sample_words = random.sample(list(word_to_id.keys()), min(30, len(word_to_id)))
            sample_words = [w for w in sample_words if not w.startswith('<')]
            print(f"Sample words: {', '.join(sample_words)}")
            continue
        
        # Generate
        print(f"[Settings] temp={temperature}, top_k={top_k}, max={max_tokens}")
        
        result = generator.complete(
            user_input,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )
        
        print(f"\nGenerated: {result}\n")


if __name__ == "__main__":
    interactive_generate()



