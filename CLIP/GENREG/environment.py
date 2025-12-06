# ================================================================
# GENREG Tiered Learning - Fill-in-the-Blank Environment
# ================================================================
# Environment that presents sentences with blanks for prediction
# ================================================================

import json
import random
import os
import torch

from config import CONFIG, DATA_DIR, DEVICE
from vocabulary import TieredVocabulary


class FillInBlankEnv:
    """
    Environment for fill-in-the-blank learning.
    
    Presents sentences with a blank, provides bidirectional context,
    and evaluates predictions.
    """
    
    def __init__(self, vocab, tier=1, device=None):
        self.vocab = vocab
        self.current_tier = tier
        self.device = device if device else DEVICE
        
        self.context_window = CONFIG["context_window"]
        self.embedding_dim = CONFIG["embedding_dim"]
        
        # Load dataset
        self.sentences = []
        self.load_tier(tier)
        
        # Current state
        self.current_sentence_idx = 0
        self.shuffle_sentences()
    
    def load_tier(self, tier):
        """Load sentences for a tier."""
        data_path = os.path.join(DATA_DIR, f"tier{tier}.json")
        
        if not os.path.exists(data_path):
            print(f"[Env] Warning: {data_path} not found. Run dataset_generator.py first.")
            self.sentences = []
            return
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        self.sentences = data["sentences"]
        self.current_tier = tier
        
        # Ensure vocab is at correct tier
        if self.vocab.current_tier < tier:
            self.vocab.set_tier(tier)
        
        print(f"[Env] Loaded tier {tier}: {len(self.sentences)} sentences")
    
    def shuffle_sentences(self):
        """Shuffle sentence order."""
        random.shuffle(self.sentences)
        self.current_sentence_idx = 0
    
    def get_sentence_count(self):
        """Return number of sentences in current tier."""
        return len(self.sentences)
    
    def reset(self):
        """Reset environment for new episode."""
        self.shuffle_sentences()
        return self.get_current_task()
    
    def get_current_task(self):
        """
        Get current fill-in-the-blank task.
        
        Returns:
            dict with:
                - words: list of words in sentence
                - blank_pos: position of blank
                - answer: correct word
                - left_context: list of word IDs before blank
                - right_context: list of word IDs after blank
                - target_id: ID of answer word
        """
        if self.current_sentence_idx >= len(self.sentences):
            self.shuffle_sentences()
        
        sentence = self.sentences[self.current_sentence_idx]
        
        words = sentence["words"]
        blank_pos = sentence["blank_pos"]
        answer = sentence["answer"]
        
        # Get context word IDs
        left_words = words[:blank_pos]
        right_words = words[blank_pos + 1:]
        
        # Pad if needed
        left_ids = [self.vocab.get_id(w) for w in left_words]
        right_ids = [self.vocab.get_id(w) for w in right_words]
        
        # Pad left context (pad at beginning)
        while len(left_ids) < self.context_window:
            left_ids.insert(0, self.vocab.pad_id)
        
        # Pad right context (pad at end)
        while len(right_ids) < self.context_window:
            right_ids.append(self.vocab.pad_id)
        
        # Take only context_window items
        left_ids = left_ids[-self.context_window:]
        right_ids = right_ids[:self.context_window]
        
        target_id = self.vocab.get_id(answer)
        
        return {
            "words": words,
            "blank_pos": blank_pos,
            "answer": answer,
            "left_context": left_ids,
            "right_context": right_ids,
            "target_id": target_id,
            "sentence_idx": self.current_sentence_idx
        }
    
    def step(self):
        """Advance to next sentence."""
        self.current_sentence_idx += 1
        if self.current_sentence_idx >= len(self.sentences):
            return None  # Episode done
        return self.get_current_task()
    
    def advance_tier(self):
        """Advance to next tier."""
        next_tier = self.current_tier + 1
        self.load_tier(next_tier)
        self.shuffle_sentences()
        return next_tier
    
    def evaluate_prediction(self, predicted_embedding, target_id, genome):
        """
        Evaluate a prediction.
        
        Args:
            predicted_embedding: Predicted embedding tensor
            target_id: ID of correct answer
            genome: Genome being evaluated (to get target embedding)
        
        Returns:
            dict with:
                - distance: Euclidean distance to target
                - hit: Whether prediction was correct
                - target_embedding: Target embedding tensor
        """
        target_embedding = genome.get_embedding_tensor(target_id)
        
        distance = torch.sqrt(torch.sum((predicted_embedding - target_embedding) ** 2)).item()
        
        hit_threshold = CONFIG["vector_hit_threshold"]
        hit = distance < hit_threshold
        
        return {
            "distance": distance,
            "hit": hit,
            "target_embedding": target_embedding
        }
    
    def get_task_display(self, task):
        """Get displayable string for task."""
        words = task["words"].copy()
        words[task["blank_pos"]] = "___"
        return " ".join(words)


class TieredCurriculum:
    """
    Manages curriculum progression through tiers.
    """
    
    def __init__(self, vocab, device=None):
        self.vocab = vocab
        self.device = device if device else DEVICE
        
        self.current_tier = 1
        self.env = FillInBlankEnv(vocab, tier=1, device=device)
        
        self.mastery_threshold = CONFIG["mastery_threshold"]
        
        # Track tier progress
        self.tier_generations = {}  # tier -> generations spent
        self.tier_best_accuracy = {}  # tier -> best accuracy achieved
    
    def get_env(self):
        """Get current environment."""
        return self.env
    
    def check_mastery(self, avg_accuracy):
        """
        Check if current tier is mastered and advance if so.
        
        Returns:
            True if advanced to new tier, False otherwise
        """
        from config import TIER_VOCAB
        
        # Track best accuracy for current tier
        if self.current_tier not in self.tier_best_accuracy:
            self.tier_best_accuracy[self.current_tier] = 0.0
        self.tier_best_accuracy[self.current_tier] = max(
            self.tier_best_accuracy[self.current_tier],
            avg_accuracy
        )
        
        if avg_accuracy >= self.mastery_threshold:
            # Check if next tier exists
            next_tier = self.current_tier + 1
            if next_tier in TIER_VOCAB:
                print(f"\n>>> TIER MASTERY! Accuracy: {avg_accuracy:.1%}")
                print(f">>> Advancing from Tier {self.current_tier} to Tier {next_tier}")
                
                # Advance vocabulary
                self.vocab.set_tier(next_tier)
                
                # Advance environment
                self.current_tier = next_tier
                self.env.load_tier(next_tier)
                self.env.shuffle_sentences()
                
                return True
        
        return False
    
    def advance_population(self, population):
        """Advance population to new tier vocabulary."""
        population.advance_tier(self.vocab)


if __name__ == "__main__":
    # Test environment
    vocab = TieredVocabulary()
    env = FillInBlankEnv(vocab, tier=1, device=DEVICE)
    
    print("\nTest Environment:")
    print(f"Sentences: {env.get_sentence_count()}")
    
    # Get a task
    task = env.get_current_task()
    print(f"\nTask: {env.get_task_display(task)}")
    print(f"Answer: {task['answer']}")
    print(f"Left context IDs: {task['left_context']}")
    print(f"Right context IDs: {task['right_context']}")
    print(f"Target ID: {task['target_id']}")
    
    # Test a few more
    print("\nSample tasks:")
    for _ in range(5):
        task = env.step()
        if task:
            print(f"  {env.get_task_display(task)} -> {task['answer']}")

