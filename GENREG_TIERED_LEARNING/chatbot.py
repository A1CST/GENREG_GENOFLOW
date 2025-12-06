# ================================================================
# GENREG Phase 6 - Simple Single-Turn Chatbot
# ================================================================
# Interactive chatbot using trained predictor and generator
# ================================================================

import os
import sys
import random

from config import DEVICE, PREDICTOR_CONFIG, PREDICTOR_CHECKPOINT_DIR
from generator import (
    find_best_predictor_checkpoint,
    load_predictor,
    TextGenerator
)


class SimplePatternMatcher:
    """
    Simple pattern matching for common conversational intents.
    Maps user input patterns to response starters.
    """
    
    def __init__(self, word_to_id):
        self.word_to_id = word_to_id
        self.vocab_words = set(word_to_id.keys())
        
        # Define patterns and response starters
        # Format: (trigger_words, response_starters, use_generation)
        self.patterns = [
            # Greetings
            ({"hello", "hi", "hey", "greetings"}, 
             ["hello", "hi", "the"], True),
            
            # Questions about identity
            ({"who", "are", "you"},
             ["i", "the"], True),
            
            # Questions starting with what
            ({"what", "is", "the"},
             ["the", "a", "it"], True),
            
            # Questions starting with where  
            ({"where", "is"},
             ["in", "at", "the"], True),
            
            # Questions starting with how
            ({"how", "do", "does"},
             ["it", "the", "you"], True),
            
            # Farewell
            ({"bye", "goodbye", "later"},
             ["goodbye", "bye"], False),
            
            # Thanks
            ({"thank", "thanks"},
             ["you", "the"], True),
            
            # Feelings/states
            ({"happy", "sad", "angry", "tired", "hungry"},
             ["i", "the", "it"], True),
            
            # Weather
            ({"weather", "rain", "snow", "sun", "cold", "hot"},
             ["the", "it", "today"], True),
            
            # Animals
            ({"cat", "dog", "bird", "fish", "horse", "cow"},
             ["the", "a"], True),
            
            # Food
            ({"food", "eat", "hungry", "bread", "milk", "apple"},
             ["the", "i", "it"], True),
        ]
    
    def find_pattern(self, words):
        """
        Find a matching pattern for the input words.
        
        Returns:
            (response_starters, use_generation) or None
        """
        input_set = set(w.lower() for w in words)
        
        for trigger_words, starters, use_gen in self.patterns:
            if input_set & trigger_words:  # Any intersection
                # Filter starters to those in vocabulary
                valid_starters = [s for s in starters if s in self.vocab_words]
                if valid_starters:
                    return valid_starters, use_gen
        
        return None
    
    def get_fallback_starters(self):
        """Get generic response starters when no pattern matches."""
        fallbacks = ["the", "i", "it", "a", "he", "she", "they", "we", "you"]
        return [s for s in fallbacks if s in self.vocab_words]


class Chatbot:
    """
    Simple single-turn chatbot.
    
    Flow:
    1. Receive user input
    2. Tokenize and check for known patterns
    3. Generate response using pattern starter or user context
    4. Return response
    """
    
    def __init__(self, generator, pattern_matcher=None):
        self.generator = generator
        self.pattern_matcher = pattern_matcher or SimplePatternMatcher(generator.word_to_id)
        
        self.word_to_id = generator.word_to_id
        self.id_to_word = generator.id_to_word
        
        # Response settings
        self.max_response_length = 12
        self.temperature = 0.8
        self.top_k = 5
        
        # Fallback responses for when generation fails
        self.fallback_responses = [
            "i do not know",
            "the answer is not clear",
            "i am not sure",
        ]
    
    def preprocess_input(self, text):
        """Clean and tokenize user input."""
        # Lowercase and split
        words = text.lower().split()
        
        # Remove punctuation from each word
        clean_words = []
        for word in words:
            word = word.strip('.,!?;:"\'-')
            if word:
                clean_words.append(word)
        
        return clean_words
    
    def get_known_words(self, words):
        """Filter to words in vocabulary."""
        known = []
        unknown = []
        
        for word in words:
            if word in self.word_to_id:
                known.append(word)
            else:
                unknown.append(word)
        
        return known, unknown
    
    def respond(self, user_input):
        """
        Generate a response to user input.
        
        Args:
            user_input: User's message (string)
        
        Returns:
            Response string
        """
        # Preprocess
        words = self.preprocess_input(user_input)
        
        if not words:
            return "please say something"
        
        known_words, unknown_words = self.get_known_words(words)
        
        # Check for patterns
        pattern_result = self.pattern_matcher.find_pattern(words)
        
        if pattern_result:
            starters, use_generation = pattern_result
            
            if not use_generation:
                # Direct response (like "goodbye")
                return random.choice(starters)
            
            # Use pattern starter for generation
            starter = random.choice(starters)
            starter_id = self.word_to_id[starter]
            
            response = self.generator.generate(
                [starter_id],
                max_tokens=self.max_response_length,
                temperature=self.temperature,
                top_k=self.top_k
            )
            
            if response:
                return f"{starter} {response}"
        
        # No pattern match - use user's known words as context
        if known_words:
            # Use last few known words as context
            context_words = known_words[-self.generator.context_length:]
            context_ids = [self.word_to_id[w] for w in context_words]
            
            response = self.generator.generate(
                context_ids,
                max_tokens=self.max_response_length,
                temperature=self.temperature,
                top_k=self.top_k
            )
            
            if response:
                return response
        
        # Fallback: try with generic starters
        fallback_starters = self.pattern_matcher.get_fallback_starters()
        if fallback_starters:
            starter = random.choice(fallback_starters)
            starter_id = self.word_to_id[starter]
            
            response = self.generator.generate(
                [starter_id],
                max_tokens=self.max_response_length,
                temperature=self.temperature,
                top_k=self.top_k
            )
            
            if response:
                return f"{starter} {response}"
        
        # Ultimate fallback
        return random.choice(self.fallback_responses)


def chat():
    """Main chat loop."""
    print("=" * 60)
    print("GENREG Chatbot")
    print("=" * 60)
    print()
    
    # Find and load checkpoint
    checkpoint_info = find_best_predictor_checkpoint()
    
    if checkpoint_info is None:
        print("[Error] No predictor checkpoint found!")
        print("Please run train_predictor.py first to train the predictor.")
        print()
        print("Training flow:")
        print("  1. Complete fill-in-blank training (train.py)")
        print("  2. Train predictor (train_predictor.py)")
        print("  3. Run chatbot (chatbot.py)")
        sys.exit(1)
    
    filepath, acc, gen = checkpoint_info
    print(f"[Load] Loading predictor: accuracy={acc}%, generation={gen}")
    
    controller, embeddings, word_to_id, id_to_word = load_predictor(filepath)
    
    # Create generator and chatbot
    generator = TextGenerator(controller, embeddings, word_to_id, id_to_word)
    chatbot = Chatbot(generator)
    
    print()
    print(f"[Ready] Vocabulary: {len(word_to_id)} words")
    print()
    print("Chat with me! (Type 'quit' to exit, 'help' for commands)")
    print("-" * 40)
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        lower_input = user_input.lower()
        
        if lower_input == 'quit' or lower_input == 'exit':
            print("Bot: goodbye")
            break
        
        if lower_input == 'help':
            print("\nCommands:")
            print("  'quit' or 'exit' - Exit chatbot")
            print("  'vocab' - Show sample vocabulary words")
            print("  'temp X' - Set temperature (e.g., 'temp 0.5')")
            print("  'settings' - Show current settings")
            print()
            continue
        
        if lower_input == 'vocab':
            sample = random.sample(
                [w for w in word_to_id.keys() if not w.startswith('<')],
                min(40, len(word_to_id))
            )
            print(f"\nSample vocabulary: {', '.join(sorted(sample))}\n")
            continue
        
        if lower_input.startswith('temp '):
            try:
                chatbot.temperature = float(lower_input.split()[1])
                print(f"Temperature set to {chatbot.temperature}")
            except:
                print("Usage: temp 0.5")
            continue
        
        if lower_input == 'settings':
            print(f"\nSettings:")
            print(f"  Temperature: {chatbot.temperature}")
            print(f"  Top-K: {chatbot.top_k}")
            print(f"  Max response length: {chatbot.max_response_length}")
            print()
            continue
        
        # Generate response
        response = chatbot.respond(user_input)
        print(f"Bot: {response}")
        print()


if __name__ == "__main__":
    chat()



