# ================================================================
# GENREG Tiered Learning - Vocabulary Management
# ================================================================
# Manages tiered vocabulary that grows as the model advances
# ================================================================

from config import TIER_VOCAB


class TieredVocabulary:
    """
    Vocabulary that grows with tiers.
    
    Tier 1: 20 core words
    Tier 2: 20 more words + all tier 1 (40 total)
    etc.
    """
    
    def __init__(self):
        self.current_tier = 1
        self.word_to_id = {}
        self.id_to_word = {}
        self._build_vocab_for_tier(1)
    
    def _build_vocab_for_tier(self, tier):
        """Build cumulative vocabulary up to specified tier."""
        self.word_to_id = {}
        self.id_to_word = {}
        
        word_id = 0
        
        # Add special tokens first
        self.word_to_id["<PAD>"] = word_id
        self.id_to_word[word_id] = "<PAD>"
        word_id += 1
        
        self.word_to_id["<BLANK>"] = word_id
        self.id_to_word[word_id] = "<BLANK>"
        word_id += 1
        
        # Add words from all tiers up to current
        for t in range(1, tier + 1):
            if t in TIER_VOCAB:
                for word in TIER_VOCAB[t]:
                    if word not in self.word_to_id:
                        self.word_to_id[word] = word_id
                        self.id_to_word[word_id] = word
                        word_id += 1
        
        self.current_tier = tier
        print(f"[Vocab] Built tier {tier} vocabulary: {len(self.word_to_id)} words")
    
    def advance_tier(self):
        """Advance to the next tier, adding new vocabulary."""
        next_tier = self.current_tier + 1
        if next_tier in TIER_VOCAB:
            self._build_vocab_for_tier(next_tier)
            return True
        else:
            print(f"[Vocab] No tier {next_tier} defined. Staying at tier {self.current_tier}")
            return False
    
    def set_tier(self, tier):
        """Set vocabulary to specific tier."""
        self._build_vocab_for_tier(tier)
    
    def get_id(self, word):
        """Get ID for a word. Returns None if not in vocabulary."""
        return self.word_to_id.get(word, None)
    
    def get_word(self, word_id):
        """Get word for an ID. Returns <UNK> if not found."""
        return self.id_to_word.get(word_id, "<UNK>")
    
    def __len__(self):
        """Return current vocabulary size."""
        return len(self.word_to_id)
    
    def __contains__(self, word):
        """Check if word is in current vocabulary."""
        return word in self.word_to_id
    
    @property
    def size(self):
        """Current vocabulary size."""
        return len(self.word_to_id)
    
    @property
    def pad_id(self):
        """ID of padding token."""
        return self.word_to_id["<PAD>"]
    
    @property
    def blank_id(self):
        """ID of blank token."""
        return self.word_to_id["<BLANK>"]
    
    def get_tier_words(self, tier):
        """Get list of words introduced in a specific tier."""
        if tier in TIER_VOCAB:
            return TIER_VOCAB[tier]
        return []
    
    def get_all_words(self):
        """Get all words in current vocabulary (excluding special tokens)."""
        return [word for word in self.word_to_id.keys() 
                if word not in ["<PAD>", "<BLANK>"]]
    
    def is_word_in_tier(self, word, tier):
        """Check if a word was introduced in a specific tier."""
        if tier in TIER_VOCAB:
            return word in TIER_VOCAB[tier]
        return False
    
    def encode_sentence(self, words):
        """Convert list of words to list of IDs."""
        return [self.get_id(w) for w in words]
    
    def decode_sentence(self, ids):
        """Convert list of IDs to list of words."""
        return [self.get_word(i) for i in ids]


# Singleton instance for easy import
_vocab_instance = None

def get_vocabulary():
    """Get the singleton vocabulary instance."""
    global _vocab_instance
    if _vocab_instance is None:
        _vocab_instance = TieredVocabulary()
    return _vocab_instance

def reset_vocabulary():
    """Reset vocabulary to tier 1."""
    global _vocab_instance
    _vocab_instance = TieredVocabulary()
    return _vocab_instance


if __name__ == "__main__":
    # Test vocabulary
    vocab = TieredVocabulary()
    print(f"\nTier 1 size: {vocab.size}")
    print(f"Words: {vocab.get_all_words()}")
    
    # Test encoding
    test_sentence = ["the", "cat", "is", "red"]
    encoded = vocab.encode_sentence(test_sentence)
    print(f"\nEncoded '{test_sentence}': {encoded}")
    decoded = vocab.decode_sentence(encoded)
    print(f"Decoded back: {decoded}")
    
    # Advance to tier 2
    vocab.advance_tier()
    print(f"\nTier 2 size: {vocab.size}")
    print(f"All words: {vocab.get_all_words()}")

