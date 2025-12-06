# ================================================================
# GENREG Tiered Learning - Dataset Generator v2
# ================================================================
# Generates fill-in-the-blank training sentences for all tiers
# with increasing sentence complexity
# ================================================================

import json
import random
import os
from config import TIER_VOCAB, TIER_SENTENCE_LENGTHS, TIER_SENTENCE_COUNTS, DATA_DIR


def get_cumulative_vocab(tier):
    """Get all vocabulary words up to and including specified tier."""
    words = set()
    for t in range(1, tier + 1):
        if t in TIER_VOCAB:
            words.update(TIER_VOCAB[t])
    return list(words)


def categorize_words(vocab):
    """Categorize words by type for sentence generation."""
    # Define word categories
    articles = {"the", "a", "an"}
    pronouns = {"he", "she", "it", "they", "we", "you", "i", "this", "that", 
                "these", "those", "each", "every", "any", "another", "such", "same"}
    prepositions = {"in", "on", "at", "to", "from", "with", "by", "for", "under",
                   "over", "behind", "near", "between", "into", "out", "above",
                   "across", "against", "along", "among", "around", "aside",
                   "below", "beneath", "beside", "beyond", "despite", "except",
                   "inside", "outside", "throughout", "toward", "upon", "within", "without"}
    conjunctions = {"and", "but", "or", "so", "because", "if", "before", "after",
                   "also", "too", "either", "neither", "both", "however", "therefore",
                   "although", "unless", "while"}
    question_words = {"what", "where", "when", "who", "why", "how", "which", "can", "will", "do"}
    negation = {"not", "nothing", "nobody", "none", "never"}
    adverbs = {"very", "really", "always", "often", "sometimes", "quickly",
              "slowly", "well", "badly", "here", "there", "up", "down", "away",
              "already", "still", "yet", "soon", "later", "early", "late", "once",
              "twice", "again", "yesterday", "tomorrow", "forever", "ago", "during",
              "now", "then", "today"}
    numbers = {"one", "two", "three", "four", "five", "six", "seven", "eight",
              "nine", "ten", "many", "few", "some", "all", "no", "more", "less", "most",
              "hundred", "half", "dozen", "couple", "pair", "single", "third", "quarter"}
    
    # Categorize vocabulary
    cats = {
        "articles": [w for w in vocab if w in articles],
        "pronouns": [w for w in vocab if w in pronouns],
        "prepositions": [w for w in vocab if w in prepositions],
        "conjunctions": [w for w in vocab if w in conjunctions],
        "question_words": [w for w in vocab if w in question_words],
        "negation": [w for w in vocab if w in negation],
        "adverbs": [w for w in vocab if w in adverbs],
        "numbers": [w for w in vocab if w in numbers],
        "nouns": [],
        "verbs": [],
        "adjectives": [],
    }
    
    # Words not in known categories - classify by ending patterns
    uncategorized = set(vocab) - articles - pronouns - prepositions - conjunctions - \
                   question_words - negation - adverbs - numbers
    
    for word in uncategorized:
        if word.endswith(('s', 'es')) and len(word) > 3:
            # Likely a verb (third person singular)
            cats["verbs"].append(word)
        elif word.endswith(('ly',)) and len(word) > 4:
            cats["adverbs"].append(word)
        elif word.endswith(('ful', 'less', 'ous', 'ive', 'able', 'ible', 'al', 'ent', 'ant')):
            cats["adjectives"].append(word)
        elif word.endswith(('tion', 'sion', 'ment', 'ness', 'ity', 'er', 'or', 'ist')):
            cats["nouns"].append(word)
        else:
            # Default categorization based on common patterns
            if word in {"is", "has", "runs", "sits", "eats", "jumps", "walks", "flies", 
                       "swims", "grows", "sees", "hears", "goes", "comes", "takes",
                       "gives", "makes", "puts", "opens", "closes", "holds", "drops",
                       "pulls", "pushes", "throws", "catches", "kicks", "hits", "cuts",
                       "breaks", "fixes", "builds", "cleans", "plays", "works", "reads",
                       "writes", "draws", "sings", "dances", "sleeps", "wakes", "waits",
                       "watches", "listens", "speaks", "talks", "calls", "moves"}:
                cats["verbs"].append(word)
            elif word in {"red", "blue", "big", "small", "green", "tall", "fast", "slow",
                         "hot", "cold", "yellow", "black", "white", "brown", "old", "new",
                         "long", "short", "dark", "light", "heavy", "soft", "hard", "wet",
                         "dry", "clean", "dirty", "quiet", "happy", "sad", "angry", "scared",
                         "tired", "hungry", "thirsty", "sick", "pretty", "ugly", "nice",
                         "bad", "good", "strong", "weak", "young", "full", "empty"}:
                cats["adjectives"].append(word)
            else:
                cats["nouns"].append(word)
    
    return cats


def generate_sentence_pattern(cats, min_len, max_len, tier):
    """Generate a sentence with a blank based on available vocabulary."""
    # Choose sentence length
    length = random.randint(min_len, max_len)
    
    # Sentence patterns by complexity
    patterns = []
    
    # Basic patterns (all tiers)
    if cats["articles"] and cats["nouns"] and cats["adjectives"]:
        # "the cat is red"
        patterns.append(("article", "noun", "verb_be", "adjective"))
        # "a big cat"
        patterns.append(("article", "adjective", "noun"))
    
    if cats["articles"] and cats["nouns"] and cats["verbs"]:
        # "the cat runs"
        patterns.append(("article", "noun", "verb"))
    
    # Medium patterns (tier 3+)
    if tier >= 3 and cats["prepositions"]:
        # "the cat sits on the mat"
        patterns.append(("article", "noun", "verb", "preposition", "article", "noun"))
        # "in the house"
        patterns.append(("preposition", "article", "noun"))
    
    if tier >= 3 and cats["pronouns"]:
        # "he is big"
        patterns.append(("pronoun", "verb_be", "adjective"))
    
    # Complex patterns (tier 5+)
    if tier >= 5 and cats["adverbs"]:
        # "the cat runs quickly"
        patterns.append(("article", "noun", "verb", "adverb"))
        # "very big"
        patterns.append(("adverb", "adjective"))
    
    # Question patterns (tier 6+)
    if tier >= 6 and cats["question_words"]:
        # "what is the cat"
        patterns.append(("question", "verb_be", "article", "noun"))
        # "where is the dog"
        patterns.append(("question", "verb_be", "article", "noun"))
    
    # Compound patterns (tier 7+)
    if tier >= 7 and cats["conjunctions"]:
        # "the cat runs and the dog walks"
        patterns.append(("article", "noun", "verb", "conjunction", "article", "noun", "verb"))
    
    if not patterns:
        # Fallback to simple pattern
        if cats["nouns"] and cats["adjectives"]:
            patterns.append(("noun", "adjective"))
        else:
            return None, None, None
    
    # Choose a pattern
    pattern = random.choice(patterns)
    
    # Build sentence from pattern
    words = []
    for slot in pattern:
        if slot == "article" and cats["articles"]:
            words.append(random.choice(cats["articles"]))
        elif slot == "noun" and cats["nouns"]:
            words.append(random.choice(cats["nouns"]))
        elif slot == "verb" and cats["verbs"]:
            words.append(random.choice(cats["verbs"]))
        elif slot == "verb_be":
            words.append("is")
        elif slot == "adjective" and cats["adjectives"]:
            words.append(random.choice(cats["adjectives"]))
        elif slot == "adverb" and cats["adverbs"]:
            words.append(random.choice(cats["adverbs"]))
        elif slot == "preposition" and cats["prepositions"]:
            words.append(random.choice(cats["prepositions"]))
        elif slot == "pronoun" and cats["pronouns"]:
            words.append(random.choice(cats["pronouns"]))
        elif slot == "conjunction" and cats["conjunctions"]:
            words.append(random.choice(cats["conjunctions"]))
        elif slot == "question" and cats["question_words"]:
            words.append(random.choice(cats["question_words"]))
        else:
            # Skip if category empty
            continue
    
    if len(words) < 2:
        return None, None, None
    
    # Choose blank position
    blank_pos = random.randint(0, len(words) - 1)
    answer = words[blank_pos]
    
    return words, blank_pos, answer


def generate_tier_sentences(tier, count=500):
    """Generate fill-in-the-blank sentences for a tier."""
    vocab = get_cumulative_vocab(tier)
    cats = categorize_words(vocab)
    
    # Get sentence length range for this tier
    min_len, max_len = TIER_SENTENCE_LENGTHS.get(tier, (3, 5))
    
    sentences = []
    attempts = 0
    max_attempts = count * 10
    
    # Track unique sentences
    seen = set()
    
    while len(sentences) < count and attempts < max_attempts:
        attempts += 1
        
        words, blank_pos, answer = generate_sentence_pattern(cats, min_len, max_len, tier)
        
        if words is None:
            continue
        
        # Check uniqueness
        key = (tuple(words), blank_pos)
        if key in seen:
            continue
        seen.add(key)
        
        # Validate all words are in vocabulary
        if not all(w in vocab or w == "is" for w in words):
            continue
        
        sentences.append({
            "words": words,
            "blank_pos": blank_pos,
            "answer": answer
        })
    
    return sentences


def generate_tier_dataset(tier, output_path=None, count=None):
    """Generate and save dataset for a tier."""
    if count is None:
        count = TIER_SENTENCE_COUNTS.get(tier, 500)
    
    if output_path is None:
        output_path = os.path.join(DATA_DIR, f"tier{tier}.json")
    
    print(f"\n[DataGen] Generating tier {tier} dataset...")
    print(f"  Target: {count} sentences")
    
    vocab = get_cumulative_vocab(tier)
    sentences = generate_tier_sentences(tier, count)
    
    print(f"  Generated: {len(sentences)} sentences")
    
    # Analyze
    blank_positions = {}
    for s in sentences:
        pos = s["blank_pos"]
        blank_positions[pos] = blank_positions.get(pos, 0) + 1
    print(f"  Blank positions: {dict(sorted(blank_positions.items()))}")
    
    answers = {}
    for s in sentences:
        ans = s["answer"]
        answers[ans] = answers.get(ans, 0) + 1
    print(f"  Unique answers: {len(answers)}")
    
    # Sentence length distribution
    lengths = {}
    for s in sentences:
        l = len(s["words"])
        lengths[l] = lengths.get(l, 0) + 1
    print(f"  Sentence lengths: {dict(sorted(lengths.items()))}")
    
    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else DATA_DIR, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            "tier": tier,
            "vocab_size": len(vocab),
            "vocabulary": vocab,
            "sentence_count": len(sentences),
            "sentences": sentences
        }, f, indent=2)
    
    print(f"  Saved to {output_path}")
    return sentences


def generate_all_tiers():
    """Generate datasets for all defined tiers."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for tier in sorted(TIER_VOCAB.keys()):
        generate_tier_dataset(tier)


if __name__ == "__main__":
    print("=" * 60)
    print("GENREG Tiered Dataset Generator")
    print("=" * 60)
    
    generate_all_tiers()
    
    # Show samples from each tier
    print("\n" + "=" * 60)
    print("SAMPLE SENTENCES PER TIER")
    print("=" * 60)
    
    for tier in sorted(TIER_VOCAB.keys()):
        filepath = os.path.join(DATA_DIR, f"tier{tier}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            print(f"\nTier {tier} ({data['sentence_count']} sentences, {data['vocab_size']} words):")
            for s in data["sentences"][:5]:
                words_display = s["words"].copy()
                words_display[s["blank_pos"]] = f"[{s['answer']}]"
                print(f"  {' '.join(words_display)}")
