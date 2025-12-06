# ================================================================
# GENREG - Language Learning Protein Network
# ================================================================
# Proteins for shaped trust signals in language learning.
# Instead of binary hit/miss, proteins provide gradient-rich feedback.
#
# The Problem:
#   Controller predicts -> binary hit/miss -> trust = accuracy * constant
#   Result: No gradient for evolution, random embeddings despite 93% accuracy
#
# The Solution:
#   Controller predicts -> proteins read signals -> shaped trust
#   Signals: prediction_distance, distance_trend, category_match
#   Result: Evolution has a gradient to follow
# ================================================================

import math
import torch
from proteins import Protein, SensorProtein, TrendProtein, TrustModifierProtein, run_protein_cascade


# ================================================================
# CATEGORY PROTEIN
# Checks if prediction lands in same semantic category as target
# ================================================================

class CategoryProtein(Protein):
    """
    Provides bonus trust when prediction is in the same semantic category
    as the target word.
    
    Even if we miss the exact word, predicting "dog" when target was "cat"
    shows semantic understanding and deserves partial credit.
    """
    
    def __init__(self, name="category_match"):
        super().__init__(name, "category")
        # Bonus multiplier when categories match
        self.params["match_bonus"] = 1.0
        # Partial bonus when in related category
        self.params["related_bonus"] = 0.3
    
    def forward(self, signals, protein_outputs):
        """
        Expects signal 'category_match' with value:
            1.0 = exact category match
            0.5 = related category
            0.0 = unrelated
        """
        match_level = signals.get(self.name, 0.0)
        
        if match_level >= 0.9:
            # Exact category match
            self.output = self.params["match_bonus"]
        elif match_level >= 0.4:
            # Related category
            self.output = self.params["related_bonus"]
        else:
            # Unrelated
            self.output = 0.0
        
        return self.output


# ================================================================
# PROXIMITY REWARD PROTEIN
# Partial credit based on distance (not binary hit/miss)
# ================================================================

class ProximityRewardProtein(Protein):
    """
    Provides shaped reward based on prediction distance.
    Closer predictions get more reward, even if they don't "hit".
    
    This gives evolution a gradient to follow:
    - Distance 0.5 is better than distance 1.0
    - Distance 1.0 is better than distance 2.0
    - Even "misses" contribute information
    """
    
    def __init__(self, name="proximity_reward"):
        super().__init__(name, "proximity")
        # Maximum reward for perfect prediction (distance = 0)
        self.params["max_reward"] = 1.0
        # Distance at which reward falls to 50%
        self.params["half_life_distance"] = 1.0
        # Minimum reward (even for far predictions)
        self.params["min_reward"] = 0.0
        # Steepness of falloff
        self.params["steepness"] = 2.0
    
    def forward(self, signals, protein_outputs):
        """
        Expects signal 'prediction_distance' (Euclidean distance to target).
        
        Uses exponential decay: reward = max * exp(-steepness * distance / half_life)
        """
        distance = signals.get("prediction_distance", 10.0)
        
        # Exponential decay based on distance
        half_life = max(0.01, self.params["half_life_distance"])
        steepness = self.params["steepness"]
        
        decay = math.exp(-steepness * distance / half_life)
        reward = self.params["max_reward"] * decay
        
        # Apply minimum
        self.output = max(self.params["min_reward"], reward)
        
        return self.output


# ================================================================
# IMPROVEMENT BONUS PROTEIN
# Extra reward when predictions are improving over time
# ================================================================

class ImprovementBonusProtein(Protein):
    """
    Tracks prediction quality over time and rewards improvement.
    
    If distance is decreasing on average, give bonus trust.
    This encourages learning trajectories, not just absolute performance.
    """
    
    def __init__(self, name="improvement_bonus"):
        super().__init__(name, "improvement")
        # Bonus when improving
        self.params["improvement_bonus"] = 0.5
        # Penalty when getting worse (optional, can be 0)
        self.params["regression_penalty"] = 0.1
        # Smoothing factor for trend detection
        self.params["momentum"] = 0.7
        
        # State tracking
        self.state["last_distance"] = None
        self.state["trend"] = 0.0  # Negative = improving, Positive = worsening
    
    def forward(self, signals, protein_outputs):
        """
        Expects signal 'prediction_distance'.
        Compares to previous predictions to detect trend.
        """
        current_distance = signals.get("prediction_distance", 10.0)
        
        if self.state["last_distance"] is None:
            self.state["last_distance"] = current_distance
            self.output = 0.0
            return self.output
        
        # Calculate change (negative = getting closer = good)
        delta = current_distance - self.state["last_distance"]
        self.state["last_distance"] = current_distance
        
        # Smooth the trend with momentum
        m = self.params["momentum"]
        self.state["trend"] = m * self.state["trend"] + (1 - m) * delta
        
        # Convert trend to reward/penalty
        if self.state["trend"] < -0.01:
            # Improving - give bonus
            self.output = self.params["improvement_bonus"] * min(1.0, abs(self.state["trend"]))
        elif self.state["trend"] > 0.01:
            # Getting worse - small penalty
            self.output = -self.params["regression_penalty"] * min(1.0, self.state["trend"])
        else:
            # Stable
            self.output = 0.0
        
        return self.output
    
    def reset(self):
        """Reset state for new episode."""
        self.state["last_distance"] = None
        self.state["trend"] = 0.0


# ================================================================
# HIT BONUS PROTEIN
# Still reward actual hits, but as part of the network
# ================================================================

class HitBonusProtein(Protein):
    """
    Provides bonus for actual hits (within threshold).
    
    This maintains backward compatibility with the hit-based system
    while being part of the protein network.
    """
    
    def __init__(self, name="hit_bonus"):
        super().__init__(name, "hit")
        # Bonus for hitting exactly
        self.params["hit_reward"] = 2.0
    
    def forward(self, signals, protein_outputs):
        """
        Expects signal 'token_hit' (1.0 if hit, 0.0 if miss).
        """
        hit = signals.get("token_hit", 0.0)
        
        if hit > 0.5:
            self.output = self.params["hit_reward"]
        else:
            self.output = 0.0
        
        return self.output


# ================================================================
# LANGUAGE PROTEIN NETWORK FACTORY
# ================================================================

def create_language_protein_network(config):
    """
    Create the complete protein network for language learning.
    
    Network structure:
        SensorProtein("prediction_distance") 
            -> ProximityRewardProtein (partial credit)
            -> ImprovementBonusProtein (trend reward)
        
        SensorProtein("category_match")
            -> CategoryProtein (semantic bonus)
        
        SensorProtein("token_hit")
            -> HitBonusProtein (hit reward)
        
        All -> TrustModifierProtein (combines into trust delta)
    
    Returns:
        list: Ordered list of proteins for cascade execution
    """
    proteins = []
    
    # === SENSORS (read raw signals) ===
    
    # Distance sensor
    distance_sensor = SensorProtein("prediction_distance")
    distance_sensor.params["norm_scale"] = config.get("protein_distance_scale", 2.0)
    proteins.append(distance_sensor)
    
    # Category sensor  
    category_sensor = SensorProtein("category_match")
    category_sensor.params["norm_scale"] = 1.0  # Already 0-1
    proteins.append(category_sensor)
    
    # Hit sensor
    hit_sensor = SensorProtein("token_hit")
    hit_sensor.params["norm_scale"] = 1.0  # Already 0-1
    proteins.append(hit_sensor)
    
    # === PROCESSING PROTEINS ===
    
    # Proximity reward (shaped distance signal)
    proximity = ProximityRewardProtein("proximity_reward")
    proximity.params["max_reward"] = config.get("protein_proximity_max", 1.0)
    proximity.params["half_life_distance"] = config.get("protein_proximity_halflife", 1.5)
    proximity.params["steepness"] = config.get("protein_proximity_steepness", 1.5)
    proteins.append(proximity)
    
    # Improvement tracking
    improvement = ImprovementBonusProtein("improvement_bonus")
    improvement.params["improvement_bonus"] = config.get("protein_improvement_bonus", 0.3)
    improvement.params["regression_penalty"] = config.get("protein_regression_penalty", 0.1)
    proteins.append(improvement)
    
    # Category matching
    category = CategoryProtein("category_match_protein")
    category.params["match_bonus"] = config.get("protein_category_bonus", 0.5)
    category.params["related_bonus"] = config.get("protein_related_bonus", 0.2)
    # Bind to category sensor output
    category.bind_inputs(["category_match"])
    proteins.append(category)
    
    # Hit bonus
    hit_bonus = HitBonusProtein("hit_bonus")
    hit_bonus.params["hit_reward"] = config.get("protein_hit_reward", 1.5)
    hit_bonus.bind_inputs(["token_hit"])
    proteins.append(hit_bonus)
    
    # === TRUST MODIFIERS (combine signals into trust) ===
    
    # Proximity-based trust (main gradient signal)
    proximity_trust = TrustModifierProtein("proximity_trust")
    proximity_trust.params["scale"] = config.get("protein_proximity_scale", 1000.0)
    proximity_trust.bind_inputs(["proximity_reward"])
    proteins.append(proximity_trust)
    
    # Improvement-based trust
    improvement_trust = TrustModifierProtein("improvement_trust")
    improvement_trust.params["scale"] = config.get("protein_improvement_scale", 200.0)
    improvement_trust.bind_inputs(["improvement_bonus"])
    proteins.append(improvement_trust)
    
    # Category-based trust
    category_trust = TrustModifierProtein("category_trust")
    category_trust.params["scale"] = config.get("protein_category_scale", 500.0)
    category_trust.bind_inputs(["category_match_protein"])
    proteins.append(category_trust)
    
    # Hit-based trust (backward compatibility)
    hit_trust = TrustModifierProtein("hit_trust")
    hit_trust.params["scale"] = config.get("protein_hit_scale", 1000.0)
    hit_trust.bind_inputs(["hit_bonus"])
    proteins.append(hit_trust)
    
    return proteins


def reset_protein_network(proteins):
    """Reset all proteins in the network for a new episode."""
    for p in proteins:
        if hasattr(p, 'reset'):
            p.reset()
        # Reset any stateful proteins
        if 'last_distance' in p.state:
            p.state['last_distance'] = None
        if 'trend' in p.state:
            p.state['trend'] = 0.0
        if 'last' in p.state:
            p.state['last'] = None
        if 'velocity' in p.state:
            p.state['velocity'] = 0.0


# ================================================================
# CATEGORY MATCHING HELPER
# ================================================================

# Word categories for semantic matching
WORD_CATEGORIES = {
    "animal": {"cat", "dog", "bird", "fish", "horse", "cow", "pig", "sheep", 
               "chicken", "duck", "rabbit", "mouse"},
    "color": {"red", "blue", "green", "yellow", "black", "white", "brown", "orange"},
    "verb": {"is", "has", "runs", "sits", "eats", "jumps", "walks", "flies", 
             "swims", "grows", "sees", "hears", "goes", "comes", "takes", "gives",
             "makes", "puts", "opens", "closes", "holds", "drops", "pulls", "pushes",
             "throws", "catches", "kicks", "hits", "cuts", "breaks", "fixes", "builds",
             "cleans", "plays", "works", "reads", "writes", "draws", "sings", "dances",
             "sleeps", "wakes", "waits", "watches", "listens", "speaks", "talks", "calls",
             "moves", "loves", "likes", "wants", "needs", "knows", "thinks", "feels", "says"},
    "adjective": {"big", "small", "hot", "cold", "tall", "fast", "slow", 
                  "old", "new", "long", "short", "heavy", "soft", "hard", "wet", "dry",
                  "clean", "dirty", "quiet", "happy", "sad", "angry", "scared", "tired",
                  "hungry", "thirsty", "sick", "pretty", "ugly", "nice", "bad", "good",
                  "strong", "weak", "young", "full", "empty", "dark", "light"},
    "nature": {"tree", "grass", "sun", "sky", "water", "fire", "rock", "leaf",
               "flower", "plant", "forest", "lake", "sea", "beach", "hill", "mountain",
               "river", "rain", "snow", "wind", "cloud", "storm"},
    "body": {"hand", "foot", "head", "eye", "arm", "leg", "nose", "mouth", "ear", "hair"},
    "food": {"food", "bread", "milk", "egg", "meat", "fruit", "apple", "orange",
             "rice", "soup", "cake", "candy"},
    "place": {"house", "room", "kitchen", "garden", "street", "park", "school",
              "store", "road", "city", "town", "village", "country"},
    "article": {"the", "a", "an"},
    "preposition": {"in", "on", "at", "to", "from", "with", "by", "for",
                    "under", "over", "behind", "near", "between", "into", "out"},
    "pronoun": {"he", "she", "it", "they", "we", "you", "this", "that"},
    "time": {"now", "then", "today", "morning", "night", "day", "time", "year",
             "month", "week", "hour", "minute"},
    "number": {"one", "two", "three", "four", "five", "six", "seven", "eight",
               "nine", "ten", "many", "few", "some", "all"},
}

# Build reverse lookup
WORD_TO_CATEGORY = {}
for cat, words in WORD_CATEGORIES.items():
    for word in words:
        WORD_TO_CATEGORY[word] = cat

# Related categories (for partial credit)
RELATED_CATEGORIES = {
    "animal": {"nature"},
    "nature": {"animal", "place"},
    "color": {"adjective"},
    "adjective": {"color"},
    "verb": set(),
    "food": {"nature"},
    "body": {"animal"},
    "place": {"nature"},
    "time": {"number"},
    "number": {"time"},
}


def get_word_category(word):
    """Get the semantic category of a word."""
    return WORD_TO_CATEGORY.get(word, "other")


def check_category_match(predicted_word, target_word):
    """
    Check if predicted and target words are in the same category.
    
    Returns:
        float: 1.0 = same category, 0.5 = related category, 0.0 = unrelated
    """
    pred_cat = get_word_category(predicted_word)
    target_cat = get_word_category(target_word)
    
    if pred_cat == "other" or target_cat == "other":
        return 0.0
    
    if pred_cat == target_cat:
        return 1.0
    
    # Check if related
    related = RELATED_CATEGORIES.get(target_cat, set())
    if pred_cat in related:
        return 0.5
    
    return 0.0


def find_nearest_word(predicted_embedding, embeddings, id_to_word, exclude_ids=None):
    """
    Find the nearest word to a predicted embedding.
    
    Args:
        predicted_embedding: The predicted embedding tensor
        embeddings: All embeddings tensor [vocab_size, dim]
        id_to_word: Dict mapping IDs to words
        exclude_ids: Set of IDs to exclude (e.g., PAD, BLANK)
    
    Returns:
        tuple: (word, word_id, distance)
    """
    if exclude_ids is None:
        exclude_ids = {0, 1}  # PAD, BLANK
    
    # Compute distances to all embeddings
    distances = torch.sqrt(torch.sum((embeddings - predicted_embedding.unsqueeze(0)) ** 2, dim=1))
    
    # Mask excluded IDs with large distance
    for eid in exclude_ids:
        if eid < len(distances):
            distances[eid] = float('inf')
    
    # Find nearest
    min_dist, min_idx = torch.min(distances, dim=0)
    min_idx = min_idx.item()
    
    word = id_to_word.get(min_idx, f"<id:{min_idx}>")
    
    return word, min_idx, min_dist.item()


