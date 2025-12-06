# ================================================================
# GENREG Tiered Learning System - Configuration
# ================================================================
# Child-like vocabulary acquisition with fill-in-the-blank learning
# ================================================================

import torch

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    # --- Model Architecture ---
    "embedding_dim": 32,              # Dimensions per word vector
    "context_window": 4,              # Words on each side of blank (4 left + 4 right = 8 context)
    "controller_hidden_size": 64,     # Hidden layer size in controller MLP
    "blank_marker_dim": 16,           # Dimension of the blank position marker
    
    # --- Population & Training ---
    "population_size": 200,            # Genomes per generation
    "generations": 10000,             # Max generations to train
    "mastery_threshold": 0.90,        # Accuracy to advance to next tier
    "checkpoint_interval": 50,        # Save checkpoint every N generations
    
    # --- Trust System ---
    "base_trust_penalty": 50000.0,    # Trust penalty at 0% accuracy
    "trust_scale_trigger": 1000000.0, # Scale all trust when max exceeds this
    "trust_scale_target": 10000.0,    # Scale down to this value
    
    # --- Hit Detection ---
    "vector_hit_threshold": 1.7,      # Distance threshold for correct prediction
    
    # --- Mutation Rates ---
    "elite_mutation_rate": 0.1,     # Mutation rate for best genome
    "standard_mutation_rate": 0.3,   # Standard mutation rate
    "hyper_mutation_rate": 0.5,      # High mutation for exploration
    "mutation_scale": 0.3,           # Gaussian scale for weight mutations
    
    # --- Embedding Mutation ---
    "embedding_mutation_prob": 0.7,   # Probability to mutate each dimension
    "embedding_mutation_scale": 2.5, # Gaussian scale for embedding mutations
    "embedding_min_value": -1.0,      # Minimum embedding value (clamp)
    "embedding_max_value": 1.0,       # Maximum embedding value (clamp)
    
    # --- Evolution Strategy ---
    "survival_rate": 0.4,             # Top X% survive to reproduce
    "hyper_mutation_start": 0.1,      # Bottom X% get hyper-mutation
    
    # --- Guided Mutation ---
    "guided_mutation_strength": 0.5,  # 0=random, 0.5=balanced, 1.0=pure guided
    "guided_mutation_scale_multiplier": 0.1,  # Multiplier for guided mutation scale
    
    # --- Shared Embeddings ---
    "shared_embeddings": True,
    "shared_embedding_mix_ratio": 0.7,  # 0.0 = 100% private embeddings (for debugging)
    "shared_embedding_contribution_rate": 0.01,
    "shared_embedding_mutation_rate": 0.3,
    "shared_embedding_mutation_scale": 0.25,  # Scale for shared embedding mutations (higher for phase 2+)
    "shared_embedding_top_percent": 0.1,
    
    # --- Crossover ---
    "crossover_rate": 0.4,
    "crossover_compatibility_threshold": 0.3,
    "crossover_neuron_swap_rate": 0.1,
    "crossover_protein_swap_rate": 0.2,
    "crossover_blend_ratio_strong": 0.7,  # Blend ratio when parent_a is stronger
    "crossover_blend_ratio_weak": 0.3,    # Blend ratio when parent_b is stronger
    
    # --- Protein-Based Trust Signals ---
    # Shaped trust signals instead of binary hit/miss
    "protein_distance_scale": 2.0,           # Normalize distance for proteins
    "protein_proximity_max": 1.0,            # Max reward for proximity
    "protein_proximity_halflife": 1.5,       # Distance at which reward halves
    "protein_proximity_steepness": 1.5,      # Steepness of proximity decay
    "protein_proximity_scale": 1000.0,       # Scale proximity trust contribution
    "protein_improvement_bonus": 0.3,        # Bonus for improving predictions
    "protein_regression_penalty": 0.1,       # Penalty for worsening predictions
    "protein_improvement_scale": 200.0,      # Scale improvement trust contribution
    "protein_category_bonus": 0.5,           # Bonus for same category prediction
    "protein_related_bonus": 0.2,            # Bonus for related category
    "protein_category_scale": 500.0,         # Scale category trust contribution
    "protein_hit_reward": 1.5,               # Bonus for exact hit
    "protein_hit_scale": 1000.0,             # Scale hit trust contribution

    # --- Triplet Protein (Semantic Clustering) ---
    "triplet_samples_per_genome": 100,       # Number of triplets to test per genome
    "triplet_base_trust": 1000000.0,         # Base trust value for multiplicative calc
    "triplet_mutation_scale": 0.1,           # Scale for triplet-guided mutation nudges
    
    # --- Geometric Predictor ---
    "predictor_mutation_rate": 0.05,         # Default mutation rate for GeometricPredictor
    "predictor_mutation_scale": 0.1,          # Default mutation scale for GeometricPredictor
    "predictor_position_weight_min": 0.01,   # Minimum position weight (clamp)
    "predictor_position_weight_max": 10.0,   # Maximum position weight (clamp)
    "predictor_dim_weight_min": 0.1,         # Minimum dimension weight (clamp)
    "predictor_dim_weight_max": 3.0,         # Maximum dimension weight (clamp)
    "predictor_dim_mutation_prob": 0.1,      # Probability to mutate dim_weights
    "predictor_dim_mutation_scale_mult": 0.5, # Scale multiplier for dim_weights
    "predictor_mode_mutation_prob": 0.01,    # Probability to change prediction mode
    
    # --- Stability Protein ---
    "stability_inactive_mutation_rate": 0.0,  # Mutation rate for inactive embeddings (0.0 = frozen)
    "stability_active_mutation_rate": 1.0,   # Mutation rate for active embeddings
    "stability_cooldown_generations": 5,      # Generations of cooldown after embedding becomes inactive
    "stability_mask_threshold": 0.01,         # Threshold for considering embedding active
    
    # --- Shared Embedding Mix Ratio Mutation ---
    "mix_ratio_mutation_rate": 0.05,          # Probability to mutate shared_mix_ratio
    "mix_ratio_mutation_scale": 0.05,         # Gaussian scale for mix_ratio mutations
    "mix_ratio_min": 0.0,                     # Minimum mix ratio
    "mix_ratio_max": 1.0,                     # Maximum mix ratio
    
    # --- Error Contribution ---
    "error_contribution_multiplier": 0.5,     # Multiplier for error direction in contributions
    
    # --- Architecture Scaling ---
    "architecture_scale_init_std": 0.1,       # Standard deviation for initializing new dimensions
    
    # --- Phase 3 Settings ---
    "phase_3_mastery_threshold": 0.85,       # Relaxed mastery threshold for phase 3
    
    # --- Random Genome Injection (Phase 2) ---
    "random_injection_percent": 0.1,         # Percent of population to replace with random genomes each gen (0.0 = disabled)
}

# ================================================================
# TIER DEFINITIONS
# ================================================================

TIER_VOCAB = {
    # ================================================================
    # TIER 1: 20 words - Core vocabulary
    # ================================================================
    1: [
        # Articles (2)
        "the", "a",
        # Nouns (8)
        "cat", "dog", "bird", "fish", "tree", "grass", "sun", "sky",
        # Verbs (5)
        "is", "has", "runs", "sits", "eats",
        # Adjectives (5)
        "red", "blue", "big", "small", "green",
    ],
    
    # ================================================================
    # TIER 2: 20 new words (40 total) - Extended basics
    # ================================================================
    2: [
        # New Nouns (8)
        "house", "car", "ball", "book", "water", "fire", "rock", "leaf",
        # New Verbs (5)
        "jumps", "walks", "flies", "swims", "grows",
        # New Adjectives (7)
        "tall", "fast", "slow", "hot", "cold", "yellow", "black",
    ],
    
    # ================================================================
    # TIER 3: 40 new words (80 total) - Prepositions & Pronouns
    # ================================================================
    3: [
        # Prepositions (8)
        "in", "on", "at", "to", "from", "with", "by", "for",
        # Pronouns (6)
        "he", "she", "it", "they", "we", "you",
        # More nouns (12)
        "man", "woman", "boy", "girl", "baby", "hand", "foot", "head",
        "eye", "door", "window", "floor",
        # More verbs (8)
        "sees", "hears", "goes", "comes", "takes", "gives", "makes", "puts",
        # More adjectives (6)
        "white", "brown", "old", "new", "long", "short",
    ],
    
    # ================================================================
    # TIER 4: 70 new words (150 total) - Actions & Places
    # ================================================================
    4: [
        # Places (10)
        "room", "kitchen", "garden", "street", "park", "school", "store", "road",
        "river", "mountain",
        # Animals (8)
        "horse", "cow", "pig", "sheep", "chicken", "duck", "rabbit", "mouse",
        # Body parts (6)
        "arm", "leg", "nose", "mouth", "ear", "hair",
        # Verbs (15)
        "opens", "closes", "holds", "drops", "pulls", "pushes", "throws", "catches",
        "kicks", "hits", "cuts", "breaks", "fixes", "builds", "cleans",
        # Time words (6)
        "now", "then", "today", "morning", "night", "day",
        # More adjectives (10)
        "dark", "light", "heavy", "soft", "hard", "wet", "dry", "clean",
        "dirty", "quiet",
        # Numbers (8)
        "one", "two", "three", "four", "five", "six", "seven", "eight",
        # More prepositions (7)
        "under", "over", "behind", "near", "between", "into", "out",
    ],
    
    # ================================================================
    # TIER 5: 100 new words (250 total) - Descriptions & Adverbs
    # ================================================================
    5: [
        # Adverbs (15)
        "very", "really", "always", "never", "often", "sometimes", "quickly",
        "slowly", "well", "badly", "here", "there", "up", "down", "away",
        # Food (12)
        "food", "bread", "milk", "egg", "meat", "fruit", "apple", "orange",
        "rice", "soup", "cake", "candy",
        # Clothing (8)
        "hat", "shirt", "pants", "shoes", "coat", "dress", "sock", "bag",
        # Furniture (8)
        "table", "chair", "bed", "desk", "lamp", "clock", "mirror", "box",
        # Weather (6)
        "rain", "snow", "wind", "cloud", "storm", "fog",
        # More verbs (18)
        "loves", "likes", "wants", "needs", "knows", "thinks", "feels", "says",
        "tells", "asks", "answers", "helps", "finds", "loses", "keeps", "leaves",
        "starts", "stops",
        # Feelings (8)
        "happy", "sad", "angry", "scared", "tired", "hungry", "thirsty", "sick",
        # More adjectives (10)
        "pretty", "ugly", "nice", "bad", "good", "strong", "weak", "young",
        "full", "empty",
        # Question words (5)
        "what", "where", "when", "who", "why",
        # More numbers (10)
        "nine", "ten", "many", "few", "some", "all", "no", "more", "less", "most",
    ],
    
    # ================================================================
    # TIER 6: 150 new words (400 total) - Questions & Conjunctions
    # ================================================================
    6: [
        # Conjunctions (8)
        "and", "but", "or", "so", "because", "if", "before", "after",
        # Question helpers (5)
        "how", "which", "can", "will", "do",
        # Negation (4)
        "not", "nothing", "nobody", "none",
        # Family (10)
        "mother", "father", "sister", "brother", "son", "daughter", "family",
        "friend", "people", "child",
        # Nature (12)
        "flower", "plant", "forest", "lake", "sea", "beach", "hill", "field",
        "farm", "earth", "moon", "star",
        # Materials (8)
        "wood", "metal", "glass", "paper", "cloth", "stone", "plastic", "gold",
        # Tools (8)
        "key", "knife", "fork", "spoon", "cup", "plate", "bottle", "pen",
        # Transport (8)
        "bus", "train", "boat", "plane", "bike", "truck", "wheel", "road",
        # Verbs (25)
        "plays", "works", "reads", "writes", "draws", "sings", "dances", "sleeps",
        "wakes", "waits", "watches", "listens", "speaks", "talks", "calls", "moves",
        "carries", "brings", "sends", "buys", "sells", "pays", "costs", "counts", "tries",
        # Time (12)
        "time", "year", "month", "week", "hour", "minute", "second", "summer",
        "winter", "spring", "fall", "season",
        # Actions (15)
        "begins", "ends", "changes", "turns", "follows", "leads", "meets", "joins",
        "lives", "dies", "born", "grows", "learns", "teaches", "shows",
        # Descriptive (20)
        "beautiful", "wonderful", "terrible", "important", "different", "same",
        "other", "first", "last", "next", "only", "own", "real", "true", "false",
        "right", "wrong", "easy", "hard", "possible",
        # Misc (15)
        "thing", "place", "part", "side", "end", "way", "name", "word", "number",
        "kind", "sort", "lot", "bit", "piece", "group",
    ],
    
    # ================================================================
    # TIER 7: 200 new words (600 total) - Compound sentences
    # ================================================================
    7: [
        # Abstract nouns (20)
        "love", "hate", "fear", "hope", "life", "death", "truth", "peace",
        "war", "power", "money", "work", "job", "idea", "problem", "answer",
        "question", "reason", "mind", "heart",
        # Actions (30)
        "believes", "remembers", "forgets", "decides", "chooses", "agrees",
        "refuses", "accepts", "expects", "hopes", "wishes", "dreams", "plans",
        "prepares", "finishes", "continues", "happens", "appears", "disappears",
        "arrives", "returns", "enters", "exits", "crosses", "passes", "reaches",
        "touches", "covers", "fills", "pours",
        # Descriptions (25)
        "amazing", "incredible", "fantastic", "horrible", "strange", "normal",
        "special", "common", "rare", "perfect", "complete", "simple", "complex",
        "natural", "artificial", "modern", "ancient", "future", "past", "present",
        "public", "private", "free", "busy", "ready",
        # Body/Health (15)
        "body", "skin", "blood", "bone", "brain", "muscle", "tooth", "heart",
        "stomach", "breath", "pain", "health", "medicine", "doctor", "hospital",
        # Communication (15)
        "language", "voice", "sound", "noise", "music", "song", "story", "news",
        "message", "letter", "email", "phone", "computer", "screen", "picture",
        # Society (20)
        "city", "town", "village", "country", "world", "nation", "government",
        "law", "rule", "police", "army", "company", "office", "market", "bank",
        "church", "temple", "museum", "library", "theater",
        # More verbs (25)
        "creates", "destroys", "protects", "attacks", "defends", "supports",
        "develops", "improves", "increases", "decreases", "spreads", "collects",
        "gathers", "separates", "combines", "connects", "divides", "shares",
        "borrows", "lends", "owns", "belongs", "contains", "includes", "requires",
        # Connectors (10)
        "also", "too", "either", "neither", "both", "however", "therefore",
        "although", "unless", "while",
        # Pronouns/Determiners (10)
        "this", "that", "these", "those", "each", "every", "any", "another",
        "such", "same",
        # Time expressions (15)
        "already", "still", "yet", "soon", "later", "early", "late", "once",
        "twice", "again", "yesterday", "tomorrow", "forever", "ago", "during",
        # Misc (15)
        "example", "case", "fact", "point", "matter", "issue", "situation",
        "result", "effect", "cause", "chance", "choice", "control", "level", "rate",
    ],
    
    # ================================================================
    # TIER 8: 300 new words (900 total) - Complex structures
    # ================================================================
    8: [
        # Education (20)
        "student", "teacher", "class", "lesson", "test", "exam", "grade",
        "subject", "math", "science", "history", "art", "sport", "game",
        "team", "player", "winner", "loser", "score", "prize",
        # Emotions (20)
        "joy", "sorrow", "anger", "surprise", "excitement", "boredom",
        "confusion", "confidence", "doubt", "worry", "stress", "relief",
        "pride", "shame", "guilt", "jealousy", "sympathy", "respect",
        "trust", "patience",
        # Physical (20)
        "size", "shape", "color", "weight", "height", "length", "width",
        "depth", "distance", "speed", "force", "energy", "heat", "light",
        "shadow", "surface", "edge", "corner", "center", "middle",
        # Actions (50)
        "achieves", "accomplishes", "admits", "advises", "allows", "announces",
        "apologizes", "appreciates", "approaches", "argues", "arranges", "assumes",
        "attempts", "avoids", "behaves", "blames", "celebrates", "challenges",
        "claims", "compares", "complains", "considers", "convinces", "cries",
        "delays", "delivers", "demands", "denies", "describes", "deserves",
        "discusses", "doubts", "encourages", "enjoys", "escapes", "examines",
        "excuses", "explains", "expresses", "faces", "fails", "fears",
        "fights", "forces", "gains", "grabs", "handles", "hates", "hesitates", "hides",
        # Descriptions (40)
        "absolute", "accurate", "active", "actual", "additional", "afraid",
        "alike", "alive", "alone", "appropriate", "available", "aware", "basic",
        "blind", "brave", "brief", "bright", "broad", "calm", "capable",
        "careful", "careless", "certain", "cheap", "clear", "clever", "close",
        "comfortable", "conscious", "constant", "correct", "crazy", "creative",
        "critical", "curious", "current", "dangerous", "dead", "deaf", "dear",
        # Places/Things (40)
        "airport", "apartment", "area", "bridge", "building", "camp", "capital",
        "castle", "cave", "ceiling", "channel", "coast", "college", "court",
        "desert", "engine", "entrance", "equipment", "exit", "factory",
        "fence", "flag", "furniture", "garage", "gate", "harbor", "highway",
        "island", "jungle", "kingdom", "ladder", "machine", "mall", "palace",
        "platform", "prison", "restaurant", "roof", "stadium", "station",
        # Time/Sequence (20)
        "beginning", "ending", "finally", "gradually", "immediately", "meanwhile",
        "nowadays", "occasionally", "previously", "recently", "regularly",
        "repeatedly", "seldom", "shortly", "suddenly", "temporarily",
        "ultimately", "usually", "whenever", "wherever",
        # Quantities (20)
        "amount", "average", "balance", "bunch", "couple", "dozen", "extra",
        "half", "hundred", "majority", "maximum", "minimum", "pair", "percent",
        "plenty", "quarter", "single", "third", "total", "triple",
        # Relations (20)
        "above", "across", "against", "along", "among", "around", "aside",
        "below", "beneath", "beside", "beyond", "despite", "except", "inside",
        "outside", "throughout", "toward", "upon", "within", "without",
        # Abstract (50)
        "ability", "accident", "action", "advantage", "advice", "affair",
        "agreement", "aim", "appearance", "argument", "arrangement", "attention",
        "attitude", "authority", "basis", "behavior", "belief", "benefit",
        "birth", "blame", "border", "bottom", "burden", "career", "category",
        "claim", "collection", "comment", "communication", "community",
        "comparison", "competition", "concern", "condition", "connection",
        "consequence", "content", "context", "contract", "contribution",
        "conversation", "crime", "criticism", "culture", "custom", "damage",
        "danger", "date", "deal", "debate",
    ],
    
    # ================================================================
    # TIER 9: 400 new words (1300 total) - Near-conversational  
    # ================================================================
    9: [
        # Common verbs (60)
        "accepts", "adds", "adjusts", "affects", "applies", "arranges",
        "assists", "assumes", "attracts", "belongs", "bends", "bites",
        "blows", "boils", "borrows", "breathes", "brushes", "burns",
        "calculates", "cancels", "causes", "charges", "chases", "checks",
        "climbs", "closes", "collects", "communicates", "completes", "concerns",
        "confirms", "connects", "considers", "consists", "contains", "copies",
        "corrects", "crashes", "crosses", "damages", "dances", "deals",
        "defines", "delays", "delivers", "depends", "designs", "determines",
        "develops", "differs", "discovers", "discusses", "dislikes", "displays",
        "distributes", "dives", "drags", "draws", "drives", "earns",
        # Common nouns (80)
        "access", "account", "activity", "administration", "adult", "afternoon",
        "agency", "agent", "air", "album", "alcohol", "alternative", "analysis",
        "animal", "application", "appointment", "article", "artist", "aspect",
        "assessment", "assistant", "association", "atmosphere", "audience",
        "background", "band", "base", "bathroom", "bedroom", "beer", "benefit",
        "bill", "block", "board", "boss", "bottom", "bowl", "boyfriend",
        "branch", "brand", "breakfast", "brother", "budget", "button", "cabinet",
        "calendar", "camera", "campaign", "candidate", "capacity", "captain",
        "card", "carpet", "category", "chain", "chairman", "challenge", "champion",
        "championship", "chapter", "character", "chemical", "childhood", "chip",
        "chocolate", "cigarette", "circle", "citizen", "client", "climate",
        "clothes", "club", "coach", "coffee", "coin", "combination", "comedy", "comfort",
        # Adjectives (60)
        "acceptable", "accessible", "actual", "additional", "administrative",
        "advanced", "aggressive", "alternative", "angry", "annual", "anxious",
        "apparent", "appropriate", "artistic", "attractive", "automatic",
        "awful", "bare", "basic", "bitter", "blind", "bloody", "boring",
        "brilliant", "broken", "cheap", "chemical", "civil", "classic",
        "classical", "clean", "clinical", "close", "cold", "comfortable",
        "commercial", "common", "competitive", "complex", "comprehensive",
        "concerned", "confident", "conscious", "considerable", "consistent",
        "constant", "contemporary", "content", "convenient", "conventional",
        "convinced", "cool", "corporate", "correct", "crazy", "creative",
        "criminal", "critical", "cultural", "curious",
        # Adverbs (40)
        "absolutely", "accordingly", "actually", "additionally", "adequately",
        "almost", "altogether", "apparently", "approximately", "automatically",
        "barely", "basically", "carefully", "certainly", "clearly", "closely",
        "commonly", "completely", "considerably", "constantly", "currently",
        "deeply", "definitely", "deliberately", "directly", "easily", "effectively",
        "entirely", "equally", "especially", "essentially", "eventually",
        "exactly", "extremely", "fairly", "firmly", "fortunately", "frankly",
        "freely", "frequently",
        # More nouns (80)
        "commitment", "committee", "companion", "complaint", "component",
        "composition", "concentration", "concept", "conclusion", "confidence",
        "conflict", "confusion", "congress", "consciousness", "consensus",
        "consequence", "consideration", "construction", "consumer", "contact",
        "contemporary", "contest", "contribution", "convention", "cooperation",
        "corporation", "correspondence", "cost", "council", "counter", "county",
        "coverage", "creation", "creature", "credit", "crew", "crisis", "criterion",
        "critic", "criticism", "crowd", "currency", "curriculum", "customer",
        "cycle", "database", "daughter", "dealer", "death", "decade", "decision",
        "declaration", "decline", "definition", "degree", "democracy", "demonstration",
        "department", "description", "desire", "destination", "detail", "determination",
        "device", "dialogue", "diet", "difficulty", "dimension", "dinner",
        "direction", "director", "disaster", "discipline", "discount", "discovery",
        "discrimination", "discussion", "disease", "disk", "display",
        # More verbs (80)
        "educates", "eliminates", "emerges", "emphasizes", "employs", "enables",
        "encounters", "ends", "engages", "enhances", "enjoys", "ensures",
        "enters", "entertains", "equips", "escapes", "establishes", "estimates",
        "evaluates", "evolves", "examines", "exceeds", "exchanges", "excludes",
        "executes", "exercises", "exhibits", "exists", "expands", "expects",
        "experiences", "experiments", "explores", "exports", "exposes", "extends",
        "extracts", "facilitates", "fails", "falls", "favors", "features",
        "feeds", "files", "finances", "fits", "fixes", "floats", "flows",
        "focuses", "folds", "forbids", "forecasts", "forgives", "forms",
        "formulates", "forwards", "founds", "frames", "freezes", "functions",
        "funds", "furnishes", "generates", "governs", "grades", "grants",
        "grasps", "greets", "grounds", "guarantees", "guards", "guides",
        "hangs", "harvests", "heads", "heals", "hears", "heats", "highlights", "hires",
    ],
    
    # ================================================================
    # TIER 10: 700 new words (2000 total) - Conversational vocabulary
    # ================================================================
    10: [
        # Common words to reach 2000 (700 new)
        # Verbs (150)
        "abandons", "abolishes", "absorbs", "accelerates", "accompanies",
        "accumulates", "accuses", "acknowledges", "acquires", "activates",
        "adapts", "addresses", "administers", "admires", "adopts", "advances",
        "advertises", "advocates", "affords", "ages", "aids", "aims", "alerts",
        "allocates", "alters", "amazes", "amends", "amplifies", "analyzes",
        "anchors", "animates", "anticipates", "appeals", "applauds", "appoints",
        "appraises", "appreciates", "approves", "archives", "arises", "arms",
        "arrests", "articulates", "ascends", "assembles", "asserts", "assigns",
        "associates", "attaches", "attains", "attends", "authorizes", "awards",
        "backs", "bakes", "balances", "bands", "banks", "bargains", "bases",
        "batches", "battles", "beams", "bears", "beats", "becomes", "begs",
        "begins", "benchmarks", "bends", "benefits", "bets", "binds", "bites",
        "blanks", "blends", "blesses", "blocks", "blogs", "boards", "boasts",
        "boils", "bolts", "bombs", "bonds", "books", "boosts", "boots", "borders",
        "bothers", "bounces", "bounds", "boxes", "brackets", "branches", "brands",
        "braves", "breaches", "breaks", "breeds", "bridges", "broadcasts", "browses",
        "brushes", "budgets", "buffers", "bugs", "builds", "bumps", "bundles",
        "burdens", "burns", "bursts", "buries", "buses", "busts", "buzzes",
        "bypasses", "cables", "caches", "calculates", "calibrates", "campaigns",
        "camps", "cancels", "captures", "cares", "carves", "casts", "catalogs",
        "catches", "caters", "causes", "cautions", "ceases", "celebrates", "certifies",
        "chains", "chairs", "challenges", "champions", "channels", "characterizes",
        "charges", "charms", "charts", "chases", "chats", "cheats", "checks",
        # Nouns (200)
        "absence", "abundance", "accent", "acceptance", "accessibility", "accessory",
        "accommodation", "accomplishment", "accountability", "accountant", "accuracy",
        "accusation", "achievement", "acquisition", "acre", "activation", "activist",
        "actor", "actress", "adaptation", "addiction", "addition", "administrator",
        "admission", "adolescent", "adoption", "advancement", "adventure", "adversity",
        "advertisement", "advisor", "advocacy", "aesthetic", "affection", "affiliation",
        "aftermath", "agenda", "aggression", "agriculture", "aide", "airline",
        "alarm", "album", "alert", "algorithm", "alignment", "allegation", "alliance",
        "allocation", "allowance", "ally", "altar", "alteration", "amateur",
        "ambassador", "ambiguity", "ambition", "amendment", "amount", "amusement",
        "analyst", "ancestor", "anchor", "angel", "animation", "ankle", "anniversary",
        "announcement", "annoyance", "anticipation", "anxiety", "apology", "apparatus",
        "apparel", "appeal", "appendix", "appetite", "applause", "appliance",
        "applicant", "appreciation", "apprentice", "approval", "arc", "arch",
        "architect", "architecture", "archive", "arena", "armor", "aroma", "array",
        "arrow", "artery", "ash", "assault", "assembly", "assertion", "asset",
        "assignment", "assistance", "assumption", "assurance", "astronaut", "athlete",
        "athletics", "atlas", "attachment", "attacker", "attendance", "attorney",
        "attraction", "attribute", "auction", "audit", "aunt", "aura", "authenticity",
        "autobiography", "automation", "automobile", "autonomy", "availability",
        "avalanche", "avenue", "awareness", "axis", "bachelor", "backing", "backup",
        "bacon", "badge", "bakery", "ballot", "bamboo", "banana", "bandwidth",
        "banner", "bargain", "barn", "barrel", "barrier", "basement", "basin",
        "basket", "batch", "battalion", "battery", "battlefield", "bay", "beacon",
        "beam", "bean", "beard", "beast", "beauty", "beaver", "bedroom", "belly",
        "belt", "bench", "benchmark", "bend", "beneath", "beneficiary", "berry",
        "beverage", "bias", "bicycle", "bid", "bin", "biography", "biology", "bishop",
        "blade", "blanket", "blast", "blessing", "blindness", "bliss", "blogger",
        "bloom", "blueprint", "blur", "blush", "boardroom", "boast",
        # Adjectives (150)
        "abandoned", "abnormal", "absent", "absolute", "abstract", "absurd",
        "abundant", "academic", "acceptable", "acclaimed", "accomplished",
        "accountable", "accurate", "accused", "acoustic", "activated", "acute",
        "adaptable", "addicted", "adequate", "adjacent", "adjustable", "administrative",
        "admirable", "adolescent", "adorable", "adult", "adverse", "aesthetic",
        "affectionate", "affordable", "aged", "aggressive", "agile", "alarming",
        "alert", "alien", "aligned", "alike", "alleged", "allergic", "allied",
        "alphabetical", "alternate", "amateur", "ambitious", "ample", "amusing",
        "analytical", "ancient", "animated", "annoyed", "anonymous", "anticipated",
        "antique", "anxious", "applicable", "appointed", "appropriate", "approximate",
        "arbitrary", "archaeological", "architectural", "armed", "aromatic", "arranged",
        "artificial", "artistic", "ashamed", "assembled", "associated", "assumed",
        "astonishing", "athletic", "atmospheric", "attached", "attractive", "authentic",
        "authorized", "automatic", "autonomous", "available", "average", "awake",
        "awarded", "awesome", "awful", "awkward", "bacterial", "balanced", "bald",
        "bare", "baseline", "basic", "beautiful", "behavioral", "beloved", "beneficial",
        "benign", "best", "better", "biblical", "bigger", "biggest", "bilateral",
        "binary", "binding", "biological", "bizarre", "blank", "blessed", "blind",
        "blond", "bloody", "blue", "blunt", "bold", "bored", "boring", "botanical",
        "bound", "brave", "breathtaking", "brief", "bright", "brilliant", "broad",
        "broken", "bronze", "brutal", "budgetary", "built", "bulk", "bureaucratic",
        "burning", "busy", "calm", "capable", "capital", "captive", "cardiac",
        "careful", "careless", "casual", "catastrophic", "cautious", "celebrated",
        # Adverbs and misc (200)
        "abroad", "abruptly", "absently", "abstractly", "abundantly", "academically",
        "acceptably", "accidentally", "accordingly", "accurately", "actively",
        "acutely", "additionally", "adequately", "administratively", "admirably",
        "admittedly", "adversely", "aesthetically", "affectionately", "affordably",
        "aggressively", "allegedly", "alphabetically", "alternatively", "amazingly",
        "ambitiously", "amusingly", "analytically", "angrily", "annually", "anonymously",
        "anxiously", "apparently", "appropriately", "approximately", "arbitrarily",
        "architecturally", "arguably", "artificially", "artistically", "astonishingly",
        "athletically", "atmospherically", "attractively", "authentically", "automatically",
        "autonomously", "averagely", "awfully", "awkwardly", "backwards", "badly",
        "barely", "basically", "beautifully", "behaviorally", "beneficially", "best",
        "better", "biblically", "biologically", "bitterly", "bizarrely", "blandly",
        "blatantly", "bleakly", "blindly", "blissfully", "boldly", "botanically",
        "bravely", "breathlessly", "breezily", "briefly", "brightly", "brilliantly",
        "briskly", "broadly", "brutally", "busily", "calmly", "candidly", "capably",
        "carefully", "carelessly", "casually", "catastrophically", "cautiously",
        "centrally", "ceremonially", "certainly", "characteristically", "charitably",
        "cheaply", "cheerfully", "chemically", "chiefly", "chronologically", "civilly",
        "classically", "cleanly", "cleverly", "clinically", "clockwise", "closely",
        "coherently", "coincidentally", "coldly", "collaboratively", "collectively",
        "colorfully", "comfortably", "commercially", "commonly", "comparatively",
        "compassionately", "competently", "competitively", "completely", "comprehensively",
        "conceptually", "concisely", "conclusively", "concretely", "concurrently",
        "conditionally", "confidentially", "confidently", "consequently", "conservatively",
        "considerably", "consistently", "conspicuously", "constantly", "constitutionally",
        "constructively", "continually", "continuously", "contractually", "contrarily",
        "controversially", "conveniently", "conventionally", "conversely", "convincingly",
        "coolly", "cooperatively", "correctly", "correspondingly", "costly", "courageously",
        "courteously", "covertly", "crazily", "creatively", "credibly", "criminally",
        "crisply", "critically", "crucially", "cruelly", "culturally", "cunningly",
        "curiously", "currently", "customarily", "cynically", "daily", "dangerously",
        "darkly", "dearly", "decently", "decisively", "deeply", "defensively",
        "defiantly", "definitely", "deliberately", "delicately", "delightfully",
        "democratically", "densely", "dependently", "desperately", "destructively",
        "detailedly", "determinedly", "diagonally", "differently", "difficultly",
        "digitally", "diligently", "diplomatically", "directly", "dirtily", "disappointingly",
        "discreetly", "dishonestly", "distinctly", "diversely", "divinely", "domestically",
        "dominantly", "doubtfully", "downward", "downwards", "dramatically", "drastically",
        "dreadfully", "dryly", "dually", "dubiously", "duly", "durably", "dynamically",
    ],
}

# Tier sentence length ranges (min, max words per sentence)
TIER_SENTENCE_LENGTHS = {
    1: (3, 5),
    2: (3, 5),
    3: (4, 6),
    4: (5, 7),
    5: (5, 8),
    6: (6, 10),
    7: (7, 12),
    8: (8, 12),
    9: (8, 15),
    10: (10, 15),
}

# Sentence counts per tier (more sentences as vocab grows)
TIER_SENTENCE_COUNTS = {
    1: 500,
    2: 500,
    3: 800,
    4: 1000,
    5: 1500,
    6: 2000,
    7: 2500,
    8: 3000,
    9: 4000,
    10: 5000,
}

# Data paths
DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"

# Auto-scaling configuration
AUTO_SCALE_CONFIG = {
    "enabled": True,
    "plateau_threshold": 0.85,      # Stuck between this and mastery
    "plateau_generations": 50,       # Generations to wait before scaling
    "scale_factor": 1.5,            # Multiply dimensions by this
    "max_embedding_dim": 128,       # Cap on embedding dimension
    "max_hidden_size": 256,         # Cap on controller hidden size
}

# Threshold phases - complete all tiers at each threshold before tightening
THRESHOLD_PHASES = [2.5, 1.7, 1.0]  # Run all tiers at each threshold in sequence

# ================================================================
# PHASE 4+: NEXT-WORD PREDICTION / GENERATION CONFIG
# ================================================================
PREDICTOR_CONFIG = {
    "context_length": 4,              # Words of left context for prediction
    "predictor_hidden_size": 64,      # Hidden layer size for CausalController
    "predictor_generations": 5000,    # Max generations to train predictor
    "mastery_threshold": 0.85,        # Accuracy to consider predictor trained
    "checkpoint_interval": 100,       # Save checkpoint every N generations
    
    # Embedding fine-tuning
    "unfreeze_embeddings_at": 1000,   # Fine-tune embeddings after N gens (0 = always frozen)
    "embedding_learning_rate": 0.01,  # How much to adjust embeddings when unfrozen
    
    # Generation settings
    "max_generation_length": 20,      # Max tokens to generate
    "temperature": 1.0,               # Sampling temperature (1.0 = greedy, higher = more random)
    "top_k": 5,                        # Consider top K nearest words for sampling
    "min_confidence": 0.3,            # Stop if best match similarity < this
    
    # Mutation rates for predictor training
    "elite_mutation_rate": 0.005,
    "standard_mutation_rate": 0.1,
    "hyper_mutation_rate": 0.2,
}

# Predictor checkpoint directory
PREDICTOR_CHECKPOINT_DIR = "checkpoints_predictor"

def get_cumulative_vocab_size(tier):
    """Get total vocabulary size up to and including specified tier."""
    total = 0
    for t in range(1, tier + 1):
        if t in TIER_VOCAB:
            total += len(TIER_VOCAB[t])
    return total

# Print tier summary
print(f"[Config] Device: {DEVICE}")
print(f"[Config] Tiers defined: {len(TIER_VOCAB)}")
for tier in sorted(TIER_VOCAB.keys()):
    new_words = len(TIER_VOCAB[tier])
    total_words = get_cumulative_vocab_size(tier)
    length_range = TIER_SENTENCE_LENGTHS.get(tier, (3, 5))
    print(f"  Tier {tier}: +{new_words} words = {total_words} total, sentences {length_range[0]}-{length_range[1]} words")

