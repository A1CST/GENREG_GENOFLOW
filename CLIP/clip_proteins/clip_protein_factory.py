# ================================================================
# CLIP Protein Network Factory
# ================================================================
# Creates protein network for CLIP training with gradient-rich signals
# Adapted from GENREG.language_proteins for CLIP context
# ================================================================

import sys
import os

# Add GENREG directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'GENREG'))

from proteins import SensorProtein, TrustModifierProtein
from language_proteins import (
    ProximityRewardProtein,
    ImprovementBonusProtein,
    CategoryProtein
)


# Configuration for CLIP protein network
CLIP_PROTEIN_CONFIG = {
    # Distance normalization (CLIP space has different scale than word embeddings)
    "protein_distance_scale": 3.0,

    # Proximity reward (main gradient signal)
    # Gives partial credit based on distance - closer = better
    "protein_proximity_max": 1.0,
    "protein_proximity_halflife": 2.0,  # Distance at which reward halves
    "protein_proximity_steepness": 1.0,
    "protein_proximity_penalty_baseline": 0.4,  # Subtract from reward (makes dist>2.0 negative)
    "protein_proximity_min": -1.0,  # Minimum floor (changed from -0.3)
    "protein_proximity_scale": 6.0,  # Reduced from 10.0 to balance with category penalties

    # Improvement tracking
    # Rewards genomes that are improving over time
    "protein_improvement_bonus": 0.5,
    "protein_regression_penalty": 0.3,  # Increased from 0.1 to make regression more costly
    "protein_improvement_scale": 5.0,

    # Category matching
    # Semantic similarity bonus (same category, related category)
    "protein_category_bonus": 0.8,
    "protein_related_bonus": 0.3,
    "protein_category_mismatch_penalty": 0.2,  # Penalty for wrong category
    "protein_category_scale": 12.0,  # Increased from 10.0 to strengthen category signal
}


def create_clip_protein_network(config):
    """
    Create protein network for CLIP training.

    Network structure:
        Sensors (read raw signals)
            ↓
        Processing Proteins (shape signals)
            ↓
        Trust Modifiers (convert to trust delta)

    Args:
        config: Configuration dict with protein parameters

    Returns:
        list: Ordered list of proteins for cascade execution
    """
    proteins = []

    # ================================================================
    # SENSORS - Read raw signals from environment
    # ================================================================

    # Distance sensor
    distance_sensor = SensorProtein("prediction_distance")
    distance_sensor.params["norm_scale"] = config["protein_distance_scale"]
    proteins.append(distance_sensor)

    # Category match sensor
    category_sensor = SensorProtein("category_match")
    category_sensor.params["norm_scale"] = 1.0  # Already 0-1
    proteins.append(category_sensor)

    # Hit sensor
    hit_sensor = SensorProtein("token_hit")
    hit_sensor.params["norm_scale"] = 1.0  # Already 0-1
    proteins.append(hit_sensor)

    # ================================================================
    # PROCESSING PROTEINS - Shape reward signals
    # ================================================================

    # Proximity reward - partial credit based on distance
    # This is the KEY protein for gradient-rich learning
    proximity = ProximityRewardProtein("proximity_reward")
    proximity.params["max_reward"] = config["protein_proximity_max"]
    proximity.params["half_life_distance"] = config["protein_proximity_halflife"]
    proximity.params["steepness"] = config["protein_proximity_steepness"]
    proximity.params["penalty_baseline"] = config.get("protein_proximity_penalty_baseline", 0.0)
    proximity.params["min_reward"] = config.get("protein_proximity_min", -2.0)
    proteins.append(proximity)

    # Improvement bonus - rewards learning trajectories
    # Tracks distance trend over time
    improvement = ImprovementBonusProtein("improvement_bonus")
    improvement.params["improvement_bonus"] = config["protein_improvement_bonus"]
    improvement.params["regression_penalty"] = config["protein_regression_penalty"]
    proteins.append(improvement)

    # Category matching - semantic similarity bonus
    # Gives partial credit for predicting related classes
    category = CategoryProtein("category_match_protein")
    category.params["match_bonus"] = config["protein_category_bonus"]
    category.params["related_bonus"] = config["protein_related_bonus"]
    category.params["mismatch_penalty"] = config.get("protein_category_mismatch_penalty", 0.0)
    category.bind_inputs(["category_match"])
    proteins.append(category)

    # ================================================================
    # TRUST MODIFIERS - Convert signals to trust delta
    # ================================================================

    # Proximity-based trust (main gradient signal)
    proximity_trust = TrustModifierProtein("proximity_trust")
    proximity_trust.params["scale"] = config["protein_proximity_scale"]
    proximity_trust.bind_inputs(["proximity_reward"])
    proteins.append(proximity_trust)

    # Improvement-based trust (learning trajectory bonus)
    improvement_trust = TrustModifierProtein("improvement_trust")
    improvement_trust.params["scale"] = config["protein_improvement_scale"]
    improvement_trust.bind_inputs(["improvement_bonus"])
    proteins.append(improvement_trust)

    # Category-based trust (semantic similarity bonus)
    category_trust = TrustModifierProtein("category_trust")
    category_trust.params["scale"] = config["protein_category_scale"]
    category_trust.bind_inputs(["category_match_protein"])
    proteins.append(category_trust)

    return proteins
