# ================================================================
# Protein Signal Computation for CLIP
# ================================================================
# Compute signals for protein network from CLIP predictions
# ================================================================

import torch
from .caltech_categories import check_caltech_category_match


def get_predicted_class(student_embedding, all_txt_tensor):
    """
    Get predicted class index by finding nearest text embedding.

    Args:
        student_embedding: [512] - Student's predicted text embedding
        all_txt_tensor: [101, 512] - All CLIP text embeddings for Caltech-101

    Returns:
        int: Predicted class index (0-100)
    """
    # Compute similarities with all text embeddings
    sims = (student_embedding @ all_txt_tensor.t())
    return sims.argmax().item()


def compute_protein_signals(student_embedding, correct_text_embedding,
                            predicted_class_idx, true_class_idx, labels):
    """
    Compute signals for protein network in CLIP context.

    Args:
        student_embedding: [512] - Student's predicted text embedding
        correct_text_embedding: [512] - Ground truth text embedding
        predicted_class_idx: int - Student's predicted Caltech class
        true_class_idx: int - Ground truth Caltech class
        labels: list - Caltech-101 class names

    Returns:
        dict with keys:
            - prediction_distance: float (Euclidean distance)
            - category_match: float (0.0, 0.5, or 1.0)
            - token_hit: float (1.0 if distance < threshold, else 0.0)
    """
    signals = {}

    # 1. Prediction distance (Euclidean)
    # This is the main gradient signal - lower is better
    distance = torch.dist(student_embedding, correct_text_embedding, p=2).item()
    signals['prediction_distance'] = distance

    # 2. Category match (semantic similarity)
    # 1.0 = same semantic group (both animals)
    # 0.5 = related groups (animal + nature)
    # 0.0 = unrelated groups
    if predicted_class_idx == true_class_idx:
        # Exact match
        signals['category_match'] = 1.0
    else:
        # Check semantic grouping
        pred_label = labels[predicted_class_idx]
        true_label = labels[true_class_idx]
        signals['category_match'] = check_caltech_category_match(pred_label, true_label)

    # 3. Hit detection (binary threshold)
    # 1.0 if distance below threshold (backward compatibility)
    # 0.0 otherwise
    hit_threshold = 1.5  # Tunable threshold
    signals['token_hit'] = 1.0 if distance < hit_threshold else 0.0

    return signals
