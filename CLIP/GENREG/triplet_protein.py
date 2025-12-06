import random
from typing import Dict, List, Tuple

import torch

from config import DEVICE
from language_proteins import get_word_category


class TripletProtein:
    """
    Triplet-based semantic clustering evaluator.

    For each genome, samples (anchor, positive, negative) triplets from the
    current tier vocabulary and checks whether the anchor embedding is closer
    to the positive than to the negative.

    - anchor, positive share the same semantic category (via get_word_category)
    - negative is drawn from a different category

    The final score is:
        triplet_score = correct_triplets / total_triplets   (0.0 - 1.0)
    """

    def __init__(self, config):
        # Number of triplets to sample per genome per generation
        self.samples_per_genome: int = int(config.get("triplet_samples_per_genome", 100))

    @staticmethod
    def _build_category_index(vocab) -> Tuple[Dict[str, List[int]], List[int]]:
        """
        Build mapping from semantic category -> list of word IDs for
        the current vocabulary, and a flat list of all candidate word IDs.
        """
        category_to_ids: Dict[str, List[int]] = {}
        all_ids: List[int] = []

        for word in vocab.get_all_words():
            word_id = vocab.get_id(word)
            if word_id is None:
                continue
            category = get_word_category(word)
            category_to_ids.setdefault(category, []).append(word_id)
            all_ids.append(word_id)

        return category_to_ids, all_ids

    def evaluate(self, genome, vocab) -> float:
        """
        Evaluate a single genome on triplet consistency.

        Args:
            genome: TieredGenome instance (must implement get_embedding_tensor(word_id))
            vocab:  TieredVocabulary instance (current tier vocabulary)

        Returns:
            float: triplet_score in [0.0, 1.0]
        """
        category_to_ids, all_ids = self._build_category_index(vocab)

        if not all_ids or self.samples_per_genome <= 0:
            return 0.0

        correct = 0
        total = 0

        # Precompute list of categories that have at least 2 words
        viable_anchor_cats = [
            cat for cat, ids in category_to_ids.items() if len(ids) >= 2
        ]
        if not viable_anchor_cats:
            return 0.0

        # For each sample, try a few times to construct a valid triplet
        max_attempts_per_sample = 10

        for _ in range(self.samples_per_genome):
            attempts = 0
            while attempts < max_attempts_per_sample:
                attempts += 1

                anchor_cat = random.choice(viable_anchor_cats)
                positive_cats = [anchor_cat]

                # Negative categories must be different from anchor_cat
                negative_cats = [
                    cat for cat in category_to_ids.keys() if cat != anchor_cat and category_to_ids[cat]
                ]
                if not negative_cats:
                    continue

                anchor_ids = category_to_ids[anchor_cat]
                if len(anchor_ids) < 2:
                    # Need at least two IDs to choose distinct anchor and positive
                    continue

                anchor_id = random.choice(anchor_ids)

                # Choose positive ID from same category but (ideally) different from anchor
                pos_id = random.choice(anchor_ids)
                if pos_id == anchor_id and len(anchor_ids) > 1:
                    # Re-sample once; if still same, we'll just accept it
                    pos_id = random.choice(anchor_ids)

                neg_cat = random.choice(negative_cats)
                neg_ids = category_to_ids[neg_cat]
                if not neg_ids:
                    continue
                neg_id = random.choice(neg_ids)

                # Compute distances
                with torch.no_grad():
                    anchor_emb = genome.get_embedding_tensor(anchor_id).to(DEVICE)
                    pos_emb = genome.get_embedding_tensor(pos_id).to(DEVICE)
                    neg_emb = genome.get_embedding_tensor(neg_id).to(DEVICE)

                    d_pos = torch.dist(anchor_emb, pos_emb, p=2).item()
                    d_neg = torch.dist(anchor_emb, neg_emb, p=2).item()

                total += 1
                if d_pos < d_neg:
                    correct += 1
                else:
                    # FAIL - anchor closer to wrong word
                    # Calculate nudge direction: push anchor toward positive
                    nudge = pos_emb - anchor_emb
                    genome.record_triplet_error(anchor_id, nudge)

                # Successfully formed a triplet; break attempt loop
                break

        if total == 0:
            return 0.0

        return correct / float(total)


