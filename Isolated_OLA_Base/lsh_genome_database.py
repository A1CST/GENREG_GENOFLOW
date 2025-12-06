"""
Simplified LSH-based genome database for storing and comparing genome vectors
Maps genome outputs to genome IDs for similarity tracking
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import List, Dict, Set, Tuple
import time
import numpy as np


class LSHGenomeDatabase:
    """
    LSH database for tracking genome output similarities
    Stores recent genome outputs and computes pairwise similarities
    """

    def __init__(self, vector_dim: int, n_bits: int = 64, seed: int = 42):
        """
        Initialize LSH genome database

        Args:
            vector_dim: Dimension of genome output vectors
            n_bits: Number of hash bits
            seed: Random seed for LSH projection
        """
        self.vector_dim = vector_dim
        self.n_bits = n_bits

        # Random projection for LSH
        torch.manual_seed(seed)
        self.R = torch.randn(vector_dim, n_bits)

        # Storage: genome_id -> list of recent vectors
        self.genome_vectors: Dict[int, List[torch.Tensor]] = {}
        self.max_vectors_per_genome = 20

        # Hash storage for fast lookup
        self.hash_to_genomes: Dict[int, Set[int]] = {}

    def hash_vector(self, vec: torch.Tensor) -> int:
        """
        Hash a vector using LSH

        Args:
            vec: Vector to hash [vector_dim]

        Returns:
            hash: Integer hash value
        """
        if vec.dim() > 1:
            vec = vec.squeeze()

        # Normalize
        vec_norm = F.normalize(vec.float(), dim=0, eps=1e-8)

        # Project and sign
        proj = vec_norm @ self.R
        bits = (proj >= 0).to(torch.int64)

        # Pack bits to int
        h = 0
        for i in range(self.n_bits):
            h |= (int(bits[i].item()) << i)

        return h

    def store_genome_vector(self, genome_id: int, vec: torch.Tensor):
        """
        Store a genome output vector

        Args:
            genome_id: Genome ID
            vec: Output vector [vector_dim]
        """
        if vec.dim() > 1:
            vec = vec.squeeze()

        vec_cpu = vec.detach().cpu().float()

        if genome_id not in self.genome_vectors:
            self.genome_vectors[genome_id] = []

        self.genome_vectors[genome_id].append(vec_cpu)

        # Keep only recent vectors
        if len(self.genome_vectors[genome_id]) > self.max_vectors_per_genome:
            self.genome_vectors[genome_id].pop(0)

        # Update hash index
        h = self.hash_vector(vec_cpu)
        if h not in self.hash_to_genomes:
            self.hash_to_genomes[h] = set()
        self.hash_to_genomes[h].add(genome_id)

    def compute_genome_similarity(self, genome_id1: int, genome_id2: int) -> float:
        """
        Compute similarity between two genomes based on their recent outputs

        Args:
            genome_id1: First genome ID
            genome_id2: Second genome ID

        Returns:
            similarity: Cosine similarity in [0, 1]
        """
        if genome_id1 not in self.genome_vectors or genome_id2 not in self.genome_vectors:
            return 0.0

        vecs1 = self.genome_vectors[genome_id1]
        vecs2 = self.genome_vectors[genome_id2]

        if not vecs1 or not vecs2:
            return 0.0

        # Use most recent vectors
        v1 = vecs1[-1]
        v2 = vecs2[-1]

        # Cosine similarity
        v1_norm = F.normalize(v1, dim=0, eps=1e-8)
        v2_norm = F.normalize(v2, dim=0, eps=1e-8)

        similarity = torch.dot(v1_norm, v2_norm).item()

        # Map to [0, 1]
        similarity = (similarity + 1.0) / 2.0

        return float(similarity)

    def get_all_similarities(self, genome_ids: List[int]) -> Dict[Tuple[int, int], float]:
        """
        Compute all pairwise similarities for a list of genomes

        Args:
            genome_ids: List of genome IDs

        Returns:
            similarities: Dict mapping (id1, id2) -> similarity
        """
        similarities = {}

        for i, id1 in enumerate(genome_ids):
            for id2 in genome_ids[i+1:]:
                sim = self.compute_genome_similarity(id1, id2)
                similarities[(id1, id2)] = sim
                similarities[(id2, id1)] = sim

        return similarities

    def get_similar_genomes(self, genome_id: int, threshold: float = 0.7) -> List[int]:
        """
        Get genomes similar to a given genome

        Args:
            genome_id: Query genome ID
            threshold: Similarity threshold

        Returns:
            similar_genome_ids: List of similar genome IDs
        """
        if genome_id not in self.genome_vectors:
            return []

        all_genome_ids = list(self.genome_vectors.keys())
        similar = []

        for other_id in all_genome_ids:
            if other_id == genome_id:
                continue

            sim = self.compute_genome_similarity(genome_id, other_id)
            if sim >= threshold:
                similar.append(other_id)

        return similar

    def get_genome_diversity(self, genome_ids: List[int]) -> float:
        """
        Compute diversity metric for a set of genomes
        Higher diversity = lower average similarity

        Args:
            genome_ids: List of genome IDs

        Returns:
            diversity: Diversity score in [0, 1]
        """
        if len(genome_ids) < 2:
            return 1.0

        similarities = []
        for i, id1 in enumerate(genome_ids):
            for id2 in genome_ids[i+1:]:
                sim = self.compute_genome_similarity(id1, id2)
                similarities.append(sim)

        if not similarities:
            return 1.0

        avg_similarity = sum(similarities) / len(similarities)
        diversity = 1.0 - avg_similarity

        return float(diversity)

    def prune_old_vectors(self):
        """Remove old vectors to prevent memory growth"""
        for genome_id in list(self.genome_vectors.keys()):
            if len(self.genome_vectors[genome_id]) > self.max_vectors_per_genome:
                self.genome_vectors[genome_id] = self.genome_vectors[genome_id][-self.max_vectors_per_genome:]

    def get_stats(self) -> Dict:
        """Get database statistics"""
        total_vectors = sum(len(vecs) for vecs in self.genome_vectors.values())

        return {
            'total_genomes': len(self.genome_vectors),
            'total_vectors': total_vectors,
            'unique_hashes': len(self.hash_to_genomes),
            'avg_vectors_per_genome': total_vectors / max(1, len(self.genome_vectors))
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (f"LSHGenomeDatabase(genomes={stats['total_genomes']}, "
                f"vectors={stats['total_vectors']}, "
                f"hashes={stats['unique_hashes']})")
