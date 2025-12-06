# ================================================================
# CLIP Geometric Predictor
# ================================================================
# Adapted from GENREG's GeometricPredictor for single CLIP embeddings
# Instead of bidirectional context (left/right), uses virtual context
# duplication strategy to maintain geometric philosophy
# ================================================================

import torch
import torch.nn.functional as F


class CLIPGeometricPredictor:
    """
    CLIP-adapted geometric predictor with virtual context.

    No neural network - only evolvable geometric transformations.
    Forces ALL learning into the embedding space structure.

    Strategy:
    - Take single CLIP image embedding [512]
    - Duplicate into virtual context positions (8 default)
    - Apply evolvable position weights
    - Apply evolvable dimension weights
    - Output normalized embedding [512]
    """

    def __init__(self, embedding_dim=512, virtual_context_size=4, config=None, device=None):
        self.embedding_dim = embedding_dim
        self.virtual_context_size = virtual_context_size
        self.config = config or {}

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Evolvable parameters (no gradients, only mutation)
        # Position weights: how much each virtual context position contributes
        # Total positions = virtual_context_size * 2 (symmetric left/right)
        total_positions = virtual_context_size * 2
        # IMPORTANT: Initialize with small random noise to create initial diversity
        self.position_weights = torch.ones(total_positions, device=self.device) + torch.randn(total_positions, device=self.device) * 0.1

        # Dimension weights: per-dimension scaling in CLIP space
        # IMPORTANT: Initialize with small random noise to create initial diversity
        self.dim_weights = torch.ones(embedding_dim, device=self.device) + torch.randn(embedding_dim, device=self.device) * 0.05

        # Combination mode (can evolve to try different strategies)
        self.mode = 'weighted_mean'

    def __call__(self, image_embedding):
        """Make the predictor callable like nn.Module."""
        return self.forward(image_embedding)

    def forward(self, image_embedding):
        """
        Predict text embedding from CLIP image embedding.

        Args:
            image_embedding: [batch, 512] - CLIP image encoding

        Returns:
            [batch, 512] - Predicted text embedding (normalized)
        """
        batch_size = image_embedding.size(0)

        # Create virtual context by duplicating image embedding
        # This creates "memory" of the same visual input from multiple perspectives
        virtual_context = image_embedding.unsqueeze(1).repeat(
            1, self.virtual_context_size * 2, 1
        )
        # Shape: [batch, virtual_context_size*2, 512]

        if self.mode == 'weighted_mean':
            # Weighted average by position
            # Softmax ensures weights sum to 1
            weights = F.softmax(self.position_weights, dim=0)
            weights = weights.view(1, -1, 1)  # [1, positions, 1] for broadcasting

            # Weighted sum
            predicted = (virtual_context * weights).sum(dim=1)  # [batch, 512]

        elif self.mode == 'center_focused':
            # Weight positions by distance from center
            # Center positions get higher weight
            distances = torch.cat([
                torch.arange(self.virtual_context_size, 0, -1, device=self.device),
                torch.arange(1, self.virtual_context_size + 1, device=self.device)
            ]).float()

            weights = 1.0 / distances
            weights = weights / weights.sum()  # Normalize
            weights = weights.view(1, -1, 1)

            predicted = (virtual_context * weights).sum(dim=1)

        else:
            # Default to simple mean
            predicted = virtual_context.mean(dim=1)

        # Apply dimension weights (which CLIP dimensions matter)
        predicted = predicted * self.dim_weights

        # Normalize for CLIP space (critical!)
        return F.normalize(predicted, dim=-1)

    def mutate(self, rate=0.08, current_phase=None):
        """
        Mutate evolvable parameters.

        Args:
            rate: Mutation rate (probability of mutating each parameter)
            current_phase: Current training phase (can adjust mutation based on phase)
        """
        with torch.no_grad():
            # Mutate position weights
            mask = torch.rand_like(self.position_weights) < rate
            noise = torch.randn_like(self.position_weights) * 0.1
            self.position_weights[mask] += noise[mask]

            # Clamp to reasonable range
            self.position_weights.clamp_(0.01, 10.0)

            # Mutate dimension weights (less frequently - 20% of the time)
            if torch.rand(1).item() < rate * 0.2:
                dim_mask = torch.rand_like(self.dim_weights) < rate
                dim_noise = torch.randn_like(self.dim_weights) * 0.05
                self.dim_weights[dim_mask] += dim_noise[dim_mask]

                # Clamp dimension weights
                self.dim_weights.clamp_(0.1, 3.0)

            # Very rarely try a different mode (1% of the time)
            if torch.rand(1).item() < rate * 0.01:
                modes = ['weighted_mean', 'center_focused']
                self.mode = modes[torch.randint(len(modes), (1,)).item()]

    def clone(self):
        """Deep copy predictor for reproduction."""
        new_pred = CLIPGeometricPredictor(
            self.embedding_dim,
            self.virtual_context_size,
            self.config,
            self.device
        )
        new_pred.position_weights = self.position_weights.clone()
        new_pred.dim_weights = self.dim_weights.clone()
        new_pred.mode = self.mode
        return new_pred

    def state_dict(self):
        """For checkpointing."""
        return {
            'position_weights': self.position_weights.cpu(),
            'dim_weights': self.dim_weights.cpu(),
            'mode': self.mode,
            'virtual_context_size': self.virtual_context_size
        }

    def load_state_dict(self, state):
        """Load from checkpoint."""
        self.position_weights = state['position_weights'].to(self.device)
        self.dim_weights = state['dim_weights'].to(self.device)
        self.mode = state.get('mode', 'weighted_mean')
        self.virtual_context_size = state.get('virtual_context_size', 4)

    def get_param_count(self):
        """Return number of evolvable parameters."""
        return len(self.position_weights) + len(self.dim_weights)

    def to(self, device):
        """Move to device."""
        self.device = device
        self.position_weights = self.position_weights.to(device)
        self.dim_weights = self.dim_weights.to(device)
        return self

    def cpu(self):
        """Move to CPU."""
        return self.to('cpu')

    def cuda(self):
        """Move to CUDA."""
        return self.to('cuda')
