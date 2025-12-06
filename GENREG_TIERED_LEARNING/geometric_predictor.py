# ================================================================
# GENREG - Geometric Predictor (No Neural Controller)
# ================================================================
# Replaces the BidirectionalController with pure embedding geometry.
# 
# The prediction is a FUNCTION of the context embeddings only.
# No neural network. No learnable weights except the embeddings.
#
# This forces ALL learning into the embedding space.
# The embeddings MUST structure themselves semantically or fail.
# ================================================================

import torch
import torch.nn.functional as F
from config import DEVICE


class GeometricPredictor:
    """
    Predicts the blank embedding using only geometric operations on context.
    
    No neural network. No hidden layers. No learnable parameters.
    The only thing that can adapt is the embedding positions.
    
    Prediction strategies:
    - 'weighted_mean': Weighted average of context embeddings (position-weighted)
    - 'mean': Simple average of context embeddings
    - 'sum': Sum of context embeddings (normalized)
    - 'attention': Context embeddings weighted by learned attention vector
    
    The genome can evolve:
    - Position weights (how much each context position matters)
    - Combination coefficients
    """
    
    def __init__(self, embedding_dim, context_window, config, device=None):
        self.embedding_dim = embedding_dim
        self.context_window = context_window
        self.config = config
        self.device = device if device else DEVICE
        
        # Evolvable parameters (no gradients, just mutation)
        # Position weights: how much each context position contributes
        # [left_4, left_3, left_2, left_1, right_1, right_2, right_3, right_4]
        total_positions = 2 * context_window
        self.position_weights = torch.ones(total_positions, device=self.device) / total_positions
        
        # Optional: per-dimension scaling (allows some dims to matter more)
        self.dim_weights = torch.ones(embedding_dim, device=self.device)
        
        # Combination mode
        self.mode = 'weighted_mean'  # Can evolve to try different strategies
    
    def __call__(self, left_context, right_context):
        """Make the class callable like nn.Module."""
        return self.forward(left_context, right_context)
    
    def forward(self, left_context, right_context):
        """
        Predict the blank embedding from context.
        
        Args:
            left_context: Tensor [batch, context_window * embedding_dim]
            right_context: Tensor [batch, context_window * embedding_dim]
        
        Returns:
            Predicted embedding [batch, embedding_dim]
        """
        batch_size = left_context.size(0)
        
        # Reshape to [batch, context_window, embedding_dim]
        left = left_context.view(batch_size, self.context_window, self.embedding_dim)
        right = right_context.view(batch_size, self.context_window, self.embedding_dim)
        
        # Stack all context: [batch, 2*context_window, embedding_dim]
        # Order: [left_far...left_near, right_near...right_far]
        all_context = torch.cat([left, right], dim=1)
        
        if self.mode == 'weighted_mean':
            # Weighted average by position
            # position_weights: [2*context_window]
            weights = F.softmax(self.position_weights, dim=0)  # Normalize to sum=1
            weights = weights.view(1, -1, 1)  # [1, positions, 1] for broadcasting
            
            # Weighted sum
            predicted = (all_context * weights).sum(dim=1)  # [batch, embedding_dim]
        
        elif self.mode == 'mean':
            # Simple average
            predicted = all_context.mean(dim=1)
        
        elif self.mode == 'sum':
            # Sum, normalized
            predicted = all_context.sum(dim=1)
            predicted = F.normalize(predicted, dim=1)
        
        elif self.mode == 'center_focused':
            # Weight positions closer to the blank more heavily
            # Positions: [left_4, left_3, left_2, left_1, right_1, right_2, right_3, right_4]
            # Distances: [4, 3, 2, 1, 1, 2, 3, 4]
            distances = torch.tensor(
                list(range(self.context_window, 0, -1)) + list(range(1, self.context_window + 1)),
                device=self.device, dtype=torch.float
            )
            weights = 1.0 / distances  # Closer = higher weight
            weights = weights / weights.sum()  # Normalize
            weights = weights.view(1, -1, 1)
            
            predicted = (all_context * weights).sum(dim=1)
        
        else:
            # Default to mean
            predicted = all_context.mean(dim=1)
        
        # Apply dimension weights
        predicted = predicted * self.dim_weights
        
        return predicted
    
    def forward_single(self, left_embeddings, right_embeddings):
        """
        Forward pass for single sample with list of embeddings.
        """
        # Pad if necessary
        zero_emb = torch.zeros(self.embedding_dim, device=self.device)
        
        while len(left_embeddings) < self.context_window:
            left_embeddings.insert(0, zero_emb)
        while len(right_embeddings) < self.context_window:
            right_embeddings.append(zero_emb)
        
        # Take correct window
        left_embeddings = left_embeddings[-self.context_window:]
        right_embeddings = right_embeddings[:self.context_window]
        
        # Stack and flatten
        left_flat = torch.cat(left_embeddings).unsqueeze(0)
        right_flat = torch.cat(right_embeddings).unsqueeze(0)
        
        return self.forward(left_flat, right_flat).squeeze(0)
    
    def clone(self):
        """Deep copy predictor."""
        new_pred = GeometricPredictor(
            self.embedding_dim,
            self.context_window,
            self.config,
            self.device
        )
        new_pred.position_weights = self.position_weights.clone()
        new_pred.dim_weights = self.dim_weights.clone()
        new_pred.mode = self.mode
        return new_pred
    
    def mutate(self, rate=None, scale=None):
        """Mutate the evolvable parameters."""
        if rate is None:
            rate = self.config.get("predictor_mutation_rate", 0.05)
        if scale is None:
            scale = self.config.get("predictor_mutation_scale", 0.1)
        
        with torch.no_grad():
            # Mutate position weights
            mask = torch.rand_like(self.position_weights) < rate
            noise = torch.randn_like(self.position_weights) * scale
            self.position_weights += noise * mask.float()
            
            # Keep weights positive and bounded
            pos_min = self.config.get("predictor_position_weight_min", 0.01)
            pos_max = self.config.get("predictor_position_weight_max", 10.0)
            self.position_weights.clamp_(pos_min, pos_max)
            
            # Occasionally mutate dimension weights
            dim_mut_prob = self.config.get("predictor_dim_mutation_prob", 0.1)
            if torch.rand(1).item() < rate * dim_mut_prob:
                dim_mask = torch.rand_like(self.dim_weights) < rate
                dim_scale_mult = self.config.get("predictor_dim_mutation_scale_mult", 0.5)
                dim_noise = torch.randn_like(self.dim_weights) * scale * dim_scale_mult
                self.dim_weights += dim_noise * dim_mask.float()
                dim_min = self.config.get("predictor_dim_weight_min", 0.1)
                dim_max = self.config.get("predictor_dim_weight_max", 3.0)
                self.dim_weights.clamp_(dim_min, dim_max)
            
            # Very rarely try a different mode
            mode_mut_prob = self.config.get("predictor_mode_mutation_prob", 0.01)
            if torch.rand(1).item() < rate * mode_mut_prob:
                modes = ['weighted_mean', 'mean', 'center_focused']
                self.mode = modes[torch.randint(len(modes), (1,)).item()]
    
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
    
    def eval(self):
        """Compatibility with nn.Module interface."""
        pass  # No-op, we don't have train/eval modes
    
    def train(self):
        """Compatibility with nn.Module interface."""
        pass
    
    def state_dict(self):
        """For checkpointing."""
        return {
            'position_weights': self.position_weights.cpu(),
            'dim_weights': self.dim_weights.cpu(),
            'mode': self.mode,
        }
    
    def load_state_dict(self, state):
        """Load from checkpoint."""
        self.position_weights = state['position_weights'].to(self.device)
        self.dim_weights = state['dim_weights'].to(self.device)
        self.mode = state.get('mode', 'weighted_mean')


# ================================================================
# Drop-in replacement for BidirectionalController
# ================================================================
# To use: Replace imports and creation in genome.py
#
# Change:
#   from controller import BidirectionalController
# To:
#   from geometric_predictor import GeometricPredictor
#
# Change:
#   controller = BidirectionalController(emb_dim, context_window, hidden_size)
# To:
#   controller = GeometricPredictor(emb_dim, context_window, config)
# ================================================================


if __name__ == "__main__":
    # Test
    from config import CONFIG
    
    emb_dim = CONFIG["embedding_dim"]
    context_win = CONFIG["context_window"]
    
    predictor = GeometricPredictor(emb_dim, context_win, CONFIG)
    print(f"GeometricPredictor created on {predictor.device}")
    print(f"Embedding dim: {predictor.embedding_dim}")
    print(f"Context window: {predictor.context_window}")
    print(f"Evolvable parameters: {predictor.get_param_count()}")
    print(f"Mode: {predictor.mode}")
    
    # Test forward
    batch_size = 4
    left = torch.randn(batch_size, context_win * emb_dim, device=DEVICE)
    right = torch.randn(batch_size, context_win * emb_dim, device=DEVICE)
    
    output = predictor(left, right)
    print(f"\nTest forward:")
    print(f"  Left shape: {left.shape}")
    print(f"  Right shape: {right.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Test single
    left_embs = [torch.randn(emb_dim, device=DEVICE) for _ in range(2)]
    right_embs = [torch.randn(emb_dim, device=DEVICE) for _ in range(2)]
    
    single_output = predictor.forward_single(left_embs, right_embs)
    print(f"\nTest single:")
    print(f"  Output shape: {single_output.shape}")
    
    # Test mutation
    predictor.mutate(rate=0.2)
    print(f"\nAfter mutation:")
    print(f"  Position weights: {predictor.position_weights}")
    print(f"  Mode: {predictor.mode}")
    
    # Test clone
    clone = predictor.clone()
    print(f"Clone parameters: {clone.get_param_count()}")

