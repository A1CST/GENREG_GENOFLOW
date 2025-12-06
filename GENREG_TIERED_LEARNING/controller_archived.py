# ================================================================
# GENREG Tiered Learning - Bidirectional Controller
# ================================================================
# Controller that processes left context + blank marker + right context
# ================================================================

import torch
import torch.nn as nn
from config import DEVICE


class BidirectionalController(nn.Module):
    """
    Controller for fill-in-the-blank prediction.
    
    Input: [left_context_embeddings, blank_marker, right_context_embeddings]
    Output: predicted embedding for the blank
    
    Architecture:
        Left Context (2 words * emb_dim)  ─┐
        Blank Marker (emb_dim)            ─┼─> Concat -> FC1 -> ReLU -> FC2 -> Output
        Right Context (2 words * emb_dim) ─┘
    """
    
    def __init__(self, embedding_dim, context_window, hidden_size):
        """
        Args:
            embedding_dim: Size of word embeddings
            context_window: Number of words on each side of blank
            hidden_size: Hidden layer size
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.context_window = context_window
        self.hidden_size = hidden_size
        
        # Input size: left context + blank marker + right context
        # left: context_window * embedding_dim
        # blank marker: embedding_dim (learned marker)
        # right: context_window * embedding_dim
        self.input_size = (2 * context_window + 1) * embedding_dim
        
        # Output size: embedding dimension (predict the missing word's embedding)
        self.output_size = embedding_dim
        
        # Layers
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.output_size)
        
        # Learned blank marker embedding
        self.blank_marker = nn.Parameter(torch.randn(embedding_dim) * 0.1)
        
        # Initialize weights
        self._init_weights()
        
        # Move to device
        self.to(DEVICE)
    
    def _init_weights(self):
        """Initialize with small random weights."""
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, left_context, right_context):
        """
        Forward pass.
        
        Args:
            left_context: Tensor of shape [batch, context_window * embedding_dim]
            right_context: Tensor of shape [batch, context_window * embedding_dim]
        
        Returns:
            Predicted embedding for the blank [batch, embedding_dim]
        """
        batch_size = left_context.size(0)
        
        # Expand blank marker for batch
        blank = self.blank_marker.unsqueeze(0).expand(batch_size, -1)
        
        # Concatenate: left + blank + right
        x = torch.cat([left_context, blank, right_context], dim=1)
        
        # Forward through network
        x = torch.relu(self.fc1(x))
        output = self.fc2(x)
        
        return output
    
    def forward_single(self, left_embeddings, right_embeddings):
        """
        Forward pass for single sample with list of embeddings.
        
        Args:
            left_embeddings: List of embedding tensors (context_window items)
            right_embeddings: List of embedding tensors (context_window items)
        
        Returns:
            Predicted embedding [embedding_dim]
        """
        # Pad if necessary
        while len(left_embeddings) < self.context_window:
            left_embeddings.insert(0, torch.zeros(self.embedding_dim, device=DEVICE))
        while len(right_embeddings) < self.context_window:
            right_embeddings.append(torch.zeros(self.embedding_dim, device=DEVICE))
        
        # Take last context_window items from left, first from right
        left_embeddings = left_embeddings[-self.context_window:]
        right_embeddings = right_embeddings[:self.context_window]
        
        # Flatten
        left_flat = torch.cat(left_embeddings).unsqueeze(0)
        right_flat = torch.cat(right_embeddings).unsqueeze(0)
        
        return self.forward(left_flat, right_flat).squeeze(0)
    
    def clone(self):
        """Create a deep copy of this controller."""
        new_controller = BidirectionalController(
            self.embedding_dim,
            self.context_window,
            self.hidden_size
        )
        new_controller.load_state_dict(self.state_dict())
        return new_controller
    
    def mutate(self, rate=0.05, scale=0.07):
        """Mutate weights with Gaussian noise."""
        with torch.no_grad():
            for param in self.parameters():
                if param.requires_grad:
                    mask = torch.rand_like(param) < rate
                    noise = torch.randn_like(param) * scale
                    param.add_(noise * mask.float())
    
    def get_param_count(self):
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Test controller
    from config import CONFIG
    
    emb_dim = CONFIG["embedding_dim"]
    context_win = CONFIG["context_window"]
    hidden = CONFIG["controller_hidden_size"]
    
    controller = BidirectionalController(emb_dim, context_win, hidden)
    print(f"Controller created on {DEVICE}")
    print(f"Input size: {controller.input_size}")
    print(f"Output size: {controller.output_size}")
    print(f"Parameters: {controller.get_param_count()}")
    
    # Test forward
    batch_size = 4
    left = torch.randn(batch_size, context_win * emb_dim, device=DEVICE)
    right = torch.randn(batch_size, context_win * emb_dim, device=DEVICE)
    
    output = controller(left, right)
    print(f"\nTest forward:")
    print(f"  Left shape: {left.shape}")
    print(f"  Right shape: {right.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Test single
    left_embs = [torch.randn(emb_dim, device=DEVICE) for _ in range(2)]
    right_embs = [torch.randn(emb_dim, device=DEVICE) for _ in range(2)]
    
    single_output = controller.forward_single(left_embs, right_embs)
    print(f"\nTest single:")
    print(f"  Output shape: {single_output.shape}")
    
    # Test mutation
    controller.mutate(rate=0.1)
    print("\nMutation applied successfully")
    
    # Test clone
    clone = controller.clone()
    print(f"Clone parameters: {clone.get_param_count()}")

