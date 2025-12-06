# ================================================================
# GENREG Tiered Learning - Causal Controller for Next-Word Prediction
# ================================================================
# Left-context only controller for autoregressive generation
# ================================================================

import torch
import torch.nn as nn
from config import DEVICE


class CausalController(nn.Module):
    """
    Controller for next-word prediction (causal/autoregressive).
    
    Input: [left_context_embeddings] - previous N words
    Output: predicted embedding for the next word
    
    Architecture:
        Left Context (N words * emb_dim) -> FC1 -> ReLU -> FC2 -> Output
    
    Simpler than BidirectionalController - no blank marker, no right context.
    """
    
    def __init__(self, embedding_dim, context_length, hidden_size):
        """
        Args:
            embedding_dim: Size of word embeddings
            context_length: Number of previous words to use as context
            hidden_size: Hidden layer size
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.hidden_size = hidden_size
        
        # Input size: context_length * embedding_dim
        self.input_size = context_length * embedding_dim
        
        # Output size: embedding dimension (predict the next word's embedding)
        self.output_size = embedding_dim
        
        # Layers
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.output_size)
        
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
    
    def forward(self, context):
        """
        Forward pass.
        
        Args:
            context: Tensor of shape [batch, context_length * embedding_dim]
        
        Returns:
            Predicted embedding for next word [batch, embedding_dim]
        """
        x = torch.relu(self.fc1(context))
        output = self.fc2(x)
        return output
    
    def forward_single(self, context_embeddings):
        """
        Forward pass for single sample with list of embeddings.
        
        Args:
            context_embeddings: List of embedding tensors (up to context_length items)
        
        Returns:
            Predicted embedding [embedding_dim]
        """
        # Pad with zeros if context is shorter than context_length
        while len(context_embeddings) < self.context_length:
            context_embeddings.insert(0, torch.zeros(self.embedding_dim, device=DEVICE))
        
        # Take last context_length items
        context_embeddings = context_embeddings[-self.context_length:]
        
        # Flatten and add batch dimension
        context_flat = torch.cat(context_embeddings).unsqueeze(0)
        
        return self.forward(context_flat).squeeze(0)
    
    def clone(self):
        """Create a deep copy of this controller."""
        new_controller = CausalController(
            self.embedding_dim,
            self.context_length,
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
    context_len = 4  # Use 4 words of context
    hidden = CONFIG["controller_hidden_size"]
    
    controller = CausalController(emb_dim, context_len, hidden)
    print(f"CausalController created on {DEVICE}")
    print(f"Input size: {controller.input_size}")
    print(f"Output size: {controller.output_size}")
    print(f"Parameters: {controller.get_param_count()}")
    
    # Test forward with batch
    batch_size = 4
    context = torch.randn(batch_size, context_len * emb_dim, device=DEVICE)
    
    output = controller(context)
    print(f"\nTest forward (batch):")
    print(f"  Context shape: {context.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Test single forward
    context_embs = [torch.randn(emb_dim, device=DEVICE) for _ in range(4)]
    single_output = controller.forward_single(context_embs)
    print(f"\nTest forward (single):")
    print(f"  Output shape: {single_output.shape}")
    
    # Test with shorter context (should pad)
    short_context = [torch.randn(emb_dim, device=DEVICE) for _ in range(2)]
    padded_output = controller.forward_single(short_context)
    print(f"\nTest forward (short context, padded):")
    print(f"  Input length: 2, Output shape: {padded_output.shape}")
    
    # Test mutation
    controller.mutate(rate=0.1)
    print("\nMutation applied successfully")
    
    # Test clone
    clone = controller.clone()
    print(f"Clone parameters: {clone.get_param_count()}")



