"""
Pattern LSTM for extracting temporal features from latent vectors
"""
import torch
import torch.nn as nn


class PatternLSTM(nn.Module):
    """
    Small LSTM that extracts temporal patterns from VAE latent vectors
    """

    def __init__(self, input_dim: int = 32, hidden_dim: int = 64, output_dim: int = 32):
        """
        Initialize PatternLSTM

        Args:
            input_dim: Input dimension (VAE latent dim)
            hidden_dim: LSTM hidden dimension
            output_dim: Output dimension (pattern embedding)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

        # Hidden state
        self.h = None
        self.c = None

    def reset_state(self, batch_size: int = 1, device: str = "cpu"):
        """
        Reset LSTM hidden state

        Args:
            batch_size: Batch size
            device: Device to create state on
        """
        self.h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        self.c = torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through LSTM

        Args:
            x: Input tensor [B, T, input_dim] or [B, input_dim]

        Returns:
            output: Pattern embedding [B, output_dim]
            (h, c): New hidden state
        """
        # Handle single timestep input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, input_dim]

        # Initialize hidden state if needed
        if self.h is None or self.c is None:
            self.reset_state(x.size(0), x.device)

        # LSTM forward
        lstm_out, (h_new, c_new) = self.lstm(x, (self.h, self.c))

        # Update hidden state
        self.h = h_new.detach()
        self.c = c_new.detach()

        # Project to output
        # Use last timestep output
        last_output = lstm_out[:, -1, :]  # [B, hidden_dim]
        pattern = self.fc(last_output)  # [B, output_dim]

        return pattern, (h_new, c_new)

    def get_pattern(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get pattern embedding without updating hidden state

        Args:
            x: Input tensor [B, input_dim]

        Returns:
            pattern: Pattern embedding [B, output_dim]
        """
        with torch.no_grad():
            pattern, _ = self.forward(x)
        return pattern
