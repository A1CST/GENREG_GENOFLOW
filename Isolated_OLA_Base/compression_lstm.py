import torch
import torch.nn as nn
from typing import Tuple


class CompressionLSTM(nn.Module):
    """
    Small temporal bottleneck that predicts Δz.
    Input: z_t  [B, D]
    Output: Δẑ  [B, D]
    """

    def __init__(self, dim: int = 32, hidden: int = 64):
        super().__init__()
        self.dim = dim
        self.lstm = nn.LSTM(input_size=dim, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim)
        )
        self.h: Tuple[torch.Tensor, torch.Tensor] | None = None  # (h, c)

    def reset(self, batch_size: int = 1, device: str = "cpu"):
        h = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
        self.h = (h, c)

    @torch.no_grad()
    def step(self, z_t: torch.Tensor) -> torch.Tensor:
        """
        z_t: [B, D] at time t
        Returns Δẑ_t: [B, D]
        """
        x = z_t.unsqueeze(1)  # [B,1,D]
        y, self.h = self.lstm(x, self.h)  # [B,1,H]
        dz_hat = self.head(y[:, 0, :])    # [B,D]
        return dz_hat


