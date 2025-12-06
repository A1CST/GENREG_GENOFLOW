"""
Minimal untrained VAE for generating latent vectors
"""
import torch
import torch.nn as nn


class SimpleVAE(nn.Module):
    """
    Untrained VAE that encodes random inputs into latent vectors
    """

    def __init__(self, input_dim: int = 128, latent_dim: int = 32, hidden_dim: int = 64):
        """
        Initialize VAE

        Args:
            input_dim: Input dimension
            latent_dim: Latent space dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Mean and log variance for latent distribution
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters

        Args:
            x: Input tensor [B, input_dim]

        Returns:
            mu: Mean of latent distribution [B, latent_dim]
            logvar: Log variance of latent distribution [B, latent_dim]
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick

        Args:
            mu: Mean [B, latent_dim]
            logvar: Log variance [B, latent_dim]

        Returns:
            z: Sampled latent vector [B, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstruction

        Args:
            z: Latent vector [B, latent_dim]

        Returns:
            reconstruction: [B, input_dim]
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE

        Args:
            x: Input tensor [B, input_dim]

        Returns:
            recon: Reconstruction [B, input_dim]
            mu: Latent mean [B, latent_dim]
            logvar: Latent log variance [B, latent_dim]
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def sample_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sample latent vector from input

        Args:
            x: Input tensor [B, input_dim]

        Returns:
            z: Latent vector [B, latent_dim]
        """
        with torch.no_grad():
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
        return z
