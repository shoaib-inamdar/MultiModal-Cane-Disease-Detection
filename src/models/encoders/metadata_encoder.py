import torch
import torch.nn as nn


class MetadataEncoder(nn.Module):
    """Encode four environmental metadata values into a 256-dimensional token."""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        output_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode metadata from [B, 4] to [B, 1, 256]."""
        out = self.net(x)
        return out.unsqueeze(1)