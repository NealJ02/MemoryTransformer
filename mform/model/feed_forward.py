"""Feed Forward for Transformer."""
from __future__ import annotations

import torch
import torch.nn as nn


class GELU(nn.Module):
    """GELU for Transformer."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the GELU.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        import torch
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
             (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """Feed Forward for Transformer."""
    def __init__(self, cfg: dict) -> None:
        """Initialize the Feed Forward.

        Args:
            cfg: The configuration dictionary.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Feed Forward.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        return self.layers(x)
