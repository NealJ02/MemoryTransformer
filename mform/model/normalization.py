"""Layer Normalization for Transformer."""
from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer Normalization for Transformer."""
    def __init__(self, emb_dim: int) -> None:
        """Initialize the Layer Normalization.

        Args:
            emb_dim: The dimension of the embedding.
        """
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Layer Normalization.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
