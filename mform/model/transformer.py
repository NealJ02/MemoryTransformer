"""Transformer Block for Memory Transformer."""
from __future__ import annotations

import torch
import torch.nn as nn

from mform.model.attention import MultiHeadAttention
from mform.model.feed_forward import FeedForward
from mform.model.normalization import LayerNorm


class TransformerBlock(nn.Module):
    """Transformer Block for Memory Transformer."""
    def __init__(self, cfg: dict) -> None:
        """Initialize the Transformer Block.

        Args:
            cfg: The configuration dictionary.
        """
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Transformer Block.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        # Attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Feedforward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
