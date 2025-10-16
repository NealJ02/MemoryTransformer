"""Memory Transformer."""
from __future__ import annotations

import torch
import torch.nn as nn

from mform.model.normalization import LayerNorm
from mform.model.transformer import TransformerBlock


class MemoryTransformer(nn.Module):
    """Memory Transformer."""
    def __init__(self, cfg: dict) -> None:
        """Initialize the Memory Transformer.

        Args:
            cfg: The configuration dictionary.
        """
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Memory Transformer.

        Args:
            in_idx: The input tensor.

        Returns:
            The output tensor.
        """
        _, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = self.drop_emb(tok_embeds + pos_embeds)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
