"""Workflow for the Memory Transformer."""
from __future__ import annotations

import tiktoken
import torch

from mform.model.memory_transformer import MemoryTransformer
from mform.scripts.generation import (
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text,
)


class MemoryTransformerWorkflow:
    """Workflow for the Memory Transformer."""
    def __init__(self, ) -> None:
        """Initialize the Memory Transformer Workflow.

        Args:
            cfg: The configuration dictionary.
        """
        self.config = {
            "vocab_size": 50257,   # Vocabulary size
            "context_length": 256, # Shortened context length (orig: 1024)
            "emb_dim": 768,        # Embedding dimension
            "n_heads": 12,         # Number of attention heads
            "n_layers": 12,        # Number of layers
            "drop_rate": 0.1,      # Dropout rate
            "qkv_bias": False      # Query-key-value bias
        }


    def run(self) -> None:
        """Run the Memory Transformer Workflow."""
        torch.manual_seed(123)
        model = MemoryTransformer(self.config)
        model.eval()
        start_context = "Every effort moves you"
        tokenizer = tiktoken.get_encoding("gpt2")

        token_ids = generate_text_simple(
            model=model,
            idx=text_to_token_ids(start_context, tokenizer),
            max_new_tokens=10,
            context_size=self.config["context_length"]
        )

        print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
