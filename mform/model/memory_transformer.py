"""Memory Transformer."""
from __future__ import annotations

import torch
import torch.nn as nn

from mform.model.ema_memory import EMAMemory
from mform.model.normalization import LayerNorm
from mform.model.transformer import TransformerBlock


class MemoryTransformer(nn.Module):
    """Memory Transformer with EMA memory mechanism."""
    def __init__(self, cfg: dict) -> None:
        """Initialize the Memory Transformer.

        Args:
            cfg: The configuration dictionary. Expected keys:
                - vocab_size: Size of vocabulary
                - emb_dim: Embedding dimension
                - context_length: Maximum sequence length
                - drop_rate: Dropout rate
                - n_layers: Number of transformer layers
                - use_memory: Whether to use EMA memory (default: False)
                - memory_alpha: EMA decay factor (default: 0.9)
                - learnable_alpha: Whether alpha is learnable (default: False)
                - memory_aggregation: Aggregation method (default: "mean")
                - memory_injection: Injection method (default: "additive")
        """
        super().__init__()

        # Extract memory configuration
        self.use_memory = cfg.get("use_memory", False)
        self.memory_injection = cfg.get("memory_injection", "additive")
        self.emb_dim = cfg["emb_dim"]

        # Token and position embeddings
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # EMA Memory module
        if self.use_memory:
            self.ema_memory = EMAMemory(
                hidden_dim=cfg["emb_dim"],
                alpha=cfg.get("memory_alpha", 0.9),
                learnable_alpha=cfg.get("learnable_alpha", False),
                aggregation=cfg.get("memory_aggregation", "mean"),
            )

            # Optional: learned gate for memory injection
            if self.memory_injection == "gated":
                self.memory_gate = nn.Linear(cfg["emb_dim"] * 2, cfg["emb_dim"])

        # Output head
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def inject_memory(self, hidden_states: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Inject memory into hidden states.

        Args:
            hidden_states: Shape (batch_size, seq_len, emb_dim).
            memory: Shape (batch_size, emb_dim).

        Returns:
            Modified hidden states of shape (batch_size, seq_len, emb_dim).
        """
        # Expand memory to match sequence length
        memory_expanded = memory.unsqueeze(1).expand_as(hidden_states)

        if self.memory_injection == "additive":
            # Simple addition
            return hidden_states + memory_expanded

        elif self.memory_injection == "gated":
            # Learned gating: gate * h + (1 - gate) * memory
            combined = torch.cat([hidden_states, memory_expanded], dim=-1)
            gate = torch.sigmoid(self.memory_gate(combined))
            return gate * hidden_states + (1 - gate) * memory_expanded

        elif self.memory_injection == "concat":
            # Concatenate as extra token (increases seq_len by 1)
            memory_token = memory.unsqueeze(1)  # (batch, 1, emb_dim)
            return torch.cat([hidden_states, memory_token], dim=1)

        else:
            raise ValueError(f"Unknown injection method: {self.memory_injection}")

    def forward(
        self,
        in_idx: torch.Tensor,
        memory_state: torch.Tensor | None = None,
        return_memory: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the Memory Transformer.

        Args:
            in_idx: Input token indices of shape (batch_size, seq_len).
            memory_state: Previous memory state of shape (batch_size, emb_dim).
                         If None and use_memory=True, initializes to zeros.
            return_memory: Whether to return updated memory state.

        Returns:
            If return_memory=True and use_memory=True:
                Tuple of (logits, new_memory_state)
            Otherwise:
                Just logits
            Logits shape: (batch_size, seq_len, vocab_size)
            Memory shape: (batch_size, emb_dim)
        """
        _, seq_len = in_idx.shape

        # Embeddings
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = self.drop_emb(tok_embeds + pos_embeds)

        # Optional: Inject memory before transformer layers
        if self.use_memory and memory_state is not None:
            x = self.inject_memory(x, memory_state)

        # Transformer layers
        x = self.trf_blocks(x)

        # Update memory after final layer
        new_memory = None
        if self.use_memory:
            new_memory = self.ema_memory(x, memory_state)
            # Inject updated memory back into hidden states
            x = self.inject_memory(x, new_memory)

        # Output projection
        x = self.final_norm(x)
        logits = self.out_head(x)

        # Return logits and optionally memory
        if return_memory and self.use_memory:
            return logits, new_memory
        else:
            return logits

    def reset_memory(self, batch_size: int = 1, device: str | torch.device = "cpu") -> torch.Tensor | None:
        """Reset memory to zeros.

        Args:
            batch_size: Batch size for memory state.
            device: Device to create memory on.

        Returns:
            Zero-initialized memory state if use_memory=True, else None.
        """
        if self.use_memory:
            return self.ema_memory.reset_memory(batch_size, device)
        return None
