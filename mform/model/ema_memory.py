"""EMA Memory Module for Memory Transformer."""
from __future__ import annotations

import torch
import torch.nn as nn


class EMAMemory(nn.Module):
    """Exponential Moving Average Memory Module.

    Updates memory as: m_t = alpha * m_{t-1} + (1 - alpha) * aggregate(h_t)
    """

    def __init__(
        self,
        hidden_dim: int,
        alpha: float = 0.9,
        learnable_alpha: bool = False,
        aggregation: str = "mean",
    ) -> None:
        """Initialize the EMA Memory module.

        Args:
            hidden_dim: Dimension of hidden states.
            alpha: EMA decay factor (higher = more weight on past).
            learnable_alpha: If True, alpha is a learnable parameter.
            aggregation: How to reduce sequence dimension ('mean', 'max', 'last').
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        self.learnable_alpha = learnable_alpha

        # Alpha parameter
        if learnable_alpha:
            # Initialize as logit, then sigmoid to keep in (0, 1)
            self.alpha_logit = nn.Parameter(torch.tensor(self._inverse_sigmoid(alpha)))
        else:
            self.register_buffer("alpha", torch.tensor(alpha))

    @staticmethod
    def _inverse_sigmoid(x: float) -> float:
        """Convert probability to logit.

        Args:
            x: Probability value in (0, 1).

        Returns:
            Logit value.
        """
        x_tensor = torch.tensor(x)
        return torch.log(x_tensor / (1 - x_tensor)).item()

    def get_alpha(self) -> torch.Tensor:
        """Get current alpha value.

        Returns:
            Alpha value in range (0, 1).
        """
        if self.learnable_alpha:
            return torch.sigmoid(self.alpha_logit)
        else:
            return self.alpha

    def aggregate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Aggregate hidden states over sequence dimension.

        Args:
            hidden_states: Shape (batch_size, seq_len, hidden_dim).

        Returns:
            Aggregated tensor of shape (batch_size, hidden_dim).
        """
        if self.aggregation == "mean":
            return hidden_states.mean(dim=1)
        elif self.aggregation == "max":
            return hidden_states.max(dim=1)[0]
        elif self.aggregation == "last":
            return hidden_states[:, -1, :]
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Update memory with new hidden states.

        Args:
            hidden_states: Shape (batch_size, seq_len, hidden_dim).
            memory_state: Previous memory state of shape (batch_size, hidden_dim).
                         If None, initializes to zeros.

        Returns:
            New memory state of shape (batch_size, hidden_dim).
        """
        batch_size = hidden_states.size(0)

        # Initialize memory if None
        if memory_state is None:
            memory_state = torch.zeros(
                batch_size,
                self.hidden_dim,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        # Aggregate current hidden states
        h_aggregated = self.aggregate(hidden_states)

        alpha = self.get_alpha()
        new_memory = alpha * memory_state + (1 - alpha) * h_aggregated

        return new_memory

    def reset_memory(self, batch_size: int = 1, device: str | torch.device = "cpu") -> torch.Tensor:
        """Create fresh memory state initialized to zeros.

        Args:
            batch_size: Batch size for memory state.
            device: Device to create memory on.

        Returns:
            Zero-initialized memory state of shape (batch_size, hidden_dim).
        """
        return torch.zeros(batch_size, self.hidden_dim, device=device)

