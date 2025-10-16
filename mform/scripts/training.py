"""Training Script for Memory Transformer."""
from __future__ import annotations

import torch


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    memory_state: torch.Tensor | None = None,
    reset_memory: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Calculate the loss for a batch.

    Args:
        input_batch: The input batch of shape (batch_size, seq_len).
        target_batch: The target batch of shape (batch_size, seq_len).
        model: The model.
        device: The device.
        memory_state: Previous memory state. If None, initializes fresh memory.
        reset_memory: If True, doesn't use previous memory state.

    Returns:
        Tuple of (loss, new_memory_state).
        new_memory_state is None if model doesn't use memory.
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    # Handle memory state
    if reset_memory:
        memory_state = None
    elif memory_state is not None:
        # Detach to prevent backprop through time
        memory_state = memory_state.detach()

    # Forward pass
    if hasattr(model, 'use_memory') and model.use_memory:
        result = model(input_batch, memory_state=memory_state, return_memory=True)
        if isinstance(result, tuple):
            logits, new_memory = result
        else:
            logits = result
            new_memory = None
    else:
        logits = model(input_batch)
        new_memory = None

    # Compute loss
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())

    return loss, new_memory


def calc_loss_loader(
    data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    num_batches: int | None = None,
    reset_memory_per_batch: bool = True,
) -> float:
    """Calculate the loss for a loader.

    Args:
        data_loader: The data loader.
        model: The model.
        device: The device.
        num_batches: The number of batches to evaluate. If None, uses all batches.
        reset_memory_per_batch: If True, resets memory state for each batch.
                                If False, memory persists across batches.

    Returns:
        Average loss over the batches.
    """
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")

    num_batches = min(num_batches or len(data_loader), len(data_loader))
    memory_state = None

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss, memory_state = calc_loss_batch(
                input_batch,
                target_batch,
                model,
                device,
                memory_state=memory_state,
                reset_memory=reset_memory_per_batch,
            )
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches
