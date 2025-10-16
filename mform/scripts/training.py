"""Training Script for Memory Transformer."""
from __future__ import annotations

import torch


def calc_loss_batch(input_batch: torch.Tensor, target_batch: torch.Tensor, model: torch.nn.Module, device: torch.device) -> float:
    """Calculate the loss for a batch.

    Args:
        input_batch: The input batch.
        target_batch: The target batch.
        model: The model.
        device: The device.
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device, num_batches: int | None = None) -> float:
    """Calculate the loss for a loader.

    Args:
        data_loader: The data loader.
        model: The model.
        device: The device.
        num_batches: The number of batches.
    """
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    num_batches = min(num_batches or len(data_loader), len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()
        else:
            break
    return total_loss / num_batches
