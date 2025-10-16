"""Evaluation Script for Memory Transformer."""
from __future__ import annotations

import torch

from mform.scripts.training import calc_loss_loader


def evaluate_model(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, device: torch.device, eval_iter: int) -> tuple[float, float]:
    """Evaluate the model.

    Args:
        model: The model.
        train_loader: The train loader.
        val_loader: The val loader.
        device: The device.
        eval_iter: The number of batches to evaluate.
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
