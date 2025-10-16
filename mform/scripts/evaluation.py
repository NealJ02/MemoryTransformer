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


def evaluate_story_completion(
    model: torch.nn.Module,
    stories: list[str],
    tokenizer,
    device: torch.device,
    max_context: int = 128,
    num_stories: int = 10,
) -> dict[str, float]:
    """Evaluate model on story completion task.
    
    Args:
        model: The model to evaluate.
        stories: List of story strings.
        tokenizer: Tokenizer to use.
        device: Device to run on.
        max_context: Maximum context length.
        num_stories: Number of stories to evaluate on.
    
    Returns:
        Dictionary with completion metrics.
    """
    model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    total_tokens = 0

    with torch.no_grad():
        for story in stories[:num_stories]:
            # Tokenize the full story
            tokens = tokenizer.encode(story)

            if len(tokens) < 20:  # Skip very short stories
                continue

            # Split: first half = context, second half = target
            split_idx = len(tokens) // 2
            context_tokens = tokens[:split_idx]
            target_tokens = tokens[split_idx:]

            # Truncate context to max_context
            if len(context_tokens) > max_context:
                context_tokens = context_tokens[-max_context:]

            # Prepare input
            context_tensor = torch.tensor([context_tokens]).to(device)
            target_tensor = torch.tensor([target_tokens]).to(device)

            # Get model predictions for the context
            if hasattr(model, 'use_memory') and model.use_memory:
                logits, _ = model(context_tensor, return_memory=True)
            else:
                logits = model(context_tensor)

            # Get predictions for next tokens
            # We'll predict one token at a time for the target length
            predicted_tokens = []
            current_context = context_tokens.copy()

            for _ in range(min(len(target_tokens), 50)):  # Limit to 50 tokens
                if len(current_context) > max_context:
                    current_context = current_context[-max_context:]

                context_tensor = torch.tensor([current_context]).to(device)

                if hasattr(model, 'use_memory') and model.use_memory:
                    logits, _ = model(context_tensor, return_memory=True)
                else:
                    logits = model(context_tensor)

                # Get next token prediction
                next_token_logits = logits[0, -1, :]
                next_token = torch.argmax(next_token_logits).item()

                predicted_tokens.append(next_token)
                current_context.append(next_token)

            # Calculate accuracy
            min_len = min(len(predicted_tokens), len(target_tokens))
            if min_len > 0:
                correct = sum([1 for i in range(min_len) if predicted_tokens[i] == target_tokens[i]])
                total_accuracy += correct
                total_tokens += min_len

    model.train()

    return {
        "completion_accuracy": total_accuracy / total_tokens if total_tokens > 0 else 0.0,
        "stories_evaluated": min(num_stories, len(stories)),
    }
