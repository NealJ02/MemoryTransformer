"""Generation Script for Memory Transformer."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import transformers


def text_to_token_ids(text: str, tokenizer: transformers.Tokenizer) -> torch.Tensor:
    """Convert text to token ids.

    Args:
        text: The text to convert.
        tokenizer: The tokenizer.
    """
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: transformers.Tokenizer) -> str:
    """Convert token ids to text.

    Args:
        token_ids: The token ids to convert.
        tokenizer: The tokenizer.
    """
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_text_simple(model: torch.nn.Module, idx: torch.Tensor, max_new_tokens: int, context_size: int) -> torch.Tensor:
    """Generate text.

    Args:
        model: The model.
        idx: The token ids to generate from.
        max_new_tokens: The maximum number of new tokens to generate.
        context_size: The context size.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_and_print_sample(model: torch.nn.Module, tokenizer: transformers.Tokenizer, device: torch.device, start_context: str) -> None:
    """Generate and print a sample.

    Args:
        model: The model.
        tokenizer: The tokenizer.
        device: The device.
        start_context: The start context.
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))
    model.train()
