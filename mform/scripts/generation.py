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


def generate_text_simple(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Generate text with optional memory persistence.

    Args:
        model: The model.
        idx: The token ids to generate from of shape (batch_size, seq_len).
        max_new_tokens: The maximum number of new tokens to generate.
        context_size: The context size.
        temperature: Temperature for sampling. Higher = more random.
        top_k: If set, only sample from top k tokens.

    Returns:
        Generated token ids of shape (batch_size, seq_len + max_new_tokens).
    """
    # Initialize memory if model uses it
    batch_size = idx.size(0)
    device = idx.device
    memory_state = None

    if hasattr(model, 'use_memory') and model.use_memory:
        memory_state = model.reset_memory(batch_size=batch_size, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            # Forward pass with memory
            if hasattr(model, 'use_memory') and model.use_memory:
                result = model(idx_cond, memory_state=memory_state, return_memory=True)
                if isinstance(result, tuple):
                    logits, memory_state = result
                else:
                    logits = result
            else:
                logits = model(idx_cond)

        # Get logits for last token
        logits = logits[:, -1, :] / temperature

        # Optional: top-k sampling
        if top_k is not None:
            top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            min_top_k = top_k_values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_top_k, torch.full_like(logits, float('-inf')), logits)

        # Sample next token
        probas = torch.softmax(logits, dim=-1)

        if temperature == 0.0:
            # Greedy decoding
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        else:
            # Sampling
            idx_next = torch.multinomial(probas, num_samples=1)

        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate_and_print_sample(
    model: torch.nn.Module,
    tokenizer: transformers.Tokenizer,
    device: torch.device,
    start_context: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> None:
    """Generate and print a sample.

    Args:
        model: The model.
        tokenizer: The tokenizer.
        device: The device.
        start_context: The start context.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Temperature for sampling.
        top_k: If set, only sample from top k tokens.
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            temperature=temperature,
            top_k=top_k,
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))

    model.train()
