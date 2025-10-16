"""Workflow for the Memory Transformer."""
from __future__ import annotations

import tiktoken
import torch

from mform.data.dataset import create_dataloader_v1
from mform.model.memory_transformer import MemoryTransformer
from mform.scripts.evaluation import evaluate_model
from mform.scripts.generation import (
    generate_and_print_sample,
)
from mform.scripts.training import calc_loss_batch


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
            "qkv_bias": False,      # Query-key-value bias
            "data_path": "mform/data/the-verdict.txt",
            "use_memory": False,           # Enable memory
            "memory_alpha": 0.9,          # EMA decay (higher = more past influence)
            "learnable_alpha": False,     # Fixed alpha for now
            "memory_aggregation": "mean", # mean/max/last
            "memory_injection": "additive", # additive/gated/concat
        }

    def train_model_simple(self, model: MemoryTransformer, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, num_epochs: int,
                       eval_freq: int, eval_iter: int, start_context: str, tokenizer: tiktoken.Tokenizer) -> tuple[list[float], list[float], list[int]]:
        """Train the model.

        Args:
            model: The model to train.
            train_loader: The train loader.
            val_loader: The val loader.
            optimizer: The optimizer.
            device: The device.
            num_epochs: The number of epochs.
            eval_freq: The frequency of evaluation.
            eval_iter: The number of iterations to evaluate.
            start_context: The start context.
            tokenizer: The tokenizer.

        Returns:
            The train losses, val losses, and track tokens seen.
        """
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1


        for epoch in range(num_epochs):
            model.train()
            for input_batch, target_batch in train_loader:
                optimizer.zero_grad()
                loss, _ = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()
                optimizer.step()
                tokens_seen += input_batch.numel()
                global_step += 1

                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            generate_and_print_sample(
                model, tokenizer, device, start_context
            )

        return train_losses, val_losses, track_tokens_seen
    def run(self) -> None:
        """Run the Memory Transformer Workflow."""
        torch.manual_seed(123)
        model = MemoryTransformer(self.config)
        model.eval()
        tokenizer = tiktoken.get_encoding("gpt2")

        """
        token_ids = generate_text_simple(
            model=model,
            idx=text_to_token_ids(start_context, tokenizer),
            max_new_tokens=10,
            context_size=self.config["context_length"]
        )

        print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
        """

        print("Loading data...")
        with open(self.config["data_path"], encoding="utf-8") as file:
            text_data = file.read()
        total_characters = len(text_data)
        total_tokens = len(tokenizer.encode(text_data))

        print("Characters:", total_characters)
        print("Tokens:", total_tokens)

        train_ratio = 0.90
        split_idx = int(train_ratio * len(text_data))
        train_data = text_data[:split_idx]
        val_data = text_data[split_idx:]


        torch.manual_seed(123)
        print("Creating data loaders...")
        train_loader = create_dataloader_v1(
            train_data,
            batch_size=2,
            max_length=self.config["context_length"],
            stride=self.config["context_length"],
            drop_last=True,
            shuffle=True,
            num_workers=0
        )

        val_loader = create_dataloader_v1(
            val_data,
            batch_size=2,
            max_length=self.config["context_length"],
            stride=self.config["context_length"],
            drop_last=False,
            shuffle=False,
            num_workers=0
        )
        print("Data loaders created.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        torch.manual_seed(123) # For reproducibility due to the shuffling in the data loader

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

        num_epochs = 10
        train_losses, val_losses, tokens_seen = self.train_model_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=num_epochs, eval_freq=5, eval_iter=5,
            start_context="Every effort moves you", tokenizer=tokenizer
        )



