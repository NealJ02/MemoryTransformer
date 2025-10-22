"""Workflow for the Memory Transformer."""
from __future__ import annotations

import tiktoken
import torch

from mform.data.dataset import load_tinystories_dataloaders
from mform.model.memory_transformer import MemoryTransformer
from mform.scripts.evaluation import evaluate_model, evaluate_story_completion
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
            "context_length": 256, # Context length for TinyStories
            "emb_dim": 256,        # Reduced from 768 for smaller model
            "n_heads": 8,          # Reduced from 12
            "n_layers": 4,         # Reduced from 12 for faster training
            "drop_rate": 0.3,      # Dropout rate (increased to prevent overfitting)
            "qkv_bias": False,     # Query-key-value bias
            # TinyStories dataset config
            "train_samples": 5000,  # Number of training stories (increased to prevent overfitting)
            "val_samples": 300,     # Number of validation stories
            # Memory config
            "use_memory": True,    # Enable memory
            "memory_alpha": 0.5,   # EMA decay (reduced to prevent feedback loops)
            "learnable_alpha": False,     # Fixed alpha for now
            "memory_aggregation": "last", # Using last token instead of mean
            "memory_injection": "gated", # Gated injection lets model learn when to use memory
            "num_epochs": 20,
            "eval_freq": 100,
            "eval_iter": 5,
            "start_context": "Once upon a time",
            "batch_size": 4,
            "num_stories": 10,
            "learning_rate": 0.0004,
            "weight_decay": 0.3,
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

        # Initialize model
        model = MemoryTransformer(self.config)
        model.eval()
        tokenizer = tiktoken.get_encoding("gpt2")

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model initialized with {total_params:,} parameters")
        print(f"Memory enabled: {self.config['use_memory']}")
        if self.config['use_memory']:
            print(f"  - Alpha: {self.config['memory_alpha']}")
            print(f"  - Aggregation: {self.config['memory_aggregation']}")
            print(f"  - Injection: {self.config['memory_injection']}")

        # Load TinyStories dataset and keep stories for evaluation
        torch.manual_seed(123)
        from datasets import load_dataset
        dataset = load_dataset("roneneldan/TinyStories")
        val_stories = [example['text'] for example in dataset['validation'].select(range(self.config["val_samples"]))]

        train_loader, val_loader = load_tinystories_dataloaders(
            train_samples=self.config["train_samples"],
            val_samples=self.config["val_samples"],
            batch_size=self.config["batch_size"],  # Increased from 2 for faster training
            max_length=self.config["context_length"],
            shuffle=True,
            num_workers=0,
        )
        print("Data loaders created.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        torch.manual_seed(123) # For reproducibility due to the shuffling in the data loader

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"])

        train_losses, val_losses, tokens_seen = self.train_model_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=self.config["num_epochs"], eval_freq=self.config["eval_freq"], eval_iter=self.config["eval_iter"],
            start_context="Once upon a time", tokenizer=tokenizer
        )

        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Final train loss: {train_losses[-1]:.3f}")
        print(f"Final val loss: {val_losses[-1]:.3f}")
        print(f"Total tokens seen: {tokens_seen[-1]:,}")

        # Evaluate story completion
        print("\n" + "="*60)
        print("Evaluating Story Completion...")
        print("="*60)
        completion_metrics = evaluate_story_completion(
            model=model,
            stories=val_stories,
            tokenizer=tokenizer,
            device=device,
            max_context=self.config["context_length"],
            num_stories=self.config["num_stories"],  # Reduced from 20 for faster evaluation
        )
        print(f"Story Completion Accuracy: {completion_metrics['completion_accuracy']*100:.2f}%")
        print(f"Stories Evaluated: {completion_metrics['stories_evaluated']}")

        # Generate a full story completion example
        print("\n" + "="*60)
        print("Example Story Completion:")
        print("="*60)
        if len(val_stories) > 0:
            example_story = val_stories[0]
            tokens = tokenizer.encode(example_story)
            split_idx = len(tokens) // 2
            context_tokens = tokens[:split_idx]

            context_text = tokenizer.decode(context_tokens)
            print(f"CONTEXT (first half):\n{context_text}\n")

            # Generate completion
            context_tensor = torch.tensor([context_tokens]).to(device)
            model.eval()
            with torch.no_grad():
                # Generate 100 tokens
                generated = context_tokens.copy()
                for _ in range(100):
                    if len(generated) > self.config["context_length"]:
                        input_tensor = torch.tensor([generated[-self.config["context_length"]:]]).to(device)
                    else:
                        input_tensor = torch.tensor([generated]).to(device)

                    if hasattr(model, 'use_memory') and model.use_memory:
                        logits, _ = model(input_tensor, return_memory=True)
                    else:
                        logits = model(input_tensor)

                    next_token = torch.argmax(logits[0, -1, :]).item()
                    generated.append(next_token)

                completion_tokens = generated[len(context_tokens):]
                completion_text = tokenizer.decode(completion_tokens)
                print(f"MODEL COMPLETION:\n{completion_text}\n")

                actual_completion = tokenizer.decode(tokens[split_idx:])
                print(f"ACTUAL COMPLETION:\n{actual_completion}")



