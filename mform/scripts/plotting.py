"""Plotting Script for Memory Transformer."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_losses(epochs_seen: list[int], tokens_seen: list[int], train_losses: list[float], val_losses: list[float], save_path: str = "loss-plot.pdf") -> None:
    """Plot the losses.

    Args:
        epochs_seen: The epochs seen.
        tokens_seen: The tokens seen.
        train_losses: The training losses.
        val_losses: The validation losses.
        save_path: The path to save the plot.
    """
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.savefig(save_path)
    plt.show()
