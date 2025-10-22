# Memory Transformer

A transformer designed to integrate EMA memory. Has the option to include EMA memory and not include EMA memory for experimentation. 

## Installation

```bash
# create and sync your uv environment
uv sync
```
## Architecture

```
mform/
├── __init__.py           # Package initialization
├── cli.py                # Main entry point
├── configs/              # Utility functions
│   └── cli_config.py     # Workflow starter
├── data/                 # Data operations
│   └──  dataset.py       # Dataset Preperation
├── model/                # Parts of the transformer
│   ├── attention.py      # Attention of transformer
│   ├── ema_memory.py     # EMA memory integration
│   ├── feed_forward.py   # Feed forward for transformer
│   ├── memory_transformer.py  # Transformer manager
│   ├── normalization.py  # normalization
│   └── transformer.py    # Transformer Configuration
├── workflow/             # Workflow managment
│   └──  workflow.py       # Workflow Manager


```

## Command Line Usage
```bash
# run full mform training pipeline
uv run mform mem-transform
```

## Results