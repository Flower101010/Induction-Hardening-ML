# Induction-Hardening-ML

An open-source pipeline for modeling induction hardening of shaft-like components using Fourier Neural Operators (FNO) and a Parallel U-FNO architecture. The repository includes data preparation, training, evaluation, and visualization utilities.

## Overview

- Predict temperature and phase evolution fields for induction hardening using neural operators.
- Includes data preprocessing from COMSOL exports, masking, normalization, and dataset splits.
- Provides training scripts, evaluation utilities, and publication-ready plotting helpers.

## Key Features

- Parallel U-FNO model that fuses global (FNO) and local (U-Net style) features.
- Sobel gradient loss to preserve sharp phase boundaries.
- Reusable CLI entrypoint [main.py](main.py) that orchestrates data prep, training, evaluation, and visualization.
- Helper scripts for dataset inspection, dummy data generation, and geometry mask plotting.

## Setup

1. Install [uv](https://github.com/astral-sh/uv) for dependency management.
2. Clone the repository and sync the environment:

```bash
git clone https://github.com/Flower101010/Induction-Hardening-ML.git
cd Induction-Hardening-ML
uv sync
```

## Data

- Raw data: Download `dataset.zip` from [Quark Drive](https://pan.quark.cn/s/fb41d8e629da), extract it, and place `dataset.csv` under `data/raw/`.
- Preprocessed outputs (.npy tensors, masks, stats) are written to `data/processed/npy_data/`.
- Geometry mask: generated as `geometry_mask.npy` during preprocessing.

## CLI Usage

The unified entrypoint is [main.py](main.py). Common commands:

```bash
# Run full data preparation pipeline
uv run main.py prepare-data

# Run a specific data step (analyze | process | preprocess | split)
uv run main.py prepare-data --step preprocess

# Train
uv run main.py train --config config/model_config.yaml

# Evaluate
uv run main.py evaluate --config config/model_config.yaml --checkpoint outputs/models_weights/best_model.pth

# Visualize fields or comparisons
uv run main.py visualize --data data/processed/npy_data/sim_f100000_i1.15.npy --mode gif

# Plot training curves
uv run main.py plot-loss --log outputs/logs/loss_history.json --out outputs/figures/paper_v2

# Generate paper figures
uv run main.py plot-paper --checkpoint outputs/models_weights/best_model.pth --output_dir outputs/figures/paper_v2

# Plot geometry mask
uv run main.py plot-mask

# Inspect dataset split distribution
uv run main.py plot-split

# Run the minimal FNO demo
uv run main.py demo

```

## Project Structure

```text
Induction-Hardening-ML/
├── config/                  # YAML configs
├── data/                    # Raw and processed data
├── docs/                    # Model guide and references
├── scripts/                 # Executable helpers (data prep, plots, demos)
├── src/                     # Core library code (data, engine, models, utils)
├── outputs/                 # Logs, figures, checkpoints (generated)
├── main.py                  # CLI entrypoint
└── README.md
```

## Contributing

Issues and pull requests are welcome. Please keep changes reproducible, add brief documentation for new commands, and avoid committing large raw datasets or model weights.
