"""
Training Script
===============

This is the main training loop for the Induction Hardening FNO model.
It initializes the dataset, dataloaders, model, optimizer, and loss functions based on the configuration.
It manages the training process, logging to TensorBoard/WandB, and saving model checkpoints.

Usage:
    uv run scripts/train.py --config configs/train_config.yaml
"""
