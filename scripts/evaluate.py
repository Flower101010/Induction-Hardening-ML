"""
Evaluation Script
=================

This script is responsible for evaluating the trained FNO model on the test dataset.
It loads the trained model weights, processes the test data, and computes performance metrics
(e.g., MSE, MAE, relative error) for temperature and phase transformation predictions.

Usage:
    uv run scripts/evaluate.py --config configs/model_config.yaml --checkpoint outputs/models_weights/best_model.pth
"""
