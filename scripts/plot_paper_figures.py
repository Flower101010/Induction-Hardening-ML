#!/usr/bin/env python
"""
Script to generate publication-quality figures for Induction Hardening ML model.
Generates:
1. Parity Plot (Ground Truth vs Prediction)
2. Profile Plot (Physical field distribution along radius)
3. Error Histogram (Error distribution)
"""

import argparse
import os
import sys
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import scipy.stats as stats

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import ParallelUFNO
from src.data.dataset import InductionHardeningDataset

# Constants
# Estimated max radius in mm based on data analysis (approx 5mm)
MAX_RADIUS_MM = 5.0
GRID_WIDTH = 64  # r dimension


def load_model(checkpoint_path, config_path, device="cuda"):
    """Load the trained model."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model = ParallelUFNO(
        n_modes=tuple(cfg["model"]["n_modes"]),
        hidden_channels=cfg["model"]["hidden_channels"],
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        n_layers=cfg["model"]["n_layers"],
        encoder_name=cfg["model"]["encoder_name"],
        encoder_weights=cfg["model"]["encoder_weights"],
    )

    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        if "_metadata" in state_dict:
            del state_dict["_metadata"]

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, cfg


def get_test_predictions(
    model, dataset, device="cuda", limit_batches=None, mask_path=None
):
    """Run inference on the dataset and collect predictions and targets."""
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load mask if provided
    mask_tensor = None
    if mask_path and os.path.exists(mask_path):
        print(f"Loading mask from {mask_path}")
        mask_np = np.load(mask_path)
        mask_tensor = torch.from_numpy(mask_np).to(device)

    all_preds = []
    all_targets = []

    print("Running inference on test set...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            if limit_batches and i >= limit_batches:
                break

            x, y = batch
            x = x.to(device)
            y = y.to(device)

            out = model(x)

            # Post-process: temp unchanged, phase -> softmax then clamp to [0,1]
            temp_pred = out[:, 0:1]
            phase_logits = out[:, 1:]
            # Check if model output is already probabilities or logits
            # Usually FNO outputs logits for classification/segmentation tasks
            # But here it's regression/multi-task.
            # Evaluate.py uses softmax, so we should too.
            phase_prob = torch.softmax(phase_logits, dim=1)
            phase_prob = torch.clamp(phase_prob, 0.0, 1.0)
            pred = torch.cat([temp_pred, phase_prob], dim=1)

            # Apply mask if available to ignore background
            if mask_tensor is not None:
                # Expand mask for channels and batch
                # Mask shape: (H, W) -> (1, 1, H, W)
                mask_expanded = mask_tensor.unsqueeze(0).unsqueeze(0).expand_as(pred)
                pred = pred * mask_expanded
                y = y * mask_expanded

            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def plot_parity(preds, targets, output_dir):
    """
    Generate Parity Plot (Ground Truth vs Prediction).
    """
    print("Generating Parity Plot...")

    # Flatten arrays
    # Select only non-zero values (assuming 0 is background/masked)
    # Or better, use the mask if we had it. Here we assume exact 0 is masked.
    mask = targets != 0
    y_true = targets[mask]
    y_pred = preds[mask]

    # Downsample for plotting if too many points
    if len(y_true) > 50000:
        indices = np.random.choice(len(y_true), 50000, replace=False)
        y_true = y_true[indices]
        y_pred = y_pred[indices]

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.1, s=1, c="blue")

    # Plot diagonal
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal (y=x)")

    plt.xlabel("COMSOL Ground Truth")
    plt.ylabel("ML Prediction")
    plt.title("Parity Plot: Ground Truth vs Prediction")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Calculate R2
    # r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    # plt.text(min_val, max_val, f'$R^2 = {r2:.4f}$', fontsize=12, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parity_plot.png"), dpi=300)
    plt.close()


def plot_profile(model, dataset, output_dir, device="cuda"):
    """
    Generate Profile Plot (Physical field distribution along radius).
    """
    print("Generating Profile Plot...")

    # Select a sample (e.g., middle of the dataset)
    idx = len(dataset) // 2
    x, y = dataset[idx]

    # Add batch dimension
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        # Post-process
        temp_pred = out[:, 0:1]
        phase_logits = out[:, 1:]
        phase_prob = torch.softmax(phase_logits, dim=1)
        phase_prob = torch.clamp(phase_prob, 0.0, 1.0)
        pred = torch.cat([temp_pred, phase_prob], dim=1)

    # Convert to numpy
    pred = pred.squeeze(0).cpu().numpy()
    target = y.squeeze(0).cpu().numpy()

    # Dimensions: (C, H, W) -> (C, z, r)
    # We want to plot along r (radius) at a specific z (height)
    # Let's pick the middle z
    z_idx = pred.shape[1] // 2

    # Extract profiles
    # Channel 0: Temperature
    # Channel 2: Martensite (usually) - Check dataset/model config
    # Based on visualize.py: 0: Temp, 1: Austenite, 2: Martensite

    r_axis = np.linspace(0, MAX_RADIUS_MM, pred.shape[2])

    temp_pred = pred[0, z_idx, :]
    temp_true = target[0, z_idx, :]

    mart_pred = pred[2, z_idx, :]
    mart_true = target[2, z_idx, :]

    # Denormalize Temperature if stats available
    if dataset.stats:
        t_min = dataset.stats["temp_min"]
        t_max = dataset.stats["temp_max"]
        temp_pred = temp_pred * (t_max - t_min) + t_min
        temp_true = temp_true * (t_max - t_min) + t_min

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Temperature (Left Axis)
    color = "tab:red"
    ax1.set_xlabel("Radius (mm)")
    ax1.set_ylabel("Temperature (K)", color=color)
    (l1,) = ax1.plot(
        r_axis, temp_true, color=color, linestyle="-", label="Temp (COMSOL)"
    )
    (l2,) = ax1.plot(r_axis, temp_pred, color=color, linestyle="--", label="Temp (ML)")
    ax1.tick_params(axis="y", labelcolor=color)

    # Plot Martensite (Right Axis)
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Martensite Fraction", color=color)
    (l3,) = ax2.plot(
        r_axis, mart_true, color=color, linestyle="-", label="Martensite (COMSOL)"
    )
    (l4,) = ax2.plot(
        r_axis, mart_pred, color=color, linestyle="--", label="Martensite (ML)"
    )
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(-0.1, 1.1)

    # Add legend
    lines = [l1, l2, l3, l4]
    labels = [l.get_label() for l in lines]  # noqa: E741
    ax1.legend(lines, labels, loc="center right")  # type: ignore

    plt.title(f"Profile Plot at z={z_idx} (Sample {idx})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "profile_plot.png"), dpi=300)
    plt.close()


def plot_error_histogram(preds, targets, output_dir):
    """
    Generate Error Histogram.
    """
    print("Generating Error Histogram...")

    mask = targets != 0
    diff = preds[mask] - targets[mask]

    # Downsample if needed
    if len(diff) > 100000:
        diff = np.random.choice(diff, 100000, replace=False)

    plt.figure(figsize=(10, 6))

    # Histogram
    n, bins, patches = plt.hist(
        diff, bins=100, density=True, alpha=0.6, color="g", label="Error Dist"
    )

    # Fit Gaussian
    mu, std = stats.norm.fit(diff)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(
        x,
        p,
        "k",
        linewidth=2,
        label=f"Normal Fit ($\mu={mu:.2e}, \sigma={std:.2e}$)",  # type: ignore
    )

    plt.xlabel("Prediction Error")
    plt.ylabel("Density")
    plt.title("Error Distribution Histogram")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_histogram.png"), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument(
        "--data_dir", type=str, default="data/processed", help="Path to processed data"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/models_weights/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_config.yaml",
        help="Path to model config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/figures/paper",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default="config/data_split.json",
        help="Path to split file",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load Dataset
    print("Loading test dataset...")
    dataset = InductionHardeningDataset(
        data_dir=args.data_dir, split="test", split_file=args.split_file
    )

    # Load Model
    model, cfg = load_model(args.checkpoint, args.config, args.device)

    # Run Inference
    mask_path = os.path.join(args.data_dir, "npy_data", "geometry_mask.npy")
    if not os.path.exists(mask_path):
        mask_path = os.path.join(args.data_dir, "geometry_mask.npy")

    preds, targets = get_test_predictions(
        model, dataset, args.device, limit_batches=50, mask_path=mask_path
    )

    # Generate Plots
    plot_parity(preds, targets, args.output_dir)
    plot_profile(model, dataset, args.output_dir, args.device)
    plot_error_histogram(preds, targets, args.output_dir)

    print(f"All figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
