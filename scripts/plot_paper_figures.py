"""
Script to generate publication-quality figures for Induction Hardening ML model.
UPDATED VERSION:
1. Auto-detects the 'hottest' sample for Profile Plot (avoids boring room-temp plots).
2. Separates Temperature and Phase for Parity/Error plots (avoids statistical mixing).
3. Corrects unit labels.
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
MAX_RADIUS_MM = 5.0


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

            # Post-process
            temp_pred = out[:, 0:1]
            phase_logits = out[:, 1:]
            phase_prob = torch.softmax(phase_logits, dim=1)
            phase_prob = torch.clamp(phase_prob, 0.0, 1.0)
            pred = torch.cat([temp_pred, phase_prob], dim=1)

            # Apply mask
            if mask_tensor is not None:
                mask_expanded = mask_tensor.unsqueeze(0).unsqueeze(0).expand_as(pred)
                pred = pred * mask_expanded
                y = y * mask_expanded

            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def plot_parity_single(y_true, y_pred, title, filename, color, output_dir):
    """Helper function to plot parity for a single physical quantity."""
    mask = y_true != 0  # Simple masking
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]

    # Downsample
    if len(y_true_masked) > 50000:
        indices = np.random.choice(len(y_true_masked), 50000, replace=False)
        y_true_masked = y_true_masked[indices]
        y_pred_masked = y_pred_masked[indices]

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_masked, y_pred_masked, alpha=0.1, s=1, c=color)

    min_val = min(y_true_masked.min(), y_pred_masked.min())
    max_val = max(y_true_masked.max(), y_pred_masked.max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.5, label="Ideal")

    # Calculate R2
    # Simple R2 implementation
    ss_res = np.sum((y_true_masked - y_pred_masked) ** 2)
    ss_tot = np.sum((y_true_masked - np.mean(y_true_masked)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    plt.xlabel("Ground Truth (COMSOL)")
    plt.ylabel("ML Prediction")
    plt.title(f"{title}\n$R^2 = {r2:.4f}$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


def plot_parity(preds, targets, output_dir):
    """Generate Separate Parity Plots for Temp and Phase."""
    print("Generating Parity Plots...")

    # 假设 Channel 0 是温度
    temp_preds = preds[:, 0, :, :].flatten()
    temp_targets = targets[:, 0, :, :].flatten()
    plot_parity_single(
        temp_targets,
        temp_preds,
        "Temperature Parity",
        "parity_temp.png",
        "red",
        output_dir,
    )

    # 假设 Channel 2 是马氏体 (根据之前的 Profile 代码逻辑)
    # 注意：如果你的模型输出通道不同，请在这里修改索引
    if preds.shape[1] > 2:
        mart_preds = preds[:, 2, :, :].flatten()
        mart_targets = targets[:, 2, :, :].flatten()
        plot_parity_single(
            mart_targets,
            mart_preds,
            "Martensite Parity",
            "parity_martensite.png",
            "blue",
            output_dir,
        )


def find_hottest_sample(dataset):
    """Loop through dataset to find the sample with maximum temperature."""
    print("Searching for the hottest sample (highest max temp)...")
    max_temp = -float("inf")
    best_idx = 0

    # Check every 10th sample to speed up (or every 1 if dataset is small)
    stride = 10 if len(dataset) > 1000 else 1

    for i in range(0, len(dataset), stride):
        _, y = dataset[i]
        # Assuming channel 0 is temperature
        curr_max = y[0].max().item()
        if curr_max > max_temp:
            max_temp = curr_max
            best_idx = i

    print(f"Found max temp {max_temp:.2f} at index {best_idx}")
    return best_idx


def plot_profile(model, dataset, output_dir, device="cuda"):
    """Generate Profile Plot for the HOTTEST sample."""
    print("Generating Profile Plot...")

    # [CHANGE] Use the hottest sample instead of a random one
    idx = find_hottest_sample(dataset)
    x, y = dataset[idx]

    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        temp_pred = out[:, 0:1]
        phase_logits = out[:, 1:]
        phase_prob = torch.softmax(phase_logits, dim=1)
        phase_prob = torch.clamp(phase_prob, 0.0, 1.0)
        pred = torch.cat([temp_pred, phase_prob], dim=1)

    pred = pred.squeeze(0).cpu().numpy()
    target = y.squeeze(0).cpu().numpy()

    # Pick middle Z
    z_idx = pred.shape[1] // 2
    r_axis = np.linspace(0, MAX_RADIUS_MM, pred.shape[2])

    temp_pred_profile = pred[0, z_idx, :]
    temp_true_profile = target[0, z_idx, :]
    mart_pred_profile = pred[2, z_idx, :]
    mart_true_profile = target[2, z_idx, :]

    # Denormalize logic (if applicable)
    if dataset.stats:
        t_min = dataset.stats["temp_min"]
        t_max = dataset.stats["temp_max"]
        temp_pred_profile = temp_pred_profile * (t_max - t_min) + t_min
        temp_true_profile = temp_true_profile * (t_max - t_min) + t_min

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = "tab:red"
    ax1.set_xlabel("Radius (mm)")
    # [CHANGE] Unit corrected to C
    ax1.set_ylabel("Temperature (°C)", color=color)
    ax1.plot(
        r_axis, temp_true_profile, color=color, ls="-", lw=2, label="Temp (COMSOL)"
    )
    ax1.plot(r_axis, temp_pred_profile, color=color, ls="--", lw=2, label="Temp (ML)")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Martensite Fraction", color=color)
    ax2.plot(
        r_axis,
        mart_true_profile,
        color=color,
        ls="-",
        lw=2,
        label="Martensite (COMSOL)",
    )
    ax2.plot(
        r_axis, mart_pred_profile, color=color, ls="--", lw=2, label="Martensite (ML)"
    )
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(-0.05, 1.05)

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center left")

    plt.title(f"Profile Plot at z={z_idx} (Sample {idx} - Peak Heating)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "profile_plot_hottest.png"), dpi=300)
    plt.close()


def plot_error_histogram_single(diff, title, filename, color, output_dir):
    """Helper for error histogram."""
    # Downsample
    if len(diff) > 100000:
        diff = np.random.choice(diff, 100000, replace=False)

    plt.figure(figsize=(8, 6))
    plt.hist(diff, bins=100, density=True, alpha=0.6, color=color, label="Error Dist")

    # Fit
    mu, std = stats.norm.fit(diff)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, "k", lw=2, label=f"Fit ($\\mu={mu:.2e}, \\sigma={std:.2e}$)")  # type: ignore

    plt.xlabel("Prediction Error")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


def plot_error_histogram(preds, targets, output_dir):
    """Generate Separate Error Histograms."""
    print("Generating Error Histograms...")

    # Temp Error
    temp_diff = (preds[:, 0, :, :] - targets[:, 0, :, :]).flatten()
    # Remove background zeros from stats if needed
    mask = targets[:, 0, :, :] != 0
    temp_diff = temp_diff[mask.flatten()]
    plot_error_histogram_single(
        temp_diff,
        "Temperature Error Distribution",
        "error_hist_temp.png",
        "green",
        output_dir,
    )

    # Martensite Error
    if preds.shape[1] > 2:
        mart_diff = (preds[:, 2, :, :] - targets[:, 2, :, :]).flatten()
        plot_error_histogram_single(
            mart_diff,
            "Martensite Error Distribution",
            "error_hist_martensite.png",
            "orange",
            output_dir,
        )


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument(
        "--checkpoint", type=str, default="outputs/models_weights/best_model.pth"
    )
    parser.add_argument("--config", type=str, default="config/model_config.yaml")
    parser.add_argument(
        "--output_dir", type=str, default="outputs/figures/paper_v2"
    )  # Changed folder name
    parser.add_argument("--split_file", type=str, default="config/data_split.json")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Dataset
    print("Loading test dataset...")
    dataset = InductionHardeningDataset(
        data_dir=args.data_dir, split="test", split_file=args.split_file
    )

    # 2. Load Model
    model, cfg = load_model(args.checkpoint, args.config, args.device)

    # 3. Inference
    mask_path = os.path.join(args.data_dir, "npy_data", "geometry_mask.npy")
    if not os.path.exists(mask_path):
        mask_path = os.path.join(args.data_dir, "geometry_mask.npy")

    preds, targets = get_test_predictions(
        model, dataset, args.device, limit_batches=50, mask_path=mask_path
    )

    # 4. Plots (Updated)
    plot_parity(preds, targets, args.output_dir)
    plot_profile(model, dataset, args.output_dir, args.device)
    plot_error_histogram(preds, targets, args.output_dir)

    print(f"All figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
