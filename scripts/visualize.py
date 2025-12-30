#!/usr/bin/env python
"""
Visualization script for Induction Hardening simulation results.
Refactored to use src.utils.plotting.Visualizer.
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.plotting import Visualizer

try:
    from src.models import ParallelUFNO
    from src.data.dataset import InductionHardeningDataset

    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False
    print("Warning: Model dependencies not found. Comparison mode will be disabled.")


def load_and_process_data(filepath, mask_path=None):
    """
    Load simulation data and apply necessary preprocessing (swapping, flipping).
    """
    print(f"Loading data from {filepath}...")
    data = np.load(filepath)

    # Flip data along Z-axis (axis 2) to fix orientation
    print("Flipping data along Z-axis (up/down)...")
    data = np.flip(data, axis=2).copy()

    mask = None
    if mask_path and os.path.exists(mask_path):
        print(f"Loading geometry mask from {mask_path}")
        mask = np.load(mask_path)
        # Flip mask along Z-axis (axis 0) to match data
        mask = np.flip(mask, axis=0).copy()

    return data, mask


def run_inference(checkpoint_path, config_path, data_path, stats_path):
    """
    Run inference on the full sequence using the model.
    """
    if not HAS_MODEL:
        raise ImportError("Model dependencies not available")

    # Load Config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Initialize Model
    model = ParallelUFNO(  # type: ignore
        n_modes=tuple(cfg["model"]["n_modes"]),
        hidden_channels=cfg["model"]["hidden_channels"],
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        n_layers=cfg["model"]["n_layers"],
        encoder_name=cfg["model"]["encoder_name"],
        encoder_weights=cfg["model"]["encoder_weights"],
    )

    # Load Checkpoint
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        if "_metadata" in state_dict:
            del state_dict["_metadata"]
    model.load_state_dict(state_dict)
    model.eval()

    # Prepare Dataset
    # Create a temporary split file to load just this file
    filename = os.path.basename(data_path)
    data_dir = os.path.dirname(data_path)

    # Handle if data_path is inside npy_data or not
    if os.path.basename(data_dir) == "npy_data":
        root_dir = os.path.dirname(data_dir)
    else:
        root_dir = data_dir

    # Create temp split file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump({"temp": [filename]}, tmp)
        temp_split_path = tmp.name

    try:
        # Determine time steps from data
        data_shape = np.load(data_path, mmap_mode="r").shape
        T = data_shape[0]

        dataset = InductionHardeningDataset(  # type: ignore
            data_dir=root_dir,
            split="temp",
            split_file=temp_split_path,
            time_steps=T,
        )

        print("Running inference...")
        preds = []
        gts = []

        with torch.no_grad():
            for t in range(T):
                x, y = dataset[t]
                x = x.unsqueeze(0)  # Add batch dim

                pred = model(x)

                # Apply Softmax to phase channels (channels 1, 2, 3) to convert logits to probabilities
                # Channel 0 is Temperature (Regression), so we leave it as is.
                pred_temp = pred[:, 0:1, :, :]
                pred_phases_logits = pred[:, 1:, :, :]
                pred_phases_probs = torch.softmax(pred_phases_logits, dim=1)
                pred = torch.cat([pred_temp, pred_phases_probs], dim=1)

                preds.append(pred)
                gts.append(y.unsqueeze(0))

        pred_tensor = torch.cat(preds, dim=0)
        gt_tensor = torch.cat(gts, dim=0)

        pred_data = pred_tensor.detach().cpu().numpy()
        gt_data = gt_tensor.detach().cpu().numpy()

        # Flip
        pred_data = np.flip(pred_data, axis=2).copy()
        gt_data = np.flip(gt_data, axis=2).copy()

        return gt_data, pred_data

    finally:
        if os.path.exists(temp_split_path):
            os.remove(temp_split_path)


def main():
    parser = argparse.ArgumentParser(
        description="Visualization for Induction Hardening Simulation"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to simulation data file (.npy)"
    )
    parser.add_argument(
        "--output", type=str, default="outputs/figures", help="Output directory"
    )
    parser.add_argument(
        "--stats",
        type=str,
        default="data/processed/npy_data/normalization_stats.json",
        help="Path to stats",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default="data/processed/npy_data/geometry_mask.npy",
        help="Path to mask",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["gif", "snapshot", "compare", "all"],
        default="all",
        help="Visualization mode",
    )
    parser.add_argument("--fps", type=int, default=10, help="FPS for GIF")
    parser.add_argument(
        "--time_idx",
        type=int,
        default=-1,
        help="Time index for snapshot (default: -1 for auto)",
    )
    parser.add_argument(
        "--snapshot_time", type=float, default=1.0, help="Time in seconds for snapshot"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Total duration of simulation in seconds",
    )

    # Model args for comparison
    parser.add_argument(
        "--checkpoint", type=str, default="outputs/models_weights/best_model.pth"
    )
    parser.add_argument("--config", type=str, default="config/model_config.yaml")
    parser.add_argument(
        "--animate", action="store_true", help="Generate animation for comparison mode"
    )

    args = parser.parse_args()

    # Load Stats
    stats = None
    if os.path.exists(args.stats):
        with open(args.stats, "r") as f:
            stats = json.load(f)
            print("Loaded normalization stats.")

    # Load Data
    data, mask = load_and_process_data(args.data, args.mask)

    # Initialize Visualizer
    viz = Visualizer(args.output, stats=stats, mask=mask)

    # Apply mask to data for visualization
    data_masked = viz._apply_mask(data)

    # Determine time indexargs.duration
    T = data.shape[0]
    if args.time_idx == -1:
        time_idx = int(args.snapshot_time / 10.0 * (T - 1))
        time_idx = max(0, min(time_idx, T - 1))
    else:
        time_idx = args.time_idx

    filename_base = os.path.splitext(os.path.basename(args.data))[0]

    # Actions
    if args.mode in ["gif", "all"]:
        print("\n=== Generating GIFs ===")
        viz.create_animation(
            data_masked,
            0,
            "Temperature Distribution",
            f"{filename_base}_temp.gif",
            args.fps,
            "hot",
        )
        viz.create_animation(
            data_masked,
            1,
            "Austenite Phase",
            f"{filename_base}_austenite.gif",
            args.fps,
            "YlOrRd",
        )
        viz.create_animation(
            data_masked,
            2,
            "Martensite Phase",
            f"{filename_base}_martensite.gif",
            args.fps,
            "Blues",
        )

    if args.mode in ["snapshot", "all"]:
        print("\n=== Generating Snapshots ===")
        viz.plot_snapshot(data_masked, time_idx, f"{filename_base}_t{time_idx}")

    if args.mode in ["compare", "all"]:
        print("\n=== Running Comparison ===")
        if HAS_MODEL:
            gt, pred = run_inference(
                args.checkpoint, args.config, args.data, args.stats
            )
            # Apply mask to predictions
            gt = viz._apply_mask(gt)
            pred = viz._apply_mask(pred)

            if not args.animate:
                viz.plot_comparison(
                    gt, pred, time_idx, f"{filename_base}_compare_t{time_idx}"
                )

            if args.animate:
                print("\n=== Generating Comparison Animation ===")
                viz.create_comparison_animation(
                    gt, pred, f"{filename_base}_compare", args.fps
                )
        else:
            print("Skipping comparison (model not available).")

    print(f"\nAll visualizations saved to: {args.output}")


if __name__ == "__main__":
    main()
