"""
Visualization Script
====================
Generates plots and animations for model predictions.
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import ParallelUFNO
from src.data.dataset import InductionHardeningDataset


def load_model(config_path, checkpoint_path, device):
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

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, cfg


def phases_to_rgb(phases_tensor):
    """
    Convert 3-channel phase probabilities to RGB image.
    Mapping:
    - R: Martensite (Channel 2) -> Hard
    - G: Austenite (Channel 1) -> Transforming
    - B: Initial/Ferrite (Channel 0 of phases, or implicit)

    Input: [C=3, H, W] (Aust, Mart, Initial) or similar
    """
    # Assuming output is [Temp, Aust, Mart, Initial] or [Temp, P1, P2, P3]
    # Let's assume prediction output is [Temp, P_Aust, P_Mart, P_Init]
    # We want RGB: R=Mart, G=Aust, B=Init

    # phases_tensor shape: [3, H, W] -> (Aust, Mart, Init)
    aust = phases_tensor[0]
    mart = phases_tensor[1]
    init = phases_tensor[2]

    rgb = np.stack([mart, aust, init], axis=-1)  # [H, W, 3]
    return np.clip(rgb, 0, 1)


def visualize_batch(model, loader, device, output_dir, num_samples=1):
    os.makedirs(output_dir, exist_ok=True)

    iter_loader = iter(loader)
    inputs, targets = next(iter_loader)
    inputs, targets = inputs.to(device), targets.to(device)

    with torch.no_grad():
        preds = model(inputs)

        # Post-process predictions
        pred_temp = preds[:, 0:1]
        pred_phases_logits = preds[:, 1:]
        pred_phases_prob = torch.softmax(pred_phases_logits, dim=1)

        # Targets
        target_temp = targets[:, 0:1]
        target_phases = targets[:, 1:]

    # Convert to numpy
    inputs = inputs.cpu().numpy()
    target_temp = target_temp.cpu().numpy()
    target_phases = target_phases.cpu().numpy()
    pred_temp = pred_temp.cpu().numpy()
    pred_phases_prob = pred_phases_prob.cpu().numpy()

    for i in range(min(num_samples, inputs.shape[0])):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # Row 1: Temperature
        # Input (t)
        im0 = axes[0, 0].imshow(inputs[i, 0], cmap="hot", vmin=0, vmax=1)
        axes[0, 0].set_title("Input Temp (t)")
        plt.colorbar(im0, ax=axes[0, 0])

        # Target (t+1)
        im1 = axes[0, 1].imshow(target_temp[i, 0], cmap="hot", vmin=0, vmax=1)
        axes[0, 1].set_title("Target Temp (t+1)")
        plt.colorbar(im1, ax=axes[0, 1])

        # Pred (t+1)
        im2 = axes[0, 2].imshow(pred_temp[i, 0], cmap="hot", vmin=0, vmax=1)
        axes[0, 2].set_title("Pred Temp (t+1)")
        plt.colorbar(im2, ax=axes[0, 2])

        # Row 2: Phases (RGB)
        # Target Phase
        rgb_target = phases_to_rgb(target_phases[i])
        axes[1, 1].imshow(rgb_target)
        axes[1, 1].set_title("Target Phase (RGB)\nR:Mart, G:Aust, B:Init")

        # Pred Phase
        rgb_pred = phases_to_rgb(pred_phases_prob[i])
        axes[1, 2].imshow(rgb_pred)
        axes[1, 2].set_title("Pred Phase (RGB)")

        # Error Map (Temp)
        err = np.abs(target_temp[i, 0] - pred_temp[i, 0])
        im_err = axes[1, 0].imshow(err, cmap="inferno")
        axes[1, 0].set_title("Temp Error |T - P|")
        plt.colorbar(im_err, ax=axes[1, 0])

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"sample_{i}.png")
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
        plt.close()


def make_animation(model, loader, device, output_dir, frames=50):
    """Creates a GIF from a sequence of predictions."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating animation ({frames} frames)...")

    # Collect frames
    inputs_list = []
    targets_list = []
    preds_list = []

    iter_loader = iter(loader)

    with torch.no_grad():
        for _ in range(frames):
            try:
                x, y = next(iter_loader)
            except StopIteration:
                break

            x = x.to(device)
            y_pred = model(x)

            # Process only the first sample in batch
            inputs_list.append(x[0, 0].cpu().numpy())  # Temp channel

            # Target Phase RGB
            t_ph = y[0, 1:].cpu().numpy()
            targets_list.append(phases_to_rgb(t_ph))

            # Pred Phase RGB
            p_logits = y_pred[0, 1:]
            p_prob = torch.softmax(p_logits, dim=0)
            preds_list.append(phases_to_rgb(p_prob.cpu().numpy()))

    # Setup Animation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im_in = axes[0].imshow(inputs_list[0], cmap="hot", vmin=0, vmax=1)
    axes[0].set_title("Input Temp")

    im_tgt = axes[1].imshow(targets_list[0])
    axes[1].set_title("Target Phase")

    im_pred = axes[2].imshow(preds_list[0])
    axes[2].set_title("Pred Phase")

    def update(frame):
        im_in.set_data(inputs_list[frame])
        im_tgt.set_data(targets_list[frame])
        im_pred.set_data(preds_list[frame])
        return im_in, im_tgt, im_pred

    ani = animation.FuncAnimation(fig, update, frames=len(inputs_list), blit=True)

    save_path = os.path.join(output_dir, "prediction_video.gif")
    ani.save(save_path, writer="pillow", fps=10)
    print(f"Animation saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument("--checkpoint", default="outputs/models_weights/best_model.pth")
    parser.add_argument("--data-dir", default="data/processed/npy_data")
    parser.add_argument("--out-dir", default="outputs/figures")
    parser.add_argument("--mode", choices=["plot", "animate"], default="plot")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model, cfg = load_model(args.config, args.checkpoint, device)

    # Load Data (Test set)
    dataset = InductionHardeningDataset(
        data_dir=args.data_dir,
        split="test",
        split_file="config/data_split.json",
        time_steps=cfg.get("data", {}).get("time_steps", 100),
    )
    # Batch size 1 for animation sequence logic (if dataset is sequential)
    # Or larger for plotting random samples
    loader = DataLoader(
        dataset, batch_size=4 if args.mode == "plot" else 1, shuffle=False
    )

    if args.mode == "plot":
        visualize_batch(model, loader, device, args.out_dir)
    elif args.mode == "animate":
        make_animation(model, loader, device, args.out_dir)


if __name__ == "__main__":
    main()
