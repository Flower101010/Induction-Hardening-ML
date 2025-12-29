import argparse
import json
import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path for local imports
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import ParallelUFNO
from src.data.dataset import InductionHardeningDataset


def build_model(cfg):
    return ParallelUFNO(
        n_modes=tuple(cfg["model"]["n_modes"]),
        hidden_channels=cfg["model"]["hidden_channels"],
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        n_layers=cfg["model"]["n_layers"],
        encoder_name=cfg["model"]["encoder_name"],
        encoder_weights=cfg["model"]["encoder_weights"],
    )


def compute_stats(pred, target, mask):
    stats = {
        "temp_mse_num": 0.0,
        "temp_mae_num": 0.0,
        "phase_mse_num": 0.0,
        "phase_mae_num": 0.0,
        "temp_den": 0.0,
        "phase_den": 0.0,
        "rel_l2_num": 0.0,
        "rel_l2_den": 0.0,
    }

    b = pred.shape[0]
    mask_bhw = None
    if mask is not None:
        mask_bhw = mask.view(1, 1, mask.shape[0], mask.shape[1]).to(pred.device)

    # Temperature channel
    temp_diff = (pred[:, 0:1] - target[:, 0:1]).abs()
    temp_diff2 = temp_diff**2
    if mask_bhw is not None:
        temp_diff = temp_diff * mask_bhw
        temp_diff2 = temp_diff2 * mask_bhw
        temp_den = mask_bhw.sum() * 1 * b
    else:
        temp_den = temp_diff.numel()

    # Phase channels
    phase_diff = (pred[:, 1:] - target[:, 1:]).abs()
    phase_diff2 = phase_diff**2
    if mask_bhw is not None:
        phase_diff = phase_diff * mask_bhw
        phase_diff2 = phase_diff2 * mask_bhw
        phase_den = mask_bhw.sum() * (pred.shape[1] - 1) * b
    else:
        phase_den = phase_diff.numel()

    # Relative L2 (all channels)
    rel_mask = mask_bhw if mask_bhw is not None else 1.0
    rel_num = (((pred - target) ** 2) * rel_mask).sum()
    rel_den = ((target**2) * rel_mask).sum().clamp_min(1e-8)

    stats["temp_mse_num"] = temp_diff2.sum().item()
    stats["temp_mae_num"] = temp_diff.sum().item()
    stats["temp_den"] = temp_den if isinstance(temp_den, float) else temp_den.item()

    stats["phase_mse_num"] = phase_diff2.sum().item()
    stats["phase_mae_num"] = phase_diff.sum().item()
    stats["phase_den"] = phase_den if isinstance(phase_den, float) else phase_den.item()

    stats["rel_l2_num"] = rel_num.item()
    stats["rel_l2_den"] = rel_den.item()
    return stats


def merge_stats(total, batch_stats):
    for k, v in batch_stats.items():
        total[k] += v


def finalize_stats(total):
    eps = 1e-12
    temp_mse = total["temp_mse_num"] / max(total["temp_den"], eps)
    phase_mse = total["phase_mse_num"] / max(total["phase_den"], eps)
    temp_mae = total["temp_mae_num"] / max(total["temp_den"], eps)
    phase_mae = total["phase_mae_num"] / max(total["phase_den"], eps)
    rel_l2 = (total["rel_l2_num"] / max(total["rel_l2_den"], eps)) ** 0.5
    overall_mse = (total["temp_mse_num"] + total["phase_mse_num"]) / max(
        total["temp_den"] + total["phase_den"], eps
    )
    overall_mae = (total["temp_mae_num"] + total["phase_mae_num"]) / max(
        total["temp_den"] + total["phase_den"], eps
    )

    return {
        "temp_mse": temp_mse,
        "temp_mae": temp_mae,
        "phase_mse": phase_mse,
        "phase_mae": phase_mae,
        "overall_mse": overall_mse,
        "overall_mae": overall_mae,
        "relative_l2": rel_l2,
    }


def evaluate(model, loader, device, mask=None):
    model.eval()
    totals = {
        "temp_mse_num": 0.0,
        "temp_mae_num": 0.0,
        "phase_mse_num": 0.0,
        "phase_mae_num": 0.0,
        "temp_den": 0.0,
        "phase_den": 0.0,
        "rel_l2_num": 0.0,
        "rel_l2_den": 0.0,
    }

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x = x.to(device)
            y = y.to(device)

            out = model(x)

            # Post-process: temp unchanged, phase -> softmax then clamp to [0,1]
            temp_pred = out[:, 0:1]
            phase_logits = out[:, 1:]
            phase_prob = torch.softmax(phase_logits, dim=1)
            phase_prob = torch.clamp(phase_prob, 0.0, 1.0)
            pred_full = torch.cat([temp_pred, phase_prob], dim=1)

            batch_stats = compute_stats(pred_full, y, mask)
            merge_stats(totals, batch_stats)

    return finalize_stats(totals)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--config", type=str, default="config/model_config.yaml")
    parser.add_argument(
        "--checkpoint", type=str, default="outputs/models_weights/best_model.pth"
    )
    parser.add_argument("--data-dir", type=str, default="data/processed/npy_data")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--output", type=str, default="outputs/logs/eval_metrics.json")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        cfg["training"]["device"] if torch.cuda.is_available() else "cpu"
    )

    # Dataset & loader
    dataset = InductionHardeningDataset(
        data_dir=args.data_dir,
        split=args.split,
        split_file="config/data_split.json",
        time_steps=cfg.get("data", {}).get("time_steps", 100),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Mask
    mask = None
    mask_path = os.path.join(dataset.data_dir, "geometry_mask.npy")
    if os.path.exists(mask_path):
        mask = torch.from_numpy(np.load(mask_path)).float().to(device)

    # Model
    model = build_model(cfg)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    # Eval
    metrics = evaluate(model, loader, device, mask)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nEvaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
