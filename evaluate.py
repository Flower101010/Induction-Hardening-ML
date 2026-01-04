import argparse
import json
import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ParallelUFNO
from dataset import InductionHardeningDataset

def build_model(cfg):
    return ParallelUFNO(
        n_modes=tuple(cfg["model"]["n_modes"]),
        hidden_channels=cfg["model"]["hidden_channels"],
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        n_layers=cfg["model"]["n_layers"],
        encoder_name=cfg["model"]["encoder_name"],
        encoder_weights=None, 
    )

def compute_stats(pred, target, mask):
    stats = {
        "temp_mse_num": 0.0, "temp_mae_num": 0.0,
        "phase_mse_num": 0.0, "phase_mae_num": 0.0,
        "temp_den": 0.0, "phase_den": 0.0,
        "rel_l2_num": 0.0, "rel_l2_den": 0.0,
    }

    b = pred.shape[0]
    mask_bhw = None
    if mask is not None:
        mask_bhw = mask.view(1, 1, mask.shape[0], mask.shape[1]).to(pred.device)

    # 1. 温度通道 (Channel 0)
    temp_diff = (pred[:, 0:1] - target[:, 0:1]).abs()
    temp_diff2 = temp_diff**2
    if mask_bhw is not None:
        temp_diff = temp_diff * mask_bhw
        temp_diff2 = temp_diff2 * mask_bhw
        temp_den = mask_bhw.sum() * 1 * b
    else:
        temp_den = float(temp_diff.numel())

    # 2. 相变通道 (Channel 1, 2, 3...)
    phase_diff = (pred[:, 1:] - target[:, 1:]).abs()
    phase_diff2 = phase_diff**2
    if mask_bhw is not None:
        phase_diff = phase_diff * mask_bhw
        phase_diff2 = phase_diff2 * mask_bhw
        phase_den = mask_bhw.sum() * (pred.shape[1] - 1) * b
    else:
        phase_den = float(phase_diff.numel())

    # 3. 相对 L2 误差
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
    
    overall_mse = (total["temp_mse_num"] + total["phase_mse_num"]) / max(total["temp_den"] + total["phase_den"], eps)
    overall_mae = (total["temp_mae_num"] + total["phase_mae_num"]) / max(total["temp_den"] + total["phase_den"], eps)

    return {
        "temp_mse": temp_mse, "temp_mae": temp_mae,
        "phase_mse": phase_mse, "phase_mae": phase_mae,
        "overall_mse": overall_mse, "overall_mae": overall_mae,
        "relative_l2": rel_l2,
    }

def evaluate(model, loader, device, mask=None, time_steps=100, num_files=0):
    model.eval()
    totals = {
        "temp_mse_num": 0.0, "temp_mae_num": 0.0,
        "phase_mse_num": 0.0, "phase_mae_num": 0.0,
        "temp_den": 0.0, "phase_den": 0.0,
        "rel_l2_num": 0.0, "rel_l2_den": 0.0,
    }

    track_files = (num_files > 0)
    if track_files:
        file_numerators = np.zeros(num_files)
        file_denominators = np.zeros(num_files)
    
    global_idx = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            batch_size = x.shape[0]
            x, y = x.to(device), y.to(device)

            # 1. 模型输出 Logits
            logits = model(x)

            # 2. 拆分通道并手动处理 Softmax
            # Channel 0: Temperature (Regression)
            temp_pred = logits[:, 0:1]
            
            # Channel 1-3: Phase (Probabilities)
            # ⚠️ 这里手动加 Softmax
            phase_logits = logits[:, 1:]
            phase_pred = torch.softmax(phase_logits, dim=1)
            
            # 3. 拼接
            pred_full = torch.cat([temp_pred, phase_pred], dim=1)

            # 4. 计算指标
            batch_stats = compute_stats(pred_full, y, mask)
            merge_stats(totals, batch_stats)

            # 5. 文件级误差追踪
            if track_files:
                diff_sq = (pred_full - y) ** 2
                target_sq = y**2

                if mask is not None:
                    mask_bhw = mask.view(1, 1, mask.shape[0], mask.shape[1])
                    diff_sq = diff_sq * mask_bhw
                    target_sq = target_sq * mask_bhw

                err_sum = diff_sq.sum(dim=[1, 2, 3]).cpu().numpy()
                target_sum = target_sq.sum(dim=[1, 2, 3]).cpu().numpy()

                for i in range(batch_size):
                    sample_idx = global_idx + i
                    file_idx = sample_idx // time_steps
                    if file_idx < num_files:
                        file_numerators[file_idx] += err_sum[i]
                        file_denominators[file_idx] += target_sum[i]

            global_idx += batch_size

    best_idx = -1
    best_error = 1.0
    
    if track_files:
        file_rel_l2 = np.sqrt(file_numerators / (file_denominators + 1e-8))
        best_idx = np.argmin(file_rel_l2)
        best_error = file_rel_l2[best_idx]

    return finalize_stats(totals), best_idx, best_error

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/models_weights/best_model.pth")
    parser.add_argument("--data-dir", type=str, default="data/processed/npy_data")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--output", type=str, default="outputs/logs/eval_metrics.json")
    args = parser.parse_args()

    print(f"🔄 Loading config from {args.config}...")
    with open(args.config, "r", encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    # 准备数据
    dataset = InductionHardeningDataset(
        data_dir=args.data_dir,
        split=args.split,
    )
    
    num_files = 0
    time_steps = 100
    if hasattr(dataset, 'file_list'):
        num_files = len(dataset.file_list)
    if hasattr(dataset, 'time_steps'):
        time_steps = dataset.time_steps
    elif 'data' in cfg and 'time_steps' in cfg['data']:
        time_steps = cfg['data']['time_steps']

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    # 加载 Mask
    mask = None
    mask_path = os.path.join(args.data_dir, "geometry_mask.npy")
    if os.path.exists(mask_path):
        print("🎭 Loading geometry mask...")
        mask = torch.from_numpy(np.load(mask_path)).float().to(device)

    # 构建模型
    print("🏗️ Building model...")
    model = build_model(cfg)
    model.to(device)

    # 加载权重
    print(f"🔄 Loading checkpoint: {args.checkpoint}")
    if os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)

        if isinstance(state, dict):
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            if "_metadata" in state:
                del state["_metadata"]

        try:
            model.load_state_dict(state)
        except RuntimeError as e:
            print(f"❌ Error loading keys (trying loose match): {e}")
            model.load_state_dict(state, strict=False)
    else:
        print(f"❌ Checkpoint file not found: {args.checkpoint}")
        return

    # 开始评估
    metrics, best_idx, best_error = evaluate(
        model,
        loader,
        device,
        mask,
        time_steps=time_steps,
        num_files=num_files,
    )

    # 保存结果
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*40)
    print("📊 Evaluation Metrics")
    print("="*40)
    for k, v in metrics.items():
        print(f"🔹 {k:<15}: {v:.6f}")

    if num_files > 0 and best_idx >= 0:
        print("-" * 40)
        best_filename = dataset.file_list[best_idx] if hasattr(dataset, 'file_list') else f"File_{best_idx}"
        print(f"🏆 Best Performing File : {best_filename}")
        print(f"🏆 Best Relative L2 Error: {best_error:.6f}")
    
    print("="*40)

    # 简单评级
    rel_l2 = metrics["relative_l2"]
    if rel_l2 < 0.02:
        print(f"\n✅ [EXCELLENT] Error ({rel_l2:.2%}) < 2%")
    elif rel_l2 < 0.05:
        print(f"\n✅ [GOOD] Error ({rel_l2:.2%}) < 5%")
    else:
        print(f"\n⚠️ [WARNING] Error ({rel_l2:.2%}) > 5%. Improvement needed.")

if __name__ == "__main__":
    main()