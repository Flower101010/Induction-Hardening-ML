import argparse
import yaml
import torch
import os
import numpy as np
import glob

# 导入你的模块
from model import ParallelUFNO
from visualizer import Visualizer

def load_config(config_path):
    with open(config_path, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Generate Paper-Ready Visualizations")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/models_weights/best_model.pth")
    parser.add_argument("--output-dir", type=str, default="outputs/figures")
    parser.add_argument("--filename", type=str, default=None, help="Specific .npy filename to visualize") 
    parser.add_argument("--time-steps", type=int, nargs="+", default=[10, 50, 90], help="Time steps to plot")
    args = parser.parse_args()

    # 1. 基础设置
    cfg = load_config(args.config)
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"🚀 Visualization running on {device}")

    # 2. 初始化 Visualizer
    viz = Visualizer(
        output_dir=args.output_dir, 
        channel_names=["Temperature", "Austenite", "Ferrite_Pearlite", "Martensite"]
    )

    # 3. 加载 Mask
    mask = None
    mask_path = os.path.join("data/processed/npy_data", "geometry_mask.npy")
    if os.path.exists(mask_path):
        print("🎭 Loading mask...")
        mask = torch.from_numpy(np.load(mask_path)).float().to(device)
        viz.plot_mask(mask)

    # 4. 加载模型
    print("🏗️  Building model...")
    model = ParallelUFNO(
        n_modes=tuple(cfg["model"]["n_modes"]),
        hidden_channels=cfg["model"]["hidden_channels"],
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        n_layers=cfg["model"]["n_layers"],
        encoder_name=cfg["model"]["encoder_name"],
        encoder_weights=None
    ).to(device)

    # 加载权重
    if os.path.exists(args.checkpoint):
        print(f"🔄 Loading weights: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if isinstance(state, dict):
            if "model_state_dict" in state: state = state["model_state_dict"]
            if "_metadata" in state: del state["_metadata"]
        
        try:
            model.load_state_dict(state)
        except Exception as e:
            print(f"⚠️ Strict loading failed, trying strict=False: {e}")
            model.load_state_dict(state, strict=False)
        
        model.eval()
    else:
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return

    # 5. 确定要画哪个文件
    data_dir = "data/processed/npy_data"
    if args.filename:
        file_path = os.path.join(data_dir, args.filename)
    else:
        files = glob.glob(os.path.join(data_dir, "sim_*.npy"))
        if not files:
            print("❌ No data files found!")
            return
        file_path = files[0]
        print(f"🔍 Auto-selected file: {os.path.basename(file_path)}")

    # 6. 加载数据
    try:
        full_data = np.load(file_path)
        print(f"📦 Loaded data shape: {full_data.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    prefix = os.path.basename(file_path).replace(".npy", "")

    # 7. 循环绘图
    for t in args.time_steps:
        if t >= full_data.shape[0]:
            continue
            
        print(f"\n🎨 Processing Time Step: {t} ...")
        
        input_np = full_data[t]
        target_np = full_data[t+1] if t < full_data.shape[0]-1 else full_data[t]
        
        # 准备数据 [1, C, H, W]
        x_in = torch.from_numpy(input_np[:4]).float().unsqueeze(0).to(device)
        y_target = torch.from_numpy(target_np[:4]).float().to(device)
        
        with torch.no_grad():
            # 获取模型输出 (Raw Logits)
            logits = model(x_in)
            
            # ⚠️ 关键修正：手动处理 Logits
            temp_pred = logits[:, 0:1]
            phase_probs = torch.softmax(logits[:, 1:], dim=1) # Softmax
            
            pred_full = torch.cat([temp_pred, phase_probs], dim=1)
            pred = pred_full.squeeze(0) # [4, H, W]

        # 画图
        viz.plot_comparison(
            target=y_target, 
            pred=pred, 
            mask=mask, 
            time_step=t+1,
            filename_prefix=prefix
        )
        
        # 画单通道
        viz.plot_single_state(y_target, channel_idx=1, time_step=t+1, filename_prefix=prefix)
        viz.plot_single_state(y_target, channel_idx=3, time_step=t+1, filename_prefix=prefix)

    print(f"\n✅ All visualizations saved to: {args.output_dir}")

if __name__ == "__main__":
    main()