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
    # 如果不指定 filename，脚本会自动在 data 目录下随便找一个
    parser.add_argument("--filename", type=str, default=None, help="Specific .npy filename to visualize") 
    parser.add_argument("--time-steps", type=int, nargs="+", default=[10, 50, 90], help="Time steps to plot")
    args = parser.parse_args()

    # 1. 基础设置
    cfg = load_config(args.config)
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"🚀 Visualization running on {device}")

    # 2. 初始化 Visualizer
    # 对应你的4个通道名称
    viz = Visualizer(
        output_dir=args.output_dir, 
        channel_names=["Temperature", "Austenite", "Ferrite_Pearlite", "Martensite"]
    )

    # 3. 加载 Mask (如果有)
    mask = None
    mask_path = os.path.join("data/processed/npy_data", "geometry_mask.npy")
    if os.path.exists(mask_path):
        print("🎭 Loading mask...")
        mask = torch.from_numpy(np.load(mask_path)).float().to(device)
        viz.plot_mask(mask) # 画 Mask

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

    # 加载权重 (带 _metadata 修复)
    if os.path.exists(args.checkpoint):
        print(f"🔄 Loading weights: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if isinstance(state, dict):
            if "model_state_dict" in state: state = state["model_state_dict"]
            if "_metadata" in state: del state["_metadata"]
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
        # 自动找一个名为 sim_... 的文件
        files = glob.glob(os.path.join(data_dir, "sim_*.npy"))
        if not files:
            print("❌ No data files found!")
            return
        file_path = files[0] # 默认取第一个
        print(f"🔍 Auto-selected file: {os.path.basename(file_path)}")

    # 6. 加载整个时间序列数据
    # Shape: [Time, Channels, H, W]
    try:
        full_data = np.load(file_path)
        print(f"📦 Loaded data shape: {full_data.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 文件名前缀 (用于保存图片)
    prefix = os.path.basename(file_path).replace(".npy", "")

    # 7. 循环指定的时间步进行绘图
    for t in args.time_steps:
        if t >= full_data.shape[0]:
            print(f"⚠️ Time step {t} out of range (Max {full_data.shape[0]-1}), skipping.")
            continue
            
        print(f"\n🎨 Processing Time Step: {t} ...")
        
        # 准备输入 (t) 和 目标 (t+1)
        # 注意：这里我们做 "t -> t+1" 的预测演示
        # Input: full_data[t]
        # Target: full_data[t+1] (如果越界就取 t)
        
        input_np = full_data[t]
        target_np = full_data[t+1] if t < full_data.shape[0]-1 else full_data[t]
        
        # 转 Tensor 并加 Batch 维度 [1, C, H, W]
        # 强制只取前4个通道 (适配你的新模型)
        x_in = torch.from_numpy(input_np[:4]).float().unsqueeze(0).to(device)
        y_target = torch.from_numpy(target_np[:4]).float().to(device) # 这里不需要 batch 维度给 visualizer, 后面处理
        
        with torch.no_grad():
            out = model(x_in) # Model output: [1, 4, H, W] (Internal Softmax Applied)
            pred = out.squeeze(0) # 去掉 Batch -> [4, H, W]

        # --- 调用 Visualizer 画图 ---
        
        # A. 画 Comparison (最重要)
        viz.plot_comparison(
            target=y_target, 
            pred=pred, 
            mask=mask, 
            time_step=t+1,  # 因为是预测 t+1
            filename_prefix=prefix
        )
        
        # B. 画 Single State (单独展示真值，用于写 Paper)
        # 画奥氏体 (Austenite, Channel 1) 和 马氏体 (Martensite, Channel 3)
        viz.plot_single_state(y_target, channel_idx=1, time_step=t+1, filename_prefix=prefix)
        viz.plot_single_state(y_target, channel_idx=3, time_step=t+1, filename_prefix=prefix)

    print(f"\n✅ All visualizations saved to: {args.output_dir}")

if __name__ == "__main__":
    main()