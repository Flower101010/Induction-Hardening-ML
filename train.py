import argparse
import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim

# 导入你的模块
from dataset import InductionHardeningDataset
from model import ParallelUFNO
from losses import CombinedLoss  # 确保 losses.py 已经是最新版

def load_config(config_path):
    with open(config_path, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)

def train():
    # 1. 基础设置
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    print(f"🔄 Loading config from {args.config}...")
    cfg = load_config(args.config)
    
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    # 2. 准备数据
    print("📂 Loading datasets...")
    # 自动获取 time_steps
    time_steps = cfg["data"].get("time_steps", 100)
    
    train_dataset = InductionHardeningDataset(
        data_dir="data/processed/npy_data",
        split="train",
        time_steps=time_steps
    )
    val_dataset = InductionHardeningDataset(
        data_dir="data/processed/npy_data",
        split="val",
        time_steps=time_steps
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=(device.type == 'cuda')
    )

    # 3. 加载几何 Mask
    mask = None
    mask_path = os.path.join("data/processed/npy_data", "geometry_mask.npy")
    if os.path.exists(mask_path):
        print("🎭 Loading geometry mask...")
        mask = torch.from_numpy(np.load(mask_path)).float().to(device)
    
    # 4. 构建模型
    print("🏗️ Building model...")
    model = ParallelUFNO(
        n_modes=tuple(cfg["model"]["n_modes"]),
        hidden_channels=cfg["model"]["hidden_channels"],
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        n_layers=cfg["model"]["n_layers"]
    ).to(device)

    # 5. 定义 Loss 和优化器
    # ⚠️ 修复点：CombinedLoss 初始化不再接受 gamma 和 mask
    criterion = CombinedLoss(
        alpha=cfg["loss"]["alpha"],
        beta=cfg["loss"]["beta"]
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"])
    )
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["training"]["scheduler_step_size"],
        gamma=cfg["training"]["scheduler_gamma"]
    )

    # 6. 训练循环
    best_val_loss = float('inf')
    epochs = cfg["training"]["epochs"]
    save_dir = "outputs/models_weights"
    os.makedirs(save_dir, exist_ok=True)

    print("🔥 Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # 训练步
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(x) # 输出 Raw Logits
            
            # ⚠️ 修复点：在计算 Loss 时传入 mask
            loss = criterion(pred, y, mask=mask)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证步
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                # 同样传入 mask
                loss = criterion(pred, y, mask=mask)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"✅ Saved best model (Loss: {best_val_loss:.6f})")

    print("🏁 Training complete!")

if __name__ == "__main__":
    train()