import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# --- 导入本地模块 (现在都在根目录，非常清爽) ---
from model import ParallelUFNO
from dataset import InductionHardeningDataset
from losses import CombinedLoss

def train(config_path="config.yaml"):
    # 1. 加载配置 (强制 UTF-8，解决 Windows 下中文注释报错)
    print(f"🔄 Loading config from {config_path}...")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. 设置设备 (GPU/CPU)
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    # 3. 准备数据
    # Windows 下 num_workers 建议设为 0，防止多进程报错
    num_workers = cfg["data"].get("num_workers", 0)
    batch_size = cfg["training"]["batch_size"]

    print("📂 Loading datasets...")
    # 这里假设数据还在原来的位置
    train_dataset = InductionHardeningDataset(data_dir="data/processed/npy_data", split="train")
    val_dataset = InductionHardeningDataset(data_dir="data/processed/npy_data", split="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 4. 加载几何 Mask (如果有)
    # 作用：计算 Loss 时只计算工件区域，忽略背景
    mask = None
    mask_path = os.path.join("data/processed/npy_data", "geometry_mask.npy")
    if os.path.exists(mask_path):
        print("🎭 Loading geometry mask...")
        mask = torch.from_numpy(np.load(mask_path)).float().to(device)

    # 5. 初始化模型
    print("🏗️ Building model...")
    model = ParallelUFNO(
        n_modes=tuple(cfg["model"]["n_modes"]),
        hidden_channels=cfg["model"]["hidden_channels"],
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        n_layers=cfg["model"]["n_layers"],
        encoder_name=cfg["model"]["encoder_name"],
        encoder_weights=cfg["model"]["encoder_weights"]
    ).to(device)

    # 6. 优化器与调度器
    optimizer = optim.Adam(
        model.parameters(), 
        lr=cfg["training"]["learning_rate"], 
        weight_decay=cfg["training"]["weight_decay"]
    )
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=cfg["training"]["scheduler"]["step_size"], 
        gamma=cfg["training"]["scheduler"]["gamma"]
    )

    # 初始化损失函数 (合并了 Alpha, Beta, Gamma 权重)
    criterion = CombinedLoss(
        alpha=cfg["loss"]["alpha"],
        beta=cfg["loss"]["beta"],
        gamma=cfg["loss"].get("gamma", 0.0),
        mask=mask
    ).to(device)

    # 7. 训练主循环
    epochs = cfg["training"]["epochs"]
    best_val_loss = float("inf")
    
    # 确保存档目录存在
    save_dir = "outputs/models_weights"
    os.makedirs(save_dir, exist_ok=True)

    print("🔥 Starting training...")
    for epoch in range(1, epochs + 1):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        # 使用 tqdm 显示进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            # 实时更新进度条上的 loss
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val ]"):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # --- 记录与保存 ---
        scheduler.step()
        print(f"📊 Epoch {epoch} Summary: Train Loss = {avg_train_loss:.6f} | Val Loss = {avg_val_loss:.6f}")

        # 策略 A: 每一轮都保存为 'last_model.pth' (用于断点续训)
        torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pth"))

        # 策略 B: 只有验证集 Loss 创新低时，才保存 'best_model.pth' (用于评估)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"🌟 New best model saved! (Loss: {best_val_loss:.6f})")
            
    print("✅ Training complete.")

if __name__ == "__main__":
    train()