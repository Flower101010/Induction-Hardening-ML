import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import sys
import os
import numpy as np

# Add project root to path
# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import ParallelUFNO
from src.data.dataset import InductionHardeningDataset
from src.utils.losses import CombinedLoss
from src.engine.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Induction Hardening Model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    # Load Config
    # 加载配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Device
    # 设备
    device = torch.device(
        config["training"]["device"] if torch.cuda.is_available() else "cpu"
    )
    config["training"]["device"] = (
        device  # Update config with device object or string? Trainer expects string or object?
    )
    # Trainer expects string in config['training']['device'] but converts to object?
    # Let's check Trainer: self.device = config['training']['device'] -> self.model.to(self.device)
    # So it can be string or object.

    # Datasets
    # 数据集
    # Use "npy_data" subdirectory if that's where data is
    data_dir = "data/processed/npy_data"
    train_dataset = InductionHardeningDataset(data_dir=data_dir, split="train")
    val_dataset = InductionHardeningDataset(data_dir=data_dir, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Load Geometry Mask
    # 加载几何掩码
    mask_path = os.path.join(data_dir, "geometry_mask.npy")
    if os.path.exists(mask_path):
        mask = torch.from_numpy(np.load(mask_path)).float()
        print(f"Loaded geometry mask from {mask_path}")
    else:
        print(
            f"Warning: Geometry mask not found at {mask_path}. Training without mask."
        )
        mask = None

    # Model
    # 模型
    model = ParallelUFNO(
        n_modes=tuple(config["model"]["n_modes"]),
        hidden_channels=config["model"]["hidden_channels"],
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        n_layers=config["model"]["n_layers"],
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=config["model"]["encoder_weights"],
    )

    # Loss
    # 损失函数
    criterion = CombinedLoss(
        alpha=config["loss"]["alpha"],
        beta=config["loss"]["beta"],
        gamma=config["loss"]["gamma"],
    )

    # Optimizer
    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Scheduler
    # 调度器
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["training"]["scheduler"]["step_size"],
        gamma=config["training"]["scheduler"]["gamma"],
    )

    # Trainer
    # 训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        mask=mask,
    )

    # Run
    # 运行
    trainer.run()


if __name__ == "__main__":
    main()
