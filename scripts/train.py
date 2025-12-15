import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import sys
import os

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
    train_dataset = InductionHardeningDataset(data_dir="data/processed", split="train")
    # Assuming we use train data for validation for now as dummy data generator only made train data
    # 假设我们暂时使用训练数据进行验证，因为虚拟数据生成器只生成了训练数据
    val_dataset = InductionHardeningDataset(data_dir="data/processed", split="train")

    train_loader = DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"], shuffle=False
    )

    # Model
    # 模型
    model = ParallelUFNO(
        n_modes=tuple(config["model"]["n_modes"]),
        hidden_channels=config["model"]["hidden_channels"],
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        n_layers=config["model"]["n_layers"],
        unet_hidden_channels=config["model"]["unet_hidden_channels"],
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
    )

    # Run
    # 运行
    trainer.run()


if __name__ == "__main__":
    main()
