import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import sys
import os
import numpy as np

# Add project root to path
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
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device(
        config["training"]["device"] if torch.cuda.is_available() else "cpu"
    )
    config["training"]["device"] = device

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
    model = ParallelUFNO(
        n_modes=tuple(config["model"]["n_modes"]),
        hidden_channels=config["model"]["hidden_channels"],
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        n_layers=config["model"]["n_layers"],
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=config["model"]["encoder_weights"],
    )

    # Load Checkpoint if specified
    checkpoint_path = config["training"].get("load_from_checkpoint")
    if checkpoint_path:
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}...")
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            print("Checkpoint loaded successfully.")
        else:
            print(
                f"Warning: Checkpoint file not found at {checkpoint_path}. Starting from scratch."
            )

    # Loss
    criterion = CombinedLoss(
        alpha=config["loss"]["alpha"],
        beta=config["loss"]["beta"],
        gamma=config["loss"]["gamma"],
    )

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["training"]["scheduler"]["step_size"],
        gamma=config["training"]["scheduler"]["gamma"],
    )

    # Trainer
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
    trainer.run()


if __name__ == "__main__":
    main()
