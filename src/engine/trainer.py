import torch
import os
import json
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        config,
        mask=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = config["training"]["device"]
        self.epochs = config["training"]["epochs"]

        self.mask = mask
        if self.mask is not None:
            self.mask = self.mask.to(self.device)

        self.model.to(self.device)

        # Create output directories
        # 创建输出目录
        os.makedirs("outputs/models_weights", exist_ok=True)
        os.makedirs("outputs/logs", exist_ok=True)

        # Initialize loss history
        self.history = {"train_loss": [], "val_loss": []}

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]")

        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y, self.mask)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        return running_loss / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.epochs} [Val]")

        with torch.no_grad():
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss = self.criterion(output, y, self.mask)
                running_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

        return running_loss / len(self.val_loader)

    def run(self):
        self.print_summary()

        best_val_loss = float("inf")

        # Early Stopping parameters
        # 早停参数
        early_stopping_cfg = self.config["training"].get("early_stopping", {})
        patience = early_stopping_cfg.get("patience", 10)
        min_delta = early_stopping_cfg.get("min_delta", 0.0)
        counter = 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            # Record and save history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            with open("outputs/logs/loss_history.json", "w") as f:
                json.dump(self.history, f, indent=2)

            if self.scheduler:
                self.scheduler.step()

            print(
                f"Epoch {epoch:03d}/{self.epochs:03d} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )

            # Save checkpoint & Early Stopping
            # 保存检查点和早停
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                counter = 0
                torch.save(
                    self.model.state_dict(), "outputs/models_weights/best_model.pth"
                )
                print(f" >> Saved best model (Loss: {best_val_loss:.6f})")
            else:
                counter += 1
                print(f" >> EarlyStopping counter: {counter}/{patience}")
                if counter >= patience:
                    print("Early stopping triggered.")
                    break

            # Save last model
            # 保存最新模型
            torch.save(self.model.state_dict(), "outputs/models_weights/last_model.pth")

    def print_summary(self):
        print("\n" + "=" * 60)
        print(f"{'Training Configuration':^60}")
        print("=" * 60)
        print(f"Device       : {self.device}")
        print(f"Model        : {self.config['model']['name']}")
        print(f"Epochs       : {self.epochs}")
        print(f"Batch Size   : {self.config['training']['batch_size']}")
        print(f"Learning Rate: {self.config['training']['learning_rate']}")
        print(f"Train Samples: {len(self.train_loader.dataset)}")
        print(f"Val Samples  : {len(self.val_loader.dataset)}")
        if self.mask is not None:
            print(f"Mask         : Enabled (Shape: {list(self.mask.shape)})")
        else:
            print("Mask         : Disabled")
        print("=" * 60 + "\n")
