import torch
import os
from tqdm import tqdm


class Trainer:
    def __init__(
        self, model, train_loader, val_loader, criterion, optimizer, scheduler, config
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

        self.model.to(self.device)

        # Create output directories
        # 创建输出目录
        os.makedirs("outputs/models_weights", exist_ok=True)
        os.makedirs("outputs/logs", exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]")

        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)
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
                loss = self.criterion(output, y)
                running_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

        return running_loss / len(self.val_loader)

    def run(self):
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

            if self.scheduler:
                self.scheduler.step()

            print(
                f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
            )

            # Save checkpoint & Early Stopping
            # 保存检查点和早停
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                counter = 0
                torch.save(
                    self.model.state_dict(), "outputs/models_weights/best_model.pth"
                )
                print("Saved best model.")
            else:
                counter += 1
                print(f"EarlyStopping counter: {counter} out of {patience}")
                if counter >= patience:
                    print("Early stopping triggered.")
                    break

            # Save last model
            # 保存最新模型
            torch.save(self.model.state_dict(), "outputs/models_weights/last_model.pth")
