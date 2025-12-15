"""
FNO Demo Script (Synthetic Data)
================================
FNO 演示脚本（合成数据）
================================

This script demonstrates a basic usage of the Fourier Neural Operator (FNO)
此脚本演示了傅里叶神经算子 (FNO) 的基本用法
on a synthetic diffusion dataset. It serves as a proof-of-concept or "smoke test"
在合成扩散数据集上。它作为一个概念验证或“冒烟测试”
to ensure the FNO model and training pipeline are working correctly before
以确保 FNO 模型和训练流程在应用之前正常工作
applying them to the more complex induction hardening data.
应用到更复杂的感应淬火数据之前。
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from neuralop.models import FNO


class DiffusionToy(Dataset):
    """Synthetic 2D diffusion dataset with smooth fields for FNO smoke tests."""

    """用于 FNO 冒烟测试的具有平滑场的合成 2D 扩散数据集。"""

    def __init__(self, n: int = 200, size: int = 64) -> None:
        self.n = n
        self.size = size
        xs = torch.linspace(0.0, 1.0, size)
        ys = torch.linspace(0.0, 1.0, size)
        X, Y = torch.meshgrid(xs, ys, indexing="ij")
        self.coord = torch.stack([X, Y], dim=-1)  # [H, W, 2]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):  # type: ignore[override]
        freq = torch.randint(1, 4, ()).float()
        phase = torch.rand(())
        base = torch.sin(freq * torch.pi * self.coord[..., 0] + phase) * torch.cos(
            freq * torch.pi * self.coord[..., 1] + phase
        )
        T0 = base + 0.05 * torch.randn_like(base)  # noisy initial temperature
        # 噪声初始温度
        T = torch.exp(-0.5 * freq) * base  # simple decay target
        # 简单衰减目标
        x = torch.cat([self.coord, T0.unsqueeze(-1)], dim=-1)  # [H, W, 3]
        y = torch.stack([T, torch.zeros_like(T)], dim=-1)  # [H, W, 2]
        return x.float(), y.float()


def get_model(
    n_modes: tuple[int, int] = (16, 16),
    hidden_channels: int = 32,
    in_channels: int = 3,
    out_channels: int = 2,
) -> FNO:
    return FNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
    )


def train_step(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    optim: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    x, y = batch
    x = x.to(device)  # [B, H, W, 3]
    y = y.to(device)  # [B, H, W, 2]
    x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
    y = y.permute(0, 3, 1, 2)
    # 强制物理范围
    pred = model(x)
    pred_phase = torch.clamp(pred[:, 1:2], 0.0, 1.0)  # enforce physical range
    pred = torch.cat([pred[:, 0:1], pred_phase], dim=1)
    loss = loss_fn(pred, y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    return float(loss.item())


def main() -> None:
    print("🚀 Starting FNO minimal example...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    print("📦 Generating synthetic data (DiffusionToy)...")
    ds = DiffusionToy(n=256, size=64)
    dl = DataLoader(ds, batch_size=8, shuffle=True)

    print("🧠 Initializing FNO model...")
    model = get_model().to(device)

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print("🔄 Starting training loop (3 epochs)...")
    initial_loss = None
    final_loss = None

    for epoch in range(3):
        losses = [train_step(model, batch, optim, loss_fn, device) for batch in dl]
        avg_loss = sum(losses) / len(losses)
        if epoch == 0:
            initial_loss = avg_loss
        final_loss = avg_loss
        print(f"   Epoch {epoch + 1}/3: loss={avg_loss:.4f}")

    print("\n✅ Training finished!")
    if initial_loss is not None and final_loss is not None:
        print(f"   Initial Loss: {initial_loss:.4f}")
        print(f"   Final Loss:   {final_loss:.4f}")

        if final_loss < initial_loss:
            print("🎉 Success: Model is learning (loss decreased).")
        else:
            print("⚠️ Warning: Loss did not decrease. Check hyperparameters or data.")


if __name__ == "__main__":
    main()
