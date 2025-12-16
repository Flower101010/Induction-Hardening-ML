import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from neuralop.models import FNO


class ParallelUFNO(nn.Module):
    """
    并行 U-FNO 架构。
    Combines FNO (Global) and U-Net (Local) in parallel.
    """

    def __init__(
        self,
        n_modes=(16, 16),
        hidden_channels=32,
        in_channels=3,
        out_channels=4,
        n_layers=4,
        encoder_name="resnet18",
        encoder_weights=None,
    ):
        super().__init__()

        # FNO Branch
        # FNO 分支
        self.fno = FNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers,
        )

        # U-Net Branch
        # U-Net 分支
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,
        )

        # Fusion Layer
        # 融合层
        self.fusion = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        # x shape: [Batch, Channels, X, Y]

        # FNO Branch
        # FNO 分支
        out_fno = self.fno(x)  # [B, C_out, H, W]

        # U-Net Branch
        # U-Net 分支
        out_unet = self.unet(x)  # [B, C_out, H, W]

        # Fusion
        # 融合
        # Fusion
        out_cat = torch.cat([out_fno, out_unet], dim=1)
        out = self.fusion(out_cat)

        return out
