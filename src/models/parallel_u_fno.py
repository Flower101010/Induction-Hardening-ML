import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from neuralop.models import FNO


class ParallelUFNO(nn.Module):
    """
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
        self.fno = FNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers,
        )

        # U-Net Branch
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,
        )

        # Fusion Layer (Dual Head Architecture)

        # 1. Shared Feature Extractor
        self.fusion_shared = nn.Sequential(
            nn.Conv2d(out_channels * 2, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(hidden_channels),
        )

        # 2. Temperature Head
        self.head_temp = nn.Conv2d(hidden_channels, 1, kernel_size=1)

        # 3. Phase Head
        self.head_phase = nn.Conv2d(hidden_channels, out_channels - 1, kernel_size=1)

    def forward(self, x):
        # x shape: [Batch, Channels, X, Y]

        # FNO Branch
        out_fno = self.fno(x)  # [B, C_out, H, W]

        # U-Net Branch
        out_unet = self.unet(x)  # [B, C_out, H, W]

        # Fusion
        out_cat = torch.cat([out_fno, out_unet], dim=1)

        # 1. Extract Shared Features
        features = self.fusion_shared(out_cat)

        # 2. Independent Predictions
        out_temp = self.head_temp(features)
        out_phase = self.head_phase(features)

        # 3. Concatenate for output (to match target shape)
        out = torch.cat([out_temp, out_phase], dim=1)

        return out
