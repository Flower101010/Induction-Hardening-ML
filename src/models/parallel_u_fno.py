import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO


class SimpleUNet(nn.Module):
    """
    A simple U-Net implementation for 2D image-like data.
    用于 2D 图像类数据的简单 U-Net 实现。
    """

    def __init__(self, in_channels, out_channels, hidden_channels=32):
        super().__init__()

        self.inc = DoubleConv(in_channels, hidden_channels)
        self.down1 = Down(hidden_channels, hidden_channels * 2)
        self.down2 = Down(hidden_channels * 2, hidden_channels * 4)
        self.up1 = Up(hidden_channels * 4, hidden_channels * 2)
        self.up2 = Up(hidden_channels * 2, hidden_channels)
        self.outc = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    """(卷积 => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    """使用最大池化进行下采样，然后进行双卷积"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    """上采样然后双卷积"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            )
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ParallelUFNO(nn.Module):
    """
    并行 U-FNO 架构。
    Combines FNO (Global) and U-Net (Local) in parallel.
    并行结合 FNO（全局）和 U-Net（局部）。
    """

    def __init__(
        self,
        n_modes=(16, 16),
        hidden_channels=32,
        in_channels=3,
        out_channels=4,
        n_layers=4,
        unet_hidden_channels=32,
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
        self.unet = SimpleUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=unet_hidden_channels,
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
