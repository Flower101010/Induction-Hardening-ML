import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from neuralop.models import FNO

class ParallelUFNO(nn.Module):
    """
    并行 U-FNO 架构 (扁平化版)
    结合了全局 FNO 和局部 U-Net 的双分支结构。
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

        # 1. FNO 分支 (捕捉全局频率特征)
        self.fno = FNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers,
        )

        # 2. U-Net 分支 (捕捉局部细节特征)
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,
        )

        # 3. 融合层 (处理双分支的输出)
        self.fusion_shared = nn.Sequential(
            nn.Conv2d(out_channels * 2, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(hidden_channels),
        )

        # 4. 输出头 (分别输出温度和相变)
        self.head_temp = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.head_phase = nn.Conv2d(hidden_channels, out_channels - 1, kernel_size=1)

    def forward(self, x):
        # FNO 前向传播
        out_fno = self.fno(x)

        # U-Net 前向传播
        out_unet = self.unet(x)

        # 拼接两个分支的结果
        out_cat = torch.cat([out_fno, out_unet], dim=1)

        # 融合特征
        features = self.fusion_shared(out_cat)

        # 独立输出
        out_temp = self.head_temp(features)
        
        # ⚠️ 关键修改：对相变通道在模型内部做 Softmax
        # 这样 train.py 计算 Loss 时用的就是归一化后的概率，和 evaluate.py 一致
        out_phase_logits = self.head_phase(features)
        out_phase = torch.softmax(out_phase_logits, dim=1)

        # 拼接最终结果
        out = torch.cat([out_temp, out_phase], dim=1)
        return out