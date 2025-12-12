"""
Fourier Neural Operator (FNO) Model Module
==========================================

This module implements the FNO architecture for 2D temperature and phase field prediction.
It typically includes the SpectralConv2d layers and the overall FNO2d model definition.

傅里叶神经算子 (FNO) 模型模块
==========================
本模块实现了用于二维温度和相场预测的 FNO 架构。 它通常包括 SpectralConv2d 层和整体 FNO2d 模型定义
"""

import torch.nn as nn
from neuralop.models import FNO


class InductionFNO(nn.Module):
    """
    Wrapper around neuralop.models.FNO for Induction Hardening tasks.

    Args:
        n_modes (tuple): Number of Fourier modes to keep in (x, y) directions.
        hidden_channels (int): Width of the hidden layers.
        in_channels (int): Number of input channels (e.g., x, y, T_initial).
        out_channels (int): Number of output channels (e.g., T_final, Phase).
        n_layers (int): Number of Fourier Layers. Default is 4.

    为感应淬火任务包装 neuralop.models.FNO。

    参数:
        n_modes (tuple): 在 (x, y) 方向上保留的傅里叶模数。
        hidden_channels (int): 隐藏层的宽度。
        in_channels (int): 输入通道数(例如,x,y,T_initial)。
        out_channels (int): 输出通道数(例如,T_final,Phase)。
        n_layers (int): 傅里叶层数。 默认值为 4。
    """

    def __init__(
        self,
        n_modes: tuple[int, int] = (16, 16),
        hidden_channels: int = 32,
        in_channels: int = 3,
        out_channels: int = 2,
        n_layers: int = 4,
    ):
        super().__init__()
        # Initialize the FNO model from neuraloperator library
        self.model = FNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers,
        )

    def forward(self, x):
        """
        Forward pass handling dimension permutation.

        Args:
            x: Input tensor of shape [Batch, X, Y, In_Channels]

        Returns:
            Output tensor of shape [Batch, X, Y, Out_Channels]

        前向传递处理维度置换。

        参数:
            x: 形状为 [Batch, X, Y, In_Channels] 的输入张量

        返回:
            形状为 [Batch, X, Y, Out_Channels] 的输出张量
        """
        # 1. Permute to [Batch, Channels, X, Y] for FNO library
        # 1. 将形状置换为 [Batch, Channels, X, Y] 以适应 FNO 库
        x = x.permute(0, 3, 1, 2)

        # 2. Forward pass
        # 2. 前向传递
        out = self.model(x)

        # 3. Permute back to [Batch, X, Y, Channels] for consistency
        # 3. 将形状置换回 [Batch, X, Y, Channels] 以保持一致性
        return out.permute(0, 2, 3, 1)
