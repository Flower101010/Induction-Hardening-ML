import torch
import torch.nn as nn
from neuralop.models import FNO

class ParallelUFNO(nn.Module):
    """
    基于 FNO 的并行预测模型
    
    [修改记录]: 
    - 移除了 forward 内部的 Softmax。
    - 现在直接输出 Raw Logits，由 Loss 函数处理归一化，
    - 推理时需在外部手动 apply softmax。
    """
    def __init__(self, n_modes, hidden_channels, in_channels, out_channels, n_layers, encoder_name="linear", encoder_weights=None):
        super().__init__()
        
        # FNO 核心层
        self.fno = FNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [Batch, In_Channels, H, W]
        Returns:
            out: Raw Logits of shape [Batch, Out_Channels, H, W]
        """
        # FNO 直接输出未归一化的 Logits
        x = self.fno(x)
        return x