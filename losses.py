import torch
import torch.nn as nn

class LpLoss(object):
    """
    Relative L2 Loss (用于评估或作为损失的一部分)
    """
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class CombinedLoss(nn.Module):
    """
    综合损失函数：温度 MSE + 相变 MSE (带 Softmax)
    [修复]: 增加了 Mask 维度的自动适配逻辑
    """
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha # 温度权重
        self.beta = beta   # 相变权重

    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: [Batch, 4, H, W] (Logits)
            target: [Batch, 4, H, W] (Ground Truth)
            mask: [H, W] or [Batch, H, W]
        """
        
        # 1. 拆分通道
        # Channel 0: 温度 (Temperature) -> 直接回归
        temp_pred = pred[:, 0:1]
        temp_target = target[:, 0:1]

        # Channel 1-3: 相变 (Phase) -> Logits 转概率
        phase_logits = pred[:, 1:]
        phase_probs = torch.softmax(phase_logits, dim=1) # Logits -> Probs
        phase_target = target[:, 1:]

        # 2. 应用几何 Mask (自动处理维度)
        if mask is not None:
            # 如果 Mask 是 2D [H, W]，扩展为 [1, 1, H, W]
            if mask.ndim == 2:
                mask_broadcast = mask.unsqueeze(0).unsqueeze(0)
            # 如果 Mask 是 3D [Batch, H, W]，扩展为 [Batch, 1, H, W]
            elif mask.ndim == 3:
                mask_broadcast = mask.unsqueeze(1)
            else:
                mask_broadcast = mask

            # 利用广播机制，自动适配 Batch 和 Channels
            temp_pred = temp_pred * mask_broadcast
            temp_target = temp_target * mask_broadcast
            
            phase_probs = phase_probs * mask_broadcast
            phase_target = phase_target * mask_broadcast

        # 3. 计算基础 MSE 损失
        loss_temp = self.mse(temp_pred, temp_target)
        loss_phase = self.mse(phase_probs, phase_target)

        # 4. 总损失
        total_loss = self.alpha * loss_temp + self.beta * loss_phase
        
        return total_loss