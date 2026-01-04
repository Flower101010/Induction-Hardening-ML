import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelFilter(nn.Module):
    """
    用于计算图像梯度的 Sobel 滤波器 (物理约束用)
    """
    def __init__(self):
        super().__init__()
        # 定义 X 方向和 Y 方向的 Sobel 核
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, x):
        # x shape: [B, C, H, W]
        b, c, h, w = x.shape
        # 对每个通道单独计算梯度
        grad_x = F.conv2d(x.reshape(b * c, 1, h, w), self.sobel_x, padding=1).view(b, c, h, w)
        grad_y = F.conv2d(x.reshape(b * c, 1, h, w), self.sobel_y, padding=1).view(b, c, h, w)
        return grad_x, grad_y

class CombinedLoss(nn.Module):
    """
    组合损失函数：
    Loss = alpha * MSE + beta * Sobel_MSE + gamma * Physics
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.0, mask=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sobel = SobelFilter()
        
        # 注册 mask 为 buffer，这样它会自动随模型移动到 GPU，但不会被更新
        if mask is not None:
            # 确保 mask 维度匹配 [1, 1, H, W] 以便广播
            if mask.dim() == 2:
                mask = mask.view(1, 1, mask.shape[0], mask.shape[1])
            self.register_buffer('mask', mask)
        else:
            self.mask = None

    def forward(self, pred, target):
        # 1. 应用 Mask (如果有)
        if self.mask is not None:
            pred = pred * self.mask
            target = target * self.mask

        # 2. 基础 MSE Loss (Data Loss)
        mse_loss = F.mse_loss(pred, target, reduction='mean')

        # 3. 梯度 Loss (Sobel Loss) - 迫使模型学习物理边界变化
        total_loss = self.alpha * mse_loss
        
        if self.beta > 0:
            pred_dx, pred_dy = self.sobel(pred)
            target_dx, target_dy = self.sobel(target)
            
            # 如果有 mask，梯度也要 mask 掉
            if self.mask is not None:
                pred_dx = pred_dx * self.mask
                pred_dy = pred_dy * self.mask
                target_dx = target_dx * self.mask
                target_dy = target_dy * self.mask
                
            grad_loss = F.mse_loss(pred_dx, target_dx) + F.mse_loss(pred_dy, target_dy)
            total_loss += self.beta * grad_loss

        return total_loss