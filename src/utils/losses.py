import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Define Sobel kernels
        # 定义 Sobel 核
        self.sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

    def forward(self, pred, target):
        # pred, target: [Batch, H, W, C] or [Batch, C, H, W]

        if pred.shape[-1] < pred.shape[1]:  # Heuristic check for [B, H, W, C]
            # 启发式检查 [B, H, W, C]
            pred = pred.permute(0, 3, 1, 2)
            target = target.permute(0, 3, 1, 2)

        b, c, h, w = pred.shape

        # Replicate kernels for each channel
        # 为每个通道复制核
        sobel_x = self.sobel_x.to(pred.device).repeat(c, 1, 1, 1)
        sobel_y = self.sobel_y.to(pred.device).repeat(c, 1, 1, 1)

        # Calculate gradients
        # 计算梯度
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1, groups=c)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1, groups=c)
        target_grad_x = F.conv2d(target, sobel_x, padding=1, groups=c)
        target_grad_y = F.conv2d(target, sobel_y, padding=1, groups=c)

        loss_x = F.mse_loss(pred_grad_x, target_grad_x)
        loss_y = F.mse_loss(pred_grad_y, target_grad_y)

        return loss_x + loss_y


class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.sobel = SobelLoss()
        self.alpha = alpha  # Weight for MSE/CE
        # MSE/CE 的权重
        self.beta = beta  # Weight for Sobel
        # Sobel 的权重
        self.gamma = gamma  # Weight for Physics
        # 物理损失的权重

    def forward(self, pred, target):
        # pred: [B, 4, H, W] (Temp, P0, P1, P2)
        # target: [B, 2, H, W] (Temp, P_idx)

        # 1. Split Pred and Target
        # 1. 分割预测和目标
        pred_temp = pred[:, 0:1, :, :]  # [B, 1, H, W]
        pred_phases = pred[:, 1:, :, :]  # [B, 3, H, W]

        target_temp = target[:, 0:1, :, :]  # [B, 1, H, W]
        target_phase_idx = target[:, 1, :, :].long()  # [B, H, W]

        # 2. MSE Loss for Temperature
        # 2. 温度的 MSE 损失
        loss_temp = self.mse(pred_temp, target_temp)

        # 3. Cross Entropy Loss for Phases
        # 3. 相的交叉熵损失
        # CE expects [B, C, H, W] for logits and [B, H, W] for target
        # CE 期望 logits 为 [B, C, H, W]，目标为 [B, H, W]
        loss_phase = self.ce(pred_phases, target_phase_idx)

        loss_main = loss_temp + loss_phase

        # 4. Sobel Loss
        # 4. Sobel 损失
        # Apply Sobel on Temp
        # 对温度应用 Sobel
        loss_sobel_temp = self.sobel(pred_temp, target_temp)

        # Apply Sobel on Phase Probabilities
        # 对相概率应用 Sobel
        target_phases_onehot = (
            F.one_hot(target_phase_idx, num_classes=3).float().permute(0, 3, 1, 2)
        )  # [B, 3, H, W]
        pred_phases_prob = F.softmax(pred_phases, dim=1)  # [B, 3, H, W]

        loss_sobel_phase = self.sobel(pred_phases_prob, target_phases_onehot)

        loss_sobel = loss_sobel_temp + loss_sobel_phase

        # 5. Physics Loss (Sum of phases = 1)
        # 5. 物理损失（相之和 = 1）
        # Implicitly satisfied by Softmax interpretation for probabilities.
        # 通过 Softmax 解释概率隐式满足。
        loss_physics = 0.0

        return (
            self.alpha * loss_main + self.beta * loss_sobel + self.gamma * loss_physics
        )
