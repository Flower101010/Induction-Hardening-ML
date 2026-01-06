import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Define Sobel kernels
        self.sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

    def forward(self, pred, target, mask=None):
        # pred, target: [Batch, H, W, C] or [Batch, C, H, W]
        # mask: [H, W] or [Batch, 1, H, W]

        if pred.shape[-1] < pred.shape[1]:  # Heuristic check for [B, H, W, C]
            pred = pred.permute(0, 3, 1, 2)
            target = target.permute(0, 3, 1, 2)

        b, c, h, w = pred.shape

        # Replicate kernels for each channel
        sobel_x = self.sobel_x.to(pred.device).repeat(c, 1, 1, 1)
        sobel_y = self.sobel_y.to(pred.device).repeat(c, 1, 1, 1)

        # Calculate gradients
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1, groups=c)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1, groups=c)
        target_grad_x = F.conv2d(target, sobel_x, padding=1, groups=c)
        target_grad_y = F.conv2d(target, sobel_y, padding=1, groups=c)

        loss_x = (pred_grad_x - target_grad_x) ** 2
        loss_y = (pred_grad_y - target_grad_y) ** 2

        if mask is not None:
            # Ensure mask is broadcastable
            if mask.dim() == 2:
                mask = mask.view(1, 1, mask.shape[0], mask.shape[1])
            loss_x = loss_x * mask
            loss_y = loss_y * mask

        return loss_x.mean() + loss_y.mean()


class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.1):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")  # Use none to apply mask manually
        # self.ce = nn.CrossEntropyLoss() # Not used for soft labels
        self.sobel = SobelLoss()
        self.alpha = alpha  # Weight for MSE
        # MSE 的权重
        self.beta = beta  # Weight for Sobel
        # Sobel 的权重
        self.gamma = gamma  # Weight for Physics (Sum=1 constraint, implicit in Softmax)
        # 物理损失的权重

    def forward(self, pred, target, mask=None):
        # pred: [B, 4, H, W] (Temp, P0, P1, P2 logits)
        # target: [B, 4, H, W] (Temp, P0, P1, P2 fractions)
        # mask: [H, W] or [B, 1, H, W]

        # 1. Split Pred and Target
        pred_temp = pred[:, 0:1, :, :]  # [B, 1, H, W]
        pred_phases_logits = pred[:, 1:, :, :]  # [B, 3, H, W]

        target_temp = target[:, 0:1, :, :]  # [B, 1, H, W]
        target_phases = target[:, 1:, :, :]  # [B, 3, H, W]

        # 2. MSE Loss for Temperature
        loss_temp_map = self.mse(pred_temp, target_temp)

        # 3. MSE Loss for Phases (Softmax applied)
        pred_phases_prob = F.softmax(pred_phases_logits, dim=1)
        loss_phase_map = self.mse(pred_phases_prob, target_phases)

        # Apply Mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.view(1, 1, mask.shape[0], mask.shape[1])
            # Ensure mask is on correct device
            mask = mask.to(pred.device)

            loss_temp = (loss_temp_map * mask).sum() / mask.sum()
            loss_phase = (loss_phase_map * mask).sum() / mask.sum()
        else:
            loss_temp = loss_temp_map.mean()
            loss_phase = loss_phase_map.mean()

        loss_main = loss_temp + loss_phase

        # 4. Sobel Loss
        # Apply Sobel on Temp
        loss_sobel_temp = self.sobel(pred_temp, target_temp, mask)

        # Apply Sobel on Phase Probabilities
        loss_sobel_phase = self.sobel(pred_phases_prob, target_phases, mask)

        loss_sobel = loss_sobel_temp + loss_sobel_phase

        # 5. Physics Loss (Sum of phases = 1)
        # Implicitly satisfied by Softmax, so we can skip or add penalty if we didn't use Softmax
        # Since we use Softmax, sum is always 1.
        loss_physics = 0.0

        return (
            self.alpha * loss_main + self.beta * loss_sobel + self.gamma * loss_physics
        )
