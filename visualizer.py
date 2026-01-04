import os
import torch
import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    """
    可视化工具类：专门负责生成科研格式的对比图和分析图。
    """
    def __init__(self, output_dir, channel_names=None):
        """
        Args:
            output_dir (str): 图片保存路径
            channel_names (list): 通道名称列表 (e.g., ['Temperature', 'Austenite', ...])
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 默认通道名称
        self.channel_names = channel_names if channel_names else [
            "Temperature", "Austenite", "Ferrite_Pearlite", "Martensite"
        ]

    def _to_numpy(self, tensor):
        """将 Tensor 或 Numpy 数组统一转为 (H, W) 的 Numpy 数组"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().squeeze().numpy()
        return tensor

    def plot_mask(self, mask, filename="geometry_mask.png"):
        """绘制几何掩码 (对应之前的 geometry_mask.png)"""
        mask_np = self._to_numpy(mask)
        plt.figure(figsize=(5, 8))
        plt.imshow(mask_np, cmap='gray')
        plt.title("Geometry Mask\n(White=Workpiece, Black=Air)")
        plt.axis('off')
        
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"🖼️  Saved mask: {save_path}")

    def plot_single_state(self, data, channel_idx, time_step, filename_prefix):
        """
        绘制单个状态的热力图 (对应之前的 sim_..._t90_martensite.png)
        """
        data_np = self._to_numpy(data)
        
        # 确保数据是 [C, H, W] 或 [H, W]
        if data_np.ndim == 3:
            img = data_np[channel_idx]
        else:
            img = data_np

        channel_name = self.channel_names[channel_idx]
        
        plt.figure(figsize=(6, 8))
        # 温度用 inferno (黑-红-黄)，相变用 viridis (蓝-绿-黄)
        cmap = 'inferno' if "Temp" in channel_name else 'viridis'
        
        plt.imshow(img, cmap=cmap, origin='lower')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(f"{channel_name}\n(t={time_step})")
        plt.axis('off')

        fname = f"{filename_prefix}_t{time_step}_{channel_name.lower()}.png"
        save_path = os.path.join(self.output_dir, fname)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"🖼️  Saved single state: {fname}")

    def plot_comparison(self, target, pred, mask=None, time_step=0, filename_prefix="compare"):
        """
        绘制核心对比图 (对应 sim_..._compare_txx_xxx.png)
        左：Ground Truth | 中：Prediction | 右：Absolute Error
        """
        target_np = self._to_numpy(target) # [C, H, W]
        pred_np = self._to_numpy(pred)     # [C, H, W]
        
        if mask is not None:
            mask_np = self._to_numpy(mask)
        
        num_channels = min(target_np.shape[0], len(self.channel_names))

        for c in range(num_channels):
            name = self.channel_names[c]
            
            gt_img = target_np[c]
            pred_img = pred_np[c]
            error_img = np.abs(gt_img - pred_img)
            
            # 如果有 Mask，把背景误差清零，让图更好看
            if mask is not None:
                error_img = error_img * mask_np

            # 创建画布 1行3列
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # 设置色卡
            cmap = 'inferno' if c == 0 else 'viridis' # 通道0(温度)用火热色，其他用冷色
            
            # 1. Truth
            im1 = axes[0].imshow(gt_img, cmap=cmap, origin='lower')
            axes[0].set_title(f"Ground Truth ({name})")
            plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
            axes[0].axis('off')

            # 2. Pred
            # 统一量程：让预测图的色标范围和真值图一致，方便肉眼对比
            vmin, vmax = gt_img.min(), gt_img.max()
            im2 = axes[1].imshow(pred_img, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
            axes[1].set_title(f"Prediction ({name})")
            plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
            axes[1].axis('off')

            # 3. Error
            # 误差图通常用 Reds 或 coolwarm
            im3 = axes[2].imshow(error_img, cmap='Reds', origin='lower')
            axes[2].set_title(f"Abs Error (Max: {error_img.max():.4f})")
            plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
            axes[2].axis('off')

            plt.suptitle(f"Time Step: {time_step} | Channel: {name}", fontsize=14)
            
            # 保存: sim_..._compare_t10_temperature.png
            fname = f"{filename_prefix}_compare_t{time_step}_{name.lower()}.png"
            save_path = os.path.join(self.output_dir, fname)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"📊 Saved comparison: {fname}")