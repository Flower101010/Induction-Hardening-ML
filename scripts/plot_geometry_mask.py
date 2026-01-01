import os
import numpy as np
import matplotlib.pyplot as plt

# 配置路径
MASK_FILE = "data/processed/npy_data/geometry_mask.npy"
OUTPUT_FILE = "outputs/figures/geometry_mask.png"


def main():
    if not os.path.exists(MASK_FILE):
        print(f"错误: 掩码文件 {MASK_FILE} 不存在。请先运行数据预处理脚本。")
        return

    try:
        # 加载掩码
        # 掩码形状通常为 (H, W) 或 (1, H, W)
        mask = np.load(MASK_FILE)
        print(f"加载掩码文件成功，形状: {mask.shape}")

        # 如果是 (1, H, W) 或 (C, H, W)，取第一个通道
        if mask.ndim == 3:
            mask = mask[0]

        # 绘图
        plt.figure(figsize=(6, 8))

        # origin='lower' 确保 z 轴 (高度) 从下往上增加，符合物理直觉
        # 假设数据存储顺序为 [z, r]，即行是 z，列是 r
        im = plt.imshow(mask, cmap="gray", origin="lower", interpolation="nearest")

        plt.title("Geometry Mask Visualization\n(White=Steel, Black=Air)", fontsize=14)
        plt.xlabel("r (Radial Coordinate)", fontsize=12)
        plt.ylabel("z (Axial Coordinate)", fontsize=12)

        # 添加颜色条
        cbar = plt.colorbar(im, label="Mask Value")
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["0 (Air)", "1 (Steel)"])

        plt.tight_layout()

        # 保存
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        plt.savefig(OUTPUT_FILE, dpi=300)
        print(f"几何掩码图已保存至: {OUTPUT_FILE}")

    except Exception as e:
        print(f"绘图过程中发生错误: {e}")


if __name__ == "__main__":
    main()
