import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os

"""
物理情景说明：
本脚本模拟了一个简化的 2D 轴类感应淬火过程 (Induction Hardening)。
- 物理方程：二维热传导方程 (Heat Equation) + 移动热源项 + 冷却项。
- 几何区域：X 轴为径向 (Radius, 左侧为轴心, 右侧为表面)，Y 轴为轴长 (Length)。
- 边界条件：
    - 左侧 (轴心): 绝热/对称边界 (Neumann BC)。
    - 右侧 (表面): 自由边界 (允许加热) + 喷水冷却。
    - 上下 (两端): Dirichlet 边界 (温度为0, 模拟夹具导热)。
- 热源 (集肤效应): 热源集中在右侧边缘 (表面)，随时间从上往下移动。
- 淬火 (Quenching): 在热源经过后，紧接着施加一个强冷却区域。
"""


def generate_shaft_data(
    num_samples=100, time_steps=50, grid_size=64, seed=42, output_dir="data"
):
    """
    生成简化的轴类加热数据。
    """
    np.random.seed(seed)

    data_inputs = []
    data_outputs = []

    print(f"Generating {num_samples} samples with seed {seed}...")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    for _ in tqdm(range(num_samples)):
        # 1. 随机化参数 (模拟不同的工况)
        power = np.random.uniform(0.5, 1.5)  # 加热功率因子
        diffusivity = 0.01  # 热扩散率

        # 网格设置
        nx, ny = grid_size, grid_size
        dx = 1.0 / nx
        dt = 0.001

        # 初始化温度场
        u = np.zeros((nx, ny))

        # 存储该样本的时间序列
        sample_trajectory = []

        # 模拟时间步
        for t in range(time_steps):
            u_new = u.copy()

            # --- 1. 计算热源 (Source) ---
            # 模拟热源移动 (从上往下)
            source_pos_y = int((t / time_steps) * ny)

            Y, X = np.ogrid[:ny, :nx]
            # 集肤效应：热源集中在右侧表面 (X = nx-1)
            # 使用较小的方差使加热更集中于表面
            dist_sq = (X - (nx - 1)) ** 2 + (Y - source_pos_y) ** 2
            source = power * np.exp(-dist_sq / 10.0)

            # --- 2. 计算扩散 (Diffusion) ---
            # 使用 np.roll 计算拉普拉斯算子
            # 注意：np.roll 是周期性的，会导致左边界和右边界互通。
            # 我们需要在后续步骤中通过强制边界条件来修正这一点。
            laplacian = (
                np.roll(u, 1, axis=0)
                + np.roll(u, -1, axis=0)
                + np.roll(u, 1, axis=1)
                + np.roll(u, -1, axis=1)
                - 4 * u
            ) / (dx**2)

            # 更新温度场: dT/dt = alpha * laplacian + Source
            u_new += dt * (diffusivity * laplacian + source)

            # --- 3. 模拟淬火 (Quenching) ---
            # 喷淋头跟在感应圈后面 15 个网格的位置
            spray_pos_y = source_pos_y - 15
            if 0 <= spray_pos_y < ny:
                # 喷水区域：表面附近
                mask = (np.abs(Y - spray_pos_y) < 6) & (X > nx - 10)
                # 强行冷却 (牛顿冷却定律的简化版，直接衰减)
                u_new[mask] *= 0.7

            # --- 4. 强制边界条件 (Boundary Conditions) ---

            # 上下边界 (两端): 夹具夹持，温度恒为 0 (Dirichlet)
            u_new[0, :] = 0
            u_new[-1, :] = 0

            # 左侧边界 (轴心): 对称轴，绝热 (Neumann, dT/dx = 0)
            # 强制左边界的值等于其右侧邻居的值，切断与右边界的周期性联系
            u_new[:, 0] = u_new[:, 1]

            # 右侧边界 (表面):
            # 之前代码强制为0是错误的，因为我们在加热表面。
            # 这里不做强制赋值，允许温度自由演化 (自然边界)。
            # 但为了切断 np.roll 带来的左侧周期性影响，我们需要修正拉普拉斯算子带来的误差
            # 或者简单地，假设最外层与倒数第二层梯度较小（绝热），除非有热源/冷却。
            # 鉴于我们已经加了 Source 和 Quench，这里不做额外操作即可，
            # 只要左侧边界被修正了，右侧就不会收到左侧传来的错误热量。

            u = u_new
            sample_trajectory.append(u.copy())

        # 转换格式
        # Input channel 0: Initial grid (zeros)
        # Input channel 1: Power map
        input_tensor = np.zeros((nx, ny, 2))
        input_tensor[:, :, 1] = power

        # Output: T时刻的温度场堆叠
        output_tensor = np.array(sample_trajectory)  # shape: (time, x, y)

        data_inputs.append(input_tensor)
        data_outputs.append(output_tensor)

    return np.array(data_inputs), np.array(data_outputs)


def main():
    parser = argparse.ArgumentParser(
        description="Generate dummy induction hardening data."
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples to generate"
    )
    parser.add_argument(
        "--time_steps", type=int, default=20, help="Number of time steps per sample"
    )
    parser.add_argument("--grid_size", type=int, default=64, help="Grid size (square)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out_dir", type=str, default="data", help="Output directory")

    args = parser.parse_args()

    inputs, outputs = generate_shaft_data(
        args.num_samples, args.time_steps, args.grid_size, args.seed, args.out_dir
    )

    # 保存路径
    save_x = os.path.join(args.out_dir, "dummy_shaft_train_x.npy")
    save_y = os.path.join(args.out_dir, "dummy_shaft_train_y.npy")

    np.save(save_x, inputs)
    np.save(save_y, outputs)

    print(f"Saved data to {args.out_dir}")
    print("Input shape:", inputs.shape)
    print("Output shape:", outputs.shape)

    # 可视化
    sample_idx = 0
    # squeeze=False 确保 axes 始终是一个 2D 数组，避免单张图时返回对象的问题
    fig, axes = plt.subplots(1, args.time_steps, figsize=(20, 3), squeeze=False)
    axes = axes.flatten()

    # 统一色标范围，方便观察
    vmax = np.max(outputs[sample_idx])

    for t in range(args.time_steps):
        # 注意：imshow 默认 origin='upper'，符合我们矩阵的定义 (0行在上面)
        # 我们转置一下显示，让 X轴(径向)水平，Y轴(轴长)垂直 -> 看起来像横放的轴
        # 或者保持原样：X轴垂直，Y轴水平？
        # 原代码：axes[t].imshow(outputs[sample_idx, t]...)
        # outputs shape: (time, nx, ny) -> (time, radius, length)
        # 通常画图习惯：横轴是轴长(Length, Y)，纵轴是径向(Radius, X)
        # 所以我们显示时转置一下，并且把径向翻转（让表面在上面或下面）

        img_data = outputs[sample_idx, t]  # (nx, ny) = (Radius, Length)

        # 现在的显示：行是Radius，列是Length。
        # 也就是 图像的高度是半径，宽度是轴长。
        # 我们的热源在 X=nx-1 (图像底部)。
        # 移动方向是 Y 增加 (图像从左到右)。

        axes[t].imshow(img_data, cmap="hot", vmin=0, vmax=vmax, aspect="auto")
        axes[t].set_title(f"T={t}")
        axes[t].axis("off")

    # 确保输出目录存在
    fig_dir = "outputs/figures"
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "dummy_shaft_sample.png"))
    print(f"Saved visualization to {fig_dir}/dummy_shaft_sample.png")


if __name__ == "__main__":
    main()
