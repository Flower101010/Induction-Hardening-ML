"""
Improved Dummy Data Generator for Induction Hardening
Improvements over v1:
1. Fixed Boundary Conditions: Removed np.roll wrapping (Surface no longer touches Center).
2. Cylindrical Coordinates: Added 1/r * dT/dr term for radial symmetry.
3. Phase Field: Added a dummy 'Phase' channel (simulating Austenite/Martensite evolution).
   - Logic: If T > Ac3 (threshold), transforms to Austenite.
   - If Austenite cools down, transforms to Martensite (simplified).
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os


def generate_shaft_data_v2(
    num_samples=100, time_steps=50, grid_size=64, seed=42, output_dir="data"
):
    np.random.seed(seed)

    # Output shapes:
    # 输出形状：
    # Input: (nx, ny, 2) -> [Initial_Temp, Power_Map]
    # 输入：(nx, ny, 2) -> [初始温度, 功率图]
    # Output: (time, nx, ny, 2) -> [Temp, Phase]
    # 输出：(time, nx, ny, 2) -> [温度, 相]

    data_inputs = []
    data_outputs = []

    print(f"Generating {num_samples} samples (v2) with seed {seed}...")
    os.makedirs(output_dir, exist_ok=True)

    # Physical constants (Normalized)
    # 物理常数（归一化）
    Ac3_temp = 0.5  # Austenitization temperature threshold
    # 奥氏体化温度阈值
    Ms_temp = 0.2  # Martensite start temperature
    # 马氏体开始温度

    for _ in tqdm(range(num_samples)):
        power = np.random.uniform(50, 80)
        diffusivity = 0.01

        nx, ny = grid_size, grid_size
        dx = 1.0 / nx
        dt = 0.005

        # Coordinates for cylindrical term
        # 圆柱坐标项的坐标
        # x is radius (0 to 1)
        # x 是半径（0 到 1）
        r = np.linspace(dx / 2, 1.0 - dx / 2, nx)  # Avoid r=0 singularity
        # 避免 r=0 的奇点
        r_matrix = r[:, np.newaxis]  # (nx, 1)

        # Fields
        # 场
        u = np.zeros((nx, ny))  # Temperature
        # 温度
        _phase = np.zeros((nx, ny))  # 0: Ferrite, 1: Austenite, 2: Martensite
        # 0: 铁素体, 1: 奥氏体, 2: 马氏体

        # History tracker for phase
        # 相的历史追踪器
        max_temp_history = np.zeros((nx, ny))
        is_austenite = np.zeros((nx, ny), dtype=bool)

        sample_trajectory = []

        for t in range(time_steps):
            u_new = u.copy()

            # --- 1. Source (Scanning) ---
            # --- 1. 热源（扫描）---
            source_pos_y = int((t / time_steps) * ny)
            # Correct grid for (nx, ny) where nx=Radius, ny=Length
            # 修正网格 (nx, ny)，其中 nx=半径, ny=长度
            X, Y = np.ogrid[:nx, :ny]

            # Skin effect at surface (index nx-1)
            # 表面的集肤效应（索引 nx-1）
            dist_sq = (X - (nx - 1)) ** 2 + (Y - source_pos_y) ** 2
            source = power * np.exp(-dist_sq / 10.0)

            # --- 2. Diffusion (Finite Difference with Fixed Boundaries) ---
            # --- 2. 扩散（具有固定边界的有限差分）---
            # Laplacian = d2u/dx2 + d2u/dy2
            # 拉普拉斯算子 = d2u/dx2 + d2u/dy2
            # Cylindrical: 1/r * du/dx + d2u/dx2 + d2u/dy2
            # 圆柱坐标：1/r * du/dx + d2u/dx2 + d2u/dy2

            # Calculate gradients manually to avoid np.roll wrap-around
            # 手动计算梯度以避免 np.roll 环绕
            # X-direction (Radial)
            # X 方向（径向）
            d2u_dx2 = np.zeros_like(u)
            d2u_dx2[1:-1, :] = (u[2:, :] - 2 * u[1:-1, :] + u[:-2, :]) / dx**2
            # BCs for Laplacian:
            # 拉普拉斯算子的边界条件：
            # Left (Center): Symmetry (u[-1] = u[1]) => d2u/dx2 at 0 = (u[1] - 2u[0] + u[1])/dx2 = 2(u[1]-u[0])/dx2
            # 左侧（中心）：对称 (u[-1] = u[1]) => d2u/dx2 at 0 = (u[1] - 2u[0] + u[1])/dx2 = 2(u[1]-u[0])/dx2
            d2u_dx2[0, :] = 2 * (u[1, :] - u[0, :]) / dx**2
            # Right (Surface): Insulated (u[n] = u[n-1]) => d2u/dx2 at -1 = (u[-2] - 2u[-1] + u[-2])/dx2 = 2(u[-2]-u[-1])/dx2
            # 右侧（表面）：绝热 (u[n] = u[n-1]) => d2u/dx2 at -1 = (u[-2] - 2u[-1] + u[-2])/dx2 = 2(u[-2]-u[-1])/dx2
            d2u_dx2[-1, :] = 2 * (u[-2, :] - u[-1, :]) / dx**2

            # Y-direction (Axial)
            # Y 方向（轴向）
            d2u_dy2 = np.zeros_like(u)
            d2u_dy2[:, 1:-1] = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / dx**2
            # Top/Bottom BCs are Dirichlet (0), so u[-1] and u[n] are 0.
            # 顶部/底部边界条件是 Dirichlet (0)，所以 u[-1] 和 u[n] 是 0。
            d2u_dy2[:, 0] = (u[:, 1] - 2 * u[:, 0] + 0) / dx**2
            d2u_dy2[:, -1] = (0 - 2 * u[:, -1] + u[:, -2]) / dx**2

            # Radial term: 1/r * du/dx
            # 径向项：1/r * du/dx
            du_dx = np.zeros_like(u)
            du_dx[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dx)
            du_dx[0, :] = 0  # Symmetry at center
            # 中心对称
            du_dx[-1, :] = 0  # Insulated at surface
            # 表面绝热

            laplacian = d2u_dx2 + d2u_dy2 + (1.0 / r_matrix) * du_dx

            # Update Temp
            # 更新温度
            u_new += dt * (diffusivity * laplacian + source)

            # --- 3. Quenching ---
            # --- 3. 淬火 ---
            spray_pos_y = source_pos_y - 15
            if 0 <= spray_pos_y < ny:
                mask = (np.abs(Y - spray_pos_y) < 6) & (X > nx - 10)
                u_new[mask] *= 0.7

            # --- 4. Phase Transformation Logic (Dummy) ---
            # --- 4. 相变逻辑（模拟）---
            # Update max temp
            # 更新最大温度
            max_temp_history = np.maximum(max_temp_history, u_new)

            # Austenitization
            # 奥氏体化
            newly_austenite = u_new > Ac3_temp
            is_austenite = is_austenite | newly_austenite

            # Martensite Transformation (Quenching)
            # 马氏体转变（淬火）
            # If was Austenite AND Temp drops below Ms
            # 如果是奥氏体且温度降至 Ms 以下
            is_martensite = is_austenite & (u_new < Ms_temp)

            # Encode Phase Map
            # 编码相图
            # 0.0: Base, 0.5: Austenite, 1.0: Martensite
            # 0.0: 基体, 0.5: 奥氏体, 1.0: 马氏体
            phase_map = np.zeros_like(u_new)
            phase_map[is_austenite] = 0.5
            phase_map[is_martensite] = 1.0

            u = u_new

            # Stack Temp and Phase
            # 堆叠温度和相
            # Shape: (nx, ny, 2)
            # 形状: (nx, ny, 2)
            frame = np.stack([u.copy(), phase_map.copy()], axis=-1)
            sample_trajectory.append(frame)

        # Input: Initial state (zeros) + Power
        # 输入：初始状态（零）+ 功率
        input_tensor = np.zeros((nx, ny, 2))
        input_tensor[:, :, 1] = power

        # Output: (Time, nx, ny, 2)
        # 输出：(Time, nx, ny, 2)
        output_tensor = np.array(sample_trajectory)

        data_inputs.append(input_tensor)
        data_outputs.append(output_tensor)

    return np.array(data_inputs), np.array(data_outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="data/processed")
    args = parser.parse_args()

    inputs, outputs = generate_shaft_data_v2(
        num_samples=args.num_samples, output_dir=args.out_dir
    )

    all_series = np.array(outputs)  # (Num_Samples, 50, 64, 64, 2)
    grid_size = all_series.shape[2]

    # 构造 X: [Temp_t, Phase_t, Power_Map]
    X_list = []
    Y_list = []

    for i in range(len(inputs)):
        # 获取当前样本的 Power 值 (从 input_tensor 的第1个通道取一个值即可)
        p_val = inputs[i][0, 0, 1]
        # 修正 2: Power 归一化！！！(非常重要)
        # 将 50~80 的值除以 100，使其变为 0.5~0.8，与温度场(0~1)量级匹配
        p_norm = p_val / 100.0

        # 序列切片
        series = all_series[i]  # (50, 64, 64, 2)

        # x: current step, y: next step
        x_frames = series[:-1]  # t=0 to 48
        y_frames = series[1:]  # t=1 to 49

        # --- 修改开始: 生成移动的热源场 ---
        source_maps = []
        nx, ny = grid_size, grid_size
        X_grid, Y_grid = np.ogrid[:nx, :ny]

        # 重新计算每一帧的热源分布
        # 注意：这里的时间步数必须与 x_frames 对应 (0 到 48)
        # 原始生成逻辑中 time_steps=50
        time_steps = 50

        for t in range(x_frames.shape[0]):
            source_pos_y = int((t / time_steps) * ny)
            dist_sq = (X_grid - (nx - 1)) ** 2 + (Y_grid - source_pos_y) ** 2
            # 生成热源场 (使用 p_norm 缩放)
            source_frame = p_norm * np.exp(-dist_sq / 10.0)
            source_maps.append(source_frame)

        source_stack = np.array(source_maps)[..., np.newaxis]  # (49, 64, 64, 1)

        # 拼接：[Temp, Phase, Source_Map]
        x_combined = np.concatenate([x_frames, source_stack], axis=-1)
        # --- 修改结束 ---

        X_list.append(x_combined)
        Y_list.append(y_frames)

    X_final = np.concatenate(X_list, axis=0)  # (4900, 64, 64, 3)
    Y_final = np.concatenate(Y_list, axis=0)  # (4900, 64, 64, 2)

    # 调整维度以适应 PyTorch: (N, H, W, C) -> (N, C, H, W)
    X_final = np.transpose(X_final, (0, 3, 1, 2))
    Y_final = np.transpose(Y_final, (0, 3, 1, 2))

    print(f"Final Training Data Shape: X {X_final.shape}, Y {Y_final.shape}")
    # X: (4900, 3, 64, 64) -> Channels: [Temp, Phase, Power]
    # Y: (4900, 2, 64, 64) -> Channels: [Temp, Phase]

    save_x = os.path.join(args.out_dir, "train_X.npy")
    save_y = os.path.join(args.out_dir, "train_Y.npy")
    np.save(save_x, X_final)
    np.save(save_y, Y_final)

    print(f"Saved Next-Step Prediction data to {args.out_dir}")

    # Visualization
    fig, axes = plt.subplots(2, 10, figsize=(20, 5))
    sample_idx = 0

    # Plot Temp (Top Row) and Phase (Bottom Row) for 10 steps
    times = np.linspace(0, 49, 10, dtype=int)

    for i, t in enumerate(times):
        # Temp
        axes[0, i].imshow(outputs[sample_idx, t, :, :, 0], cmap="hot", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[0, i].set_title(f"T={t} Temp")

        # Phase
        axes[1, i].imshow(
            outputs[sample_idx, t, :, :, 1], cmap="viridis", vmin=0, vmax=1
        )
        axes[1, i].axis("off")
        axes[1, i].set_title("Phase")

    plt.savefig("outputs/figures/dummy_shaft_v2_sample.png")
    print("Saved visualization to outputs/figures/dummy_shaft_v2_sample.png")


if __name__ == "__main__":
    main()
