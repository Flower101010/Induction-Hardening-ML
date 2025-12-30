"""
Data Preprocessor Module
========================
This module contains logic for data preprocessing steps that might be applied
before or during data loading. This includes normalization (MinMax, Z-score),
handling missing values, or domain-specific transformations.

此模块包含可能在数据加载之前或期间应用的数据预处理步骤的逻辑。
这包括归一化（MinMax，Z-score），
处理缺失值或特定领域的转换。
"""

import pandas as pd
import numpy as np
import os
import json
import gc
from typing import Tuple, Dict


class DataPreprocessor:
    """
    Handles the preprocessing of raw CSV/Parquet data into .npy tensors for training.

    Attributes:
        input_file (str): Path to the input processed dataset (CSV/Parquet).
        output_dir (str): Directory to save the output .npy files.
        grid_size (Tuple[int, int]): Target grid size (Width/r, Height/z).
        norm_file (str): Filename for normalization statistics.
        mask_file (str): Filename for geometry mask.
    """

    def __init__(
        self,
        input_file: str = "data/processed/processed_dataset.csv",
        output_dir: str = "data/processed/npy_data",
        grid_size: Tuple[int, int] = (64, 128),
        norm_file: str = "normalization_stats.json",
        mask_file: str = "geometry_mask.npy",
    ):
        self.input_file = input_file
        self.output_dir = output_dir
        self.grid_size = grid_size
        self.norm_file = norm_file
        self.mask_file = mask_file

    def calculate_global_stats(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        第一步：扫描全局最大最小值，用于归一化
        """
        print("正在扫描全局统计数据 (用于归一化)...")

        # 只需要扫描温度，因为相含量本身就是 0-1
        # 甚至参数 (f, I) 也建议扫描一下用于输入归一化

        stats = {
            "temp_min": float(df["Temperature"].min()),
            "temp_max": float(df["Temperature"].max()),
            "freq_min": float(df["freq"].min()),
            "freq_max": float(df["freq"].max()),
            "curr_min": float(df["current"].min()),
            "curr_max": float(df["current"].max()),
        }

        print(f"统计完成: {json.dumps(stats, indent=2)}")
        return stats

    def process(self):
        """
        Main processing logic to convert data to tensors.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 1. 加载数据
        print(f"1. 加载数据: {self.input_file}")

        # === 修正读取逻辑 ===
        if self.input_file.endswith(".csv"):
            # 如果是 CSV 文件，用 read_csv
            df = pd.read_csv(self.input_file)
        else:
            # 如果是 Parquet 文件，用 read_parquet
            df = pd.read_parquet(self.input_file)
        # ===================

        # 2. 计算并保存归一化参数
        stats = self.calculate_global_stats(df)
        with open(os.path.join(self.output_dir, self.norm_file), "w") as f:
            json.dump(stats, f)

        # 3. 准备处理
        # 按工艺参数分组 (freq, current)
        groups = df.groupby(["freq", "current"])
        print(f"检测到 {len(groups)} 组实验。开始转换...")

        # 期望的尺寸
        W, H = self.grid_size  # r=64, z=128

        # 标志位：是否已经保存了几何掩码
        mask_saved = False

        for i, ((freq_val, curr_val), group_df) in enumerate(groups):
            print(
                f"\r处理进度 [{i + 1}/{len(groups)}] (f={int(freq_val)}, I={curr_val})...",
                end="",
            )

            # --- A. 严格排序 (至关重要) ---
            # 必须按照: 时间(T) -> 高度(z) -> 宽度(r) 排序
            # 这样 reshape 出来的图片才是正确的方向
            sorted_df = group_df.sort_values(by=["time", "z", "r"])

            # 获取时间步数
            n_time = sorted_df["time"].nunique()

            # --- B. 提取原始数据 ---
            # 取出我们要的列
            # 注意：这里可能会包含 NaN (因为倒角/规则格栅的缘故)
            temp_raw = sorted_df["Temperature"].values
            aust_raw = sorted_df["Austenite"].values
            mart_raw = sorted_df["Martensite"].values

            # --- C. 处理倒角 (Chamfer/Void) ---
            # COMSOL 导出的规则网格，如果有点在几何体外，通常是 NaN
            # 我们利用 Temperature 里的 NaN 来判断哪里是“空气”

            # 创建掩码：非 NaN 的地方是 1 (实体)，NaN 的地方是 0 (空气)
            # 注意：这里假设温度场里 NaN 代表空气。
            is_solid = ~np.isnan(temp_raw)

            # 如果这是第一次循环，我们保存这个掩码 (因为所有实验的几何体都是一样的)
            if not mask_saved:
                # 这里的 mask 是展平的，我们需要 reshape 成 2D 图片
                # 取第一个时间步的 mask 即可 (几何体形状不随时间改变)
                # mask shape: (H, W)
                single_step_mask = is_solid[: H * W].astype(np.float32).reshape(H, W)

                # 保存掩码
                np.save(os.path.join(self.output_dir, self.mask_file), single_step_mask)
                print("\n[Info] 几何掩码已检测并保存 (含倒角处理)。")
                mask_saved = True

            # --- D. 填充 NaN ---
            # 将 NaN 替换为 0 (或者其他值，但 0 最适合 CNN)
            # 这一步必须做，否则网络训练 Loss 会变成 NaN
            temp_raw = np.nan_to_num(temp_raw, nan=0.0)  # type: ignore
            aust_raw = np.nan_to_num(aust_raw, nan=0.0)  # pyright: ignore[reportArgumentType, reportCallIssue]
            mart_raw = np.nan_to_num(mart_raw, nan=0.0)  # pyright: ignore[reportArgumentType, reportCallIssue]

            # --- E. 计算剩余相 (Initial Phase) ---
            # Initial = 1 - (Aust + Mart)
            # 仅在实体内部计算，空气部分保持 0
            init_raw = np.zeros_like(aust_raw)

            # 只在 is_solid 的地方计算
            init_raw[is_solid] = 1.0 - (aust_raw[is_solid] + mart_raw[is_solid])

            # 截断到 [0, 1] 防止浮点误差
            init_raw = np.clip(init_raw, 0.0, 1.0)
            # 再次确保空气部分是 0 (clip可能会把负数变0，这没问题，但空气部分应该是0)
            init_raw[~is_solid] = 0.0

            # --- F. 归一化 (Normalization) ---
            # 仅对温度进行 Min-Max 归一化
            # Temp_norm = (T - T_min) / (T_max - T_min)
            # 同样，只归一化实体部分，空气部分保持 0
            t_min, t_max = stats["temp_min"], stats["temp_max"]

            temp_norm = np.zeros_like(temp_raw)
            if t_max > t_min:
                temp_norm[is_solid] = (temp_raw[is_solid] - t_min) / (t_max - t_min)

            # --- G. 组合与 Reshape ---
            # 目前我们要组合 4 个通道: [Temp_norm, Aust, Mart, Initial]
            # 此时数据形状是 (Total_Points, )

            # 堆叠
            # Shape: (Total_Points, 4)
            combined_data = np.stack([temp_norm, aust_raw, mart_raw, init_raw], axis=1)

            # 核心 Reshape
            # 目标: (Time, Height/z, Width/r, Channels)
            try:
                tensor_4d = combined_data.reshape(n_time, H, W, 4)
            except ValueError as e:
                print(f"\n❌ Reshape 失败 (f={freq_val}, I={curr_val}): {e}")
                print(
                    f"数据长度 {len(combined_data)} 无法被 {n_time}x{H}x{W}={n_time * H * W} 整除"
                )
                continue

            # 转换为 PyTorch 格式: (Batch/Time, Channels, Height, Width)
            # (T, H, W, C) -> (T, C, H, W)
            tensor_final = tensor_4d.transpose(0, 3, 1, 2)

            # --- H. 保存 ---
            # float32 足够机器学习使用，节省一半空间
            tensor_final = tensor_final.astype(np.float32)

            filename = f"sim_f{int(freq_val)}_i{curr_val:.2f}.npy"
            np.save(os.path.join(self.output_dir, filename), tensor_final)

            # 清理内存
            del (
                sorted_df,
                temp_raw,
                aust_raw,
                mart_raw,
                init_raw,
                temp_norm,
                combined_data,
                tensor_4d,
                tensor_final,
            )
            gc.collect()

        print("\n\n全部完成！")
        print(f"1. 训练数据 (.npy) 保存在: {self.output_dir}/")
        print(
            f"2. 几何掩码 (Mask) 保存在: {os.path.join(self.output_dir, self.mask_file)}"
        )
        print("   (训练时请加载此 Mask，计算 Loss 时只计算 mask=1 的区域)")
        print(f"3. 归一化参数保存在: {os.path.join(self.output_dir, self.norm_file)}")
        print("   (预测推理时，用这些参数把 0-1 的温度还原回摄氏度)")


if __name__ == "__main__":
    # 默认配置
    config = {
        "input_file": "data/processed/processed_dataset.csv",
        "output_dir": "data/processed/npy_data",
        "grid_size": (64, 128),
        "norm_file": "normalization_stats.json",
        "mask_file": "geometry_mask.npy",
    }

    preprocessor = DataPreprocessor(**config)
    preprocessor.process()
