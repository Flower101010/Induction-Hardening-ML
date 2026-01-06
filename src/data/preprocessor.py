"""
Data Preprocessor Module
========================
This module contains logic for data preprocessing steps that might be applied
before or during data loading. This includes normalization (MinMax, Z-score),
handling missing values, or domain-specific transformations.
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
        Step 1: Scan global min/max for normalization
        """
        print("Scanning global stats (for normalization)...")

        # Scan temperature (phases are 0-1)
        # Scan parameters (f, I) for input normalization

        stats = {
            "temp_min": float(df["Temperature"].min()),
            "temp_max": float(df["Temperature"].max()),
            "freq_min": float(df["freq"].min()),
            "freq_max": float(df["freq"].max()),
            "curr_min": float(df["current"].min()),
            "curr_max": float(df["current"].max()),
        }

        print(f"Stats scan complete: {json.dumps(stats, indent=2)}")
        return stats

    def process(self):
        """
        Main processing logic to convert data to tensors.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 1. Load data
        print(f"1. Loading data: {self.input_file}")

        # === Read Logic ===
        if self.input_file.endswith(".csv"):
            df = pd.read_csv(self.input_file)
        else:
            df = pd.read_parquet(self.input_file)
        # ===================

        # 2. Calculate and save stats
        stats = self.calculate_global_stats(df)
        with open(os.path.join(self.output_dir, self.norm_file), "w") as f:
            json.dump(stats, f)

        # 3. Prepare processing
        # Group by process parameters (freq, current)
        groups = df.groupby(["freq", "current"])
        print(f"Detected {len(groups)} experiment groups. Starting conversion...")

        # Target dimensions
        W, H = self.grid_size  # r=64, z=128

        # Flag: check if geometry mask is saved
        mask_saved = False

        for i, ((freq_val, curr_val), group_df) in enumerate(groups):
            print(
                f"\rProcessing [{i + 1}/{len(groups)}] (f={int(freq_val)}, I={curr_val})...",
                end="",
            )

            # --- A. Strict Sorting (Crucial) ---
            # Must sort by: Time -> Z -> R
            # This ensures correct reshape
            sorted_df = group_df.sort_values(by=["time", "z", "r"])

            # Get time steps
            n_time = sorted_df["time"].nunique()

            # --- B. Extract Raw Data ---
            # Extract desired columns
            # Values might contain NaN (due to chamfer/regular grid)
            temp_raw = sorted_df["Temperature"].values
            aust_raw = sorted_df["Austenite"].values
            mart_raw = sorted_df["Martensite"].values

            # --- C. Handle Chamfer/Void ---
            # COMSOL exports regular grids, points outside geometry are NaN
            # We use NaN in Temperature to detect "Air"

            # Create mask: 1 for Solid, 0 for Air (NaN)
            is_solid = ~np.isnan(temp_raw)

            # Save mask if first run (geometry is constant)
            if not mask_saved:
                # Mask is flattened, reshape to 2D
                # Take first time step mask
                # mask shape: (H, W)
                single_step_mask = is_solid[: H * W].astype(np.float32).reshape(H, W)

                # Save mask
                np.save(os.path.join(self.output_dir, self.mask_file), single_step_mask)
                print("\n[Info] Geometry mask detected and saved.")
                mask_saved = True

            # --- D. Fill NaNs ---
            # Replace NaN with 0.0 (clean for CNN)
            # Essential to prevent NaN loss
            temp_raw = np.nan_to_num(temp_raw, nan=0.0)  # type: ignore
            aust_raw = np.nan_to_num(aust_raw, nan=0.0)  # pyright: ignore[reportArgumentType, reportCallIssue]
            mart_raw = np.nan_to_num(mart_raw, nan=0.0)  # pyright: ignore[reportArgumentType, reportCallIssue]

            # --- E. Calculate Remaining Phase (Initial Phase) ---
            # Initial = 1 - (Aust + Mart)
            # Only inside solid, air remains 0
            init_raw = np.zeros_like(aust_raw)

            # Compute only where is_solid
            init_raw[is_solid] = 1.0 - (aust_raw[is_solid] + mart_raw[is_solid])

            # Clip to [0, 1]
            init_raw = np.clip(init_raw, 0.0, 1.0)
            # Ensure air is 0
            init_raw[~is_solid] = 0.0

            # --- F. Normalization ---
            # Min-Max Normalize Temperature
            # Temp_norm = (T - T_min) / (T_max - T_min)
            # Only normalize solid, air remains 0
            t_min, t_max = stats["temp_min"], stats["temp_max"]

            temp_norm = np.zeros_like(temp_raw)
            if t_max > t_min:
                temp_norm[is_solid] = (temp_raw[is_solid] - t_min) / (t_max - t_min)

            # --- G. Combine and Reshape ---
            # Combine 4 channels: [Temp_norm, Aust, Mart, Initial]
            # Current shape: (Total_Points, )

            # Stack
            # Shape: (Total_Points, 4)
            combined_data = np.stack([temp_norm, aust_raw, mart_raw, init_raw], axis=1)

            # Core Reshape
            # Target: (Time, Height/z, Width/r, Channels)
            try:
                tensor_4d = combined_data.reshape(n_time, H, W, 4)
            except ValueError as e:
                print(f"\nReshape failed (f={freq_val}, I={curr_val}): {e}")
                print(
                    f"Data length {len(combined_data)} not divisible by {n_time}x{H}x{W}={n_time * H * W}"
                )
                continue

            tensor_final = tensor_4d.transpose(0, 3, 1, 2)

            tensor_final = tensor_final.astype(np.float32)

            filename = f"sim_f{int(freq_val)}_i{curr_val:.2f}.npy"
            np.save(os.path.join(self.output_dir, filename), tensor_final)

            # Clean memory
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

        print("\n\nAll tasks completed!")
        print(f"1. Training data (.npy) saved at: {self.output_dir}/")
        print(
            f"2. Geometry Mask saved at: {os.path.join(self.output_dir, self.mask_file)}"
        )
        print(
            "   (Load this mask during training to compute loss only on mask=1 regions)"
        )
        print(
            f"3. Normalization stats saved at: {os.path.join(self.output_dir, self.norm_file)}"
        )
        print("   (Use these stats to denormalize temperature during inference)")


if __name__ == "__main__":
    config = {
        "input_file": "data/processed/processed_dataset.csv",
        "output_dir": "data/processed/npy_data",
        "grid_size": (64, 128),
        "norm_file": "normalization_stats.json",
        "mask_file": "geometry_mask.npy",
    }

    preprocessor = DataPreprocessor(**config)
    preprocessor.process()
