import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json


class InductionHardeningDataset(Dataset):
    """
    Dataset for Induction Hardening.
    Loads .npy files based on a split configuration file.
    """

    def __init__(
        self,
        data_dir,
        split="train",
        transform=None,
        split_file="config/data_split.json",
        time_steps=100,  # Number of time steps per simulation
    ):
        """
        Args:
            data_dir (str): Path to the processed data directory (containing .npy files).
            split (str): "train", "val", or "test".
            transform (callable, optional): Optional transform to be applied on a sample.
            split_file (str): Path to the JSON file containing data splits.
            time_steps (int): Number of time steps in each simulation file.
        """
        # Handle data_dir path (if it points to processed root instead of npy folder)
        if os.path.isdir(os.path.join(data_dir, "npy_data")):
            self.data_dir = os.path.join(data_dir, "npy_data")
        elif os.path.isdir(os.path.join(data_dir, "npy")):
            self.data_dir = os.path.join(data_dir, "npy")
        else:
            self.data_dir = data_dir

        self.split = split
        self.transform = transform
        self.time_steps = time_steps

        # Load split config
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, "r") as f:
            splits = json.load(f)

        if split not in splits:
            raise ValueError(f"Split '{split}' not found in {split_file}")

        self.file_list = splits[split]

        # Load normalization stats
        stats_path = os.path.join(self.data_dir, "normalization_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                self.stats = json.load(f)
        else:
            print(
                f"Warning: Normalization stats not found at {stats_path}. Using raw values."
            )
            self.stats = None

        # Pre-calculate grid (assuming fixed size 128x64)
        self.H, self.W = 128, 64
        self.grid_z, self.grid_r = torch.meshgrid(
            torch.linspace(0, 1, self.H), torch.linspace(0, 1, self.W), indexing="ij"
        )

    def _parse_params(self, filename):
        """
        Parse frequency and current from filename.
        Example: 'sim_f50000_i1.00.npy' -> (50000.0, 1.0)
        """
        try:
            name = os.path.basename(filename).replace(".npy", "")
            parts = name.split("_")
            # parts: ['sim', 'f50000', 'i1.00']
            f_val = float(parts[1].replace("f", ""))
            i_val = float(parts[2].replace("i", ""))
            return f_val, i_val
        except Exception as e:
            print(f"Error parsing params from {filename}: {e}")
            return 0.0, 0.0

    def __len__(self):
        # Total samples = number of files * time steps
        return len(self.file_list) * self.time_steps

    def __getitem__(self, idx):
        # Map linear index to (file_index, time_index)
        file_idx = idx // self.time_steps
        time_idx = idx % self.time_steps

        filename = self.file_list[file_idx]
        filepath = os.path.join(self.data_dir, filename)

        # Load data: Shape (T, C, H, W)
        # C=4: [Temp_norm, Aust, Mart, Initial]
        # Use mmap_mode to avoid loading full file if possible, though for 100 steps it might be small enough
        try:
            # We only need one time step, so mmap is good
            data = np.load(filepath, mmap_mode="r")
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {filepath}")

        # Get specific time step
        # shape: (4, H, W)
        frame = data[time_idx].copy()

        # Convert to tensor
        target = torch.from_numpy(frame).float()  # [4, H, W]

        # --- Construct Input X ---
        # Channels: [freq, current, time, z, r] -> 5 channels

        # 1. Parse Params
        f_val, i_val = self._parse_params(filename)

        # 2. Normalize Params
        if self.stats:
            f_norm = (f_val - self.stats["freq_min"]) / (
                self.stats["freq_max"] - self.stats["freq_min"]
            )
            i_norm = (i_val - self.stats["curr_min"]) / (
                self.stats["curr_max"] - self.stats["curr_min"]
            )
        else:
            f_norm = f_val
            i_norm = i_val

        # Normalize time (0 to 1)
        t_norm = time_idx / (self.time_steps - 1) if self.time_steps > 1 else 0.0

        # 3. Create Constant Maps
        f_map = torch.full((self.H, self.W), f_norm, dtype=torch.float32)
        i_map = torch.full((self.H, self.W), i_norm, dtype=torch.float32)
        t_map = torch.full((self.H, self.W), t_norm, dtype=torch.float32)

        # 4. Stack Inputs
        # Order: [f, i, t, z, r]
        input_x = torch.stack(
            [f_map, i_map, t_map, self.grid_z, self.grid_r], dim=0
        )  # [5, H, W]

        if self.transform:
            # Apply transform if needed (usually for augmentation, but maybe not here)
            pass

        return input_x, target
