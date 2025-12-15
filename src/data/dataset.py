import torch
from torch.utils.data import Dataset
import numpy as np
import os


class InductionHardeningDataset(Dataset):
    """
    Dataset for Induction Hardening.
    Loads .npy files from the processed data directory.

    感应淬火数据集。
    从处理后的数据目录加载 .npy 文件。
    """

    def __init__(self, data_dir, split="train", transform=None):
        """
        Args:
            data_dir (str): Path to the processed data directory.
            split (str): "train", "val", or "test".
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        # Load data
        # 加载数据
        try:
            self.X = np.load(os.path.join(data_dir, f"{split}_X.npy"))
            self.Y = np.load(os.path.join(data_dir, f"{split}_Y.npy"))
        except FileNotFoundError:
            if split != "train":
                print(f"Warning: {split} data not found. Loading train data instead.")
                # 警告：未找到 {split} 数据。加载训练数据代替。
                self.X = np.load(os.path.join(data_dir, "train_X.npy"))
                self.Y = np.load(os.path.join(data_dir, "train_Y.npy"))
            else:
                raise

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.from_numpy(self.Y[idx]).float()

        # x: [3, 64, 64]
        # y: [2, 64, 64]

        if self.transform:
            pass

        return x, y
