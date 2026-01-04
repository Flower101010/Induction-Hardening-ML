import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class InductionHardeningDataset(Dataset):
    """
    感应淬火数据集加载器 (最终修复版 V2)
    包含：
    1. 自动 split 生成
    2. 下一时刻预测逻辑 (t -> t+1)
    3. 动态通道截取
    """
    def __init__(self, data_dir, split="train", split_file="data_split.json", time_steps=100):
        self.data_dir = data_dir
        self.split = split
        self.time_steps = time_steps
        
        # 1. 尝试加载或生成数据划分
        self.file_list = self._load_or_create_split(split_file, split)
        
        # 2. 预先计算总长度
        self.num_files = len(self.file_list)
        self.total_samples = self.num_files * time_steps

        if self.num_files == 0:
            print(f"⚠️ Warning: No files found for split '{split}' in {data_dir}")

    def _load_or_create_split(self, split_file, split):
        if not os.path.exists(split_file):
            print(f"⚠️ Split file '{split_file}' not found. Generating a new one...")
            self._generate_split_file(self.data_dir, split_file)
            
        with open(split_file, "r", encoding='utf-8') as f:
            splits = json.load(f)
            
        if split not in splits:
            if split == 'test' and 'val' in splits:
                return splits['val']
            raise ValueError(f"Split '{split}' not found. Available: {list(splits.keys())}")
            
        return splits[split]

    def _generate_split_file(self, data_dir, output_path):
        all_files = glob.glob(os.path.join(data_dir, "*.npy"))
        all_files = [os.path.basename(f) for f in all_files if "geometry_mask" not in f]
        all_files.sort()
        
        if not all_files:
            raise RuntimeError(f"No .npy files found in {data_dir}!")

        np.random.seed(42)
        np.random.shuffle(all_files)
        
        num_total = len(all_files)
        n_train = int(num_total * 0.8)
        n_val = int(num_total * 0.1)
        
        splits = {
            "train": all_files[:n_train],
            "val": all_files[n_train:n_train+n_val],
            "test": all_files[n_train+n_val:]
        }
        
        # 兜底防止空集
        if len(splits["val"]) == 0 and num_total > 1: splits["val"] = [splits["train"].pop()]
        if len(splits["test"]) == 0 and num_total > 2: splits["test"] = [splits["train"].pop()]
        if len(splits["val"]) == 0: splits["val"] = splits["train"]
        if len(splits["test"]) == 0: splits["test"] = splits["val"]

        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(splits, f, indent=4)
        
        print(f"✅ Generated new split file: {output_path}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # 1. 确定文件与时间索引
        file_idx = idx // self.time_steps
        time_idx = idx % self.time_steps
        
        filename = self.file_list[file_idx]
        filepath = os.path.join(self.data_dir, filename)
        
        # 2. 加载数据 (这步绝不能少！)
        try:
            data = np.load(filepath) 
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            raise e

        # 3. 获取输入与目标
        # 假设 data 形状为 [Time, Channels, H, W]
        if data.ndim == 4:
            # 输入 X: 当前时刻 t
            x_raw = torch.from_numpy(data[time_idx]).float()
            
            # 目标 Y: 下一时刻 t+1
            # 如果已经是最后一步，就预测它自己（或者是零变化），防止越界
            if time_idx < self.time_steps - 1:
                y_raw = torch.from_numpy(data[time_idx + 1]).float()
            else:
                y_raw = x_raw # 边界情况：最后一步保持不变
        else:
            # 如果是单步数据 [C, H, W]，无法做时间预测，只能做自编码
            x_raw = torch.from_numpy(data).float()
            y_raw = x_raw

        # 4. 截取通道
        # Config 改成了 in_channels: 4, out_channels: 4
        # 所以我们需要确保只取前 4 个通道
        x = x_raw[:4]
        y = y_raw[:4]
        
        return x, y