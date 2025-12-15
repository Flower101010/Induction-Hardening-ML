"""
Raw Data Processing Script
==========================
原始数据处理脚本
==========================

This script handles the preprocessing of raw data provided by the experimental/simulation team.
此脚本处理实验/模拟团队提供的原始数据的预处理。
It converts raw files (e.g., CSV, MAT) into the standard numpy format (.npy) used by the Dataset class.
它将原始文件（例如 CSV、MAT）转换为 Dataset 类使用的标准 numpy 格式 (.npy)。
It may also perform initial cleaning, normalization, or regridding if necessary.
如有必要，它还可以执行初始清理、归一化或重新网格化。

Usage:
    uv run scripts/process_raw_data.py --input_dir data/raw --output_dir data/processed
"""
