"""
Raw Data Processing Script
==========================

This script handles the preprocessing of raw data provided by the experimental/simulation team.
It converts raw files (e.g., CSV, MAT) into the standard numpy format (.npy) used by the Dataset class.
It may also perform initial cleaning, normalization, or regridding if necessary.

Usage:
    uv run scripts/process_raw_data.py --input_dir data/raw --output_dir data/processed
"""
