import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

DATA_DIR = "data/processed/npy_data"
SPLIT_CONFIG_FILE = "config/data_split.json"
OUTPUT_FILE = "outputs/figures/data_split_distribution.png"


def parse_filename(filename):
    match = re.search(r"f(\d+)_i([\d\.]+)\.npy", filename)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def main():
    if not os.path.exists(SPLIT_CONFIG_FILE):
        print(f"error: Config file {SPLIT_CONFIG_FILE} does not exist.")
        return

    with open(SPLIT_CONFIG_FILE, "r") as f:
        split_data = json.load(f)

    all_points = []

    datasets = {"train": [], "val": [], "test": []}

    for split_name, filenames in split_data.items():
        if split_name not in datasets:
            continue

        for fname in filenames:
            freq, curr = parse_filename(fname)
            if freq is not None:
                pt = [freq, curr]
                datasets[split_name].append(pt)
                all_points.append(pt)

    if not all_points:
        print("No valid data points found.")
        return

    all_points = np.array(all_points)

    for k in datasets:
        datasets[k] = np.array(datasets[k])  # type: ignore

    hull = ConvexHull(all_points)

    plt.figure(figsize=(10, 6))

    if len(datasets["train"]) > 0:
        plt.scatter(
            datasets["train"][:, 0],  # type: ignore
            datasets["train"][:, 1],  # type: ignore
            c="tab:blue",
            label="Training Set (Boundary Protected)",
            alpha=0.7,
            s=50,
        )

    if len(datasets["val"]) > 0:
        plt.scatter(
            datasets["val"][:, 0],  # type: ignore
            datasets["val"][:, 1],  # type: ignore
            c="tab:orange",
            label="Validation Set",
            marker="s",
            s=50,
        )

    if len(datasets["test"]) > 0:
        plt.scatter(
            datasets["test"][:, 0],  # type: ignore
            datasets["test"][:, 1],  # type: ignore
            c="tab:green",
            label="Test Set",
            marker="^",
            s=50,
        )

    for simplex in hull.simplices:
        plt.plot(all_points[simplex, 0], all_points[simplex, 1], "k--", lw=1, alpha=0.5)

    plt.title("Dataset Split Strategy: Parameter Space Coverage", fontsize=14)
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Current (A)", fontsize=12)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"figure saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
