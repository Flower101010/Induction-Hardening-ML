import os
import numpy as np
import matplotlib.pyplot as plt

# 配置路径
MASK_FILE = "data/processed/npy_data/geometry_mask.npy"
OUTPUT_FILE = "outputs/figures/geometry_mask.png"


def main():
    if not os.path.exists(MASK_FILE):
        print(
            "error: Mask file does not exist. Please run the data preprocessing script first."
        )
        return

    try:
        mask = np.load(MASK_FILE)
        print(f"Mask file loaded successfully, shape: {mask.shape}")

        if mask.ndim == 3:
            mask = mask[0]

        plt.figure(figsize=(6, 8))

        im = plt.imshow(mask, cmap="gray", origin="lower", interpolation="nearest")

        plt.title("Geometry Mask Visualization\n(White=Steel, Black=Air)", fontsize=14)
        plt.xlabel("r (Radial Coordinate)", fontsize=12)
        plt.ylabel("z (Axial Coordinate)", fontsize=12)

        cbar = plt.colorbar(im, label="Mask Value")
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["0 (Air)", "1 (Steel)"])

        plt.tight_layout()

        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        plt.savefig(OUTPUT_FILE, dpi=300)
        print(f"Geometry mask figure saved to: {OUTPUT_FILE}")

    except Exception as e:
        print(f"error occurred while plotting: {e}")


if __name__ == "__main__":
    main()
