import json
import matplotlib.pyplot as plt
import os
import argparse


def plot_loss(log_file, output_dir):
    if not os.path.exists(log_file):
        print(f"Error: Log file not found at {log_file}")
        return

    with open(log_file, "r") as f:
        history = json.load(f)

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_loss, label="Validation Loss", linewidth=2, linestyle="--")

    plt.yscale("log")  # Set y-axis to logarithmic scale

    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Log Scale)")
    plt.legend()
    plt.grid(
        True, alpha=0.3, which="both"
    )  # 'both' adds grid lines for minor ticks in log scale

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(save_path, dpi=300)
    print(f"Loss curve saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="outputs/logs/loss_history.json")
    parser.add_argument("--out", type=str, default="outputs/figures/paper_v2")
    args = parser.parse_args()

    plot_loss(args.log, args.out)
