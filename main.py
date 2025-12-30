"""
Main Entry Point
================

This script serves as the unified CLI for the Induction Hardening ML project.
It wraps the various scripts in `scripts/` and `src/` to provide a convenient way to run the pipeline.

Usage:
    python main.py <command> [arguments]

Available commands:
    prepare-data    Run the data preparation pipeline
    train           Train the model
    evaluate        Evaluate the model
    visualize       Visualize simulation results
    plot-loss       Plot training loss curve
    plot-paper      Generate publication-quality figures
"""

import argparse
import subprocess
import sys


def run_command(command_args):
    """Run a command using subprocess."""
    try:
        # Use sys.executable to ensure we use the current python interpreter
        cmd = [sys.executable] + command_args
        print(f"🚀 Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Induction Hardening ML Pipeline Manager"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Data Preparation ---
    prep_parser = subparsers.add_parser(
        "prepare-data", help="Run the full data preparation pipeline"
    )
    prep_parser.add_argument(
        "--step",
        choices=["all", "analyze", "process", "preprocess", "split"],
        default="all",
        help="Specific step to run (default: all)",
    )

    # --- Training ---
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--config", default="config/model_config.yaml", help="Path to config file"
    )

    # --- Evaluation ---
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument(
        "--config", default="config/model_config.yaml", help="Path to config file"
    )
    eval_parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint"
    )
    eval_parser.add_argument(
        "--output",
        default="outputs/logs/eval_metrics.json",
        help="Path to output metrics file",
    )

    # --- Visualization ---
    viz_parser = subparsers.add_parser("visualize", help="Visualize simulation results")
    viz_parser.add_argument("--data", required=True, help="Path to .npy data file")
    viz_parser.add_argument(
        "--mode",
        choices=["gif", "snapshot", "compare", "all"],
        default="all",
        help="Visualization mode",
    )
    viz_parser.add_argument(
        "--checkpoint", help="Path to model checkpoint (required for compare mode)"
    )
    viz_parser.add_argument(
        "--output", default="outputs/figures", help="Output directory"
    )
    viz_parser.add_argument("--fps", type=int, default=10, help="GIF FPS")
    viz_parser.add_argument(
        "--snapshot_time", type=float, default=5.0, help="Snapshot time"
    )
    viz_parser.add_argument(
        "--duration", type=float, default=10.0, help="Total duration of simulation in seconds"
    )
    viz_parser.add_argument(
        "--animate", action="store_true", help="Generate comparison animation"
    )

    # --- Plotting ---
    plot_loss_parser = subparsers.add_parser("plot-loss", help="Plot training loss")
    plot_loss_parser.add_argument(
        "--log", default="outputs/logs/loss_history.json", help="Path to loss log file"
    )
    plot_loss_parser.add_argument(
        "--out", default="outputs/figures/paper_v2", help="Output directory"
    )

    plot_paper_parser = subparsers.add_parser(
        "plot-paper", help="Generate paper figures"
    )
    plot_paper_parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint"
    )
    plot_paper_parser.add_argument(
        "--output_dir", default="outputs/figures/paper_v2", help="Output directory"
    )

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Dispatch commands
    if args.command == "prepare-data":
        steps = []
        if args.step == "all":
            steps = ["analyze", "process", "preprocess", "split"]
        else:
            steps = [args.step]

        if "analyze" in steps:
            run_command(["scripts/analyze_structure.py"])
        if "process" in steps:
            run_command(["scripts/process_raw_data.py"])
        if "preprocess" in steps:
            run_command(["src/data/preprocessor.py"])
        if "split" in steps:
            run_command(["scripts/split_data.py"])

    elif args.command == "train":
        run_command(["scripts/train.py", "--config", args.config])

    elif args.command == "evaluate":
        cmd = [
            "scripts/evaluate.py",
            "--config",
            args.config,
            "--checkpoint",
            args.checkpoint,
            "--output",
            args.output,
        ]
        run_command(cmd)

    elif args.command == "visualize":
        cmd = [
            "scripts/visualize.py",
            "--data",
            args.data,
            "--mode",
            args.mode,
            "--output",
            args.output,
            "--fps",
            str(args.fps),
            "--snapshot_time",
            str(args.snapshot_time),
            "--duration",
            str(args.duration),
        ]
        if args.checkpoint:
            cmd.extend(["--checkpoint", args.checkpoint])
        if args.animate:
            cmd.append("--animate")
        run_command(cmd)

    elif args.command == "plot-loss":
        run_command(["scripts/plot_loss.py", "--log", args.log, "--out", args.out])

    elif args.command == "plot-paper":
        run_command(
            [
                "scripts/plot_paper_figures.py",
                "--checkpoint",
                args.checkpoint,
                "--output_dir",
                args.output_dir,
            ]
        )


if __name__ == "__main__":
    main()
