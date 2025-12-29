import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Visualizer:
    """
    Helper class for visualizing Induction Hardening simulation results.
    Handles 2D reconstruction, plotting, and animation.
    """

    def __init__(self, output_dir, stats=None, mask=None):
        """
        Args:
            output_dir (str): Directory to save outputs.
            stats (dict, optional): Normalization statistics for denormalization.
            mask (np.ndarray, optional): Geometry mask.
        """
        self.output_dir = output_dir
        self.stats = stats
        self.mask = mask
        os.makedirs(self.output_dir, exist_ok=True)

    def _apply_mask(self, data):
        """Apply geometry mask to data (set masked values to NaN)."""
        if self.mask is None:
            return data

        # Ensure mask is on the same device/type if needed, but here we assume numpy
        masked_data = data.copy()
        # Expand mask to match data shape if necessary (assuming data is ..., H, W)
        # Mask is (H, W)
        if masked_data.shape[-2:] != self.mask.shape:
            # Try to flip mask if it doesn't match (common issue in this project)
            pass

        # Broadcast mask
        mask_expanded = self.mask
        while mask_expanded.ndim < masked_data.ndim:
            mask_expanded = np.expand_dims(mask_expanded, 0)

        # Use np.where for broadcasting support
        # masked_data[mask_expanded == 0] = np.nan  # This fails if shapes don't match exactly
        return np.where(mask_expanded == 0, np.nan, masked_data)

    def _reconstruct_full_2d(self, quarter_data):
        """
        Reconstruct full 2D cross-section from 1/4 data using symmetry.
        Input shape: (H, W)
        Output shape: (2*H, 2*W)
        """
        H, W = quarter_data.shape
        full = np.zeros((2 * H, 2 * W))

        # Upper right (original)
        full[:H, W:] = quarter_data
        # Upper left (mirror horizontal)
        full[:H, :W] = np.flip(quarter_data, axis=1)
        # Lower right (mirror vertical)
        full[H:, W:] = np.flip(quarter_data, axis=0)
        # Lower left (mirror both)
        full[H:, :W] = np.flip(np.flip(quarter_data, axis=0), axis=1)

        return full

    def _denormalize(self, data, channel_name):
        """Denormalize data based on channel name."""
        if self.stats is None:
            return data, 0, 1

        if channel_name == "Temperature":
            vmin = self.stats.get("temp_min", 0)
            vmax = self.stats.get("temp_max", 1)
            return data * (vmax - vmin) + vmin, vmin, vmax

        return data, 0, 1

    def create_animation(self, data, channel_idx, title, filename, fps=10, cmap="hot"):
        """
        Create animated GIF for a single channel.

        Args:
            data: (T, C, H, W)
            channel_idx: Channel index to visualize.
            title: Plot title.
            filename: Output filename.
        """
        T, C, H, W = data.shape
        channel_data = data[:, channel_idx, :, :]

        # Determine channel name for denormalization
        channel_names = {0: "Temperature", 1: "Austenite", 2: "Martensite"}
        name = channel_names.get(channel_idx, "Unknown")

        # Denormalize for colorbar limits
        _, vmin, vmax = self._denormalize(channel_data[0], name)

        # If Temperature, we want to denormalize the data itself for the plot
        if name == "Temperature" and self.stats:
            channel_data, _, _ = self._denormalize(channel_data, name)

        fig, ax = plt.subplots(figsize=(8, 8))

        # Initial frame
        frame_data = self._reconstruct_full_2d(channel_data[0])
        extent = [-W, W, -H, H]

        im = ax.imshow(
            frame_data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            origin="lower",
            extent=extent,  # type: ignore
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(title)

        ax.set_xlabel("Radial Position (r)")
        ax.set_ylabel("Axial Position (z)")
        time_text = ax.set_title(f"{title} - t = 0.00 s")

        # Symmetry lines
        ax.axhline(y=0, color="white", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color="white", linestyle="--", linewidth=0.5, alpha=0.5)

        def update(frame_idx):
            frame = self._reconstruct_full_2d(channel_data[frame_idx])
            im.set_array(frame)
            # Assuming 10s total time
            time = frame_idx * 10.0 / (T - 1)
            time_text.set_text(f"{title} - t = {time:.2f} s")
            return [im, time_text]

        anim = animation.FuncAnimation(
            fig, update, frames=T, interval=1000 / fps, blit=True
        )

        save_path = os.path.join(self.output_dir, filename)
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer="pillow", fps=fps)
        plt.close(fig)

    def plot_snapshot(self, data, time_idx, filename_prefix):
        """Plot snapshots for all channels at a specific time."""
        T, C, H, W = data.shape

        channels = [
            (0, "Temperature", "hot"),
            (1, "Austenite", "YlOrRd"),
            (2, "Martensite", "Blues"),
        ]

        for ch_idx, name, cmap in channels:
            channel_data = data[time_idx, ch_idx]

            # Denormalize
            plot_data, vmin, vmax = self._denormalize(channel_data, name)

            fig, ax = plt.subplots(figsize=(8, 8))
            frame = self._reconstruct_full_2d(plot_data)
            extent = [-W, W, -H, H]

            im = ax.imshow(
                frame,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
                origin="lower",
                extent=extent,  # type: ignore
            )

            plt.colorbar(im, ax=ax, label=name)
            ax.set_xlabel("Radial Position (r)")
            ax.set_ylabel("Axial Position (z)")

            time = time_idx * 10.0 / (T - 1)
            ax.set_title(f"{name} - t = {time:.2f} s")

            output_path = os.path.join(
                self.output_dir, f"{filename_prefix}_{name.lower()}.png"
            )
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved snapshot: {output_path}")

    def plot_comparison(self, gt_data, pred_data, time_idx, filename_prefix):
        """Plot comparison between Ground Truth and Prediction."""
        T, C, H, W = gt_data.shape
        channels = [
            (0, "Temperature", "hot"),
            (1, "Austenite", "YlOrRd"),
            (2, "Martensite", "Blues"),
        ]

        for ch_idx, name, cmap in channels:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            gt = gt_data[time_idx, ch_idx]
            pred = pred_data[time_idx, ch_idx]

            # Denormalize
            gt_plot, vmin, vmax = self._denormalize(gt, name)
            pred_plot, _, _ = self._denormalize(pred, name)

            diff = np.abs(gt_plot - pred_plot)

            # Reconstruct
            gt_full = self._reconstruct_full_2d(gt_plot)
            pred_full = self._reconstruct_full_2d(pred_plot)
            diff_full = self._reconstruct_full_2d(diff)

            extent = [-W, W, -H, H]

            # Plot GT
            im0 = axes[0].imshow(
                gt_full, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", extent=extent
            )
            axes[0].set_title(f"Ground Truth - {name}")
            plt.colorbar(im0, ax=axes[0])

            # Plot Pred
            im1 = axes[1].imshow(
                pred_full,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                origin="lower",
                extent=extent,
            )
            axes[1].set_title(f"Prediction - {name}")
            plt.colorbar(im1, ax=axes[1])

            # Plot Diff
            im2 = axes[2].imshow(diff_full, cmap="Reds", origin="lower", extent=extent)
            axes[2].set_title(f"Absolute Error - {name}")
            plt.colorbar(im2, ax=axes[2])

            time = time_idx * 10.0 / (T - 1)
            fig.suptitle(f"{name} Comparison - t = {time:.2f} s")

            output_path = os.path.join(
                self.output_dir, f"{filename_prefix}_{name.lower()}.png"
            )
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved comparison: {output_path}")

    def create_comparison_animation(self, gt_data, pred_data, filename_prefix, fps=10):
        """
        Create animated GIF for comparison (GT vs Pred vs Error).
        """
        T, C, H, W = gt_data.shape
        channels = [
            (0, "Temperature", "hot"),
            (1, "Austenite", "YlOrRd"),
            (2, "Martensite", "Blues"),
        ]

        for ch_idx, name, cmap in channels:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Pre-calculate denormalization params from first frame
            _, vmin, vmax = self._denormalize(gt_data[0, ch_idx], name)
            extent = [-W, W, -H, H]

            # Initial frame setup
            ims = []
            for ax in axes:
                im = ax.imshow(
                    np.zeros((2 * H, 2 * W)),
                    origin="lower",
                    extent=extent,
                    aspect="equal",
                )
                ims.append(im)

            # Set colormaps and limits
            ims[0].set_cmap(cmap)
            ims[0].set_clim(vmin, vmax)
            ims[1].set_cmap(cmap)
            ims[1].set_clim(vmin, vmax)
            ims[2].set_cmap("Reds")
            # Error limit is dynamic, but let's fix it to a reasonable range or let it auto-scale per frame?
            # For animation, fixed scale is better. Let's assume error is small, e.g., 10% of max range
            err_max = (vmax - vmin) * 0.2 if name == "Temperature" else 0.2
            ims[2].set_clim(0, err_max)

            # Titles and Colorbars
            axes[0].set_title(f"Ground Truth - {name}")
            plt.colorbar(ims[0], ax=axes[0])

            axes[1].set_title(f"Prediction - {name}")
            plt.colorbar(ims[1], ax=axes[1])

            axes[2].set_title(f"Absolute Error - {name}")
            plt.colorbar(ims[2], ax=axes[2])

            fig.suptitle(f"{name} Comparison", fontsize=16)
            time_text = axes[1].text(
                0.5,
                1.15,
                "t = 0.00 s",
                transform=axes[1].transAxes,
                ha="center",
                fontsize=14,
            )

            def update(frame_idx):
                gt = gt_data[frame_idx, ch_idx]
                pred = pred_data[frame_idx, ch_idx]

                gt_plot, _, _ = self._denormalize(gt, name)
                pred_plot, _, _ = self._denormalize(pred, name)
                diff = np.abs(gt_plot - pred_plot)

                gt_full = self._reconstruct_full_2d(gt_plot)
                pred_full = self._reconstruct_full_2d(pred_plot)
                diff_full = self._reconstruct_full_2d(diff)

                ims[0].set_array(gt_full)
                ims[1].set_array(pred_full)
                ims[2].set_array(diff_full)

                time = frame_idx * 10.0 / (T - 1)
                time_text.set_text(f"t = {time:.2f} s")

                return ims + [time_text]

            anim = animation.FuncAnimation(
                fig, update, frames=T, interval=1000 / fps, blit=False
            )

            output_path = os.path.join(
                self.output_dir, f"{filename_prefix}_{name.lower()}.gif"
            )
            print(f"Saving comparison animation to {output_path}...")
            anim.save(output_path, writer="pillow", fps=fps)
            plt.close(fig)
