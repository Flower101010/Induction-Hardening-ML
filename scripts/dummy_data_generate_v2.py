import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os

"""
Improved Dummy Data Generator for Induction Hardening
Improvements over v1:
1. Fixed Boundary Conditions: Removed np.roll wrapping (Surface no longer touches Center).
2. Cylindrical Coordinates: Added 1/r * dT/dr term for radial symmetry.
3. Phase Field: Added a dummy 'Phase' channel (simulating Austenite/Martensite evolution).
   - Logic: If T > Ac3 (threshold), transforms to Austenite.
   - If Austenite cools down, transforms to Martensite (simplified).
"""


def generate_shaft_data_v2(
    num_samples=100, time_steps=50, grid_size=64, seed=42, output_dir="data"
):
    np.random.seed(seed)

    # Output shapes:
    # Input: (nx, ny, 2) -> [Initial_Temp, Power_Map]
    # Output: (time, nx, ny, 2) -> [Temp, Phase]

    data_inputs = []
    data_outputs = []

    print(f"Generating {num_samples} samples (v2) with seed {seed}...")
    os.makedirs(output_dir, exist_ok=True)

    # Physical constants (Normalized)
    Ac3_temp = 0.5  # Austenitization temperature threshold
    Ms_temp = 0.2  # Martensite start temperature

    for _ in tqdm(range(num_samples)):
        power = np.random.uniform(50, 80)
        diffusivity = 0.01

        nx, ny = grid_size, grid_size
        dx = 1.0 / nx
        dt = 0.005

        # Coordinates for cylindrical term
        # x is radius (0 to 1)
        r = np.linspace(dx / 2, 1.0 - dx / 2, nx)  # Avoid r=0 singularity
        r_matrix = r[:, np.newaxis]  # (nx, 1)

        # Fields
        u = np.zeros((nx, ny))  # Temperature
        _phase = np.zeros((nx, ny))  # 0: Ferrite, 1: Austenite, 2: Martensite

        # History tracker for phase
        max_temp_history = np.zeros((nx, ny))
        is_austenite = np.zeros((nx, ny), dtype=bool)

        sample_trajectory = []

        for t in range(time_steps):
            u_new = u.copy()

            # --- 1. Source (Scanning) ---
            source_pos_y = int((t / time_steps) * ny)
            # Correct grid for (nx, ny) where nx=Radius, ny=Length
            X, Y = np.ogrid[:nx, :ny]

            # Skin effect at surface (index nx-1)
            dist_sq = (X - (nx - 1)) ** 2 + (Y - source_pos_y) ** 2
            source = power * np.exp(-dist_sq / 10.0)

            # --- 2. Diffusion (Finite Difference with Fixed Boundaries) ---
            # Laplacian = d2u/dx2 + d2u/dy2
            # Cylindrical: 1/r * du/dx + d2u/dx2 + d2u/dy2

            # Calculate gradients manually to avoid np.roll wrap-around
            # X-direction (Radial)
            d2u_dx2 = np.zeros_like(u)
            d2u_dx2[1:-1, :] = (u[2:, :] - 2 * u[1:-1, :] + u[:-2, :]) / dx**2
            # BCs for Laplacian:
            # Left (Center): Symmetry (u[-1] = u[1]) => d2u/dx2 at 0 = (u[1] - 2u[0] + u[1])/dx2 = 2(u[1]-u[0])/dx2
            d2u_dx2[0, :] = 2 * (u[1, :] - u[0, :]) / dx**2
            # Right (Surface): Insulated (u[n] = u[n-1]) => d2u/dx2 at -1 = (u[-2] - 2u[-1] + u[-2])/dx2 = 2(u[-2]-u[-1])/dx2
            d2u_dx2[-1, :] = 2 * (u[-2, :] - u[-1, :]) / dx**2

            # Y-direction (Axial)
            d2u_dy2 = np.zeros_like(u)
            d2u_dy2[:, 1:-1] = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / dx**2
            # Top/Bottom BCs are Dirichlet (0), so u[-1] and u[n] are 0.
            d2u_dy2[:, 0] = (u[:, 1] - 2 * u[:, 0] + 0) / dx**2
            d2u_dy2[:, -1] = (0 - 2 * u[:, -1] + u[:, -2]) / dx**2

            # Radial term: 1/r * du/dx
            du_dx = np.zeros_like(u)
            du_dx[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dx)
            du_dx[0, :] = 0  # Symmetry at center
            du_dx[-1, :] = 0  # Insulated at surface

            laplacian = d2u_dx2 + d2u_dy2 + (1.0 / r_matrix) * du_dx

            # Update Temp
            u_new += dt * (diffusivity * laplacian + source)

            # --- 3. Quenching ---
            spray_pos_y = source_pos_y - 15
            if 0 <= spray_pos_y < ny:
                mask = (np.abs(Y - spray_pos_y) < 6) & (X > nx - 10)
                u_new[mask] *= 0.7

            # --- 4. Phase Transformation Logic (Dummy) ---
            # Update max temp
            max_temp_history = np.maximum(max_temp_history, u_new)

            # Austenitization
            newly_austenite = u_new > Ac3_temp
            is_austenite = is_austenite | newly_austenite

            # Martensite Transformation (Quenching)
            # If was Austenite AND Temp drops below Ms
            is_martensite = is_austenite & (u_new < Ms_temp)

            # Encode Phase Map
            # 0.0: Base, 0.5: Austenite, 1.0: Martensite
            phase_map = np.zeros_like(u_new)
            phase_map[is_austenite] = 0.5
            phase_map[is_martensite] = 1.0

            u = u_new

            # Stack Temp and Phase
            # Shape: (nx, ny, 2)
            frame = np.stack([u.copy(), phase_map.copy()], axis=-1)
            sample_trajectory.append(frame)

        # Input: Initial state (zeros) + Power
        input_tensor = np.zeros((nx, ny, 2))
        input_tensor[:, :, 1] = power

        # Output: (Time, nx, ny, 2)
        output_tensor = np.array(sample_trajectory)

        data_inputs.append(input_tensor)
        data_outputs.append(output_tensor)

    return np.array(data_inputs), np.array(data_outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="data/processed")
    args = parser.parse_args()

    inputs, outputs = generate_shaft_data_v2(
        num_samples=args.num_samples, output_dir=args.out_dir
    )

    save_x = os.path.join(args.out_dir, "dummy_shaft_v2_train_x.npy")
    save_y = os.path.join(args.out_dir, "dummy_shaft_v2_train_y.npy")
    np.save(save_x, inputs)
    np.save(save_y, outputs)

    print(f"Saved V2 data to {args.out_dir}")
    print("Output shape:", outputs.shape)  # (Samples, Time, nx, ny, 2)

    # Visualization
    fig, axes = plt.subplots(2, 10, figsize=(20, 5))
    sample_idx = 0

    # Plot Temp (Top Row) and Phase (Bottom Row) for 10 steps
    times = np.linspace(0, 49, 10, dtype=int)

    for i, t in enumerate(times):
        # Temp
        axes[0, i].imshow(outputs[sample_idx, t, :, :, 0], cmap="hot", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[0, i].set_title(f"T={t} Temp")

        # Phase
        axes[1, i].imshow(
            outputs[sample_idx, t, :, :, 1], cmap="viridis", vmin=0, vmax=1
        )
        axes[1, i].axis("off")
        axes[1, i].set_title("Phase")

    plt.savefig("outputs/figures/dummy_shaft_v2_sample.png")
    print("Saved visualization to outputs/figures/dummy_shaft_v2_sample.png")


if __name__ == "__main__":
    main()
