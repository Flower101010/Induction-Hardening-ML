import pandas as pd
import numpy as np
import os


def generate_ml_dataset(
    raw_file_path, map_file_path, output_file="final_ml_dataset.parquet"
):
    print(f"1. Loading column map: {map_file_path}")
    df_map = pd.read_csv(map_file_path)

    # --- Step A: Extract metadata indices ---
    # Data is grouped in 3s: Temp, Aust, Mart
    # Extract temperature rows as base time steps
    base_map = df_map[df_map["variable"] == "Temperature"].reset_index(drop=True)

    # 获取每一类变量的列索引 (Excel/CSV中的列号)
    # 注意：你的map里 col_idx 是绝对索引
    idx_temp = df_map[df_map["variable"] == "Temperature"]["col_idx"].values
    idx_aust = df_map[df_map["variable"] == "Austenite"]["col_idx"].values
    idx_mart = df_map[df_map["variable"] == "Martensite"]["col_idx"].values

    # Validate lengths match
    n_steps = len(base_map)
    if not (len(idx_temp) == len(idx_aust) == len(idx_mart)):
        print("Error: Inconsistent column counts for variables, cannot align!")
        return

    # Extract feature vectors (t, f, I) -> Shape: (N_steps, )
    meta_time = base_map["time"].values
    meta_freq = base_map["f_set"].values
    meta_curr = base_map["I_factor"].values

    print(f"   - Detected time/parameter steps: {n_steps}")
    print(
        f"   - Expected output rows: 8192 nodes * {n_steps} steps ~ {8192 * n_steps // 1000000} Million rows"
    )

    # --- Step B: Process raw file in chunks ---
    # Assuming first two columns are coords (r, z)
    coord_cols = [0, 1]

    # Chunk size in rows (nodes)
    # Process 1000 nodes at a time. 1000 * 6400 steps = 6.4M rows.
    chunk_size = 1000

    # Determine header row (header=8 based on previous analysis)
    header_row = 8

    print(f"2. Reading raw data ({raw_file_path})...")
    reader = pd.read_csv(raw_file_path, header=header_row, chunksize=chunk_size)

    # Prepare output file (remove if exists)
    if os.path.exists(output_file):
        os.remove(output_file)

    total_nodes_processed = 0

    for chunk_i, chunk in enumerate(reader):
        print(
            f"\r   Processing chunk {chunk_i + 1} (Processed {total_nodes_processed} nodes)...",
            end="",
        )

        coords = chunk.iloc[:, coord_cols].values
        n_nodes = len(coords)

        data_temp = chunk.iloc[:, idx_temp].values
        data_aust = chunk.iloc[:, idx_aust].values
        data_mart = chunk.iloc[:, idx_mart].values

        # [r1, r1... r2, r2...]
        batch_r = np.repeat(coords[:, 0], n_steps)
        batch_z = np.repeat(coords[:, 1], n_steps)

        # [t1, t2... t1, t2...]
        batch_time = np.tile(meta_time, n_nodes)  # pyright: ignore[reportArgumentType, reportCallIssue]
        batch_freq = np.tile(meta_freq, n_nodes)  # pyright: ignore[reportCallIssue, reportArgumentType]
        batch_curr = np.tile(meta_curr, n_nodes)  # pyright: ignore[reportArgumentType, reportCallIssue]

        flat_temp = data_temp.flatten()
        flat_aust = data_aust.flatten()
        flat_mart = data_mart.flatten()

        df_batch = pd.DataFrame(
            {
                "r": batch_r,
                "z": batch_z,
                "time": batch_time,
                "freq": batch_freq,
                "current": batch_curr,
                "Temperature": flat_temp,
                "Austenite": flat_aust,
                "Martensite": flat_mart,
            }
        )

        if output_file.endswith(".parquet"):
            if not os.path.exists(output_file):
                df_batch.to_parquet(output_file, engine="pyarrow", index=False)
            else:
                part_file = output_file.replace(".parquet", f"_part_{chunk_i}.parquet")
                df_batch.to_parquet(part_file, index=False)
        else:
            header = chunk_i == 0
            df_batch.to_csv(output_file, mode="a", header=header, index=False)

        total_nodes_processed += n_nodes

    print("\n\n[DONE] Data conversion finished!")
    print(f"File saved to: {output_file} (or partitions)")
    print("You can now use this file for ML training.")
    print("-" * 30)
    print("ML Input Features (X): ['r', 'z', 'time', 'freq', 'current']")
    print("ML Targets (Y): ['Temperature', 'Austenite', 'Martensite']")


# ==========================================
# Configuration Paths
# ==========================================
raw_csv = "data/raw/dataset.csv"  # COMSOL 原始大文件
map_csv = "data/analyze/comsol_column_map.csv"  # 刚才生成的映射表
output_name = "data/processed/processed_dataset.csv"  # 输出文件名

if __name__ == "__main__":
    generate_ml_dataset(raw_csv, map_csv, output_name)
