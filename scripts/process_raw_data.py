import pandas as pd
import numpy as np
import os


def generate_ml_dataset(
    raw_file_path, map_file_path, output_file="final_ml_dataset.parquet"
):
    print(f"1. 加载映射表: {map_file_path}")
    df_map = pd.read_csv(map_file_path)

    # --- 步骤 A: 提取元数据索引 ---
    # 我们知道数据是每3列一组：Temp, Aust, Mart
    # 提取 Temperature 对应的行，作为“基准时间步”
    base_map = df_map[df_map["variable"] == "Temperature"].reset_index(drop=True)

    # 获取每一类变量的列索引 (Excel/CSV中的列号)
    # 注意：你的map里 col_idx 是绝对索引
    idx_temp = df_map[df_map["variable"] == "Temperature"]["col_idx"].values
    idx_aust = df_map[df_map["variable"] == "Austenite"]["col_idx"].values
    idx_mart = df_map[df_map["variable"] == "Martensite"]["col_idx"].values

    # 验证长度一致
    n_steps = len(base_map)
    if not (len(idx_temp) == len(idx_aust) == len(idx_mart)):
        print("错误：三种变量的列数不一致，无法对齐！")
        return

    # 提取对应的特征向量 (每个时间步的 t, f, I) -> Shape: (N_steps, )
    meta_time = base_map["time"].values
    meta_freq = base_map["f_set"].values
    meta_curr = base_map["I_factor"].values

    print(f"   - 检测到时间步/参数组合数: {n_steps}")
    print(
        f"   - 预计转换后的总行数: 8192节点 * {n_steps}步 ≈ {8192 * n_steps // 1000000} 百万行"
    )

    # --- 步骤 B: 分块处理原始大文件 ---
    # 假设前两列是坐标 (r, z)，如果不是请修改这里
    coord_cols = [0, 1]

    # 这里的 chunksize 是“行数”（即网格节点数）。
    # 一次处理 1000 个节点。1000 * 6400步 = 640万行输出，内存占用约 500MB，很安全。
    chunk_size = 1000

    # 确定表头行号 (根据你之前的日志，表头在第9行，即 header=8)
    header_row = 8

    print(f"2. 开始读取原始数据 ({raw_file_path})...")
    reader = pd.read_csv(raw_file_path, header=header_row, chunksize=chunk_size)

    # 准备输出文件 (如果存在先删除)
    if os.path.exists(output_file):
        os.remove(output_file)

    total_nodes_processed = 0

    for chunk_i, chunk in enumerate(reader):
        print(
            f"\r   正在处理节点块 {chunk_i + 1} (已处理 {total_nodes_processed} 节点)...",
            end="",
        )

        # 1. 提取坐标 (N_nodes, 2)
        # 假设第0列是r，第1列是z。如果你的CSV里坐标列名包含 '%'，pandas可能读成 '% r'
        # 我们直接按位置取前两列最稳
        coords = chunk.iloc[:, coord_cols].values
        n_nodes = len(coords)

        # 2. 提取三种变量的数据矩阵 (N_nodes, N_steps)
        # 使用 numpy 的切片读取，速度极快
        # 注意：iloc 需要整数位置索引
        data_temp = chunk.iloc[:, idx_temp].values
        data_aust = chunk.iloc[:, idx_aust].values
        data_mart = chunk.iloc[:, idx_mart].values

        # 3. 核心：构建宽变长的输出矩阵
        # 目标：我们需要把 (N_nodes, N_steps) 拉直

        # A. 坐标重复: 每个节点重复 N_steps 次
        # [r1, r1... r2, r2...]
        batch_r = np.repeat(coords[:, 0], n_steps)
        batch_z = np.repeat(coords[:, 1], n_steps)

        # B. 参数重复: 每个节点都经历完整的 meta_time 序列
        # [t1, t2... t1, t2...]
        batch_time = np.tile(meta_time, n_nodes)  # pyright: ignore[reportArgumentType, reportCallIssue]
        batch_freq = np.tile(meta_freq, n_nodes)  # pyright: ignore[reportCallIssue, reportArgumentType]
        batch_curr = np.tile(meta_curr, n_nodes)  # pyright: ignore[reportArgumentType, reportCallIssue]

        # C. 变量拉直: (N_nodes * N_steps)
        # flatten 默认按行优先，正好对应 repeat 坐标
        flat_temp = data_temp.flatten()
        flat_aust = data_aust.flatten()
        flat_mart = data_mart.flatten()

        # 4. 组装 DataFrame
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

        # 5. 写入文件
        # 推荐使用 Parquet 格式，因为它支持分块追加 (append) 且体积小
        # 如果没有安装 pyarrow，可以改用 to_csv (mode='a')
        if output_file.endswith(".parquet"):
            if not os.path.exists(output_file):
                df_batch.to_parquet(output_file, engine="pyarrow", index=False)
            else:
                # Append 模式稍微麻烦点，通常建议写成多个小parquet文件
                # 为了简单，这里我们将每个 chunk 写为一个单独的文件，或者使用 fastparquet 的 append
                # 【简易方案】写成多个文件，后续读取时 dataset 读取文件夹即可
                part_file = output_file.replace(".parquet", f"_part_{chunk_i}.parquet")
                df_batch.to_parquet(part_file, index=False)
        else:
            # CSV 追加模式
            header = chunk_i == 0
            df_batch.to_csv(output_file, mode="a", header=header, index=False)

        total_nodes_processed += n_nodes

    print("\n\n[完成] 数据转换结束！")
    print(f"文件已保存为: {output_file} (或其分块文件)")
    print("现在你可以直接读取这个文件进行机器学习训练了。")
    print("-" * 30)
    print("ML 模型输入特征 (X): ['r', 'z', 'time', 'freq', 'current']")
    print("ML 模型预测目标 (Y): ['Temperature', 'Austenite', 'Martensite']")


# ==========================================
# 配置路径
# ==========================================
raw_csv = "data/raw/dataset.csv"  # COMSOL 原始大文件
map_csv = "data/analyze/comsol_column_map.csv"  # 刚才生成的映射表
output_name = "data/processed/processed_dataset.csv"  # 输出文件名 (建议用 .csv 方便查看，大数据建议 .parquet)

if __name__ == "__main__":
    generate_ml_dataset(raw_csv, map_csv, output_name)
