import pandas as pd
import re
import os

# ================= 用户配置区 =================
# 输入文件路径 (请修改为你实际的文件名)
INPUT_FILE = "data/raw/training_data.csv"

# 输出文件夹路径 (脚本会自动创建)
OUTPUT_DIR = "data/processed_parquet"

# 每次处理的行数 (2000行是一个在速度和内存之间很好的平衡点)
CHUNK_SIZE = 2000
# ============================================


def extract_metadata_from_header(header_line):
    """
    解析 COMSOL 复杂的表头，生成列名到参数的映射字典。
    """
    # 去掉开头的 %，按逗号分割，并去除首尾空格
    raw_cols = [c.strip() for c in header_line.replace("%", "").strip().split(",")]

    meta_map = {}

    # 定义坐标列 (通常前两列是坐标，根据你的数据调整)
    coord_cols = ["r", "z"]

    print(f"正在解析 {len(raw_cols)} 个列名信息，请稍候...")

    for col in raw_cols:
        if col in coord_cols:
            continue

        # --- 正则表达式提取核心逻辑 ---
        # 1. 提取物理量名称 (截取 @ 符号前面的部分)
        # 例如 "T (degC) @ ..." -> "T"
        if "@" in col:
            phys_name_part = col.split("@")[0].strip()
        else:
            # 应对某些没写 @ 的异常情况，直接用整个列名
            phys_name_part = col

        # 去掉括号内的单位，例如 "T (degC)" -> "T"
        phys_name = re.sub(r"\s*.∗?.*?.∗?", "", phys_name_part).strip()

        # 2. 提取参数数值 (支持整数、小数、科学计数法)
        # 查找 t=..., f_set=..., I_factor=...
        t_match = re.search(r"t=\s*([-+]?[\d\.eE]+)", col)
        f_match = re.search(r"f_set=\s*([-+]?[\d\.eE]+)", col)
        i_match = re.search(r"I_factor=\s*([-+]?[\d\.eE]+)", col)

        meta_map[col] = {
            "variable": phys_name,  # 物理量名称 (T, audc.phase5.xi 等)
            "t": float(t_match.group(1)) if t_match else 0.0,
            "f": float(f_match.group(1)) if f_match else 0.0,
            "I": float(i_match.group(1)) if i_match else 0.0,
        }

    return raw_cols, coord_cols, meta_map


def process_big_csv():
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")

    # --- 第一步：稳健地查找并读取表头 ---
    print("Step 1: 扫描文件表头...")

    last_comment = None
    header_row_index = -1

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            # 扫描前 100 行寻找表头
            for i in range(100):
                line = f.readline()
                if not line:
                    break

                # 记录最后一行以 % 开头的行
                if line.strip().startswith("%"):
                    last_comment = line
                    header_row_index = i

                # 如果遇到非注释且非空的行，说明注释区结束
                if not line.strip().startswith("%") and line.strip():
                    break
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 '{INPUT_FILE}'")
        return

    # 检查是否找到表头
    if last_comment is None:
        print("❌ 错误: 未找到以 '%' 开头的表头行。请检查 CSV 格式。")
        return

    print(f"✅ 找到表头 (位于第 {header_row_index + 1} 行)")

    # 解析表头元数据
    raw_cols, coord_cols, meta_map = extract_metadata_from_header(last_comment)
    print(f"✅ 解析完成！将处理 {len(meta_map)} 个数据变量列。")

    # --- 第二步：分块读取并清洗 ---
    print(f"\nStep 2: 开始分块转换 (Chunk Size: {CHUNK_SIZE})...")

    # 使用 Pandas 读取，跳过前面的注释行
    # 注意：names=raw_cols 强制指定列名，避免 pandas 再次去读表头
    reader = pd.read_csv(
        INPUT_FILE, comment="%", header=None, names=raw_cols, chunksize=CHUNK_SIZE
    )

    chunk_id = 0
    total_rows_processed = 0

    for chunk in reader:
        print(f"   >>> 正在处理第 {chunk_id + 1} 块...", end="\r")

        # 1. 宽表转长表 (Melt)
        # 将 [r, z, T@t1, T@t2...] 转换为 [r, z, 原列名, 数值]
        melted = pd.melt(
            chunk, id_vars=coord_cols, var_name="original_col", value_name="value"
        )

        # 2. 映射元数据
        # 将 original_col 替换为具体的 t, f, I, variable
        # 为了性能，先将 meta_map 转换为 DataFrame 进行 Merge
        meta_df = pd.DataFrame.from_dict(meta_map, orient="index")
        meta_df.index.name = "original_col"

        # 合并数据
        processed_chunk = melted.merge(meta_df, on="original_col", how="left")

        # 3. 清理不需要的列
        processed_chunk.drop(columns=["original_col"], inplace=True)

        # 4. 数据类型优化 (Float64 -> Float32)
        # 这对于 ML 至关重要，能节省 50% 内存
        cols_to_float32 = ["t", "f", "I", "value"] + coord_cols
        for col in cols_to_float32:
            if col in processed_chunk.columns:
                processed_chunk[col] = processed_chunk[col].astype("float32")

        # 5. 保存为 Parquet 小文件
        save_name = f"{OUTPUT_DIR}/part_{chunk_id:04d}.parquet"
        processed_chunk.to_parquet(save_name, index=False)

        total_rows_processed += len(chunk)
        chunk_id += 1

    print("\n\n✅ 全部完成！")
    print(f"📁 清洗后的数据已保存在: {OUTPUT_DIR}/")
    print(
        f"🧠 接下来的建议: 使用 pd.read_parquet('{OUTPUT_DIR}') 即可读取整个数据集用于训练。"
    )


if __name__ == "__main__":
    process_big_csv()
