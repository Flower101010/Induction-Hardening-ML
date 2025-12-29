import pandas as pd
import re
import os

# ==========================================
# 用户配置区
# ==========================================
file_path = "data/raw/training_data.csv"  #
# ==========================================


def analyze_comsol_structure(filepath):
    print(f"正在分析文件: {filepath} ...\n")

    # --- 步骤 1: 纯文本探测 (寻找表头行) ---
    header_line_index = -1
    header_raw = ""
    comment_lines_count = 0

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            # 只读前 50 行找规律，避免读取大文件
            for i in range(50):
                line = f.readline()
                if not line:
                    break

                if line.strip().startswith("%"):
                    comment_lines_count += 1
                    # COMSOL 的表头通常是最后一个带 % 的行
                    header_line_index = i
                    header_raw = line.strip()
                else:
                    # 一旦遇到没有 % 的行，说明数据开始了
                    break
    except FileNotFoundError:
        print("❌ 错误: 找不到文件，请检查路径。")
        return

    print("=== 1. 文件元数据分析 ===")
    if header_line_index != -1:
        print(f"✅ 找到 COMSOL 注释头，共 {comment_lines_count} 行")
        print(f"✅ 表头位于第 {header_line_index + 1} 行")
        print(
            f"ℹ️ 原始表头内容: {header_raw[:100]}..."
            + (" (内容过长已截断)" if len(header_raw) > 100 else "")
        )
    else:
        print("⚠️ 未找到以 '%' 开头的标准 COMSOL 表头，尝试作为普通 CSV 读取。")

    # --- 步骤 2: Pandas 采样读取 (只读 5 行) ---
    print("\n=== 2. 数据结构采样 (只读前 5 行) ===")
    try:
        # 如果找到了表头，用 header=None 读取，因为我们自己处理列名会更灵活
        # skiprows 跳过除最后一行注释外的所有注释
        skip_rows = range(header_line_index) if header_line_index > 0 else None

        # 尝试读取
        df_sample = pd.read_csv(
            filepath,
            skiprows=skip_rows,
            nrows=5,
            header=None if header_line_index != -1 else "infer",
        )

        # 如果是 COMSOL 格式，第一行通常包含 %，需要清理
        if header_line_index != -1:
            # 获取读取进来的第一行作为列名
            raw_columns = df_sample.iloc[0].astype(str).tolist()
            # 清理列名中的 % 和空格
            clean_columns = [col.replace("%", "").strip() for col in raw_columns]
            df_sample.columns = clean_columns
            df_sample = df_sample[1:].reset_index(drop=True)  # 去掉变成列名的那一行数据

        num_rows_sample, num_cols = df_sample.shape
        print(f"📊 列总数: {num_cols} 列")
        print("   (如果这是参数化扫描，列数通常 = 坐标列数 + 变量数 * 参数组数)")

    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    # --- 步骤 3: 列名模式识别 ---
    print("\n=== 3. 列内容智能识别 ===")

    cols = df_sample.columns.tolist()

    # 1. 识别坐标列 (通常是 x, y, z, r, phi 等)
    coord_cols = [c for c in cols if c.lower() in ["x", "y", "z", "r", "phi", "theta"]]
    print(f"📍 坐标列 ({len(coord_cols)} 个): {coord_cols}")

    # 2. 识别参数数据列 (包含 @, =, freq, time 等特征)
    # COMSOL 典型格式: "Temperature (K) @ t=0.1" 或 "B_z @ freq=50"
    data_cols = [c for c in cols if c not in coord_cols]

    if len(data_cols) > 0:
        example_col = data_cols[0]
        print(f"📉 数据列 ({len(data_cols)} 个)")
        print(f"   示例列名: '{example_col}'")

        # 尝试解析参数
        # 匹配规则: 找 = 后面的数字
        match = re.search(r"=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", example_col)
        if match:
            param_val = match.group(1)
            print(f"   ✅ 成功从示例列名中提取出参数值: {param_val}")
            print("   🧠 推测: 这是一个宽表，每一列对应一个参数步。")
            if len(data_cols) % 64 == 0:
                print("   🔍 发现数据列数是 64 的倍数，与你提到的 '64组数据' 吻合！")
        else:
            print("   ⚠️ 无法自动从列名提取参数，可能列名格式较特殊，或者没有参数标签。")

    else:
        print("⚠️ 没有检测到数据列，请检查文件内容。")

    # --- 步骤 4: 内存估算 ---
    # 假设 float64 占 8 bytes
    # 我们不知道总行数，但可以通过文件大小估算
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print("\n=== 4. 资源估算 ===")
    print(f"💾 文件大小: {file_size_mb:.2f} MB")

    if file_size_mb > 1000:
        print(
            "🚨 文件超过 1GB，建议使用分块处理 (Chunking) 或 Dask，不要一次性读入 Pandas。"
        )
    elif file_size_mb > 200:
        print("⚠️ 文件较大，处理时请留意内存。")
    else:
        print("✅ 文件大小适中，可以直接加载。")


# 运行分析
if __name__ == "__main__":
    analyze_comsol_structure(file_path)
