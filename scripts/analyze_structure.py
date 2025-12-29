import pandas as pd
import re
import csv
import io


def parse_specific_comsol_header(file_path):
    print(f"正在读取文件头: {file_path}")

    header_line = ""
    # 1. 寻找表头 (包含 'f_set=' 和 '@' 的行)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if "f_set=" in line and "@" in line:
                header_line = line.strip()
                print(f"✅ 在第 {i + 1} 行找到表头。")
                break

    if not header_line:
        print("❌ 未找到符合格式的表头。")
        return None

    # 2. 分割列
    reader = csv.reader(io.StringIO(header_line), delimiter=",")
    columns = next(reader)
    print(f"总列数: {len(columns)}")

    parsed_data = []
    coord_cols = []

    # 3. 针对性正则匹配
    # 目标格式: T (°C) @ t=0; f_set=50000; I_factor=0.85
    # 提取逻辑:
    #   Group 1 (变量):  (.*?)             -> T (°C)
    #   Group 2 (时间):  t=([0-9.E+-]+)    -> 0
    #   Group 3 (频率):  f_set=([0-9.E+-]+) -> 50000
    #   Group 4 (电流):  I_factor=([0-9.E+-]+) -> 0.85

    pattern = re.compile(
        r"(.*?)@\s*t=([0-9.E+-]+);\s*f_set=([0-9.E+-]+);\s*I_factor=([0-9.E+-]+)"
    )

    for idx, col_str in enumerate(columns):
        col_clean = col_str.strip()

        # 尝试匹配参数列
        match = pattern.search(col_clean)
        if match:
            var_raw = match.group(1).strip()
            t_val = float(match.group(2))
            f_val = float(match.group(3))
            i_val = float(match.group(4))

            # 变量名标准化
            if "T" in var_raw:
                var_name = "Temperature"
            elif "phase1" in var_raw:
                var_name = "Martensite"  # 假设
            elif "phase5" in var_raw:
                var_name = "Austenite"  # 假设
            else:
                var_name = var_raw

            parsed_data.append(
                {
                    "col_idx": idx,
                    "variable": var_name,
                    "time": t_val,
                    "f_set": f_val,
                    "I_factor": i_val,
                }
            )
        else:
            # 记录坐标列 (% r, z 等)
            if idx < 5:  # 通常坐标在前几列
                coord_cols.append({"col_idx": idx, "name": col_clean})

    # 4. 保存结果
    df_map = pd.DataFrame(parsed_data)

    if not df_map.empty:
        print("-" * 50)
        print(f"解析成功！捕获参数列: {len(df_map)}")
        print(f"坐标列: {[c['name'] for c in coord_cols]}")
        print(f"变量: {df_map['variable'].unique()}")
        print(f"时间步数: {df_map['time'].nunique()}")
        print(f"频率组 (f_set): {df_map['f_set'].nunique()}")
        print(f"电流组 (I_factor): {df_map['I_factor'].nunique()}")

        # 保存映射表
        df_map.to_csv("data/comsol_column_map.csv", index=False)
        print("\n✅ 映射表已保存为 'data/comsol_column_map.csv'")

        # 保存坐标列索引
        pd.DataFrame(coord_cols).to_csv("data/comsol_coord_map.csv", index=False)
    else:
        print("❌ 解析失败，正则未匹配到任何列。")

    return df_map


# ============================
file_path = "data/raw/training_data.csv"  # 请确保路径正确
# ============================

if __name__ == "__main__":
    parse_specific_comsol_header(file_path)
