import json
import random
from pathlib import Path

# ================= 配置 =================
DATA_DIR = "data/processed/npy_data"  # 你的 .npy 文件夹
OUTPUT_JSON = "config/data_split.json"
SEED = 42  # 固定随机种子，保证每次划分结果一样
TRAIN_RATIO = 0.75
VAL_RATIO = 0.125
# Test Ratio = 1 - Train - Val
# ========================================


def parse_params(filename):
    """
    从文件名解析参数，例如 'sim_f50000_i1.00.npy' -> (50000, 1.0)
    用于后续分析划分是否均匀（可选）
    """
    try:
        # 去掉 .npy
        name = filename.replace(".npy", "")
        # 简单分割
        parts = name.split("_")
        f_val = float(parts[1].replace("f", ""))
        i_val = float(parts[2].replace("i", ""))
        return f_val, i_val
    except Exception as e:
        print(f"无法解析参数从文件名: {filename}，错误: {e}")
        return None, None


def create_split():
    # 1. 获取所有数据文件
    data_path = Path(DATA_DIR)
    all_files = [
        f.name for f in data_path.glob("*.npy") if "geometry_mask" not in f.name
    ]

    total_files = len(all_files)
    if total_files == 0:
        print(f"错误：在 {DATA_DIR} 没找到 .npy 文件")
        return

    print(f"找到 {total_files} 个仿真案例。")

    # 2. 随机打乱
    random.seed(SEED)
    random.shuffle(all_files)

    # 3. 计算数量
    n_train = int(total_files * TRAIN_RATIO)
    n_val = int(total_files * VAL_RATIO)

    # 4. 切片划分
    train_files = all_files[:n_train]
    val_files = all_files[n_train : n_train + n_val]
    test_files = all_files[n_train + n_val :]

    # --- 进阶检查：确保边界值（最大最小频率/电流）在训练集中 ---
    # 这是一个简单的优化策略：如果测试集里包含了极值，尝试交换一下
    # (为了简化，这里先不做强制交换，只做打印提示)

    print("-" * 30)
    print(f"训练集: {len(train_files)} 个")
    print(f"验证集: {len(val_files)} 个")
    print(f"测试集: {len(test_files)} 个")

    # 打印测试集的参数，让你确认它们是否在参数空间的“中间”
    print("-" * 30)
    print("【测试集案例预览】(模型将从未见过这些参数):")
    for f in test_files:
        p_f, p_i = parse_params(f)
        print(f"  - Freq: {p_f}, Curr: {p_i} ({f})")

    # 5. 保存索引
    split_dict = {"train": train_files, "val": val_files, "test": test_files}

    with open(OUTPUT_JSON, "w") as f:
        json.dump(split_dict, f, indent=2)

    print("-" * 30)
    print(f"划分索引已保存至: {OUTPUT_JSON}")
    print("训练 PyTorch Dataset 时请读取此 JSON 加载对应文件。")


if __name__ == "__main__":
    create_split()
