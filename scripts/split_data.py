import json
import random
from pathlib import Path

# ================= 配置 =================
DATA_DIR = "data/processed/npy_data"
OUTPUT_JSON = "config/data_split.json"
SEED = 42
TRAIN_RATIO = 0.75
VAL_RATIO = 0.125
# ========================================


def parse_params(filename):
    """解析参数，返回 (freq, current)"""
    try:
        name = filename.replace(".npy", "")
        parts = name.split("_")
        f_val = float(parts[1].replace("f", ""))
        i_val = float(parts[2].replace("i", ""))
        return f_val, i_val
    except Exception:
        print(f"解析失败: {filename}")
        return None, None


def create_smart_split():
    data_path = Path(DATA_DIR)
    all_files = [
        f.name for f in data_path.glob("*.npy") if "geometry_mask" not in f.name
    ]

    total_files = len(all_files)
    if total_files == 0:
        return

    print(f"找到 {total_files} 个文件，开始【智能边界保护】划分...")

    # 1. 解析所有文件的参数，找出边界值
    file_params = []
    freqs = []
    currs = []

    for f in all_files:
        f_val, i_val = parse_params(f)
        if f_val is not None:
            file_params.append({"name": f, "f": f_val, "i": i_val})
            freqs.append(f_val)
            currs.append(i_val)

    # 找出最大最小值 (你的边界)
    min_f, max_f = min(freqs), max(freqs)
    min_i, max_i = min(currs), max(currs)

    print(f"参数范围检测: 频率[{min_f}, {max_f}], 电流[{min_i}, {max_i}]")

    # 2. 分类：是“边界数据”还是“中间数据”？
    boundary_files = []  # 必须进训练集
    middle_files = []  # 可以随机分配

    for item in file_params:
        is_boundary = False
        # 如果达到了任何一个极值，就是边界数据
        if (
            item["f"] == min_f
            or item["f"] == max_f
            or item["i"] == min_i
            or item["i"] == max_i
        ):
            is_boundary = True

        if is_boundary:
            boundary_files.append(item["name"])
        else:
            middle_files.append(item["name"])

    print(f"-> 发现边界文件(强制训练): {len(boundary_files)} 个")
    print(f"-> 发现中间文件(随机分配): {len(middle_files)} 个")

    # 3. 开始分配
    # 目标总数
    n_total_train = int(total_files * TRAIN_RATIO)
    n_total_val = int(total_files * VAL_RATIO)

    # A. 训练集 = 所有边界文件 + 从中间文件里补齐剩余名额
    train_files = list(boundary_files)  # 先把边界全拿走

    # 还需要补充多少个中间数据？
    slots_needed = n_total_train - len(train_files)

    if slots_needed < 0:
        print(
            "警告：边界文件太多，超过了训练集比例！所有边界都将放入训练集，验证集会变少。"
        )
        slots_needed = 0

    # 打乱中间文件
    random.seed(SEED)
    random.shuffle(middle_files)

    # 补齐训练集
    train_files.extend(middle_files[:slots_needed])

    # 剩下的中间文件给验证集和测试集
    remaining = middle_files[slots_needed:]
    val_files = remaining[:n_total_val]
    test_files = remaining[n_total_val:]

    # 4. 打印验证信息（这就是你的定心丸）
    print("-" * 30)
    print(
        f"最终划分结果: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}"
    )

    # 检查测试集里有没有混入边界（理论上不应该有）
    print("【测试集质量检查】:")
    for f in test_files:
        pf, pi = parse_params(f)
        is_bad = pf == min_f or pf == max_f or pi == min_i or pi == max_i
        status = "❌ 危险边界!" if is_bad else "✅ 安全中间值"
        print(f"  {f} -> {status}")

    # 5. 保存
    split_dict = {"train": train_files, "val": val_files, "test": test_files}
    with open(OUTPUT_JSON, "w") as f:
        json.dump(split_dict, f, indent=2)
    print(f"配置已保存至 {OUTPUT_JSON}")


if __name__ == "__main__":
    create_smart_split()
