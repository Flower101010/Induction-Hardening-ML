import json
import random
from pathlib import Path

# ================= Configuration =================
DATA_DIR = "data/processed/npy_data"
OUTPUT_JSON = "config/data_split.json"
SEED = 42
TRAIN_RATIO = 0.75
VAL_RATIO = 0.125
# ========================================


def parse_params(filename):
    """Parse parameters, return (freq, current)"""
    try:
        name = filename.replace(".npy", "")
        parts = name.split("_")
        f_val = float(parts[1].replace("f", ""))
        i_val = float(parts[2].replace("i", ""))
        return f_val, i_val
    except Exception:
        print(f"Parsing failed: {filename}")
        return None, None


def create_smart_split():
    data_path = Path(DATA_DIR)
    all_files = [
        f.name for f in data_path.glob("*.npy") if "geometry_mask" not in f.name
    ]

    total_files = len(all_files)
    if total_files == 0:
        return

    print(f"Found {total_files} files, starting [Smart Boundary Protection] split...")

    # 1. Parse parameters of all files to find boundary values
    file_params = []
    freqs = []
    currs = []

    for f in all_files:
        f_val, i_val = parse_params(f)
        if f_val is not None:
            file_params.append({"name": f, "f": f_val, "i": i_val})
            freqs.append(f_val)
            currs.append(i_val)

    # Find min/max values (your boundaries)
    min_f, max_f = min(freqs), max(freqs)
    min_i, max_i = min(currs), max(currs)

    print(f"Parameter range detection: Freq[{min_f}, {max_f}], Curr[{min_i}, {max_i}]")

    # 2. Classification: Is it "boundary data" or "middle data"?
    boundary_files = []  # Must go into training set
    middle_files = []  # Can be randomly assigned

    for item in file_params:
        is_boundary = False
        # If any extreme value is reached, it is boundary data
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

    print(f"-> Found boundary files (forced training): {len(boundary_files)}")
    print(f"-> Found middle files (random assignment): {len(middle_files)}")

    # 3. Start allocation
    # Target totals
    n_total_train = int(total_files * TRAIN_RATIO)
    n_total_val = int(total_files * VAL_RATIO)

    # A. Training set = All boundary files + Fill remaining slots from middle files
    train_files = list(boundary_files)  # Take all boundaries first

    # How many middle data files are needed?
    slots_needed = n_total_train - len(train_files)

    if slots_needed < 0:
        print(
            "Warning: Too many boundary files, exceeding training set ratio! All boundaries will be put into training set, validation set will be smaller."
        )
        slots_needed = 0

    # Shuffle middle files
    random.seed(SEED)
    random.shuffle(middle_files)

    # Fill training set
    train_files.extend(middle_files[:slots_needed])

    # Remaining middle files for validation and test sets
    remaining = middle_files[slots_needed:]
    val_files = remaining[:n_total_val]
    test_files = remaining[n_total_val:]

    # 4. Print validation info
    print("-" * 30)
    print(
        f"Final Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}"
    )

    # Check that test set has no boundaries
    print("[Test Set Quality Check]:")
    for f in test_files:
        pf, pi = parse_params(f)
        is_bad = pf == min_f or pf == max_f or pi == min_i or pi == max_i
        status = "[FAIL] Boundary Data" if is_bad else "[OK] Inner Data"
        print(f"  {f} -> {status}")

    # 5. Save
    split_dict = {"train": train_files, "val": val_files, "test": test_files}
    with open(OUTPUT_JSON, "w") as f:
        json.dump(split_dict, f, indent=2)
    print(f"Config saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    create_smart_split()
