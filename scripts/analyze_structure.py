import pandas as pd
import re
import csv
import io


def parse_specific_comsol_header(file_path):
    print(f"Reading file header: {file_path}")

    header_line = ""
    # 1. Find header (containing 'f_set=' and '@')
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if "f_set=" in line and "@" in line:
                header_line = line.strip()
                print(f"Found header (line {i + 1}).")
                break

    if not header_line:
        print("No valid header found.")
        return None

    # 2. Split columns
    reader = csv.reader(io.StringIO(header_line), delimiter=",")
    columns = next(reader)
    print(f"Total columns: {len(columns)}")

    parsed_data = []
    coord_cols = []

    pattern = re.compile(
        r"(.*?)@\s*t=([0-9.E+-]+);\s*f_set=([0-9.E+-]+);\s*I_factor=([0-9.E+-]+)"
    )

    for idx, col_str in enumerate(columns):
        col_clean = col_str.strip()

        match = pattern.search(col_clean)
        if match:
            var_raw = match.group(1).strip()
            t_val = float(match.group(2))
            f_val = float(match.group(3))
            i_val = float(match.group(4))

            if "T" in var_raw:
                var_name = "Temperature"
            elif "phase1" in var_raw:
                var_name = "Austenite"
            elif "phase5" in var_raw:
                var_name = "Martensite"
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
            if idx < 5:
                coord_cols.append({"col_idx": idx, "name": col_clean})

    # 4. Save results
    df_map = pd.DataFrame(parsed_data)

    if not df_map.empty:
        print("-" * 50)
        print(f"Parsing successful! Captured parameter columns: {len(df_map)}")
        print(f"Coordinate columns: {[c['name'] for c in coord_cols]}")
        print(f"Variables: {df_map['variable'].unique()}")
        print(f"Time steps: {df_map['time'].nunique()}")
        print(f"Frequency groups (f_set): {df_map['f_set'].nunique()}")
        print(f"Current groups (I_factor): {df_map['I_factor'].nunique()}")

        # Save map
        df_map.to_csv("data/analyze/comsol_column_map.csv", index=False)
        print("\nMap saved to 'data/analyze/comsol_column_map.csv'")

        # Save coordinate column indices
        pd.DataFrame(coord_cols).to_csv(
            "data/analyze/comsol_coord_map.csv", index=False
        )
    else:
        print("Parsing failed, regex did not match any columns.")

    return df_map


# ============================
file_path = "data/raw/dataset.csv"
# ============================

if __name__ == "__main__":
    parse_specific_comsol_header(file_path)
