"""
reorganize_tactile_data.py

Reads Google Forms response CSVs from the tactile perception study and outputs
a clean CSV sorted by object_id, then participant_id.

File layout:
  part1.csv  — participant ID at col index 3, ratings start at index 8,  5 objects × 18 adjectives
  part2–5.csv — participant ID at col index 2, ratings start at index 3, 10 objects × 18 adjectives

Presentation orders are hard-coded per participant (P01–P20) from the study protocol.
"""

import csv
import os
import sys
from collections import defaultdict

# ── Configuration ──────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402

# Inputs and output (session1_final.csv) live in HUMAN_DATA_DIR.
BASE_DIR = paths.human_data_dir()
OUTPUT_FILE = os.path.join(BASE_DIR, "session1_final.csv")

ADJECTIVES = [
    "hard", "soft", "rough", "smooth", "bumpy", "flat",
    "sticky", "slippery", "cold", "warm", "wet", "dry",
    "compressible", "firm", "fuzzy", "scratchy", "pleasant", "irritating"
]

# Presentation order per participant (which objects were shown, in order).
# Part 1 covers the first 5 objects in each participant's list.
# Parts 2–5 cover the next 10 each (objects 6–15, 16–25, 26–35, 36–45).
PRESENTATION_ORDERS = {
    "P01": ["obj_01","obj_02","obj_03","obj_04","obj_05","obj_06","obj_07","obj_08","obj_09","obj_10","obj_11","obj_12","obj_13","obj_14","obj_15","obj_16","obj_17","obj_18","obj_19","obj_20","obj_21","obj_22","obj_23","obj_24","obj_25","obj_26","obj_27","obj_28","obj_29","obj_30","obj_31","obj_32","obj_33","obj_34","obj_35","obj_36","obj_37","obj_38","obj_39","obj_40","obj_41","obj_42","obj_43","obj_44","obj_45"],
    "P02": ["obj_01","obj_02","obj_03","obj_04","obj_05","obj_06","obj_07","obj_08","obj_09","obj_10","obj_11","obj_12","obj_13","obj_14","obj_15","obj_16","obj_17","obj_18","obj_19","obj_20","obj_21","obj_22","obj_23","obj_24","obj_25","obj_26","obj_27","obj_28","obj_29","obj_30","obj_31","obj_32","obj_33","obj_34","obj_35","obj_36","obj_37","obj_38","obj_39","obj_40","obj_41","obj_42","obj_43","obj_44","obj_45"],
    "P03": ["obj_01","obj_02","obj_03","obj_04","obj_05","obj_06","obj_07","obj_08","obj_09","obj_10","obj_11","obj_12","obj_13","obj_14","obj_15","obj_16","obj_17","obj_18","obj_19","obj_20","obj_21","obj_22","obj_23","obj_24","obj_25","obj_26","obj_28","obj_29","obj_30","obj_31","obj_32","obj_33","obj_34","obj_35","obj_36","obj_27","obj_37","obj_38","obj_39","obj_40","obj_41","obj_43","obj_42","obj_44","obj_45"],
    "P04": ["obj_01","obj_02","obj_03","obj_04","obj_05","obj_06","obj_07","obj_08","obj_09","obj_10","obj_11","obj_12","obj_13","obj_14","obj_15","obj_16","obj_17","obj_18","obj_19","obj_20","obj_21","obj_22","obj_23","obj_24","obj_25","obj_26","obj_27","obj_28","obj_29","obj_30","obj_31","obj_32","obj_33","obj_34","obj_35","obj_36","obj_37","obj_38","obj_39","obj_40","obj_41","obj_42","obj_43","obj_44","obj_45"],
    "P05": ["obj_01","obj_02","obj_03","obj_04","obj_05","obj_06","obj_07","obj_08","obj_09","obj_10","obj_11","obj_12","obj_13","obj_14","obj_15","obj_16","obj_17","obj_18","obj_19","obj_20","obj_21","obj_22","obj_23","obj_24","obj_25","obj_26","obj_27","obj_28","obj_29","obj_30","obj_31","obj_32","obj_33","obj_34","obj_35","obj_36","obj_37","obj_38","obj_39","obj_40","obj_41","obj_42","obj_43","obj_44","obj_45"],
    "P06": ["obj_45","obj_19","obj_15","obj_22","obj_09","obj_16","obj_43","obj_23","obj_24","obj_11","obj_08","obj_27","obj_30","obj_35","obj_41","obj_42","obj_06","obj_13","obj_33","obj_44","obj_12","obj_34","obj_01","obj_32","obj_20","obj_07","obj_02","obj_39","obj_14","obj_40","obj_37","obj_03","obj_17","obj_05","obj_25","obj_31","obj_10","obj_28","obj_04","obj_38","obj_29","obj_26","obj_21","obj_18","obj_36"],
    "P07": ["obj_45","obj_19","obj_15","obj_22","obj_09","obj_16","obj_43","obj_23","obj_24","obj_11","obj_08","obj_27","obj_30","obj_35","obj_41","obj_42","obj_06","obj_13","obj_33","obj_44","obj_12","obj_34","obj_01","obj_32","obj_20","obj_07","obj_02","obj_39","obj_14","obj_40","obj_37","obj_03","obj_17","obj_05","obj_25","obj_31","obj_10","obj_28","obj_04","obj_29","obj_26","obj_21","obj_18","obj_38","obj_36"],
    "P08": ["obj_45","obj_19","obj_15","obj_22","obj_09","obj_16","obj_43","obj_23","obj_24","obj_11","obj_08","obj_27","obj_30","obj_31","obj_41","obj_42","obj_06","obj_13","obj_33","obj_44","obj_12","obj_34","obj_01","obj_32","obj_20","obj_07","obj_02","obj_39","obj_14","obj_40","obj_37","obj_03","obj_17","obj_05","obj_25","obj_35","obj_10","obj_28","obj_04","obj_29","obj_26","obj_21","obj_18","obj_38","obj_36"],
    "P09": ["obj_45","obj_19","obj_15","obj_22","obj_09","obj_16","obj_43","obj_23","obj_24","obj_11","obj_08","obj_27","obj_30","obj_35","obj_41","obj_42","obj_06","obj_13","obj_33","obj_44","obj_12","obj_34","obj_01","obj_32","obj_20","obj_07","obj_02","obj_39","obj_14","obj_40","obj_37","obj_03","obj_17","obj_05","obj_25","obj_31","obj_10","obj_28","obj_04","obj_29","obj_26","obj_21","obj_18","obj_38","obj_36"],
    "P10": ["obj_45","obj_19","obj_15","obj_22","obj_09","obj_16","obj_43","obj_23","obj_24","obj_11","obj_08","obj_27","obj_30","obj_35","obj_41","obj_42","obj_06","obj_13","obj_33","obj_44","obj_12","obj_34","obj_01","obj_32","obj_20","obj_07","obj_02","obj_39","obj_14","obj_40","obj_37","obj_03","obj_17","obj_05","obj_25","obj_31","obj_10","obj_28","obj_04","obj_29","obj_26","obj_21","obj_18","obj_38","obj_36"],
    "P11": ["obj_38","obj_19","obj_17","obj_39","obj_31","obj_01","obj_03","obj_37","obj_27","obj_07","obj_22","obj_14","obj_11","obj_09","obj_42","obj_15","obj_16","obj_26","obj_12","obj_13","obj_10","obj_45","obj_35","obj_02","obj_43","obj_23","obj_08","obj_06","obj_33","obj_29","obj_34","obj_24","obj_20","obj_04","obj_32","obj_05","obj_25","obj_30","obj_41","obj_28","obj_44","obj_21","obj_36","obj_40","obj_18"],
    "P12": ["obj_38","obj_19","obj_17","obj_39","obj_31","obj_01","obj_03","obj_37","obj_27","obj_07","obj_22","obj_14","obj_09","obj_42","obj_11","obj_15","obj_16","obj_26","obj_12","obj_13","obj_10","obj_45","obj_35","obj_02","obj_43","obj_23","obj_08","obj_06","obj_33","obj_29","obj_34","obj_24","obj_20","obj_04","obj_32","obj_05","obj_25","obj_30","obj_41","obj_28","obj_44","obj_21","obj_36","obj_40","obj_18"],
    "P13": ["obj_38","obj_19","obj_17","obj_39","obj_31","obj_01","obj_03","obj_37","obj_27","obj_07","obj_22","obj_14","obj_09","obj_42","obj_11","obj_15","obj_16","obj_26","obj_12","obj_13","obj_10","obj_45","obj_35","obj_02","obj_43","obj_23","obj_08","obj_06","obj_33","obj_29","obj_34","obj_24","obj_20","obj_04","obj_32","obj_05","obj_25","obj_30","obj_41","obj_28","obj_44","obj_21","obj_36","obj_40","obj_18"],
    "P14": ["obj_38","obj_19","obj_17","obj_39","obj_31","obj_01","obj_03","obj_37","obj_27","obj_07","obj_22","obj_14","obj_09","obj_42","obj_11","obj_15","obj_16","obj_26","obj_12","obj_13","obj_10","obj_45","obj_35","obj_02","obj_43","obj_23","obj_08","obj_06","obj_33","obj_29","obj_34","obj_24","obj_20","obj_04","obj_32","obj_05","obj_25","obj_30","obj_41","obj_28","obj_44","obj_21","obj_36","obj_40","obj_18"],
    "P15": ["obj_38","obj_19","obj_17","obj_39","obj_31","obj_01","obj_03","obj_37","obj_27","obj_07","obj_22","obj_14","obj_11","obj_09","obj_42","obj_15","obj_16","obj_26","obj_12","obj_13","obj_10","obj_45","obj_35","obj_02","obj_43","obj_23","obj_08","obj_06","obj_33","obj_29","obj_34","obj_24","obj_20","obj_04","obj_32","obj_05","obj_25","obj_30","obj_41","obj_28","obj_44","obj_21","obj_36","obj_40","obj_18"],
    "P16": ["obj_23","obj_19","obj_32","obj_09","obj_35","obj_28","obj_08","obj_17","obj_07","obj_41","obj_42","obj_13","obj_37","obj_16","obj_10","obj_05","obj_44","obj_38","obj_01","obj_45","obj_15","obj_11","obj_39","obj_24","obj_22","obj_03","obj_43","obj_30","obj_33","obj_36","obj_29","obj_27","obj_31","obj_34","obj_14","obj_12","obj_26","obj_20","obj_02","obj_25","obj_40","obj_06","obj_04","obj_18","obj_21"],
    "P17": ["obj_23","obj_19","obj_32","obj_09","obj_35","obj_28","obj_08","obj_17","obj_07","obj_41","obj_42","obj_13","obj_37","obj_16","obj_10","obj_05","obj_44","obj_38","obj_01","obj_45","obj_15","obj_11","obj_39","obj_24","obj_22","obj_03","obj_43","obj_30","obj_33","obj_36","obj_29","obj_27","obj_31","obj_34","obj_14","obj_12","obj_26","obj_20","obj_02","obj_25","obj_40","obj_06","obj_04","obj_18","obj_21"],
    "P18": ["obj_23","obj_19","obj_32","obj_09","obj_35","obj_28","obj_08","obj_17","obj_07","obj_41","obj_42","obj_13","obj_37","obj_16","obj_10","obj_05","obj_44","obj_38","obj_01","obj_45","obj_15","obj_11","obj_39","obj_24","obj_22","obj_03","obj_43","obj_30","obj_33","obj_36","obj_29","obj_27","obj_31","obj_34","obj_14","obj_12","obj_26","obj_20","obj_02","obj_25","obj_40","obj_06","obj_04","obj_18","obj_21"],
    "P19": ["obj_23","obj_19","obj_32","obj_09","obj_35","obj_28","obj_08","obj_17","obj_07","obj_41","obj_42","obj_13","obj_37","obj_16","obj_10","obj_05","obj_44","obj_38","obj_01","obj_45","obj_15","obj_11","obj_39","obj_24","obj_22","obj_03","obj_43","obj_30","obj_33","obj_36","obj_29","obj_27","obj_31","obj_34","obj_14","obj_12","obj_26","obj_20","obj_02","obj_25","obj_40","obj_06","obj_04","obj_18","obj_21"],
    "P20": ["obj_23","obj_19","obj_32","obj_09","obj_35","obj_28","obj_08","obj_17","obj_07","obj_41","obj_42","obj_13","obj_37","obj_16","obj_10","obj_05","obj_44","obj_38","obj_01","obj_45","obj_15","obj_11","obj_39","obj_24","obj_22","obj_03","obj_43","obj_30","obj_33","obj_36","obj_29","obj_27","obj_31","obj_34","obj_14","obj_12","obj_26","obj_20","obj_02","obj_25","obj_40","obj_06","obj_04","obj_18","obj_21"],
}

# CSV file specs: (filename, pid_col_index, ratings_start_index, n_objects_in_file)
CSV_SPECS = {
    "part1.csv": {"pid_col": 3, "ratings_start": 26, "n_objects": 5,  "form_slot": slice(0, 5)},
    "part2.csv": {"pid_col": 2, "ratings_start": 3,  "n_objects": 10, "form_slot": slice(5, 15)},
    "part3.csv": {"pid_col": 2, "ratings_start": 3,  "n_objects": 10, "form_slot": slice(15, 25)},
    "part4.csv": {"pid_col": 2, "ratings_start": 3,  "n_objects": 10, "form_slot": slice(25, 35)},
    "part5.csv": {"pid_col": 2, "ratings_start": 3,  "n_objects": 10, "form_slot": slice(35, 45)},
}

# ── Parsing ────────────────────────────────────────────────────────────────────

# results[object_id][participant_id] = {adjective: value, ...}
results = defaultdict(dict)

for filename, spec in CSV_SPECS.items():
    filepath = os.path.join(BASE_DIR, filename)
    if not os.path.exists(filepath):
        print(f"  [SKIP] {filename} not found at {filepath}")
        continue

    pid_col      = spec["pid_col"]
    ratings_start = spec["ratings_start"]
    n_objects    = spec["n_objects"]
    form_slot    = spec["form_slot"]  # which positional slot (0-44) in the presentation order

    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header row

        expected_rating_cols = n_objects * len(ADJECTIVES)
        actual_rating_cols   = len(header) - ratings_start
        if actual_rating_cols != expected_rating_cols:
            print(f"  [WARN] {filename}: expected {expected_rating_cols} rating columns "
                  f"after index {ratings_start}, found {actual_rating_cols}. Check layout.")

        for row_num, row in enumerate(reader, start=2):
            if not any(row):
                continue  # skip blank rows

            participant_id = row[pid_col].strip()
            if not participant_id:
                print(f"  [WARN] {filename} row {row_num}: empty participant ID, skipping.")
                continue

            if participant_id not in PRESENTATION_ORDERS:
                print(f"  [WARN] {filename} row {row_num}: unknown participant ID '{participant_id}', skipping.")
                continue

            order = PRESENTATION_ORDERS[participant_id]  # full 45-object order for this participant
            objects_in_this_form = order[form_slot]      # the slice relevant to this CSV

            for obj_idx, obj_id in enumerate(objects_in_this_form):
                col_start = ratings_start + obj_idx * len(ADJECTIVES)
                col_end   = col_start + len(ADJECTIVES)
                rating_values = row[col_start:col_end]

                if len(rating_values) != len(ADJECTIVES):
                    print(f"  [WARN] {filename} row {row_num}, {obj_id}: "
                          f"got {len(rating_values)} values, expected {len(ADJECTIVES)}. Skipping.")
                    continue

                ratings_dict = {adj: val.strip() for adj, val in zip(ADJECTIVES, rating_values)}
                results[obj_id][participant_id] = ratings_dict

    print(f"  [OK] Parsed {filename}")

# ── Output ─────────────────────────────────────────────────────────────────────

# Sort objects numerically (obj_01 < obj_02 … < obj_45)
sorted_objects = sorted(results.keys(), key=lambda x: int(x.split("_")[1]))

# Sort participants numerically within each object
output_rows = []
for obj_id in sorted_objects:
    participant_ratings = results[obj_id]
    sorted_pids = sorted(participant_ratings.keys(), key=lambda x: int(x[1:]))
    for pid in sorted_pids:
        row = {"object_id": obj_id, "participant_id": pid}
        row.update(participant_ratings[pid])
        output_rows.append(row)

fieldnames = ["object_id", "participant_id"] + ADJECTIVES

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)

print(f"\nDone. {len(output_rows)} rows written to:\n  {OUTPUT_FILE}")

# ── Sanity check ───────────────────────────────────────────────────────────────
print("\n── Coverage summary ──")
missing = []
all_pids = sorted(PRESENTATION_ORDERS.keys())
for obj_id in sorted_objects:
    found_pids = set(results[obj_id].keys())
    missing_pids = [p for p in all_pids if p not in found_pids]
    n = len(found_pids)
    status = "✓" if n == 20 else f"✗ missing: {', '.join(missing_pids)}"
    print(f"  {obj_id}: {n}/20 participants  {status}")
    if missing_pids:
        missing.append((obj_id, missing_pids))

if not missing:
    print("\nAll 45 objects have complete ratings from all 20 participants.")
else:
    print(f"\n{len(missing)} object(s) have incomplete data — see above.")
