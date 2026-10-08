"""
reorganize_session2_data.py

Reads 4 Google Forms CSVs from the Session 2 tactile perception study and
outputs a clean CSV sorted by object_id, then participant_id.

File layout (all 4 parts):
  col 0: Horodateur
  col 1: Your Name
  col 2: Participant ID
  cols 3+: 18 ratings × N objects

Adjective order IN the file (Session 2 form order):
  warm, scratchy, firm, wet, smooth, bumpy, pleasant, dry,
  hard, fuzzy, slippery, cold, compressible, rough, sticky, soft, flat, irritating

Standard output order (same as Session 1 output):
  hard, soft, rough, smooth, bumpy, flat, sticky, slippery,
  cold, warm, wet, dry, compressible, firm, fuzzy, scratchy, pleasant, irritating
"""

import csv
import os
import sys
from collections import defaultdict

# ── Configuration ──────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402

# Inputs and output (session2_final.csv) live in HUMAN_DATA_DIR.
BASE_DIR   = paths.human_data_dir()
OUTPUT_FILE = os.path.join(BASE_DIR, "session2_final.csv")

# Adjective order as they appear in the Session 2 CSV columns
FORM_ADJECTIVES = [
    "warm", "scratchy", "firm", "wet", "smooth", "bumpy",
    "pleasant", "dry", "hard", "fuzzy", "slippery", "cold",
    "compressible", "rough", "sticky", "soft", "flat", "irritating",
]

# Standard output order (matches Session 1)
OUTPUT_ADJECTIVES = [
    "hard", "soft", "rough", "smooth", "bumpy", "flat",
    "sticky", "slippery", "cold", "warm", "wet", "dry",
    "compressible", "firm", "fuzzy", "scratchy", "pleasant", "irritating",
]

# Indices that reorder FORM_ADJECTIVES → OUTPUT_ADJECTIVES
REORDER = [FORM_ADJECTIVES.index(adj) for adj in OUTPUT_ADJECTIVES]

# ── Presentation orders ────────────────────────────────────────────────────────

PRESENTATION_ORDERS = {
    "P06": ["obj_46","obj_47","obj_48","obj_49","obj_50","obj_51","obj_52","obj_53","obj_54","obj_55","obj_56","obj_57","obj_58","obj_59","obj_60","obj_61","obj_62","obj_63","obj_64","obj_65","obj_66","obj_67","obj_68","obj_69","obj_70","obj_71","obj_72","obj_73","obj_74","obj_75","obj_76","obj_77","obj_78","obj_79","obj_80","obj_81","obj_82","obj_83","obj_84","obj_85"],
    "P04": ["obj_46","obj_47","obj_48","obj_49","obj_50","obj_51","obj_53","obj_52","obj_54","obj_55","obj_56","obj_57","obj_58","obj_59","obj_60","obj_61","obj_62","obj_63","obj_64","obj_65","obj_66","obj_67","obj_68","obj_69","obj_70","obj_71","obj_72","obj_73","obj_74","obj_75","obj_76","obj_77","obj_78","obj_79","obj_80","obj_81","obj_82","obj_83","obj_84","obj_85"],
    "P10": ["obj_46","obj_47","obj_48","obj_49","obj_50","obj_51","obj_52","obj_53","obj_54","obj_55","obj_56","obj_57","obj_58","obj_59","obj_60","obj_62","obj_61","obj_63","obj_64","obj_65","obj_66","obj_67","obj_68","obj_69","obj_70","obj_71","obj_72","obj_73","obj_74","obj_75","obj_76","obj_77","obj_78","obj_79","obj_80","obj_81","obj_82","obj_83","obj_84","obj_85"],
    "P05": ["obj_46","obj_47","obj_48","obj_49","obj_50","obj_51","obj_53","obj_52","obj_54","obj_55","obj_56","obj_57","obj_58","obj_59","obj_60","obj_61","obj_62","obj_63","obj_64","obj_65","obj_66","obj_67","obj_68","obj_69","obj_70","obj_71","obj_72","obj_73","obj_74","obj_75","obj_76","obj_77","obj_78","obj_79","obj_80","obj_81","obj_82","obj_83","obj_84","obj_85"],
    "P12": ["obj_46","obj_47","obj_48","obj_49","obj_50","obj_51","obj_53","obj_52","obj_54","obj_55","obj_56","obj_57","obj_58","obj_59","obj_60","obj_61","obj_62","obj_63","obj_64","obj_65","obj_66","obj_67","obj_68","obj_69","obj_70","obj_72","obj_73","obj_74","obj_71","obj_75","obj_76","obj_77","obj_78","obj_79","obj_80","obj_81","obj_82","obj_83","obj_84","obj_85"],
    "P02": ["obj_46","obj_47","obj_48","obj_49","obj_50","obj_51","obj_53","obj_52","obj_54","obj_55","obj_56","obj_57","obj_58","obj_59","obj_60","obj_61","obj_62","obj_63","obj_64","obj_65","obj_66","obj_67","obj_68","obj_69","obj_70","obj_71","obj_72","obj_73","obj_74","obj_75","obj_76","obj_77","obj_78","obj_79","obj_80","obj_81","obj_82","obj_83","obj_84","obj_85"],
    "P11": ["obj_69","obj_78","obj_63","obj_52","obj_83","obj_53","obj_48","obj_75","obj_79","obj_85","obj_66","obj_55","obj_74","obj_56","obj_77","obj_50","obj_59","obj_46","obj_65","obj_84","obj_73","obj_51","obj_60","obj_68","obj_62","obj_57","obj_82","obj_54","obj_70","obj_71","obj_58","obj_64","obj_72","obj_80","obj_67","obj_47","obj_49","obj_61","obj_81","obj_76"],
    "P14": ["obj_69","obj_78","obj_63","obj_52","obj_83","obj_53","obj_48","obj_75","obj_79","obj_85","obj_66","obj_55","obj_74","obj_56","obj_77","obj_50","obj_59","obj_46","obj_65","obj_84","obj_60","obj_73","obj_51","obj_68","obj_62","obj_57","obj_82","obj_54","obj_70","obj_71","obj_58","obj_64","obj_72","obj_80","obj_67","obj_47","obj_49","obj_61","obj_81","obj_76"],
    "P13": ["obj_69","obj_78","obj_63","obj_52","obj_83","obj_53","obj_48","obj_75","obj_79","obj_85","obj_66","obj_55","obj_74","obj_56","obj_77","obj_50","obj_59","obj_46","obj_65","obj_84","obj_60","obj_73","obj_51","obj_68","obj_62","obj_57","obj_82","obj_54","obj_70","obj_71","obj_58","obj_64","obj_72","obj_80","obj_67","obj_47","obj_49","obj_61","obj_81","obj_76"],
    "P03": ["obj_69","obj_78","obj_63","obj_52","obj_83","obj_53","obj_48","obj_75","obj_79","obj_85","obj_66","obj_55","obj_74","obj_56","obj_77","obj_50","obj_59","obj_46","obj_65","obj_84","obj_60","obj_73","obj_51","obj_68","obj_62","obj_57","obj_82","obj_54","obj_70","obj_71","obj_58","obj_64","obj_72","obj_80","obj_67","obj_47","obj_49","obj_61","obj_81","obj_76"],
    "P01": ["obj_69","obj_78","obj_63","obj_52","obj_83","obj_53","obj_48","obj_75","obj_79","obj_85","obj_66","obj_55","obj_74","obj_56","obj_77","obj_50","obj_59","obj_46","obj_65","obj_84","obj_60","obj_73","obj_68","obj_62","obj_57","obj_82","obj_54","obj_70","obj_71","obj_58","obj_64","obj_51","obj_72","obj_80","obj_67","obj_47","obj_49","obj_61","obj_81","obj_76"],
    "P07": ["obj_69","obj_78","obj_63","obj_52","obj_83","obj_53","obj_48","obj_75","obj_79","obj_85","obj_66","obj_55","obj_74","obj_56","obj_77","obj_50","obj_59","obj_46","obj_65","obj_84","obj_60","obj_73","obj_68","obj_62","obj_57","obj_82","obj_54","obj_70","obj_71","obj_58","obj_64","obj_51","obj_72","obj_80","obj_67","obj_47","obj_49","obj_61","obj_81","obj_76"],
    "P17": ["obj_65","obj_63","obj_83","obj_66","obj_82","obj_72","obj_75","obj_59","obj_67","obj_47","obj_46","obj_48","obj_73","obj_74","obj_69","obj_77","obj_60","obj_70","obj_52","obj_50","obj_51","obj_56","obj_62","obj_80","obj_79","obj_53","obj_68","obj_54","obj_55","obj_85","obj_58","obj_71","obj_64","obj_57","obj_49","obj_81","obj_84","obj_78","obj_76","obj_61"],
    "P18": ["obj_65","obj_63","obj_83","obj_66","obj_82","obj_72","obj_75","obj_59","obj_67","obj_47","obj_46","obj_48","obj_73","obj_74","obj_69","obj_77","obj_60","obj_70","obj_52","obj_50","obj_51","obj_56","obj_62","obj_80","obj_79","obj_53","obj_68","obj_54","obj_55","obj_85","obj_58","obj_71","obj_64","obj_57","obj_49","obj_81","obj_84","obj_78","obj_76","obj_61"],
    "P09": ["obj_65","obj_63","obj_83","obj_66","obj_82","obj_72","obj_75","obj_59","obj_67","obj_47","obj_46","obj_48","obj_73","obj_74","obj_69","obj_77","obj_60","obj_70","obj_52","obj_50","obj_51","obj_56","obj_62","obj_80","obj_79","obj_53","obj_68","obj_54","obj_55","obj_85","obj_58","obj_71","obj_64","obj_57","obj_49","obj_81","obj_84","obj_78","obj_76","obj_61"],
    "P08": ["obj_65","obj_63","obj_83","obj_66","obj_82","obj_72","obj_75","obj_59","obj_67","obj_47","obj_46","obj_48","obj_73","obj_74","obj_69","obj_77","obj_60","obj_70","obj_52","obj_50","obj_51","obj_56","obj_62","obj_80","obj_79","obj_53","obj_68","obj_54","obj_55","obj_85","obj_58","obj_71","obj_64","obj_57","obj_49","obj_81","obj_84","obj_78","obj_76","obj_61"],
    "P16": ["obj_54","obj_69","obj_76","obj_53","obj_56","obj_75","obj_82","obj_59","obj_78","obj_52","obj_68","obj_77","obj_62","obj_84","obj_85","obj_55","obj_72","obj_50","obj_83","obj_64","obj_67","obj_71","obj_79","obj_65","obj_48","obj_63","obj_66","obj_51","obj_57","obj_49","obj_70","obj_60","obj_61","obj_58","obj_73","obj_74","obj_47","obj_81","obj_46","obj_80"],
    "P19": ["obj_54","obj_69","obj_76","obj_53","obj_56","obj_75","obj_82","obj_59","obj_78","obj_52","obj_68","obj_77","obj_62","obj_84","obj_85","obj_55","obj_72","obj_50","obj_83","obj_64","obj_67","obj_71","obj_79","obj_65","obj_48","obj_63","obj_66","obj_51","obj_57","obj_49","obj_70","obj_60","obj_61","obj_58","obj_73","obj_74","obj_47","obj_81","obj_46","obj_80"],
    "P15": ["obj_54","obj_69","obj_76","obj_53","obj_56","obj_75","obj_82","obj_59","obj_78","obj_52","obj_68","obj_77","obj_62","obj_84","obj_85","obj_55","obj_72","obj_50","obj_83","obj_64","obj_67","obj_71","obj_79","obj_65","obj_48","obj_63","obj_66","obj_51","obj_57","obj_49","obj_70","obj_60","obj_61","obj_58","obj_73","obj_74","obj_47","obj_81","obj_46","obj_80"],
    "P20": ["obj_54","obj_69","obj_76","obj_53","obj_56","obj_75","obj_82","obj_59","obj_78","obj_52","obj_68","obj_77","obj_62","obj_84","obj_85","obj_55","obj_72","obj_50","obj_83","obj_64","obj_67","obj_71","obj_79","obj_65","obj_48","obj_63","obj_66","obj_51","obj_57","obj_49","obj_70","obj_60","obj_61","obj_58","obj_73","obj_74","obj_47","obj_81","obj_46","obj_80"],
}

# CSV file specs: (filename, form_slot = which positions in the presentation order)
CSV_SPECS = {
    "session2_part1.csv": {"pid_col": 2, "ratings_start": 3, "n_objects": 10, "form_slot": slice(0, 10)},
    "session2_part2.csv": {"pid_col": 2, "ratings_start": 3, "n_objects": 10, "form_slot": slice(10, 20)},
    "session2_part3.csv": {"pid_col": 2, "ratings_start": 3, "n_objects": 10, "form_slot": slice(20, 30)},
    "session2_part4.csv": {"pid_col": 2, "ratings_start": 3, "n_objects": 10, "form_slot": slice(30, 40)},
}

# ── Parsing ────────────────────────────────────────────────────────────────────

# results[object_id][participant_id] = {adjective: value, ...}
results = defaultdict(dict)

for filename, spec in CSV_SPECS.items():
    filepath = os.path.join(BASE_DIR, filename)
    if not os.path.exists(filepath):
        print(f"  [SKIP] {filename} not found")
        continue

    pid_col       = spec["pid_col"]
    ratings_start = spec["ratings_start"]
    n_objects     = spec["n_objects"]
    form_slot     = spec["form_slot"]

    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)

        expected_rating_cols = n_objects * len(FORM_ADJECTIVES)
        actual_rating_cols   = len(header) - ratings_start
        if actual_rating_cols != expected_rating_cols:
            print(f"  [WARN] {filename}: expected {expected_rating_cols} rating cols, "
                  f"found {actual_rating_cols}. Check layout.")

        for row_num, row in enumerate(reader, start=2):
            if not any(row):
                continue

            participant_id = row[pid_col].strip()
            if not participant_id:
                print(f"  [WARN] {filename} row {row_num}: empty participant ID, skipping.")
                continue

            if participant_id not in PRESENTATION_ORDERS:
                print(f"  [WARN] {filename} row {row_num}: unknown participant '{participant_id}', skipping.")
                continue

            order = PRESENTATION_ORDERS[participant_id]
            objects_in_this_form = order[form_slot]

            for obj_idx, obj_id in enumerate(objects_in_this_form):
                col_start = ratings_start + obj_idx * len(FORM_ADJECTIVES)
                col_end   = col_start + len(FORM_ADJECTIVES)
                raw_values = row[col_start:col_end]

                if len(raw_values) != len(FORM_ADJECTIVES):
                    print(f"  [WARN] {filename} row {row_num}, {obj_id}: "
                          f"got {len(raw_values)} values, expected {len(FORM_ADJECTIVES)}. Skipping.")
                    continue

                # Reorder from form order to standard output order
                reordered = [raw_values[i] for i in REORDER]
                ratings_dict = {adj: val.strip() for adj, val in zip(OUTPUT_ADJECTIVES, reordered)}
                results[obj_id][participant_id] = ratings_dict

    print(f"  [OK] Parsed {filename}")

# ── Output ─────────────────────────────────────────────────────────────────────

sorted_objects = sorted(results.keys(), key=lambda x: int(x.split("_")[1]))

output_rows = []
for obj_id in sorted_objects:
    participant_ratings = results[obj_id]
    sorted_pids = sorted(participant_ratings.keys(), key=lambda x: int(x[1:]))
    for pid in sorted_pids:
        row = {"object_id": obj_id, "participant_id": pid}
        row.update(participant_ratings[pid])
        output_rows.append(row)

fieldnames = ["object_id", "participant_id"] + OUTPUT_ADJECTIVES

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)

print(f"\nDone. {len(output_rows)} rows written to:\n  {OUTPUT_FILE}")

# ── Sanity check ───────────────────────────────────────────────────────────────
print("\n── Coverage summary ──")
all_pids = sorted(PRESENTATION_ORDERS.keys(), key=lambda x: int(x[1:]))
missing = []
for obj_id in sorted_objects:
    found_pids = set(results[obj_id].keys())
    missing_pids = [p for p in all_pids if p not in found_pids]
    n = len(found_pids)
    status = "✓" if n == 20 else f"✗ missing: {', '.join(missing_pids)}"
    print(f"  {obj_id}: {n}/20 participants  {status}")
    if missing_pids:
        missing.append((obj_id, missing_pids))

if not missing:
    print("\nAll 40 objects have complete ratings from all 20 participants.")
else:
    print(f"\n{len(missing)} object(s) have incomplete data — see above.")
