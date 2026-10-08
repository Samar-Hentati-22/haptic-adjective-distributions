"""
build_label_distributions.py

Converts raw per-participant ratings into normalized rating distributions
for ML training.

Input:  session1_final.csv  (45 objects × 20 participants)
        session2_final.csv  (40 objects × 20 participants)
Output: labels_distribution.csv  (85 objects × 127 columns)
        Columns: object_id, hard_1..hard_7, soft_1..soft_7, ..., irritating_1..irritating_7
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402

# Inputs and output (labels_distribution.csv) live in HUMAN_DATA_DIR.
BASE_DIR    = paths.human_data_dir()
OUTPUT_FILE = os.path.join(BASE_DIR, "labels_distribution.csv")

ADJECTIVES = [
    "hard", "soft", "rough", "smooth", "bumpy", "flat",
    "sticky", "slippery", "cold", "warm", "wet", "dry",
    "compressible", "firm", "fuzzy", "scratchy", "pleasant", "irritating",
]

# ── Load and merge ─────────────────────────────────────────────────────────────

s1 = pd.read_csv(os.path.join(BASE_DIR, "session1_final.csv"))
s2 = pd.read_csv(os.path.join(BASE_DIR, "session2_final.csv"))
df = pd.concat([s1, s2], ignore_index=True)
print(f"Merged: {len(df)} rows, {df['object_id'].nunique()} objects")

# ── Build distributions ────────────────────────────────────────────────────────

output_cols = ["object_id"] + [f"{adj}_{level}" for adj in ADJECTIVES for level in range(1, 8)]
output_rows = []

objects = sorted(df["object_id"].unique(), key=lambda x: int(x.split("_")[1]))

for obj_id in objects:
    obj_df = df[df["object_id"] == obj_id]
    row = {"object_id": obj_id}

    for adj in ADJECTIVES:
        ratings = obj_df[adj].dropna().astype(int).values

        dist = np.zeros(7)
        for r in ratings:
            dist[int(r) - 1] += 1
        dist = dist / dist.sum()

        for level in range(1, 8):
            row[f"{adj}_{level}"] = round(dist[level - 1], 6)

    output_rows.append(row)

out_df = pd.DataFrame(output_rows, columns=output_cols)

# ── Save and verify ────────────────────────────────────────────────────────────

out_df.to_csv(OUTPUT_FILE, index=False)
print(f"Output shape: {out_df.shape}")
assert out_df.shape == (85, 127), f"Unexpected shape: {out_df.shape}"
print(f"Shape check passed: (85, 127) ✓")
print(f"Saved to: {OUTPUT_FILE}")

# Quick sanity: each adjective's 7 levels should sum to 1.0 per object
for adj in ADJECTIVES:
    cols = [f"{adj}_{l}" for l in range(1, 8)]
    sums = out_df[cols].sum(axis=1)
    assert np.allclose(sums, 1.0), f"Distribution does not sum to 1 for {adj}"
print("Distribution sums verified: all adjectives sum to 1.0 per object ✓")
