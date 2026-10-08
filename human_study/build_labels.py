"""
build_labels.py

Converts labels_distribution.csv (85 objects x 127 columns, one row per
object with a 7-bin rating distribution per adjective) into the .npy label
arrays consumed by dataset.py / train.py.

This is the same logic used to build Phase 1's 45-object label files
(Training/labels/), just without the "first 45 objects" filter, so all 85
objects present in the CSV are included.

Input:  --csv      labels_distribution.csv (85, 127)
                   (default: HUMAN_DATA_DIR/labels_distribution.csv, written by
                   build_label_distributions.py)
Output: --out-dir  labels_phase2.npy       (85, 18, 7)  float32
                   labels_mean_phase2.npy  (85, 18)     float32
                   object_ids_phase2.npy   (85,)        str
(--out-dir is what TACTILE_LABELS_DIR should point to for training.)

Usage: python build_labels.py --out-dir /path/to/labels [--csv /path/to/labels_distribution.csv]
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402


ADJECTIVES = [
    'hard', 'soft', 'rough', 'smooth', 'bumpy', 'flat',
    'sticky', 'slippery', 'cold', 'warm', 'wet', 'dry',
    'compressible', 'firm', 'fuzzy', 'scratchy', 'pleasant', 'irritating',
]

LEVELS = np.arange(1, 8, dtype=np.float32)  # [1, 2, ..., 7]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=None,
                        help='labels_distribution.csv (default: HUMAN_DATA_DIR/labels_distribution.csv)')
    parser.add_argument('--out-dir', required=True,
                        help='folder to write labels_phase2.npy, labels_mean_phase2.npy, object_ids_phase2.npy')
    args = parser.parse_args()
    CSV_PATH = args.csv or os.path.join(paths.human_data_dir(), 'labels_distribution.csv')
    LABELS_DIR = args.out_dir

    df = pd.read_csv(CSV_PATH)
    df = df.sort_values('object_id', key=lambda s: s.str.split('_').str[1].astype(int)) \
           .reset_index(drop=True)

    n = len(df)
    object_ids = df['object_id'].values  # (n,)

    labels_dist = np.zeros((n, len(ADJECTIVES), 7), dtype=np.float32)
    labels_mean = np.zeros((n, len(ADJECTIVES)), dtype=np.float32)

    for adj_idx, adj in enumerate(ADJECTIVES):
        cols = [f'{adj}_{lvl}' for lvl in range(1, 8)]
        dist = df[cols].values.astype(np.float32)          # (n, 7)
        labels_dist[:, adj_idx, :] = dist
        labels_mean[:, adj_idx] = (dist * LEVELS).sum(axis=1)

    os.makedirs(LABELS_DIR, exist_ok=True)
    np.save(os.path.join(LABELS_DIR, 'labels_phase2.npy'), labels_dist)
    np.save(os.path.join(LABELS_DIR, 'labels_mean_phase2.npy'), labels_mean)
    np.save(os.path.join(LABELS_DIR, 'object_ids_phase2.npy'), object_ids)

    print(f"labels_phase2.npy        : {labels_dist.shape}  {labels_dist.dtype}")
    print(f"labels_mean_phase2.npy   : {labels_mean.shape}  {labels_mean.dtype}")
    print(f"object_ids_phase2.npy    : {object_ids.shape}   {object_ids.dtype}")
    print(f"\nfirst object_id: {object_ids[0]}   last object_id: {object_ids[-1]}")

    # Sanity: each adjective's 7 bins should sum to 1.0 per object
    sums = labels_dist.sum(axis=2)  # (n, 18)
    assert np.allclose(sums, 1.0), "Distribution does not sum to 1 for some object/adjective"
    assert labels_mean.min() >= 1.0 and labels_mean.max() <= 7.0, "Mean rating out of [1,7] range"
    print("\nSanity checks passed: all distributions sum to 1.0, means within [1, 7].")
    print(f"\nSaved to {LABELS_DIR}/")


if __name__ == '__main__':
    main()
