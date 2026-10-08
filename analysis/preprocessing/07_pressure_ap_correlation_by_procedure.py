"""
Pipeline v3 — Step B: per-procedure Pearson correlation of pressure_ap
contact-event features vs. true adjective ratings, for the k-sensitivity
comparison (k=3, 5, 8) and then the final validated k.

Diagnostic only: no changes to dataset.py / model.py / train.py / any
existing pipeline.

Aggregates per-trial features (from 06_extract_pressure_ap_features.py)
to one row per (object, procedure) via nanmean across that object's
trials for the procedure, then correlates each feature against each of
the 18 true adjective mean ratings across the 85 objects, separately per
procedure.
"""
import argparse
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402

# Reads the pressure_ap_features_k{k}.csv files written by 06_extract_pressure_ap_features.py.
OUTPUT_DIR = paths.output_dir('06_extract_pressure_ap_features.py', create=False)
LABELS_DIR = paths.labels_dir()

sys.path.insert(0, BASE_DIR)
from pressure_ap_event_features import FEATURE_NAMES  # noqa: E402

ACTIONS = ['tap', 'squeeze', 'slide', 'hold']
ADJECTIVES = [
    'hard', 'soft', 'rough', 'smooth', 'bumpy', 'flat',
    'sticky', 'slippery', 'cold', 'warm', 'wet', 'dry',
    'compressible', 'firm', 'fuzzy', 'scratchy', 'pleasant', 'irritating',
]


def load_features_csv(path):
    """object_id, procedure -> {feature: [values across trials]}"""
    data = {}
    with open(path) as fp:
        header = fp.readline().strip().split(',')
        for line in fp:
            parts = line.strip().split(',')
            row = dict(zip(header, parts))
            key = (row['object_id'], row['procedure'])
            if key not in data:
                data[key] = {name: [] for name in FEATURE_NAMES}
            for name in FEATURE_NAMES:
                v = row[name]
                data[key][name].append(float(v) if v != '' else np.nan)
    return data


def aggregate_per_object(data, action, object_ids):
    """object_ids order -> (n_objects, n_features) array, nanmean across trials."""
    out = np.full((len(object_ids), len(FEATURE_NAMES)), np.nan)
    for i, obj_id in enumerate(object_ids):
        key = (obj_id, action)
        if key not in data:
            continue
        for j, name in enumerate(FEATURE_NAMES):
            vals = np.array(data[key][name], dtype=float)
            if np.all(np.isnan(vals)):
                continue
            out[i, j] = np.nanmean(vals)
    return out


def pearson_with_nan(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 10:  # require a reasonable number of objects
        return np.nan, mask.sum()
    xm, ym = x[mask], y[mask]
    if xm.std() == 0 or ym.std() == 0:
        return np.nan, mask.sum()
    r = np.corrcoef(xm, ym)[0, 1]
    return r, mask.sum()


def top5_per_procedure(features_csv_path):
    """{procedure: [(adjective, feature, r, n), ...] top 5 by |r|}"""
    mean = np.load(os.path.join(LABELS_DIR, 'labels_mean_phase2.npy'))
    object_ids = [str(o) for o in np.load(
        os.path.join(LABELS_DIR, 'object_ids_phase2.npy'), allow_pickle=True)]

    data = load_features_csv(features_csv_path)

    results = {}
    for action in ACTIONS:
        feat_arr = aggregate_per_object(data, action, object_ids)  # (85, 7)
        pairs = []
        for j, fname in enumerate(FEATURE_NAMES):
            for a_idx, adj in enumerate(ADJECTIVES):
                r, n = pearson_with_nan(feat_arr[:, j], mean[:, a_idx])
                if not np.isnan(r):
                    pairs.append((adj, fname, r, n))
        pairs.sort(key=lambda t: -abs(t[2]))
        results[action] = pairs[:5]
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=None,
                         help='single k to run (uses pressure_ap_features_k{k}.csv)')
    parser.add_argument('--compare-k', nargs='+', type=int, default=None,
                         help='multiple k values to run and compare for stability')
    parser.add_argument('--csv', type=str, default=None,
                         help='explicit features CSV path (overrides --k)')
    args = parser.parse_args()

    if args.compare_k:
        all_results = {}
        for k in args.compare_k:
            path = os.path.join(OUTPUT_DIR, f'pressure_ap_features_k{k}.csv')
            all_results[k] = top5_per_procedure(path)

        for action in ACTIONS:
            print(f"\n=== {action} ===")
            for k in args.compare_k:
                print(f"  k={k}:")
                for adj, fname, r, n in all_results[k][action]:
                    print(f"    {adj:14s} {fname:24s} r={r:+.3f} (n={n})")
            sets = [set((adj, fname) for adj, fname, r, n in all_results[k][action])
                    for k in args.compare_k]
            overlap_all = set.intersection(*sets)
            union_all = set.union(*sets)
            print(f"  overlap across all k in top5: {len(overlap_all)}/5 "
                  f"({sorted(overlap_all)})")
            print(f"  union size: {len(union_all)}")
        return

    csv_path = args.csv or os.path.join(OUTPUT_DIR, f'pressure_ap_features_k{args.k}.csv')
    results = top5_per_procedure(csv_path)
    for action in ACTIONS:
        print(f"\n=== {action} (top 5 |r|) ===")
        for adj, fname, r, n in results[action]:
            print(f"  {adj:14s} {fname:24s} r={r:+.3f} (n={n})")


if __name__ == '__main__':
    main()
