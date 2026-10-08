"""
Pipeline v3 — final spot check on fragment-merged crops.

Selects 5 fragment-merged trials from image_crop_full_run_log.csv (min radius,
max radius, and ~p25/p50/p75 of merged-trial radii), re-runs detection on the
original image for the overlay (circle + padded crop box), and compares
against the already-saved cropped+resized result in
DATA_Final/hdf5_cropped_images/. Does NOT modify DATA_Final/hdf5/ or
hdf5_cropped_images/ — read-only spot check.
"""
import csv
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_prep'))  # <repo>/data_prep

HDF5_DIR = paths.hdf5_dir()
CROPPED_DIR = paths.cropped_dir()
OUTPUT_DIR = paths.output_dir(__file__)
# Log written by data_prep/crop_images.py.
LOG_CSV = os.path.join(paths.output_dir('crop_images.py', create=False), 'image_crop_full_run_log.csv')

from image_crop import detect_circle, PADDING_FRAC  # noqa: E402


def load_merged_rows():
    rows = []
    with open(LOG_CSV, newline='') as f:
        for row in csv.DictReader(f):
            if row['status'] == 'success' and row['fragment_merged'] == 'True':
                row['radius'] = float(row['radius'])
                rows.append(row)
    return rows


def pick_spotcheck_rows(rows):
    rows_sorted = sorted(rows, key=lambda r: r['radius'])
    n = len(rows_sorted)
    idx_min = 0
    idx_max = n - 1
    idx_p25 = int(round(0.25 * (n - 1)))
    idx_p50 = int(round(0.50 * (n - 1)))
    idx_p75 = int(round(0.75 * (n - 1)))
    picks = {}
    for label, idx in [('min', idx_min), ('p25', idx_p25), ('p50', idx_p50),
                        ('p75', idx_p75), ('max', idx_max)]:
        picks.setdefault(idx, label)
    selected = [(picks[idx], rows_sorted[idx]) for idx in sorted(picks.keys())]
    return selected


def main():
    rows = load_merged_rows()
    print(f"Found {len(rows)} fragment-merged trials in log")
    selected = pick_spotcheck_rows(rows)

    results = []
    for label, row in selected:
        obj_id, proc, trial = row['object_id'], row['procedure'], row['trial']
        orig_path = os.path.join(HDF5_DIR, f'{obj_id}.h5')
        cropped_path = os.path.join(CROPPED_DIR, f'{obj_id}.h5')

        with h5py.File(orig_path, 'r') as f:
            orig_img = f[proc][trial]['image'][()].astype(np.float32)
        with h5py.File(cropped_path, 'r') as f:
            cropped_img = f[proc][trial]['image'][()].astype(np.float32)

        gray_u8 = (np.clip(orig_img.mean(axis=2), 0, 1) * 255).astype(np.uint8)
        cx, cy, r, n_fragments = detect_circle(gray_u8)

        results.append({
            'label': label, 'obj_id': obj_id, 'procedure': proc, 'trial': trial,
            'radius_logged': row['radius'], 'n_fragments_logged': row['n_fragments'],
            'orig_img': orig_img, 'cropped_img': cropped_img,
            'cx': cx, 'cy': cy, 'r': r, 'n_fragments': n_fragments,
        })
        print(f"[{label}] {obj_id}/{proc}/{trial}: radius(log)={row['radius']:.2f} "
              f"radius(recomputed)={r:.2f} n_fragments={n_fragments}")

    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(2.8 * n, 6.0))
    for col, res in enumerate(results):
        ax = axes[0, col]
        ax.imshow(np.clip(res['orig_img'], 0, 1))
        circle = plt.Circle((res['cx'], res['cy']), res['r'], color='lime',
                             fill=False, linewidth=1.5)
        ax.add_patch(circle)
        r_pad = res['r'] * (1.0 + PADDING_FRAC)
        rect = plt.Rectangle((res['cx'] - r_pad, res['cy'] - r_pad),
                              2 * r_pad, 2 * r_pad, edgecolor='yellow',
                              fill=False, linewidth=1.2, linestyle='--')
        ax.add_patch(rect)
        ax.set_title(f"{res['label']}: {res['obj_id']}/{res['procedure']}/{res['trial']}\n"
                     f"r={res['r']:.1f}px (orig)", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[1, col]
        ax.imshow(np.clip(res['cropped_img'], 0, 1))
        ax.set_title('cropped+resized', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle('Fragment-merged trials spot check: original w/ detected circle (green) + '
                 'crop box (yellow, dashed) vs cropped+resized 224x224')
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'image_crop_merged_spotcheck.png')
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")

    print("\nSelected trials:")
    for res in results:
        print(f"  [{res['label']}] {res['obj_id']} / {res['procedure']} / {res['trial']} "
              f"— radius={res['radius_logged']:.2f}px, n_fragments={res['n_fragments_logged']}")


if __name__ == '__main__':
    main()
