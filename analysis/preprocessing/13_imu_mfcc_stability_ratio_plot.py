"""
Pipeline v3 -- MFCC cross-trial stability RATIO figure.

Companion to 12_imu_feature_stability_check.py. That script's figure
(imu_mfcc_stability_check.png) plots raw per-trial MFCC coefficient
values, which visually makes c0-c4 look *more* spread than c5-c12 --
but the actual "keep c0-c4, drop c5-c12" decision was made on a
normalized stability ratio (std across trials / clipped |mean| across
trials), not raw spread. This script plots that ratio directly, one
line per object, so the metric that drove the decision is visible
instead of implied.

n_mfcc is forced to 13 here via _mfcc_features(..., n_mfcc=13):
src/imu_features.py now hardcodes N_MFCC=5 in production (c5-c12
already dropped downstream), so the default path no longer emits
enough coefficients to reproduce this check.

Read-only, does not modify DATA_Final/.
"""
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402

HDF5_DIR = paths.hdf5_dir()
OUTPUT_DIR = paths.output_dir(__file__)

from imu_features import (  # noqa: E402
    deinterleave_imu, dft321, _highpass, _mfcc_features, DEFAULT_FS,
)

FS = DEFAULT_FS
PROCEDURE = 'squeeze'
OBJECTS = ['obj_01', 'obj_05', 'obj_30', 'obj_45']
N_MFCC_FULL = 13


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.set_printoptions(precision=3, suppress=True, linewidth=140)

    per_object = {}  # obj_id -> {'trials': [...], 'mfcc': (n_trials, 13)}
    for obj_id in OBJECTS:
        with h5py.File(os.path.join(HDF5_DIR, f'{obj_id}.h5'), 'r') as f:
            trial_names = sorted(f[PROCEDURE].keys())
            mfccs, kept_trials = [], []
            for t in trial_names:
                imu_raw = f[PROCEDURE][t]['imu_raw'][()].astype(np.float32)
                acc = deinterleave_imu(imu_raw)[:, 0:3]
                if acc.shape[0] == 0:
                    print(f"  skipping {obj_id}/{PROCEDURE}/{t}: dropout trial (empty ACC substream)")
                    continue
                combined = dft321(acc.astype(np.float64))
                filtered = _highpass(combined, fs=FS)
                mfcc = _mfcc_features(filtered, fs=FS, n_mfcc=N_MFCC_FULL)
                mfccs.append(mfcc)
                kept_trials.append(t)
        per_object[obj_id] = {
            'trials': kept_trials,
            'mfcc': np.array(mfccs),  # (n_trials, 13)
        }
        print(f"{obj_id}/{PROCEDURE}: {len(kept_trials)} usable trials -> {kept_trials}")

    # ── Stability ratio per object: std across trials / clip(|mean|, 0.5) ──
    print("\n" + "=" * 100)
    print("Cross-trial coefficient stability ratio (std / clip(|mean|, min=0.5)) "
          f"across trials of the SAME object/{PROCEDURE}")
    print("=" * 100)
    all_ratios = []
    for obj_id, d in per_object.items():
        mfcc = d['mfcc']
        mean = mfcc.mean(axis=0)
        std = mfcc.std(axis=0)
        ratio = std / np.abs(mean).clip(min=0.5)
        all_ratios.append(ratio)
        print(f"\n{obj_id} ({mfcc.shape[0]} trials):")
        print(f"  ratio: {ratio}")
        print(f"  c0-c4  avg ratio: {ratio[:5].mean():.3f}   c5-c12 avg ratio: {ratio[5:].mean():.3f}")
    per_object_ratio = {obj_id: r for obj_id, r in zip(per_object.keys(), all_ratios)}

    all_ratios = np.array(all_ratios)
    print(f"\nPooled across {len(OBJECTS)} objects — mean ratio per coefficient:")
    print(f"  {all_ratios.mean(axis=0)}")
    print(f"  c0-c4 avg: {all_ratios[:, :5].mean():.3f}   c5-c12 avg: {all_ratios[:, 5:].mean():.3f}")

    # ── Plot: stability ratio, one line per object, same 4-panel layout ────
    n_obj = len(OBJECTS)
    fig, axes = plt.subplots(2, (n_obj + 1) // 2, figsize=(6 * ((n_obj + 1) // 2), 8), squeeze=False)
    axes = axes.flatten()
    for i, (obj_id, d) in enumerate(per_object.items()):
        ax = axes[i]
        ratio = per_object_ratio[obj_id]
        ax.plot(range(N_MFCC_FULL), ratio, marker='o', markersize=4,
                 color='tab:blue', label=f'{obj_id}/{PROCEDURE} ({len(d["trials"])} trials)')
        ax.axvspan(4.5, N_MFCC_FULL - 0.5, color='gray', alpha=0.08)
        ax.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.5,
                    label='ratio = 1 (std = mean)')
        ax.set_xticks(range(N_MFCC_FULL))
        ax.set_xlabel('MFCC coefficient index')
        ax.set_ylabel('Stability ratio (std / clipped mean)')
        ax.set_title(f'{obj_id}/{PROCEDURE} ({len(d["trials"])} trials)')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    for i in range(len(per_object), len(axes)):
        axes[i].axis('off')
    fig.suptitle('MFCC coefficient stability ratio across repeat trials, same object/procedure '
                 '(shaded region = c5-c12)')
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'imu_mfcc_stability_ratio.png')
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
