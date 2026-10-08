"""
Pipeline v3 — smoke test for src/pressure_ap_features.py on 10 real trials
(mixed objects/procedures). Prints input shape -> output shape per trial,
confirms output is always (224, 50), and plots raw vs glitch-cleaned vs
resampled heatmaps for 3 example trials (short/medium/long duration among
the 10, same picks as before glitch cleaning was added).

Investigation script (stays outside src/). Read-only, does not modify
DATA_Final/.
"""
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402

HDF5_DIR = paths.hdf5_dir()
OUTPUT_DIR = paths.output_dir(__file__)

from pressure_ap_features import (  # noqa: E402
    extract_pressure_ap_features, clean_pressure_ap_glitches,
    DEFAULT_FS, DEFAULT_TARGET_LEN,
)

# (object_id, procedure, trial) — same set used for the imu_features smoke test
TEST_TRIALS = [
    ('obj_01', 'tap', 'trial_000'),
    ('obj_05', 'squeeze', 'trial_001'),
    ('obj_10', 'slide', 'trial_002'),
    ('obj_15', 'hold', 'trial_000'),
    ('obj_20', 'tap', 'trial_003'),
    ('obj_30', 'squeeze', 'trial_002'),
    ('obj_45', 'slide', 'trial_001'),
    ('obj_60', 'hold', 'trial_004'),
    ('obj_70', 'tap', 'trial_002'),
    ('obj_85', 'squeeze', 'trial_000'),
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    for obj_id, proc, trial in TEST_TRIALS:
        h5_path = os.path.join(HDF5_DIR, f'{obj_id}.h5')
        with h5py.File(h5_path, 'r') as f:
            pap = f[proc][trial]['pressure_ap'][()].astype(np.float32)

        cleaned = clean_pressure_ap_glitches(pap)
        out = extract_pressure_ap_features(pap, fs=DEFAULT_FS, target_len=DEFAULT_TARGET_LEN)
        assert out.shape == (DEFAULT_TARGET_LEN, 50), f"unexpected output shape {out.shape}"
        results.append({'obj_id': obj_id, 'proc': proc, 'trial': trial,
                         'pap': pap, 'cleaned': cleaned, 'out': out})
        print(f"{obj_id}/{proc}/{trial}: input shape={pap.shape} -> output shape={out.shape}  "
              f"(raw max={pap.max():.0f}, cleaned max={cleaned.max():.0f})")

    print(f"\nAll {len(results)} trials produced output shape ({DEFAULT_TARGET_LEN}, 50). Confirmed.")

    # ── Pick short/medium/long trials (by input length) — same picks as before ──
    by_len = sorted(results, key=lambda r: r['pap'].shape[0])
    picks = [by_len[0], by_len[len(by_len) // 2], by_len[-1]]

    fig, axes = plt.subplots(3, len(picks), figsize=(5 * len(picks), 10.5))
    for col, res in enumerate(picks):
        n = res['pap'].shape[0]
        duration_sec = n / DEFAULT_FS
        label = f"{res['obj_id']}/{res['proc']}/{res['trial']}"

        # Shared color scale per trial, derived from the CLEANED data's real
        # baseline range -- lets the raw panel's glitch outlier visibly
        # saturate the scale, while cleaned/resampled show genuine variation.
        vmin, vmax = res['cleaned'].min(), res['cleaned'].max()

        ax = axes[0, col]
        im = ax.imshow(res['pap'], aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax,
                        extent=[0, 50, duration_sec, 0])
        ax.set_title(f"{label}\nraw (N={n}, {duration_sec:.2f}s, max={res['pap'].max():.0f})",
                     fontsize=9)
        ax.set_xlabel('channel')
        ax.set_ylabel('time (s)')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1, col]
        im = ax.imshow(res['cleaned'], aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax,
                        extent=[0, 50, duration_sec, 0])
        ax.set_title(f"glitch-cleaned (N={n}, max={res['cleaned'].max():.0f})", fontsize=9)
        ax.set_xlabel('channel')
        ax.set_ylabel('time (s)')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[2, col]
        im = ax.imshow(res['out'], aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax,
                        extent=[0, 50, duration_sec, 0])
        ax.set_title(f"resampled (N={DEFAULT_TARGET_LEN}, max={res['out'].max():.0f})", fontsize=9)
        ax.set_xlabel('channel')
        ax.set_ylabel('time (s)')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('pressure_ap: raw (top) vs glitch-cleaned (middle) vs resampled to 224 '
                 'timesteps (bottom) — color scale shared per trial, derived from cleaned data')
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'pressure_ap_resample_check.png')
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
