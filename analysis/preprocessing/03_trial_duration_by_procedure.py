"""
Pipeline v3 — Analysis 2: true (unpadded) trial duration in seconds per
procedure, for imu_raw / pressure_ap / pressure.

Diagnostic only: no changes to dataset.py / model.py / train.py / any
existing pipeline.

Sampling rates used to convert raw sample counts to seconds (none of these
are recoverable from the data itself — no timestamps exist anywhere in the
HDF5 files or the raw dataset/ npy tree; confirmed with the user):
  imu_raw (raw interleaved message stream): 438 Hz
  pressure_ap (50-channel tactile array):   248 Hz
  pressure (2-channel barometric/reference): 56 Hz
"""
import csv
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

ACTIONS = ['tap', 'squeeze', 'slide', 'hold']
MODALITY_RATE_HZ = {
    'imu_raw': 438.0,
    'pressure_ap': 248.0,
    'pressure': 56.0,
}


def collect_durations():
    """{modality: {action: [duration_seconds, ...]}}"""
    durations = {m: {a: [] for a in ACTIONS} for m in MODALITY_RATE_HZ}
    n_trials = 0
    for obj_num in range(1, 86):
        obj_id = f'obj_{obj_num:02d}'
        h5_path = os.path.join(HDF5_DIR, f'{obj_id}.h5')
        if not os.path.isfile(h5_path):
            continue
        with h5py.File(h5_path, 'r') as f:
            for action in ACTIONS:
                if action not in f:
                    continue
                for trial_name in f[action]:
                    trial = f[action][trial_name]
                    n_trials += 1
                    for modality, rate in MODALITY_RATE_HZ.items():
                        n_samples = trial[modality].shape[0]
                        durations[modality][action].append(n_samples / rate)
    print(f"Scanned {n_trials} trials.")
    return durations


def summarize(durations):
    """[(modality, action, n, min, max, mean, median, p95), ...]"""
    rows = []
    for modality in MODALITY_RATE_HZ:
        for action in ACTIONS:
            vals = np.array(durations[modality][action])
            rows.append((
                modality, action, len(vals),
                float(vals.min()), float(vals.max()),
                float(vals.mean()), float(np.median(vals)),
                float(np.percentile(vals, 95)),
            ))
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    durations = collect_durations()
    rows = summarize(durations)

    print(f"\n{'modality':12s} {'action':8s} {'n':>5s} {'min_s':>8s} {'max_s':>8s} "
          f"{'mean_s':>8s} {'median_s':>9s} {'p95_s':>8s}")
    for modality, action, n, mn, mx, mean, median, p95 in rows:
        print(f"{modality:12s} {action:8s} {n:5d} {mn:8.3f} {mx:8.3f} "
              f"{mean:8.3f} {median:9.3f} {p95:8.3f}")

    csv_path = os.path.join(OUTPUT_DIR, 'trial_duration_summary.csv')
    with open(csv_path, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['modality', 'action', 'n_trials', 'min_s', 'max_s',
                          'mean_s', 'median_s', 'p95_s', 'sampling_rate_hz'])
        for modality, action, n, mn, mx, mean, median, p95 in rows:
            writer.writerow([modality, action, n, f'{mn:.4f}', f'{mx:.4f}',
                              f'{mean:.4f}', f'{median:.4f}', f'{p95:.4f}',
                              MODALITY_RATE_HZ[modality]])
    print(f"\nSaved summary table to {csv_path}")

    fig, axes = plt.subplots(1, len(MODALITY_RATE_HZ), figsize=(15, 5))
    for ax, modality in zip(axes, MODALITY_RATE_HZ):
        data = [durations[modality][action] for action in ACTIONS]
        ax.boxplot(data, tick_labels=ACTIONS, showfliers=True)
        ax.set_title(f'{modality} ({MODALITY_RATE_HZ[modality]:.0f} Hz)')
        ax.set_xlabel('procedure')
        ax.set_ylabel('duration (s)')
        ax.grid(True, axis='y', alpha=0.3)
    fig.suptitle('True (unpadded) trial duration by procedure, 1700 trials (85 objects x 4 actions x 5 trials)')
    fig.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, 'trial_duration_by_procedure.png')
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")


if __name__ == '__main__':
    main()
