"""
Pipeline v3 — Analysis 1: power spectral density of the IMU accelerometer
sub-stream, combined across axes via DFT321.

Diagnostic only: no changes to dataset.py / model.py / train.py / any
existing pipeline. Uses the already-implemented de-interleaving logic from
model/src/imu_features.py (imported, not modified; identical to the old v2
dataset.py version) to extract the ACC sub-stream, then combines it to 1 axis
with the verified DFT321 implementation (model/src/imu_features.py) and estimates its
PSD with scipy.signal.welch.

Sampling rate note: there is no timestamp data anywhere in this dataset
(checked HDF5 attrs/datasets and the raw dataset/ npy tree — neither has
per-sample timestamps), so the ACC rate below cannot be verified against
ground truth. It is taken as given: native combined IMU message rate
~438 Hz / 4 interleaved streams = 109.5 Hz for the de-interleaved ACC
sub-stream.
"""
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402
from imu_features import deinterleave_imu, dft321  # noqa: E402

HDF5_DIR = paths.hdf5_dir()
OUTPUT_DIR = paths.output_dir(__file__)

# Defined locally: identical to ACTIONS in the old v2 Training_Phase2/dataset.py
# (imu_features.py does not define it).
ACTIONS = ['tap', 'squeeze', 'slide', 'hold']

IMU_NATIVE_RATE_HZ = 438.0
ACC_RATE_HZ = IMU_NATIVE_RATE_HZ / 4.0  # 109.5 Hz, de-interleaved ACC stream

# Quota of trials to collect per action, totalling 10, for a mix of procedures.
ACTION_QUOTA = {'tap': 3, 'squeeze': 3, 'slide': 2, 'hold': 2}


def find_trials():
    """Scan objects/actions/trials for ones with a valid (non-empty)
    de-interleaved ACC sub-stream, until ACTION_QUOTA is satisfied. Each
    action starts scanning from a different offset into the object range
    (spread ~evenly across 1-85) so the 10 trials cover diverse objects,
    not just the first few."""
    n_objects = 85
    found = {action: [] for action in ACTIONS}
    used_objects = set()

    for action_idx, action in enumerate(ACTIONS):
        offset = action_idx * (n_objects // len(ACTIONS))
        obj_order = [((offset + i) % n_objects) + 1 for i in range(n_objects)]
        # First pass: only objects not already used by another action.
        for prefer_unused in (True, False):
            if len(found[action]) >= ACTION_QUOTA[action]:
                break
            for obj_num in obj_order:
                if len(found[action]) >= ACTION_QUOTA[action]:
                    break
                if prefer_unused and obj_num in used_objects:
                    continue
                obj_id = f'obj_{obj_num:02d}'
                h5_path = os.path.join(HDF5_DIR, f'{obj_id}.h5')
                if not os.path.isfile(h5_path):
                    continue
                with h5py.File(h5_path, 'r') as f:
                    if action not in f:
                        continue
                    for trial_name in sorted(f[action].keys()):
                        imu = f[action][trial_name]['imu_raw'][()].astype(np.float32)
                        deint = deinterleave_imu(imu)
                        if deint.shape[0] >= 16:  # enough cycles for a usable PSD
                            found[action].append((obj_id, trial_name, deint[:, 0:3]))
                            used_objects.add(obj_num)
                            break

    trials = []
    for action in ACTIONS:
        trials.extend([(obj_id, action, trial_name, acc)
                        for obj_id, trial_name, acc in found[action]])
    return trials


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trials = find_trials()
    print(f"Selected {len(trials)} trials:")
    for obj_id, action, trial_name, acc in trials:
        print(f"  {obj_id}/{action}/{trial_name}: {acc.shape[0]} ACC cycles")

    fig, ax = plt.subplots(figsize=(10, 7))
    summary_rows = []
    for obj_id, action, trial_name, acc in trials:
        combined = dft321(acc.astype(np.float64))
        nperseg = min(256, len(combined))
        freqs, psd = signal.welch(combined, fs=ACC_RATE_HZ, nperseg=nperseg)
        label = f"{obj_id}/{action}/{trial_name} (N={acc.shape[0]})"
        ax.semilogy(freqs, psd, label=label, linewidth=1.2, alpha=0.85)

        # peak frequency (excluding DC bin) for the report
        if len(freqs) > 1:
            peak_idx = 1 + np.argmax(psd[1:])
            peak_freq = freqs[peak_idx]
        else:
            peak_freq = 0.0
        dc_frac = psd[0] / psd.sum() if psd.sum() > 0 else 0.0
        summary_rows.append((obj_id, action, trial_name, acc.shape[0], peak_freq, dc_frac))

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (power / Hz)')
    ax.set_title(
        f'DFT321-combined ACC sub-stream PSD, 10 trials (fs = {ACC_RATE_HZ:.1f} Hz '
        f'= {IMU_NATIVE_RATE_HZ:.0f} Hz native / 4)')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, 'imu_dft321_psd.png')
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved figure to {out_path}")

    print("\nPer-trial summary (peak non-DC frequency, DC power fraction):")
    print(f"{'trial':45s} {'N':>5s} {'peak_hz':>8s} {'dc_frac':>8s}")
    for obj_id, action, trial_name, n, peak_freq, dc_frac in summary_rows:
        label = f"{obj_id}/{action}/{trial_name}"
        print(f"{label:45s} {n:5d} {peak_freq:8.2f} {dc_frac:8.3f}")


if __name__ == '__main__':
    main()
