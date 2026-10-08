"""
Pipeline v3 — 3-panel illustration of DFT321 on one real trial's ACC
sub-stream: raw 3-axis input, DFT321-combined output, and its power
spectrum. Uses dft321() from src/imu_features.py (not reimplemented).
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

HDF5_DIR = paths.hdf5_dir()
OUTPUT_DIR = paths.output_dir(__file__)

from imu_features import deinterleave_imu, dft321  # noqa: E402

ACC_RATE_HZ = 438.0 / 4.0  # 109.5 Hz, de-interleaved ACC stream

OBJ_ID = 'obj_49'
ACTION = 'squeeze'
TRIAL_NAME = 'trial_000'


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    h5_path = os.path.join(HDF5_DIR, f'{OBJ_ID}.h5')
    with h5py.File(h5_path, 'r') as f:
        imu_raw = f[ACTION][TRIAL_NAME]['imu_raw'][()].astype(np.float32)
    acc = deinterleave_imu(imu_raw)[:, 0:3].astype(np.float64)

    combined = dft321(acc)
    t = np.arange(acc.shape[0]) / ACC_RATE_HZ
    freqs, psd = signal.welch(combined, fs=ACC_RATE_HZ, nperseg=min(256, len(combined)))

    fig, axes = plt.subplots(3, 1, figsize=(9, 10))

    ax = axes[0]
    for i, name in enumerate(['x', 'y', 'z']):
        ax.plot(t, acc[:, i], label=name, linewidth=1.0)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Raw ACC (counts)')
    ax.set_title(f'Raw 3-axis ACC sub-stream — {OBJ_ID}/{ACTION}/{TRIAL_NAME}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, combined, color='tab:purple', linewidth=1.0)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('DFT321-combined signal')
    ax.set_title('DFT321-combined output')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.semilogy(freqs, psd, color='tab:red')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (power / Hz)')
    ax.set_title(f'Power spectrum of combined output (fs = {ACC_RATE_HZ:.1f} Hz)')
    ax.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'dft321_real_example.png')
    fig.savefig(out_path, dpi=150)
    print(out_path)


if __name__ == '__main__':
    main()
