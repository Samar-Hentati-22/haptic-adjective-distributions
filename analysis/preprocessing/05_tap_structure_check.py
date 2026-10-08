"""
Pipeline v3 — Step A: verify tap's expected quiet -> contact structure
across pressure_ap, IMU (ACC via DFT321), and audio, on a shared time axis.

Diagnostic only: no changes to dataset.py / model.py / train.py / any
existing pipeline.

Sampling rates (all user-provided; none recoverable from the data itself —
no timestamps and no mcap_to_dataset.py or equivalent conversion script
exist anywhere in this environment to verify against):
  pressure_ap: 248 Hz
  imu_raw ACC sub-stream (de-interleaved, native 438 Hz / 4): 109.5 Hz
  audio mel spectrogram: 48000 / 512 = 93.75 Hz (SAMPLE_RATE=48000, HOP_LENGTH=512)
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
from imu_features import deinterleave_imu, dft321  # noqa: E402

HDF5_DIR = paths.hdf5_dir()
OUTPUT_DIR = paths.output_dir(__file__)

PRESSURE_AP_RATE_HZ = 248.0
IMU_ACC_RATE_HZ = 438.0 / 4.0
AUDIO_RATE_HZ = 48000.0 / 512.0

CANDIDATE_OBJECT_NUMS = [1, 10, 20, 30, 40, 50, 60, 70, 80, 5, 15, 25, 35, 45, 55, 65, 75, 85]
N_TRIALS_TARGET = 10


def contact_intensity(pap):
    n_base = max(3, min(10, pap.shape[0] // 10))
    baseline = pap[:n_base, :].mean(axis=0)
    return np.abs(pap - baseline[None, :]).mean(axis=1)


def try_trial(f, trial_name):
    trial = f['tap'][trial_name]
    pap = trial['pressure_ap'][()].astype(np.float64)
    imu = trial['imu_raw'][()].astype(np.float64)
    audio = trial['audio'][()].astype(np.float64)

    if pap.shape[0] < 10:
        return None
    deint = deinterleave_imu(imu.astype(np.float32))
    if deint.shape[0] < 16:
        return None
    if audio.shape[-1] < 5 or np.all(audio == 0):
        return None

    pap_intensity = contact_intensity(pap)
    acc_combined = dft321(deint[:, 0:3].astype(np.float64))
    audio_energy = np.abs(audio).mean(axis=(0, 1))

    return {
        'pap_t': np.arange(len(pap_intensity)) / PRESSURE_AP_RATE_HZ,
        'pap_intensity': pap_intensity,
        'acc_t': np.arange(len(acc_combined)) / IMU_ACC_RATE_HZ,
        'acc_combined': acc_combined,
        'audio_t': np.arange(len(audio_energy)) / AUDIO_RATE_HZ,
        'audio_energy': audio_energy,
    }


def select_trials():
    records = []
    for obj_num in CANDIDATE_OBJECT_NUMS:
        obj_id = f'obj_{obj_num:02d}'
        h5_path = os.path.join(HDF5_DIR, f'{obj_id}.h5')
        if not os.path.isfile(h5_path):
            continue
        with h5py.File(h5_path, 'r') as f:
            if 'tap' not in f:
                continue
            for trial_name in sorted(f['tap'].keys()):
                result = try_trial(f, trial_name)
                if result is not None:
                    records.append({'obj_id': obj_id, 'trial_name': trial_name, **result})
                    break
        if len(records) >= N_TRIALS_TARGET:
            break
    return records


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    records = select_trials()
    print(f"Selected {len(records)} tap trials: "
          f"{[(r['obj_id'], r['trial_name']) for r in records]}")

    n = len(records)
    fig, axes = plt.subplots(3, n, figsize=(3.2 * n, 8), sharey='row')

    for col, r in enumerate(records):
        ax = axes[0, col]
        ax.plot(r['pap_t'], r['pap_intensity'], color='tab:blue', linewidth=1)
        ax.set_title(f"{r['obj_id']}\n{r['trial_name']}", fontsize=10)
        if col == 0:
            ax.set_ylabel('pressure_ap\nintensity')

        ax = axes[1, col]
        ax.plot(r['acc_t'], r['acc_combined'], color='tab:green', linewidth=1)
        if col == 0:
            ax.set_ylabel('IMU ACC\n(DFT321 combined)')

        ax = axes[2, col]
        ax.plot(r['audio_t'], r['audio_energy'], color='tab:red', linewidth=1)
        ax.set_xlabel('time (s)')
        if col == 0:
            ax.set_ylabel('audio energy\n(mean |mel|)')

    fig.suptitle('Tap trial structure across modalities: pressure_ap intensity / IMU ACC (DFT321) / audio energy')
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'tap_structure_check.png')
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
