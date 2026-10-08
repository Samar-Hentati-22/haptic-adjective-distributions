"""
Pipeline v3 — cross-trial stability check for the fixed-length whole-trial
MFCC adaptation in src/imu_features.py.

Investigation/exploration script (not production code, stays outside
src/). Compares repeat trials of the same object/procedure: if a
coefficient reflects real, reproducible signal, trials of the *same*
object/procedure should cluster; if it's noise, they should scatter as
much as trials of *different* objects would.

Read-only, does not modify DATA_Final/.
"""
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import librosa

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402

HDF5_DIR = paths.hdf5_dir()
OUTPUT_DIR = paths.output_dir(__file__)

from imu_features import (  # noqa: E402
    deinterleave_imu, dft321, _highpass, extract_imu_features,
    DEFAULT_FS, N_MFCC, FIXED_ANALYSIS_SEC,
)

FS = DEFAULT_FS
PROCEDURE = 'squeeze'
OBJECTS = ['obj_01', 'obj_05', 'obj_30', 'obj_45']


def mel_and_mfcc(acc):
    """Reproduce src/imu_features.py's fixed-window MFCC, exposing the
    intermediate mel-band energy for plotting."""
    combined = dft321(acc.astype(np.float64))
    filtered = _highpass(combined, fs=FS)

    fixed_len = max(8, int(round(FIXED_ANALYSIS_SEC * FS)))
    n = filtered.shape[0]
    if n >= fixed_len:
        x = filtered[:fixed_len]
    else:
        x = np.pad(filtered, (0, fixed_len - n))

    n_freq_bins = fixed_len // 2 + 1
    n_mels = max(min(26, n_freq_bins - 1), N_MFCC)

    S = librosa.stft(x.astype(np.float64), n_fft=fixed_len, hop_length=fixed_len,
                      win_length=fixed_len, window='hann', center=False)
    mag2 = np.abs(S) ** 2
    mel_fb = librosa.filters.mel(sr=FS, n_fft=fixed_len, n_mels=n_mels)
    mel_energy = (mel_fb @ mag2)[:, 0]
    log_mel_db = librosa.power_to_db(mel_energy[:, None])
    mfcc = librosa.feature.mfcc(S=log_mel_db, n_mfcc=N_MFCC)[:, 0]
    return mel_energy, mfcc


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.set_printoptions(precision=3, suppress=True, linewidth=140)

    per_object = {}  # obj_id -> {'trials': [...], 'mfcc': (n_trials,13), 'mel': (n_trials,n_mels)}
    for obj_id in OBJECTS:
        with h5py.File(os.path.join(HDF5_DIR, f'{obj_id}.h5'), 'r') as f:
            trial_names = sorted(f[PROCEDURE].keys())
            mfccs, mels, kept_trials = [], [], []
            for t in trial_names:
                imu_raw = f[PROCEDURE][t]['imu_raw'][()].astype(np.float32)
                acc = deinterleave_imu(imu_raw)[:, 0:3]
                if acc.shape[0] == 0:
                    print(f"  skipping {obj_id}/{PROCEDURE}/{t}: dropout trial (empty ACC substream)")
                    continue
                mel_energy, mfcc = mel_and_mfcc(acc)
                mfccs.append(mfcc)
                mels.append(mel_energy)
                kept_trials.append(t)
        per_object[obj_id] = {
            'trials': kept_trials,
            'mfcc': np.array(mfccs),   # (n_trials, 13)
            'mel': np.array(mels),     # (n_trials, n_mels)
        }
        print(f"{obj_id}/{PROCEDURE}: {len(kept_trials)} usable trials -> {kept_trials}")

    # ── Cross-trial stability report (std/|mean| per coefficient, per object) ──
    print("\n" + "=" * 100)
    print(f"Cross-trial coefficient stability (std/|mean| across trials of the SAME "
          f"object/{PROCEDURE})")
    print("=" * 100)
    all_ratios = []
    for obj_id, d in per_object.items():
        mfcc = d['mfcc']
        mean = mfcc.mean(axis=0)
        std = mfcc.std(axis=0)
        ratio = std / np.abs(mean).clip(min=0.5)
        all_ratios.append(ratio)
        print(f"\n{obj_id} ({mfcc.shape[0]} trials):")
        print(f"  mean : {mean}")
        print(f"  std  : {std}")
        print(f"  ratio: {ratio}")
        print(f"  c0-c4  avg ratio: {ratio[:5].mean():.3f}   c5-c12 avg ratio: {ratio[5:].mean():.3f}")

    all_ratios = np.array(all_ratios)
    print(f"\nPooled across {len(OBJECTS)} objects — mean ratio per coefficient:")
    print(f"  {all_ratios.mean(axis=0)}")
    print(f"  c0-c4 avg: {all_ratios[:, :5].mean():.3f}   c5-c12 avg: {all_ratios[:, 5:].mean():.3f}")

    # ── Plot 1: mel-band energy spectrum, one example trial per object ─────
    fig, ax = plt.subplots(figsize=(9, 5))
    for obj_id, d in per_object.items():
        mel_db = librosa.power_to_db(d['mel'][0])
        ax.plot(mel_db, marker='o', markersize=3, label=f"{obj_id}/{PROCEDURE}/{d['trials'][0]}")
    ax.set_xlabel('mel band index')
    ax.set_ylabel('log-mel energy (dB)')
    ax.set_title(f'Mel-band energy spectrum, fixed {FIXED_ANALYSIS_SEC:.0f}s window '
                 f'(fs={FS}Hz) — one example trial per object')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    mel_path = os.path.join(OUTPUT_DIR, 'imu_mel_spectrum_check.png')
    fig.savefig(mel_path, dpi=150)
    print(f"\nSaved {mel_path}")

    # ── Plot 2: MFCC coefficient stability, trials overlaid per object ─────
    n_obj = len(OBJECTS)
    fig, axes = plt.subplots(2, (n_obj + 1) // 2, figsize=(6 * ((n_obj + 1) // 2), 8), squeeze=False)
    axes = axes.flatten()
    for i, (obj_id, d) in enumerate(per_object.items()):
        ax = axes[i]
        for j, t in enumerate(d['trials']):
            ax.plot(range(N_MFCC), d['mfcc'][j], marker='o', markersize=4,
                     label=t, alpha=0.85)
        ax.axvspan(4.5, N_MFCC - 0.5, color='gray', alpha=0.08)
        ax.set_xticks(range(N_MFCC))
        ax.set_xlabel('MFCC coefficient index')
        ax.set_ylabel('value')
        ax.set_title(f'{obj_id}/{PROCEDURE} ({len(d["trials"])} trials)')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
    for i in range(len(per_object), len(axes)):
        axes[i].axis('off')
    fig.suptitle('MFCC coefficients across repeat trials, same object/procedure '
                 '(shaded region = c5-c12)')
    fig.tight_layout()
    mfcc_path = os.path.join(OUTPUT_DIR, 'imu_mfcc_stability_check.png')
    fig.savefig(mfcc_path, dpi=150)
    print(f"Saved {mfcc_path}")


if __name__ == '__main__':
    main()
