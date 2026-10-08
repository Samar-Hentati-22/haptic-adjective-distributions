"""
Pipeline v3 -- improved mel-spectrogram figure, obj_35 (steel wool pad)
squeeze/trial_000, left/right mic channels + a difference panel.

No script in this codebase (pipeline_v3, sparsh-x-explore, or the older
Training_Phase2/ tree) was found to reference obj_35 specifically or to
produce a per-trial mel-spectrogram heatmap -- searched the full tree.
This is a fresh standalone script, following the same "real example"
convention as dft321_real_example.py / dft321_real_example_obj82.py:
loads the precomputed mel spectrogram directly from TACTILE_DATA_ROOT/hdf5/ (the
'audio' field is already a (2, 64, T) log-mel spectrogram in dB, range
observed here is -70.5..0.0 dB -- NOT a raw waveform, so no STFT/mel
computation happens here, only display).

Axis units: both axes are shown as raw indices (mel bin index, time frame
index), not seconds/Hz. The sample rate and hop length that would be
needed to convert frame index -> seconds, and the mel-filter fmax that
would be needed to convert bin index -> Hz, are not recorded anywhere in
this codebase (no timestamps, no mcap_to_hdf5.py or equivalent conversion
script exist to verify against -- see 05_tap_structure_check.py's module
docstring for the same gap on a different modality). Rather than assume
values for either, this figure deliberately stays in index units on both
axes so nothing here implies a calibration that hasn't been confirmed.
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

OBJ_ID = 'obj_35'
ACTION = 'squeeze'
TRIAL_NAME = 'trial_000'

GRID_EVERY_N_BINS = 12  # within the requested 8-16 range


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    h5_path = os.path.join(HDF5_DIR, f'{OBJ_ID}.h5')
    with h5py.File(h5_path, 'r') as f:
        mel = f[ACTION][TRIAL_NAME]['audio'][()].astype(np.float64)  # (2, 64, T), dB
    n_mels, n_frames = mel.shape[1], mel.shape[2]
    print(f"{OBJ_ID}/{ACTION}/{TRIAL_NAME}: audio shape {mel.shape} "
          f"({n_mels} mel bins, {n_frames} time frames), "
          f"range [{mel.min():.1f}, {mel.max():.1f}] dB")

    ch0, ch1 = mel[0], mel[1]
    diff = ch0 - ch1
    corr = np.corrcoef(ch0.ravel(), ch1.ravel())[0, 1]
    print(f"  channel correlation: {corr:.3f}, mean|diff|={np.abs(diff).mean():.2f} dB "
          f"-- channels are {'NEARLY IDENTICAL' if corr > 0.95 else 'meaningfully different'}, "
          f"showing the difference panel either way")

    # Each raw-channel panel gets its OWN 5th-95th pct color scale (not
    # pooled) -- ch0/ch1 differ by ~22 dB mean level here (see printed
    # correlation/diff above), and a shared scale would wash one channel
    # out entirely. Per-panel scaling maximizes within-panel bin contrast;
    # the difference panel is what actually shows the L/R level gap.
    vmin0, vmax0 = np.percentile(ch0, [5, 95])
    vmin1, vmax1 = np.percentile(ch1, [5, 95])
    diff_abs = np.percentile(np.abs(diff), 95)

    fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True)

    panels = [
        (ch0, 'Left channel (mic 0)', vmin0, vmax0, 'magma', 'dB'),
        (ch1, 'Right channel (mic 1)', vmin1, vmax1, 'magma', 'dB'),
        (diff, 'Difference (L - R)', -diff_abs, diff_abs, 'coolwarm', 'dB difference'),
    ]

    for ax, (data, title, vmn, vmx, cmap, cbar_label) in zip(axes, panels):
        img = ax.imshow(data, aspect='auto', origin='lower', cmap=cmap,
                         vmin=vmn, vmax=vmx,
                         extent=[0, n_frames, 0, n_mels])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Time (frames)')
        # Gridlines every GRID_EVERY_N_BINS mel bins, by bin index.
        for gb in range(0, n_mels, GRID_EVERY_N_BINS):
            ax.axhline(gb, color='white', linewidth=0.5, alpha=0.35, linestyle='--')
        fig.colorbar(img, ax=ax, shrink=0.85, label=cbar_label)

    axes[0].set_ylabel('Mel bin index')

    fig.suptitle(f'Mel spectrogram — {OBJ_ID} (steel wool pad) / {ACTION} / {TRIAL_NAME}\n'
                 f'{n_mels} mel bins x {n_frames} time frames, '
                 f'color = 5th-95th pct of dB values',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, 'audio_mel_spectrogram_obj35_squeeze_trial000.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
