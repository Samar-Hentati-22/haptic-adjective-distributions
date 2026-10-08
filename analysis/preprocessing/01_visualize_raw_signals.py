"""
Pipeline v3 — step 01: visualize raw, unprocessed sensor signals.

Picks the 2 objects with the lowest and 2 with the highest mean "hard"
rating (TACTILE_LABELS_DIR/labels_mean_phase2.npy) and, for one
squeeze trial per object, plots the raw data exactly as stored in the
HDF5 files (TACTILE_DATA_ROOT/hdf5/obj_XX.h5) — no preprocessing, no
normalization, no de-interleaving of imu_raw.

This is a standalone visualization script for the fresh pipeline_v3
effort. It does not import from or modify any existing Training_Phase2
code (dataset.py, model.py, train.py, ...).
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
LABELS_DIR = paths.labels_dir()
OUTPUT_PATH = os.path.join(paths.output_dir(__file__), '01_raw_signals_hard_vs_soft.png')

ADJECTIVES = [
    'hard', 'soft', 'rough', 'smooth', 'bumpy', 'flat',
    'sticky', 'slippery', 'cold', 'warm', 'wet', 'dry',
    'compressible', 'firm', 'fuzzy', 'scratchy', 'pleasant', 'irritating',
]
HARD_IDX = ADJECTIVES.index('hard')

ACTION = 'squeeze'  # image is valid for squeeze/hold trials
N_PER_SIDE = 2       # 2 lowest + 2 highest mean "hard" rating


def select_objects():
    mean = np.load(os.path.join(LABELS_DIR, 'labels_mean_phase2.npy'))
    obj_ids = np.load(os.path.join(LABELS_DIR, 'object_ids_phase2.npy'), allow_pickle=True)
    hard_rating = mean[:, HARD_IDX]
    order = np.argsort(hard_rating)
    chosen = np.concatenate([order[:N_PER_SIDE], order[-N_PER_SIDE:]])
    return [(str(obj_ids[i]), float(hard_rating[i])) for i in chosen]


def pick_valid_trial(obj_group, action):
    """First trial under `action` with a non-empty, non-all-zero image."""
    action_group = obj_group[action]
    for trial_name in sorted(action_group.keys()):
        trial = action_group[trial_name]
        img = trial['image'][()]
        pap = trial['pressure_ap'][()]
        if img.size > 0 and pap.shape[0] > 0 and not np.all(img == 0):
            return trial_name, trial
    raise RuntimeError(f"No valid '{action}' trial with a real image found")


def load_trial_data(obj_id):
    h5_path = os.path.join(HDF5_DIR, f'{obj_id}.h5')
    with h5py.File(h5_path, 'r') as f:
        trial_name, trial = pick_valid_trial(f, ACTION)
        data = {
            'trial_name': trial_name,
            'image': trial['image'][()].astype(np.float32),
            'pressure_ap': trial['pressure_ap'][()].astype(np.float32),
            'imu_raw': trial['imu_raw'][()].astype(np.float32),
            'pressure': trial['pressure'][()].astype(np.float32),
            'audio': trial['audio'][()].astype(np.float32),
        }
    return data


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    selected = select_objects()
    # left-to-right: softest -> hardest
    selected = sorted(selected, key=lambda x: x[1])
    print("Selected objects (object_id, mean 'hard' rating):")
    for obj_id, rating in selected:
        print(f"  {obj_id}: {rating:.2f}")

    trials = [load_trial_data(obj_id) for obj_id, _ in selected]

    n_cols = len(selected)
    row_labels = [
        'tactile image',
        'pressure_ap (raw ADC heatmap)',
        'imu_raw (raw, interleaved)',
        'pressure (raw ADC)',
        'audio mel spectrogram (raw)',
    ]
    n_rows = len(row_labels)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows),
                              constrained_layout=True)

    # Row-wide color/axis scales so objects are visually comparable within a
    # row. pressure_ap is clipped to the 1st-99th percentile (pooled across
    # the 4 objects) so rare outlier spikes don't wash out the rest of the
    # heatmap; the underlying data plotted is untouched. pressure's two
    # channels differ by ~3 orders of magnitude, so a single pooled
    # percentile range clips one channel off-screen entirely — each channel
    # gets its own percentile-based y-range instead (twin axes below).
    pap_pool = np.concatenate([d['pressure_ap'].ravel() for d in trials])
    pap_vmin, pap_vmax = np.percentile(pap_pool, [1, 99])
    pressure_ch_pool = [np.concatenate([d['pressure'][:, c] for d in trials])
                         for c in range(trials[0]['pressure'].shape[1])]
    pressure_ch_range = [np.percentile(p, [1, 99]) for p in pressure_ch_pool]
    audio_vmin = min(d['audio'][0].min() for d in trials)
    audio_vmax = max(d['audio'][0].max() for d in trials)

    pap_im = None
    audio_im = None

    for col, ((obj_id, rating), data) in enumerate(zip(selected, trials)):
        # Row 0: raw tactile camera image
        ax = axes[0, col]
        ax.imshow(np.clip(data['image'], 0.0, 1.0))
        ax.set_title(f"{obj_id}\ntrue hard = {rating:.2f}\n(trial: {data['trial_name']})",
                     fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

        # Row 1: pressure_ap heatmap (time x 50 channels), raw ADC values
        ax = axes[1, col]
        pap = data['pressure_ap']  # (T, 50)
        pap_im = ax.imshow(pap.T, aspect='auto', origin='lower', cmap='viridis',
                            vmin=pap_vmin, vmax=pap_vmax)
        ax.set_xlabel('time step')
        if col == 0:
            ax.set_ylabel('channel (0-49)')

        # Row 2: imu_raw line plot, raw interleaved, all 3 columns
        ax = axes[2, col]
        imu = data['imu_raw']  # (T, 3)
        for c in range(imu.shape[1]):
            ax.plot(imu[:, c], linewidth=0.6, label=f'col {c}')
        ax.set_xlabel('sample index (interleaved)')
        if col == 0:
            ax.set_ylabel('raw ADC value')
        ax.legend(fontsize=7, loc='upper right')

        # Row 3: pressure line plot, raw, both channels. ch0 and ch1 live on
        # very different scales, so each gets its own y-axis (twinx) with
        # its own 1st-99th percentile range pooled across the 4 objects.
        ax = axes[3, col]
        ax2 = ax.twinx()
        pr = data['pressure']  # (T, 2)
        channel_axes = [ax, ax2]
        channel_colors = ['tab:blue', 'tab:orange']
        lines = []
        for c in range(pr.shape[1]):
            line, = channel_axes[c].plot(pr[:, c], linewidth=1.0,
                                          color=channel_colors[c], label=f'ch {c}')
            lines.append(line)
            channel_axes[c].set_ylim(*pressure_ch_range[c])
        ax.set_xlabel('time step')
        if col == 0:
            ax.set_ylabel('ch 0 raw ADC (1st-99th pct)', color=channel_colors[0])
        if col == n_cols - 1:
            ax2.set_ylabel('ch 1 raw ADC (1st-99th pct)', color=channel_colors[1])
        ax.legend(lines, [l.get_label() for l in lines], fontsize=7, loc='upper right')

        # Row 4: raw audio mel spectrogram (channel 0)
        ax = axes[4, col]
        mel = data['audio'][0]  # (64 mel bins, T)
        audio_im = ax.imshow(mel, aspect='auto', origin='lower', cmap='magma',
                              vmin=audio_vmin, vmax=audio_vmax)
        ax.set_xlabel('time frame')
        if col == 0:
            ax.set_ylabel('mel bin (channel 0)')

    for row in range(n_rows):
        axes[row, 0].annotate(
            row_labels[row], xy=(0, 0.5), xytext=(-axes[row, 0].yaxis.labelpad - 45, 0),
            xycoords=axes[row, 0].yaxis.label, textcoords='offset points',
            fontsize=12, fontweight='bold', ha='right', va='center', rotation=90)

    fig.colorbar(pap_im, ax=axes[1, :].tolist(), shrink=0.8,
                 label='pressure_ap raw ADC value (clipped to 1st-99th pct)')
    fig.colorbar(audio_im, ax=axes[4, :].tolist(), shrink=0.8, label='mel spectrogram value')

    fig.suptitle(
        "Raw sensor signals — 2 softest vs 2 hardest objects (squeeze trial), "
        "no preprocessing applied", fontsize=14, fontweight='bold')

    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved figure to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
