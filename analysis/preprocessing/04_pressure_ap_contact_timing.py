"""
Pipeline v3 — Analysis 3: contact event timing in pressure_ap, to inform
windowing strategy.

Diagnostic only: no changes to dataset.py / model.py / train.py / any
existing pipeline.

For 2 trials per object (170 total, stratified across the 4 procedures),
computes a per-timestep "contact intensity" signal from pressure_ap
(mean absolute deviation from a baseline estimated at the start of the
trial), then locates the peak and measures how sharp/spread the event is.
"""
import os
import sys
import random

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
PRESSURE_AP_RATE_HZ = 248.0
MIN_VALID_SAMPLES = 10   # need at least this many timesteps for a meaningful baseline+intensity
SEED = 0


def contact_intensity(pap):
    """pap: (N, 50) -> intensity: (N,), baseline: (50,)."""
    n_base = max(3, min(10, pap.shape[0] // 10))
    baseline = pap[:n_base, :].mean(axis=0)
    intensity = np.abs(pap - baseline[None, :]).mean(axis=1)
    return intensity, baseline


def event_stats(intensity):
    """Peak index/value and the contiguous half-max width around the peak."""
    peak_idx = int(np.argmax(intensity))
    peak_val = intensity[peak_idx]
    if peak_val <= 0:
        return None
    thresh = 0.5 * peak_val
    start = peak_idx
    while start > 0 and intensity[start - 1] >= thresh:
        start -= 1
    end = peak_idx
    while end < len(intensity) - 1 and intensity[end + 1] >= thresh:
        end += 1
    width_samples = end - start + 1
    return {
        'peak_idx': peak_idx,
        'peak_val': peak_val,
        'peak_fraction': peak_idx / max(1, len(intensity) - 1),
        'width_samples': width_samples,
        'width_sec': width_samples / PRESSURE_AP_RATE_HZ,
        'thresh': thresh,
        'window': (start, end),
    }


def load_pap(f, action, trial_name):
    return f[action][trial_name]['pressure_ap'][()].astype(np.float64)


def pick_valid_trial(f, action, rng):
    """Try trial indices in random order; return (trial_name, pap, intensity, stats) or None."""
    if action not in f:
        return None
    trial_names = sorted(f[action].keys())
    rng.shuffle(trial_names)
    for trial_name in trial_names:
        pap = load_pap(f, action, trial_name)
        if pap.shape[0] < MIN_VALID_SAMPLES:
            continue
        intensity, baseline = contact_intensity(pap)
        stats = event_stats(intensity)
        if stats is None:
            continue
        return trial_name, pap, intensity, stats
    return None


def select_trials():
    """2 trials per object (170 total), stratified across procedures.
    Each object contributes to 2 of the 4 procedures, chosen via a fixed
    offset-2 cycle over object index so procedure usage stays balanced
    (~42-43 trials per procedure) across all 85 objects."""
    rng = random.Random(SEED)
    records = []
    for obj_idx in range(85):
        obj_num = obj_idx + 1
        obj_id = f'obj_{obj_num:02d}'
        h5_path = os.path.join(HDF5_DIR, f'{obj_id}.h5')
        if not os.path.isfile(h5_path):
            continue
        primary = ACTIONS[obj_idx % 4]
        secondary = ACTIONS[(obj_idx + 2) % 4]
        with h5py.File(h5_path, 'r') as f:
            for action in (primary, secondary):
                candidates = [action] + [a for a in ACTIONS if a != action]
                found = None
                for try_action in candidates:
                    result = pick_valid_trial(f, try_action, rng)
                    if result is not None:
                        trial_name, pap, intensity, stats = result
                        found = (try_action, trial_name, pap, intensity, stats)
                        break
                if found is None:
                    print(f"  WARNING: no valid pressure_ap trial found for {obj_id} "
                          f"(tried {action} + fallbacks)")
                    continue
                try_action, trial_name, pap, intensity, stats = found
                records.append({
                    'obj_id': obj_id, 'action': try_action, 'trial_name': trial_name,
                    'pap': pap, 'intensity': intensity, **stats,
                })
    return records


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    records = select_trials()
    print(f"Collected {len(records)} trials (target 170).")
    counts = {a: sum(1 for r in records if r['action'] == a) for a in ACTIONS}
    print(f"Procedure counts: {counts}")

    # ── Peak-location histogram, faceted by procedure ──────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    for ax, action in zip(axes, ACTIONS):
        fracs = [r['peak_fraction'] for r in records if r['action'] == action]
        ax.hist(fracs, bins=20, range=(0, 1), color='tab:blue', edgecolor='black')
        ax.set_title(f'{action} (n={len(fracs)})')
        ax.set_xlabel('peak location (fraction of trial)')
        if action == ACTIONS[0]:
            ax.set_ylabel('count')
    fig.suptitle('pressure_ap contact peak location, 170 trials, by procedure')
    fig.tight_layout()
    peak_path = os.path.join(OUTPUT_DIR, 'pressure_ap_peak_timing.png')
    fig.savefig(peak_path, dpi=150)
    print(f"Saved {peak_path}")

    # ── Event-duration histogram, faceted by procedure ─────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    for ax, action in zip(axes, ACTIONS):
        durs = [r['width_sec'] for r in records if r['action'] == action]
        ax.hist(durs, bins=20, color='tab:orange', edgecolor='black')
        ax.set_title(f'{action} (n={len(durs)})')
        ax.set_xlabel('event duration, half-max width (s)')
        if action == ACTIONS[0]:
            ax.set_ylabel('count')
    fig.suptitle('pressure_ap contact event duration (half-max width), 170 trials, by procedure')
    fig.tight_layout()
    dur_path = os.path.join(OUTPUT_DIR, 'pressure_ap_event_duration.png')
    fig.savefig(dur_path, dpi=150)
    print(f"Saved {dur_path}")

    # ── Example intensity[t] curves, one per procedure ──────────────────
    rng = random.Random(SEED + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, action in zip(axes.ravel(), ACTIONS):
        candidates = [r for r in records if r['action'] == action]
        r = rng.choice(candidates)
        t = np.arange(len(r['intensity'])) / PRESSURE_AP_RATE_HZ
        ax.plot(t, r['intensity'], color='black', linewidth=1.2)
        ax.axhline(r['thresh'], color='tab:red', linestyle='--', linewidth=1,
                   label='50% of peak')
        start, end = r['window']
        ax.axvspan(t[start], t[end], color='tab:orange', alpha=0.2, label='half-max window')
        ax.axvline(t[r['peak_idx']], color='tab:blue', linestyle=':', linewidth=1,
                   label='peak')
        ax.set_title(f"{r['obj_id']}/{action}/{r['trial_name']}")
        ax.set_xlabel('time (s)')
        ax.set_ylabel('contact intensity')
        ax.legend(fontsize=7)
    fig.suptitle('Example contact-intensity curves (one random trial per procedure)')
    fig.tight_layout()
    curves_path = os.path.join(OUTPUT_DIR, 'pressure_ap_example_curves.png')
    fig.savefig(curves_path, dpi=150)
    print(f"Saved {curves_path}")

    # ── Report numbers ──────────────────────────────────────────────────
    print(f"\n{'procedure':10s} {'n':>4s} {'peak_frac_median':>17s} {'peak_frac_p10':>14s} "
          f"{'peak_frac_p90':>14s} {'dur_median_s':>13s} {'dur_p95_s':>10s}")
    for action in ACTIONS:
        fracs = np.array([r['peak_fraction'] for r in records if r['action'] == action])
        durs = np.array([r['width_sec'] for r in records if r['action'] == action])
        print(f"{action:10s} {len(fracs):4d} {np.median(fracs):17.3f} "
              f"{np.percentile(fracs, 10):14.3f} {np.percentile(fracs, 90):14.3f} "
              f"{np.median(durs):13.3f} {np.percentile(durs, 95):10.3f}")

    all_fracs = np.array([r['peak_fraction'] for r in records])
    all_durs = np.array([r['width_sec'] for r in records])
    print(f"\n{'ALL':10s} {len(all_fracs):4d} {np.median(all_fracs):17.3f} "
          f"{np.percentile(all_fracs, 10):14.3f} {np.percentile(all_fracs, 90):14.3f} "
          f"{np.median(all_durs):13.3f} {np.percentile(all_durs, 95):10.3f}")


if __name__ == '__main__':
    main()
