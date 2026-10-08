"""
Pipeline v3 — quantify prevalence of the pressure_ap zero-block glitch
signature found while spot-checking resampling (see
outputs/13b_pressure_ap_resample_test/pressure_ap_resample_check.png and the
imu-features-session follow-up discussion).

Detection proxy: a timestep where >=10 contiguous channels read exactly
0.0, while (a) the OTHER channels at that same timestep are near normal
baseline (mean > BASELINE_THRESH), and (b) the same channel range at the
immediately preceding and following timestep is also near baseline (i.e.
this is a one-timestep blip, not a sustained low-signal/dropout region
or a legitimately blank trial).

Investigation only, stays outside src/. Read-only, does not modify
DATA_Final/. Does NOT fix or filter anything itself -- purely quantifies
prevalence, before and after applying model/src/pressure_ap_features.py's
clean_pressure_ap_glitches (added 2026-07-19) to confirm the fix works
at full-dataset scale.
"""
import os
import sys
import csv

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402

HDF5_DIR = paths.hdf5_dir()
OUTPUT_DIR = paths.output_dir(__file__)

from pressure_ap_features import clean_pressure_ap_glitches  # noqa: E402

ACTIONS = ('tap', 'squeeze', 'slide', 'hold')
MIN_RUN = 10
BASELINE_THRESH = 500.0  # baseline pressure_ap values observed ~2000-2200; true zero is not normal


def find_contiguous_zero_runs(row, min_run):
    """Return list of (start, end_exclusive) for runs of exact-zero >= min_run."""
    runs = []
    n = len(row)
    i = 0
    while i < n:
        if row[i] == 0.0:
            j = i
            while j < n and row[j] == 0.0:
                j += 1
            if j - i >= min_run:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def detect_glitch(pap):
    """pap: (N, 50). Returns list of (t, ch_start, ch_end, run_len) glitch events."""
    n_t, n_ch = pap.shape
    events = []
    for t in range(n_t):
        row = pap[t]
        runs = find_contiguous_zero_runs(row, MIN_RUN)
        for (a, b) in runs:
            other_ch = np.concatenate([row[:a], row[b:]])
            if other_ch.size == 0 or other_ch.mean() <= BASELINE_THRESH:
                continue  # whole row is low/blank -- not an isolated glitch, likely a real quiet/blank trial
            neighbors_ok = True
            for t_nb in (t - 1, t + 1):
                if t_nb < 0 or t_nb >= n_t:
                    neighbors_ok = False
                    break
                nb_vals = pap[t_nb, a:b]
                if nb_vals.mean() <= BASELINE_THRESH:
                    neighbors_ok = False
                    break
            if neighbors_ok:
                events.append((t, a, b, b - a))
    return events


def scan(files, apply_clean: bool):
    per_trial_rows = []
    n_trials_total = 0
    n_trials_glitched = 0
    by_object = {}
    by_procedure = {a: [0, 0] for a in ACTIONS}  # [n_glitched, n_total]

    for fname in files:
        obj_id = fname[:-3]
        with h5py.File(os.path.join(HDF5_DIR, fname), 'r') as f:
            for action in ACTIONS:
                if action not in f:
                    continue
                for trial in sorted(f[action].keys()):
                    pap = f[action][trial]['pressure_ap'][()].astype(np.float32)
                    if apply_clean:
                        pap = clean_pressure_ap_glitches(pap)
                    n_trials_total += 1
                    by_procedure[action][1] += 1
                    events = detect_glitch(pap)
                    has_glitch = len(events) > 0
                    by_object.setdefault(obj_id, [0, 0])
                    by_object[obj_id][1] += 1
                    if has_glitch:
                        n_trials_glitched += 1
                        by_procedure[action][0] += 1
                        by_object[obj_id][0] += 1
                    for (t, a, b, run_len) in events:
                        per_trial_rows.append({
                            'object_id': obj_id, 'procedure': action, 'trial': trial,
                            'n_timesteps': pap.shape[0], 't_glitch': t,
                            'ch_start': a, 'ch_end': b, 'run_len': run_len,
                        })

    return {
        'per_trial_rows': per_trial_rows, 'n_trials_total': n_trials_total,
        'n_trials_glitched': n_trials_glitched, 'by_object': by_object,
        'by_procedure': by_procedure,
    }


def report(stats, csv_path):
    per_trial_rows = stats['per_trial_rows']
    n_trials_total = stats['n_trials_total']
    n_trials_glitched = stats['n_trials_glitched']
    by_object = stats['by_object']
    by_procedure = stats['by_procedure']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['object_id', 'procedure', 'trial', 'n_timesteps',
                                                't_glitch', 'ch_start', 'ch_end', 'run_len'])
        writer.writeheader()
        writer.writerows(per_trial_rows)

    print(f"Total trials scanned: {n_trials_total}")
    print(f"Trials with >=1 glitch event: {n_trials_glitched} "
          f"({100 * n_trials_glitched / n_trials_total:.2f}%)")
    print(f"Total glitch events (a trial can have >1): {len(per_trial_rows)}")

    print("By procedure:")
    for action in ACTIONS:
        n_g, n_t = by_procedure[action]
        print(f"  {action}: {n_g}/{n_t} trials ({100 * n_g / n_t:.2f}%)")

    obj_sorted = sorted(
        [(oid, g, t) for oid, (g, t) in by_object.items() if g > 0],
        key=lambda x: -x[1]
    )
    print(f"{len(obj_sorted)}/{len(by_object)} objects have at least one glitched trial")
    print(f"Saved per-event log to {csv_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(HDF5_DIR) if f.startswith('obj_') and f.endswith('.h5'))

    print("=" * 80)
    print("BEFORE cleaning (raw pressure_ap, detection threshold MIN_RUN=10)")
    print("=" * 80)
    raw_stats = scan(files, apply_clean=False)
    report(raw_stats, os.path.join(OUTPUT_DIR, 'pressure_ap_glitch_events.csv'))

    print("\n" + "=" * 80)
    print("AFTER clean_pressure_ap_glitches (same MIN_RUN=10 detection threshold)")
    print("=" * 80)
    clean_stats = scan(files, apply_clean=True)
    report(clean_stats, os.path.join(OUTPUT_DIR, 'pressure_ap_glitch_events_postclean.csv'))


if __name__ == '__main__':
    main()
