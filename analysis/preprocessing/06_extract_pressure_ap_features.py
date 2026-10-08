"""
Pipeline v3 — extract pressure_ap contact-event features for all 1700
trials, at three candidate thresholds (k=3, 5, 8), for a k-sensitivity
check before committing to a final value.

Diagnostic/feature-engineering only: no changes to dataset.py / model.py /
train.py / any existing pipeline.

Writes one CSV per k to <repo>/outputs/06_extract_pressure_ap_features/, and reports the
median n_events/trial over a reproducible 50-trial sample for each k
(cheap sanity check requested before the full correlation comparison).
"""
import csv
import os
import random
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402
from pressure_ap_event_features import extract_features, FEATURE_NAMES  # noqa: E402

HDF5_DIR = paths.hdf5_dir()
OUTPUT_DIR = paths.output_dir(__file__)

ACTIONS = ['tap', 'squeeze', 'slide', 'hold']
K_VALUES = [3, 5, 8]
SENSITIVITY_SAMPLE_SIZE = 50
SEED = 0


def all_trial_keys():
    keys = []
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
                    keys.append((obj_id, action, trial_name))
    return keys


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    keys = all_trial_keys()
    print(f"Found {len(keys)} trials total.")

    rng = random.Random(SEED)
    sensitivity_sample = set(rng.sample(keys, SENSITIVITY_SAMPLE_SIZE))
    sensitivity_n_events = {k: [] for k in K_VALUES}

    writers = {}
    files = {}
    for k in K_VALUES:
        path = os.path.join(OUTPUT_DIR, f'pressure_ap_features_k{k}.csv')
        fp = open(path, 'w', newline='')
        writer = csv.writer(fp)
        writer.writerow(['object_id', 'procedure', 'trial_name'] + FEATURE_NAMES)
        files[k] = fp
        writers[k] = writer

    n_processed = 0
    n_too_short = 0
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
                    pap = f[action][trial_name]['pressure_ap'][()].astype(np.float64)
                    n_processed += 1
                    for k in K_VALUES:
                        feats = extract_features(pap, k)
                        if feats is None:
                            n_too_short += 1 if k == K_VALUES[0] else 0
                            row = ['' for _ in FEATURE_NAMES]
                        else:
                            row = [feats[name] for name in FEATURE_NAMES]
                            if (obj_id, action, trial_name) in sensitivity_sample:
                                sensitivity_n_events[k].append(feats['n_events'])
                        writers[k].writerow([obj_id, action, trial_name] + row)

    for fp in files.values():
        fp.close()

    print(f"Processed {n_processed} trials ({n_too_short} too short for a baseline, "
          f"< 10 pressure_ap samples).")
    for k in K_VALUES:
        path = os.path.join(OUTPUT_DIR, f'pressure_ap_features_k{k}.csv')
        print(f"Saved {path}")

    print(f"\nSensitivity check (n={SENSITIVITY_SAMPLE_SIZE} sampled trials):")
    print(f"{'k':>3s} {'median_n_events':>16s} {'mean_n_events':>14s}")
    for k in K_VALUES:
        vals = np.array(sensitivity_n_events[k])
        print(f"{k:3d} {np.median(vals):16.2f} {vals.mean():14.2f}")


if __name__ == '__main__':
    main()
