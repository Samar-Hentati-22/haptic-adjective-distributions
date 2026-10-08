"""
Pipeline v3 — one-time raw feature precompute cache.

Runs _load_raw_trial_features (unnormalized) once for every trial of all
85 objects and saves the result to TACTILE_CACHE_DIR (default <repo>/cache/)
(one .npz per object), so compute_norm_stats/TactileDatasetV3 don't have
to re-run the full HDF5 -> ResNet-50 -> MFCC -> glitch-cleaning pipeline
on every epoch of every fold. This was the dominant training cost before
caching (see the --smoke-test report: ~723s per compute_norm_stats call,
~125s per epoch, almost entirely spent redoing identical per-trial
extraction that doesn't change across epochs or folds).

Run this once, ever. It is NOT automatically invalidated by later changes
to the underlying HDF5 data or the feature extractors themselves
(imu_features.py, pressure_ap_features.py, image_features.py,
audio_ght_features.py) -- delete cache/ and re-run this script if any of
those change.

Normalization is NOT part of this cache -- it's fold-dependent
(compute_norm_stats/apply_norm_stats), computed at train time from
whichever subset of these raw cached features is that fold's training set.
"""
import os
import sys
import time

import h5py
import numpy as np

import paths  # noqa: E402

# All locations come from environment variables -- see paths.py.
HDF5_DIR = paths.hdf5_dir()
CROPPED_DIR = paths.cropped_dir()
CACHE_DIR = paths.cache_dir()

from dataset import _load_raw_trial_features, ACTIONS  # noqa: E402


def precompute_object(obj_id, hdf5_dir, cropped_dir, cache_dir, device='cpu'):
    """Extracts and saves raw features for every trial of one object.
    Returns the number of trials processed (0 if the object's HDF5 file
    doesn't exist)."""
    h5_path = os.path.join(hdf5_dir, f'{obj_id}.h5')
    if not os.path.isfile(h5_path):
        return 0

    with h5py.File(h5_path, 'r') as f:
        trial_list = [(action, trial) for action in ACTIONS if action in f
                      for trial in sorted(f[action].keys())]

    data = {}
    for action, trial in trial_list:
        feats = _load_raw_trial_features(hdf5_dir, cropped_dir, obj_id, action, trial,
                                          device=device)
        prefix = f'{action}__{trial}'
        data[f'{prefix}__imu'] = feats['imu']
        data[f'{prefix}__imu_is_real'] = np.array(feats['imu_is_real'])
        data[f'{prefix}__pressure_ap'] = feats['pressure_ap']
        data[f'{prefix}__image'] = feats['image']
        data[f'{prefix}__has_image'] = np.array(feats['has_image'])
        data[f'{prefix}__audio'] = feats['audio']
        data[f'{prefix}__has_audio'] = np.array(feats['has_audio'])

    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, f'{obj_id}.npz')
    np.savez(out_path, **data)
    return len(trial_list)


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    obj_files = sorted(f[:-3] for f in os.listdir(HDF5_DIR)
                        if f.startswith('obj_') and f.endswith('.h5'))
    print(f"Precomputing raw features for {len(obj_files)} objects -> {CACHE_DIR}")

    t0 = time.time()
    total_trials = 0
    for i, obj_id in enumerate(obj_files):
        t_obj0 = time.time()
        n = precompute_object(obj_id, HDF5_DIR, CROPPED_DIR, CACHE_DIR)
        total_trials += n
        print(f"  [{i + 1:2d}/{len(obj_files)}] {obj_id}: {n} trials in {time.time() - t_obj0:.1f}s")

    elapsed = time.time() - t0
    print(f"\nDone: {total_trials} trials across {len(obj_files)} objects in "
          f"{elapsed:.1f}s ({elapsed / 60:.1f} min)")

    cache_size = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in os.listdir(CACHE_DIR))
    print(f"Cache size on disk: {cache_size / 1e6:.1f} MB")


if __name__ == '__main__':
    main()
