"""
Pipeline v3 — verify GHT removal from src/dataset.py. Reuses the cached
fold-1 raw per-dimension stats (outputs/17_dataset_v3_test/
fold1_norm_stats_cache.npz) for imu/pressure_ap/image_resnet/image_glcm/
audio -- removing ght accumulation doesn't change those (independent
running sums), so this is equivalent to a fresh compute_norm_stats call
minus the ~650s of training-set ResNet forward passes. Only the ght
entries from the cache are dropped when rebuilding the stats dict.

Investigation script (stays outside src/). Read-only, does not modify
DATA_Final/.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402

HDF5_DIR = paths.hdf5_dir()
CROPPED_DIR = paths.cropped_dir()
LABELS_DIR = paths.labels_dir()
# Written by 17_dataset_v3_test.py.
CACHE_PATH = os.path.join(paths.output_dir('17_dataset_v3_test.py', create=False), 'fold1_norm_stats_cache.npz')

from dataset import (  # noqa: E402
    make_kfold_splits, TactileDatasetV3, GLCM_FEATURE_DIM, RESNET_DEAD_CHANNEL_STD_THRESH,
)

SEED = 42
N_SPLITS = 5


def rebuild_stats_from_cache(cache_path):
    cache = np.load(cache_path)
    stats = {}
    for key in ['imu', 'pressure_ap', 'audio', 'image_glcm']:  # ght dropped
        stats[key] = (cache[f'{key}_mean'], cache[f'{key}_std'])

    mean_r = cache['image_resnet_mean']
    std_r = cache['image_resnet_std']
    alive = std_r > RESNET_DEAD_CHANNEL_STD_THRESH
    stats['image_resnet_alive_dims'] = np.where(alive)[0].astype(np.int64)
    stats['image_resnet'] = (mean_r[alive].astype(np.float32),
                              np.maximum(std_r[alive], 1e-6).astype(np.float32))
    return stats


def main():
    object_ids = np.load(os.path.join(LABELS_DIR, 'object_ids_phase2.npy'), allow_pickle=True)
    labels_dist = np.load(os.path.join(LABELS_DIR, 'labels_phase2.npy'))
    labels_mean = np.load(os.path.join(LABELS_DIR, 'labels_mean_phase2.npy'))

    splits = make_kfold_splits(object_ids, n_splits=N_SPLITS, seed=SEED)
    train_idx, test_idx = splits[0]

    stats = rebuild_stats_from_cache(CACHE_PATH)
    print(f"Rebuilt fold-1 stats (ght excluded). Keys: {sorted(stats.keys())}")
    assert 'ght' not in stats, "ght should not be in stats"

    test_ds = TactileDatasetV3(test_idx, object_ids, labels_dist, labels_mean, stats,
                                hdf5_dir=HDF5_DIR, cropped_dir=CROPPED_DIR)
    print(f"Fold 1 test dataset: {len(test_ds)} trials")

    print("\n=== 3 sample trials via __getitem__ ===")
    for i in [0, len(test_ds) // 2, len(test_ds) - 1]:
        obj_idx, action, trial = test_ds.samples[i]
        sample = test_ds[i]
        print(f"\n[index {i}] {object_ids[obj_idx]}/{action}/{trial}")
        print(f"  keys: {sorted(sample.keys())}")
        assert 'ght' not in sample, "ght key leaked into __getitem__ output"
        for k, v in sample.items():
            if hasattr(v, 'shape'):
                print(f"  {k:12s}: shape={tuple(v.shape)}, dtype={v.dtype}")
            else:
                print(f"  {k:12s}: {v}")

    print("\nConfirmed: no 'ght' key in any sample dict. image dim = "
          f"{len(stats['image_resnet_alive_dims']) + GLCM_FEATURE_DIM} as before (unaffected by ght removal).")


if __name__ == '__main__':
    main()
