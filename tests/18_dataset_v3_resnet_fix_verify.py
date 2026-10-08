"""
Pipeline v3 — verify the dead-ResNet-dim fix in src/dataset.py using the
cached fold-1 raw per-dimension stats from
outputs/17_dataset_v3_test/fold1_norm_stats_cache.npz (produced by
17_dataset_v3_test.py before the fix). Reusing the cache is valid here:
the underlying per-dimension mean/std computation over fold 1's training
images is unchanged by the fix -- only which dims get kept afterwards
changed -- so this reproduces exactly what a fresh ~650s
compute_norm_stats(train_idx, ...) call would produce, without redoing
~600 ResNet-50 forward passes over the training images.

Only the TEST-set forward passes (17 objects) are actually re-run here,
since we need real inference output to check the post-fix normalization
sanity numbers.

Investigation script (stays outside src/). Read-only, does not modify
DATA_Final/.
"""
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402

HDF5_DIR = paths.hdf5_dir()
CROPPED_DIR = paths.cropped_dir()
LABELS_DIR = paths.labels_dir()
# Written by 17_dataset_v3_test.py.
CACHE_PATH = os.path.join(paths.output_dir('17_dataset_v3_test.py', create=False), 'fold1_norm_stats_cache.npz')

from dataset import (  # noqa: E402
    make_kfold_splits, TactileDatasetV3, RESNET_FEATURE_DIM, GLCM_FEATURE_DIM,
    RESNET_DEAD_CHANNEL_STD_THRESH,
)

SEED = 42
N_SPLITS = 5


def rebuild_stats_from_cache(cache_path):
    """Reconstruct the NEW-format stats dict (with dead ResNet dims
    dropped) from the OLD cached full-2048 raw mean/std -- mathematically
    identical to a fresh compute_norm_stats call, see module docstring."""
    cache = np.load(cache_path)
    stats = {}
    for key in ['imu', 'pressure_ap', 'audio', 'ght', 'image_glcm']:
        mean = cache[f'{key}_mean']
        std = cache[f'{key}_std']
        stats[key] = (mean, std)

    mean_r = cache['image_resnet_mean']
    std_r = cache['image_resnet_std']
    alive = std_r > RESNET_DEAD_CHANNEL_STD_THRESH
    alive_dims = np.where(alive)[0].astype(np.int64)
    stats['image_resnet_alive_dims'] = alive_dims
    stats['image_resnet'] = (mean_r[alive].astype(np.float32),
                              np.maximum(std_r[alive], 1e-6).astype(np.float32))
    return stats


def summarize(name, arr):
    if arr.size == 0:
        print(f"  {name:14s}: no samples")
        return
    print(f"  {name:14s}: mean={arr.mean():+.4f}  std={arr.std():.4f}  n={arr.size}")


def main():
    object_ids = np.load(os.path.join(LABELS_DIR, 'object_ids_phase2.npy'), allow_pickle=True)
    labels_dist = np.load(os.path.join(LABELS_DIR, 'labels_phase2.npy'))
    labels_mean = np.load(os.path.join(LABELS_DIR, 'labels_mean_phase2.npy'))

    splits = make_kfold_splits(object_ids, n_splits=N_SPLITS, seed=SEED)
    train_idx, test_idx = splits[0]

    stats = rebuild_stats_from_cache(CACHE_PATH)
    n_alive = len(stats['image_resnet_alive_dims'])
    print(f"Rebuilt fold-1 stats from cache. image_resnet: {n_alive}/2048 dims kept "
          f"(threshold std > {RESNET_DEAD_CHANNEL_STD_THRESH})")
    final_image_dim = n_alive + GLCM_FEATURE_DIM
    print(f"Final normalized image feature dim: {n_alive} (resnet) + {GLCM_FEATURE_DIM} (glcm) "
          f"= {final_image_dim}")

    test_ds = TactileDatasetV3(test_idx, object_ids, labels_dist, labels_mean, stats,
                                hdf5_dir=HDF5_DIR, cropped_dir=CROPPED_DIR)
    print(f"\nFold 1 test dataset: {len(test_ds)} trials")

    print("\n=== 3 sample trials via __getitem__ (image shape should now reflect the fix) ===")
    for i in [0, len(test_ds) // 2, len(test_ds) - 1]:
        obj_idx, action, trial = test_ds.samples[i]
        sample = test_ds[i]
        print(f"[index {i}] {object_ids[obj_idx]}/{action}/{trial}: "
              f"image shape={tuple(sample['image'].shape)}, has_image={sample['has_image'].item()}")

    print("\n=== image_resnet / image_glcm normalization sanity check (fold-1 test set) ===")
    resnet_vals, glcm_vals = [], []
    for i in range(len(test_ds)):
        sample = test_ds[i]
        if sample['has_image'].item() == 1.0:
            img = sample['image'].numpy()
            resnet_vals.append(img[:n_alive])
            glcm_vals.append(img[n_alive:])

    resnet_arr = np.stack(resnet_vals)
    glcm_arr = np.stack(glcm_vals)
    summarize('image_resnet', resnet_arr.reshape(-1))
    summarize('image_glcm', glcm_arr.reshape(-1))

    per_dim_std = resnet_arr.std(axis=0)
    worst = np.argsort(per_dim_std)[-5:][::-1]
    print(f"\n  top-5 highest-std dims after fix: {worst.tolist()}")
    print(f"  their normalized std values: {per_dim_std[worst]}")
    print(f"  their normalized max abs values: {np.abs(resnet_arr[:, worst]).max(axis=0)}")

    # Verify has_image=0 placeholder now matches the reduced width too
    placeholder_found = False
    for i in range(len(test_ds)):
        sample = test_ds[i]
        if sample['has_image'].item() == 0.0:
            assert sample['image'].shape[0] == final_image_dim, (
                f"placeholder width {sample['image'].shape[0]} != real width {final_image_dim}")
            placeholder_found = True
    print(f"\nConfirmed has_image=0 placeholder width == real width ({final_image_dim}): "
          f"{'OK' if placeholder_found else 'no placeholder trial found in test set'}")


if __name__ == '__main__':
    main()
