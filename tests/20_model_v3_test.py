"""
Pipeline v3 — smoke test for src/model.py: instantiate
TactilePerceptionModelV3 with fold 1's actual image_input_dim, build a
small batch from fold 1's test set (reusing cached norm stats), run a
forward pass, and sanity-check output shape/softmax/NaN-freedom.

Investigation script (stays outside src/). Read-only, does not modify
DATA_Final/.
"""
import os
import sys

import numpy as np
import torch

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
from model import TactilePerceptionModelV3, collate_batch  # noqa: E402

SEED = 42
N_SPLITS = 5
BATCH_SIZE = 6


def rebuild_stats_from_cache(cache_path):
    """Same reconstruction used in 18_/19_ verify scripts -- ght excluded,
    mathematically identical to a fresh compute_norm_stats call for the
    remaining modalities (see those scripts for the reasoning)."""
    cache = np.load(cache_path)
    stats = {}
    for key in ['imu', 'pressure_ap', 'audio', 'image_glcm']:
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
    image_input_dim = len(stats['image_resnet_alive_dims']) + GLCM_FEATURE_DIM
    print(f"Fold 1 image_input_dim = {image_input_dim} "
          f"({len(stats['image_resnet_alive_dims'])} resnet + {GLCM_FEATURE_DIM} glcm)")

    test_ds = TactileDatasetV3(test_idx, object_ids, labels_dist, labels_mean, stats,
                                hdf5_dir=HDF5_DIR, cropped_dir=CROPPED_DIR)
    print(f"Fold 1 test dataset: {len(test_ds)} trials")

    samples = [test_ds[i] for i in range(BATCH_SIZE)]
    batch = collate_batch(samples)
    print(f"\nBatch shapes: " + ", ".join(f"{k}={tuple(v.shape)}" for k, v in batch.items()))

    model = TactilePerceptionModelV3(image_input_dim=image_input_dim)
    model.eval()

    n_image_params = sum(p.numel() for p in model.image_enc.parameters())
    print(f"\nImageEncoder parameters: {n_image_params:,}")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {n_params:,}  (trainable: {n_trainable:,})")

    with torch.no_grad():
        out = model(batch['imu'], batch['pressure_ap'], batch['image'], batch['audio'],
                     batch['has_image'], batch['has_audio'])

    print(f"\nOutput shape: {tuple(out.shape)}  (expect ({BATCH_SIZE}, 18, 7))")
    assert out.shape == (BATCH_SIZE, 18, 7), f"unexpected output shape {out.shape}"

    row_sums = out.sum(dim=2)  # (B, 18)
    print(f"Softmax row-sum range: [{row_sums.min().item():.6f}, {row_sums.max().item():.6f}] "
          f"(expect ~1.0)")
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
        "softmax rows do not sum to 1.0"

    has_nan = torch.isnan(out).any().item()
    print(f"Contains NaN: {has_nan}")
    assert not has_nan, "output contains NaN"

    print("\nAll checks passed.")


if __name__ == '__main__':
    main()
