"""
Pipeline v3 — smoke test for src/dataset.py: compute norm stats on fold 1's
training objects, build the dataset for fold 1's test objects, pull sample
trials, sanity-check normalization, and confirm has_image/has_audio flags
are correct on known dropout/blank-frame trials.

Investigation script (stays outside src/). Read-only, does not modify
DATA_Final/.
"""
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402

HDF5_DIR = paths.hdf5_dir()
CROPPED_DIR = paths.cropped_dir()
LABELS_DIR = paths.labels_dir()

from dataset import (  # noqa: E402
    make_kfold_splits, compute_norm_stats, TactileDatasetV3,
    RESNET_FEATURE_DIM, GLCM_FEATURE_DIM,
)
from imu_features import deinterleave_imu  # noqa: E402

SEED = 42
N_SPLITS = 5

# Known dropout/blank-frame trials (verified against real data in the
# audio_ght_features and image_features build sessions)
FULLY_AUDIO_DROPOUT_OBJECTS = ['obj_10', 'obj_11', 'obj_26', 'obj_57', 'obj_58',
                                'obj_59', 'obj_60', 'obj_61', 'obj_62']
BLANK_FRAME_TRIALS = [
    ('obj_06', 'tap', 'trial_002'),
    ('obj_09', 'hold', 'trial_002'),
    ('obj_20', 'slide', 'trial_004'),
    ('obj_24', 'squeeze', 'trial_003'),
    ('obj_79', 'tap', 'trial_000'),
]


def summarize(name, arr, mask=None):
    x = arr if mask is None else arr[mask]
    if x.size == 0:
        print(f"  {name:14s}: no real (non-placeholder) samples in this set")
        return
    print(f"  {name:14s}: mean={x.mean():+.4f}  std={x.std():.4f}  n={x.size}")


def main():
    t0 = time.time()
    object_ids = np.load(os.path.join(LABELS_DIR, 'object_ids_phase2.npy'), allow_pickle=True)
    labels_dist = np.load(os.path.join(LABELS_DIR, 'labels_phase2.npy'))
    labels_mean = np.load(os.path.join(LABELS_DIR, 'labels_mean_phase2.npy'))

    splits = make_kfold_splits(object_ids, n_splits=N_SPLITS, seed=SEED)
    train_idx, test_idx = splits[0]
    print(f"Fold 1: {len(train_idx)} train objects, {len(test_idx)} test objects")
    print(f"  test objects: {sorted(str(object_ids[i]) for i in test_idx)}")

    cache_path = os.path.join(paths.output_dir(__file__), 'fold1_norm_stats_cache.npz')
    print("\nComputing norm stats on fold 1 training objects (this runs a ResNet-50 "
          "forward pass per real image, ~600+ images -- may take a few minutes)...")
    norm_stats = compute_norm_stats(train_idx, object_ids, HDF5_DIR, CROPPED_DIR)
    print(f"Done in {time.time() - t0:.1f}s. Stats keys: {list(norm_stats.keys())}")
    for k, (mean, std) in norm_stats.items():
        print(f"  {k:14s}: mean shape={mean.shape}, std shape={std.shape}, "
              f"mean range=[{mean.min():.3g}, {mean.max():.3g}], "
              f"std range=[{std.min():.3g}, {std.max():.3g}]")
    np.savez(cache_path, **{f'{k}_mean': v[0] for k, v in norm_stats.items()},
             **{f'{k}_std': v[1] for k, v in norm_stats.items()})
    print(f"Cached norm_stats to {cache_path}")

    # ── Diagnose the image_resnet std floor (many post-ReLU channels are
    # near-constant/dead across a few hundred training images -> per-channel
    # z-score with a 1e-6 std floor can blow up on test-set deviation) ───────
    mean_r, std_r = norm_stats['image_resnet']
    n_floor = int((std_r <= 1e-5).sum())
    print(f"\nimage_resnet std-floor diagnostic: {n_floor}/2048 dims have std <= 1e-5 "
          f"(essentially dead/constant across training images)")
    print(f"  dims with smallest std: {np.argsort(std_r)[:10].tolist()}")
    print(f"  their std values: {np.sort(std_r)[:10]}")

    test_ds = TactileDatasetV3(test_idx, object_ids, labels_dist, labels_mean, norm_stats,
                                hdf5_dir=HDF5_DIR, cropped_dir=CROPPED_DIR)
    print(f"\nFold 1 test dataset: {len(test_ds)} trials")

    print("\n=== 3 sample trials via __getitem__ ===")
    for i in [0, len(test_ds) // 2, len(test_ds) - 1]:
        obj_idx, action, trial = test_ds.samples[i]
        sample = test_ds[i]
        print(f"\n[index {i}] {object_ids[obj_idx]}/{action}/{trial}")
        for k, v in sample.items():
            if hasattr(v, 'shape'):
                print(f"  {k:12s}: shape={tuple(v.shape)}, dtype={v.dtype}")
            else:
                print(f"  {k:12s}: {v}")

    print("\n=== Normalization sanity check (pooled over all fold-1 test trials) ===")
    imu_real_vals, imu_all_vals = [], []
    pap_vals = []
    resnet_vals, glcm_vals = [], []
    audio_vals = []
    ght_vals = []
    for i in range(len(test_ds)):
        obj_idx, action, trial = test_ds.samples[i]
        obj_id = str(object_ids[obj_idx])
        with h5py.File(os.path.join(HDF5_DIR, f'{obj_id}.h5'), 'r') as f:
            imu_raw_arr = f[action][trial]['imu_raw'][()].astype(np.float32)
        imu_is_real = deinterleave_imu(imu_raw_arr).shape[0] > 0
        sample = test_ds[i]
        imu_all_vals.append(sample['imu'].numpy())
        if imu_is_real:
            imu_real_vals.append(sample['imu'].numpy())
        pap_vals.append(sample['pressure_ap'].numpy().reshape(-1))
        if sample['has_image'].item() == 1.0:
            resnet_vals.append(sample['image'].numpy()[:RESNET_FEATURE_DIM])
            glcm_vals.append(sample['image'].numpy()[RESNET_FEATURE_DIM:])
        if sample['has_audio'].item() == 1.0:
            audio_vals.append(sample['audio'].numpy().reshape(-1))
        ght_vals.append(sample['ght'].numpy())

    summarize('imu (all, incl. dropout placeholders left at 0)', np.stack(imu_all_vals))
    summarize('imu (real trials only)', np.stack(imu_real_vals))
    summarize('pressure_ap', np.concatenate(pap_vals))
    if resnet_vals:
        resnet_arr = np.stack(resnet_vals)  # (n_real_images, 2048)
        summarize('image_resnet', resnet_arr.reshape(-1))
        # Per-dimension diagnostic: which dims are blowing up the aggregate std?
        per_dim_std = resnet_arr.std(axis=0)
        worst = np.argsort(per_dim_std)[-5:][::-1]
        print(f"    top-5 highest-std dims after normalization: {worst.tolist()}")
        print(f"    their normalized std values: {per_dim_std[worst]}")
        print(f"    their normalized max abs values: {np.abs(resnet_arr[:, worst]).max(axis=0)}")
        summarize('image_glcm', np.stack(glcm_vals).reshape(-1))
    if audio_vals:
        summarize('audio', np.concatenate(audio_vals))
    summarize('ght (all)', np.stack(ght_vals))

    print("\n=== has_image / has_audio correctness on known dropout/blank trials ===")
    # Objects landing in fold 1's test set that happen to hit these cases:
    fold1_test_objs = set(str(object_ids[i]) for i in test_idx)
    for obj_id in FULLY_AUDIO_DROPOUT_OBJECTS:
        if obj_id in fold1_test_objs:
            obj_idx = int(np.where(object_ids == obj_id)[0][0])
            with h5py.File(os.path.join(HDF5_DIR, f'{obj_id}.h5'), 'r') as f:
                action = next(a for a in ['tap', 'squeeze', 'slide', 'hold'] if a in f)
                trial = sorted(f[action].keys())[0]
            idx = test_ds.samples.index((obj_idx, action, trial))
            sample = test_ds[idx]
            print(f"  [in fold-1 test set] {obj_id}/{action}/{trial} (fully audio-dropout object): "
                  f"has_audio={sample['has_audio'].item()} (expect 0.0)")
    for obj_id, action, trial in BLANK_FRAME_TRIALS:
        if obj_id in fold1_test_objs:
            obj_idx = int(np.where(object_ids == obj_id)[0][0])
            idx = test_ds.samples.index((obj_idx, action, trial))
            sample = test_ds[idx]
            print(f"  [in fold-1 test set] {obj_id}/{action}/{trial} (known blank-frame trial, "
                  f"procedure={action}): has_image={sample['has_image'].item()} (expect 0.0)")

    # obj_09/hold/trial_002 and obj_24/squeeze/trial_003 are the two INTERESTING
    # blank-frame cases (squeeze/hold procedure, so has_image would be 1 without
    # the missing-crop check) -- neither lands in fold-1's test set for this
    # seed, so check them directly against a tiny ad-hoc dataset reusing the
    # same fold-1 norm_stats (no leakage concern: this is a flag-correctness
    # spot check, not evaluation).
    print("\n  Interesting cases (squeeze/hold blank frames) not in fold-1 test set -- "
          "checked via a small ad-hoc dataset over their objects instead:")
    extra_objs = ['obj_09', 'obj_24', 'obj_60']  # obj_60: another fully audio-dropout object
    extra_idx = [int(np.where(object_ids == oid)[0][0]) for oid in extra_objs]
    extra_ds = TactileDatasetV3(extra_idx, object_ids, labels_dist, labels_mean, norm_stats,
                                 hdf5_dir=HDF5_DIR, cropped_dir=CROPPED_DIR)
    for obj_id, action, trial in [('obj_09', 'hold', 'trial_002'), ('obj_24', 'squeeze', 'trial_003')]:
        obj_idx = int(np.where(object_ids == obj_id)[0][0])
        idx = extra_ds.samples.index((obj_idx, action, trial))
        sample = extra_ds[idx]
        print(f"  {obj_id}/{action}/{trial} (blank frame, squeeze/hold procedure): "
              f"has_image={sample['has_image'].item()} (expect 0.0)")
    for obj_id in ['obj_60']:
        obj_idx = int(np.where(object_ids == obj_id)[0][0])
        with h5py.File(os.path.join(HDF5_DIR, f'{obj_id}.h5'), 'r') as f:
            action = next(a for a in ['tap', 'squeeze', 'slide', 'hold'] if a in f)
            trial = sorted(f[action].keys())[0]
        idx = extra_ds.samples.index((obj_idx, action, trial))
        sample = extra_ds[idx]
        print(f"  {obj_id}/{action}/{trial} (fully audio-dropout object): "
              f"has_audio={sample['has_audio'].item()} (expect 0.0)")

    print(f"\nTotal test runtime: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
