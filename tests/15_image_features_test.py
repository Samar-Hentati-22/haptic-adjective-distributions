"""
Pipeline v3 — smoke test for src/image_features.py on 10 real cropped
images (mixed objects) from DATA_Final/hdf5_cropped_images/. Prints
output shape per image, confirms (2304,), and prints first/last 5 values
of a few examples to sanity-check value ranges/scales.

Investigation script (stays outside src/). Read-only, does not modify
DATA_Final/.
"""
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402

CROPPED_DIR = paths.cropped_dir()

from image_features import extract_image_features, RESNET_FEATURE_DIM, GLCM_FEATURE_DIM  # noqa: E402

# (object_id, procedure, trial) — mixed objects/procedures
TEST_IMAGES = [
    ('obj_01', 'tap', 'trial_000'),
    ('obj_05', 'squeeze', 'trial_001'),
    ('obj_10', 'slide', 'trial_002'),
    ('obj_15', 'hold', 'trial_000'),
    ('obj_20', 'tap', 'trial_003'),
    ('obj_30', 'squeeze', 'trial_002'),
    ('obj_45', 'slide', 'trial_001'),
    ('obj_60', 'hold', 'trial_004'),
    ('obj_70', 'tap', 'trial_002'),
    ('obj_85', 'squeeze', 'trial_000'),
]

EXPECTED_DIM = RESNET_FEATURE_DIM + GLCM_FEATURE_DIM


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    results = []
    for obj_id, proc, trial in TEST_IMAGES:
        h5_path = os.path.join(CROPPED_DIR, f'{obj_id}.h5')
        with h5py.File(h5_path, 'r') as f:
            img = f[proc][trial]['image'][()].astype(np.float32)

        feats = extract_image_features(img)
        assert feats.shape == (EXPECTED_DIM,), f"unexpected shape {feats.shape}"
        results.append({'obj_id': obj_id, 'proc': proc, 'trial': trial, 'feats': feats})
        print(f"{obj_id}/{proc}/{trial}: image shape={img.shape} -> features shape={feats.shape}")

    print(f"\nAll {len(results)} images produced feature shape ({EXPECTED_DIM},). Confirmed.")

    print("\nFirst 5 / last 5 values for 3 example images "
          "(first 5 are ResNet-branch, last 5 are the tail of the GLCM branch):")
    for res in results[:3]:
        feats = res['feats']
        label = f"{res['obj_id']}/{res['proc']}/{res['trial']}"
        print(f"\n{label}:")
        print(f"  resnet[0:5]  = {feats[:5]}")
        print(f"  glcm[-5:]    = {feats[-5:]}")
        print(f"  resnet range: min={feats[:RESNET_FEATURE_DIM].min():.4f} "
              f"max={feats[:RESNET_FEATURE_DIM].max():.4f} "
              f"mean={feats[:RESNET_FEATURE_DIM].mean():.4f}")
        print(f"  glcm   range: min={feats[RESNET_FEATURE_DIM:].min():.6f} "
              f"max={feats[RESNET_FEATURE_DIM:].max():.6f} "
              f"mean={feats[RESNET_FEATURE_DIM:].mean():.6f} "
              f"sum={feats[RESNET_FEATURE_DIM:].sum():.6f}")


if __name__ == '__main__':
    main()
