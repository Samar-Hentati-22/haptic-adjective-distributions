"""
Pipeline v3 — smoke test for imu_features.py on 10 real trials (mixed
objects/procedures). Prints output shape + values per trial. Read-only,
does not modify DATA_Final/.
"""
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402

HDF5_DIR = paths.hdf5_dir()

from imu_features import deinterleave_imu, extract_imu_features, DEFAULT_FS  # noqa: E402

# (object_id, procedure, trial) — spread across objects and all 4 procedures
TEST_TRIALS = [
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


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    for obj_id, proc, trial in TEST_TRIALS:
        h5_path = os.path.join(HDF5_DIR, f'{obj_id}.h5')
        with h5py.File(h5_path, 'r') as f:
            imu_raw = f[proc][trial]['imu_raw'][()].astype(np.float32)

        deint = deinterleave_imu(imu_raw)
        acc = deint[:, 0:3]
        feats = extract_imu_features(acc, fs=DEFAULT_FS)

        print(f"{obj_id}/{proc}/{trial}: imu_raw={imu_raw.shape} "
              f"acc_substream={acc.shape} -> features shape={feats.shape}")
        print(f"  {feats}")


if __name__ == '__main__':
    main()
