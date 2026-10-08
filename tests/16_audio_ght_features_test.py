"""
Pipeline v3 — smoke test for src/audio_ght_features.py on 10 real trials
(mixed objects/procedures), plus targeted checks against known audio
dropout objects.

NOTE: the task that requested this test named obj_09 among "9 objects
with full audio dropout" alongside obj_10/11/26/57/58/59/60/61/62. Direct
re-check against DATA_Final/hdf5/ (see below) found only 9 objects are
FULLY dropped (obj_10, 11, 26, 57, 58, 59, 60, 61, 62); obj_09 is mostly
but not fully dropped (15/20 trials). Both are exercised below: obj_10 as
a fully-dropped object, and obj_09 with one dropped + one non-dropped
trial to confirm has_audio is correct even within a partially-affected
object.

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

from audio_ght_features import extract_audio_features, extract_ght_features, GHT_FIELD_NAMES  # noqa: E402

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

# Confirmed fully-dropped objects (all 20 trials all-zero audio)
DROPOUT_CHECK_TRIALS = [
    ('obj_10', 'tap', 'trial_000'),
    ('obj_60', 'squeeze', 'trial_002'),
]
# obj_09 is only partially dropped (15/20): all `tap` trials are fine, all
# squeeze/slide/hold trials are dropped -- confirm has_audio distinguishes
# a dropped trial from a real one within the SAME object
PARTIAL_OBJECT_TRIALS = [
    ('obj_09', 'tap', 'trial_000'),      # expect has_audio=True
    ('obj_09', 'tap', 'trial_002'),      # expect has_audio=True
    ('obj_09', 'squeeze', 'trial_000'),  # expect has_audio=False
    ('obj_09', 'slide', 'trial_001'),    # expect has_audio=False
    ('obj_09', 'hold', 'trial_003'),     # expect has_audio=False
]


def load_trial(obj_id, proc, trial):
    with h5py.File(os.path.join(HDF5_DIR, f'{obj_id}.h5'), 'r') as f:
        audio = f[proc][trial]['audio'][()].astype(np.float32)
        ght = f[proc][trial]['ght'][()].astype(np.float32)
    return audio, ght


def main():
    np.set_printoptions(precision=3, suppress=True)

    print("=== 10 mixed trials ===")
    for obj_id, proc, trial in TEST_TRIALS:
        audio, ght = load_trial(obj_id, proc, trial)
        mel_out, has_audio = extract_audio_features(audio)
        ght_out = extract_ght_features(ght)
        print(f"{obj_id}/{proc}/{trial}: audio in={audio.shape} -> out={mel_out.shape}, "
              f"has_audio={has_audio}  |  ght in={ght.shape} -> out={ght_out.shape}, "
              f"values={dict(zip(GHT_FIELD_NAMES, ght_out.tolist()))}")

    print("\n=== Confirmed fully-dropped objects (expect has_audio=False) ===")
    for obj_id, proc, trial in DROPOUT_CHECK_TRIALS:
        audio, _ = load_trial(obj_id, proc, trial)
        _, has_audio = extract_audio_features(audio)
        print(f"{obj_id}/{proc}/{trial}: shape={audio.shape}, all_zero={np.all(audio==0)}, "
              f"has_audio={has_audio}")

    print("\n=== obj_09 (partially-dropped object, 15/20 trials) — per-trial check ===")
    for obj_id, proc, trial in PARTIAL_OBJECT_TRIALS:
        audio, _ = load_trial(obj_id, proc, trial)
        _, has_audio = extract_audio_features(audio)
        print(f"{obj_id}/{proc}/{trial}: all_zero={np.all(audio==0)}, has_audio={has_audio}")


if __name__ == '__main__':
    main()
