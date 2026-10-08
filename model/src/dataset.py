"""
Pipeline v3 — unified dataset class, tying together the feature modules
already built and tested in pipeline_v3/src/: imu_features.py,
pressure_ap_features.py, image_features.py, audio_ght_features.py.

Self-contained: no imports from outside pipeline_v3/ (make_kfold_splits is
ported, not imported, from the old Training_Phase2/dataset.py).

Per-trial output feature dims: imu (10), pressure_ap (224, 50), image
(RAW extraction is always 2304 = 2048 ResNet + 256 GLCM; the NORMALIZED
output from apply_norm_stats/__getitem__ is narrower and FOLD-DEPENDENT
-- see compute_norm_stats's RESNET_DEAD_CHANNEL_STD_THRESH note: ResNet
dims that are dead/constant across a fold's training images are dropped
entirely, not just floored, so read the actual width off
len(stats['image_resnet_alive_dims']) + GLCM_FEATURE_DIM rather than
assuming 2304 or 2048), audio (2, 64, T -- T left variable, see note in
audio_ght_features.py: the audio encoder is designed to consume variable
T directly, so no padding is applied here; batching variable-T audio is a
model.py/collate-function concern, not handled in this file).
`pressure` and `ght` are intentionally NOT loaded anywhere in this module,
per pipeline_v3's scope (ght removed 2026-07-19, matching the earlier
pressure exclusion).

Raw per-trial feature extraction (HDF5 -> ResNet/MFCC/glitch-cleaning) is
the expensive part and doesn't change across epochs or folds (only the
z-score normalization is fold-dependent) -- pass cache_dir (pointing at
precompute_features.py's output) to compute_norm_stats/TactileDatasetV3
to read precomputed raw features instead of recomputing them every call.
Falls back to the original extraction path per-trial if the cache is
missing or incomplete, see _load_trial_features_cached.
"""
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from imu_features import deinterleave_imu, extract_imu_features
from pressure_ap_features import extract_pressure_ap_features
from image_features import extract_image_features, RESNET_FEATURE_DIM, GLCM_FEATURE_DIM
from audio_ght_features import extract_audio_features

ACTIONS = ['tap', 'squeeze', 'slide', 'hold']
# Ported from Training_Phase2/dataset.py: only squeeze/hold get a real
# camera frame; tap/slide always get has_image=0 regardless of what's in
# DATA_Final/hdf5_cropped_images (most tap/slide trials aren't even cropped
# at all -- image_valid=False upstream, see image_crop_full_run_log.csv).
IMAGE_ACTIONS = frozenset({'squeeze', 'hold'})

IMU_FEATURE_DIM = 10
IMAGE_FEATURE_DIM = RESNET_FEATURE_DIM + GLCM_FEATURE_DIM  # 2304
PRESSURE_AP_SHAPE = (224, 50)

# No objects currently excluded from CV (matches Training_Phase2/dataset.py's
# current EXCLUDE_OBJECTS = frozenset()).
EXCLUDE_OBJECTS = frozenset()


# ── K-fold splitting (ported from Training_Phase2/dataset.py, unchanged) ───

def make_kfold_splits(object_ids, n_splits: int, seed: int, exclude=EXCLUDE_OBJECTS):
    """
    Return list of (train_indices, test_indices) tuples. Object-level split.

    Indices are positions into `object_ids` (and therefore into labels_dist/
    labels_mean), so they can be passed straight into TactileDatasetV3.
    Objects whose id is in `exclude` are dropped before splitting -- they
    are never placed in a train or test fold.
    """
    indices = [i for i, oid in enumerate(object_ids) if str(oid) not in exclude]
    n = len(indices)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    # Spread any remainder across the first folds (e.g. n=84, n_splits=5 →
    # sizes [17,17,17,17,16]) instead of dumping it all into the last fold.
    folds = [arr.tolist() for arr in np.array_split(np.array(indices), n_splits)]

    splits = []
    for i in range(n_splits):
        test = folds[i]
        train = [idx for j, fold in enumerate(folds) if j != i for idx in fold]
        splits.append((train, test))
    return splits


# ── Per-trial raw feature loading (shared by compute_norm_stats and __getitem__) ─

def _load_image_features(cropped_dir, obj_id, action, trial, device='cpu'):
    """
    Returns (features (2304,) float32, has_image: bool).

    has_image is False (and features is an all-zero placeholder) whenever
    action isn't squeeze/hold, OR the cropped image is missing -- covers
    both the 5 known blank-frame trials (obj_06/tap/trial_002,
    obj_09/hold/trial_002, obj_20/slide/trial_004, obj_24/squeeze/trial_003,
    obj_79/tap/trial_000, caught by image_crop.py's blank-frame guard) and
    any tap/slide trial never cropped at all (image_valid=False upstream) --
    checked generically via a missing h5py group/dataset rather than a
    hardcoded trial list, so it also covers any other current or future
    missing-image case without crashing.
    """
    if action not in IMAGE_ACTIONS:
        return np.zeros(IMAGE_FEATURE_DIM, dtype=np.float32), False

    h5_path = os.path.join(cropped_dir, f'{obj_id}.h5')
    try:
        with h5py.File(h5_path, 'r') as f:
            img = f[action][trial]['image'][()].astype(np.float32)
    except (OSError, KeyError):
        return np.zeros(IMAGE_FEATURE_DIM, dtype=np.float32), False

    feats = extract_image_features(img, device=device)
    return feats, True


def _load_raw_trial_features(hdf5_dir, cropped_dir, obj_id, action, trial, device='cpu'):
    """
    Loads one trial and runs every modality through its (already validated)
    feature extractor. Returns a dict of RAW (unnormalized) features plus
    'imu_is_real'/'has_image'/'has_audio' flags, for either direct use by
    compute_norm_stats (accumulating stats) or as input to apply_norm_stats
    (for __getitem__).
    """
    h5_path = os.path.join(hdf5_dir, f'{obj_id}.h5')
    with h5py.File(h5_path, 'r') as f:
        trial_grp = f[action][trial]
        imu_raw_arr = trial_grp['imu_raw'][()].astype(np.float32)
        pap_arr = trial_grp['pressure_ap'][()].astype(np.float32)
        audio_arr = trial_grp['audio'][()].astype(np.float32)

    deint = deinterleave_imu(imu_raw_arr)
    imu_is_real = deint.shape[0] > 0
    imu_feat = extract_imu_features(deint[:, 0:3])  # all-zero already if not imu_is_real

    pap_feat = extract_pressure_ap_features(pap_arr)  # (224, 50), always "real" (resampled)

    image_feat, has_image = _load_image_features(cropped_dir, obj_id, action, trial, device=device)

    audio_feat, has_audio = extract_audio_features(audio_arr)

    return {
        'imu': imu_feat, 'imu_is_real': imu_is_real,
        'pressure_ap': pap_feat,
        'image': image_feat, 'has_image': has_image,
        'audio': audio_feat, 'has_audio': has_audio,
    }


def _load_object_npz(cache_dir, obj_id):
    """Open one object's precomputed-feature .npz (see
    precompute_features.py), or None if it doesn't exist -- e.g. cache/
    was never built, or this object was added after the cache was built."""
    path = os.path.join(cache_dir, f'{obj_id}.npz')
    if not os.path.isfile(path):
        return None
    return np.load(path)


def _load_trial_features_cached(hdf5_dir, cropped_dir, obj_id, action, trial, device='cpu',
                                  cache_dir=None, npz_cache=None):
    """
    Like _load_raw_trial_features, but reads from precompute_features.py's
    cache when available (cache_dir given, the object's .npz exists, and
    this specific trial's keys are present in it), falling back to the
    original HDF5+ResNet+MFCC extraction path otherwise -- so a missing or
    partial cache never crashes, it's just slower for the uncached trials.

    npz_cache (optional dict, keyed by obj_id) lets a caller that loads
    many trials of the same object (compute_norm_stats, or a
    TactileDatasetV3 instance across its lifetime) avoid reopening the
    same .npz file per trial.
    """
    if cache_dir is not None:
        if npz_cache is not None:
            if obj_id not in npz_cache:
                npz_cache[obj_id] = _load_object_npz(cache_dir, obj_id)
            npz = npz_cache[obj_id]
        else:
            npz = _load_object_npz(cache_dir, obj_id)

        if npz is not None:
            prefix = f'{action}__{trial}'
            key = f'{prefix}__imu'
            if key in npz:
                return {
                    'imu': npz[f'{prefix}__imu'],
                    'imu_is_real': bool(npz[f'{prefix}__imu_is_real']),
                    'pressure_ap': npz[f'{prefix}__pressure_ap'],
                    'image': npz[f'{prefix}__image'],
                    'has_image': bool(npz[f'{prefix}__has_image']),
                    'audio': npz[f'{prefix}__audio'],
                    'has_audio': bool(npz[f'{prefix}__has_audio']),
                }

    return _load_raw_trial_features(hdf5_dir, cropped_dir, obj_id, action, trial, device=device)


# ── Normalization ────────────────────────────────────────────────────────

# Fold-1 diagnostic (2026-07-19, ~600 real training images): 14/2048
# ResNet-50 pooled-feature dimensions were exactly 0 (dead post-ReLU units)
# across every training image. Per-channel z-score with a naive epsilon
# floor blew those dims up to normalized values of 10,000-54,000 on the
# test set whenever a test image happened to have any nonzero activation
# there. Fix: dimensions with training std <= this threshold are dropped
# entirely (not just floored/zeroed) before normalizing -- see
# compute_norm_stats's 'image_resnet_alive_dims' and apply_norm_stats.
RESNET_DEAD_CHANNEL_STD_THRESH = 1e-5


def compute_norm_stats(train_object_ids, object_ids, hdf5_dir, cropped_dir, device='cpu',
                        cache_dir=None):
    """
    Scans every trial of the given TRAINING object indices only and returns
    per-modality mean/std for z-scoring. Call once per fold with that
    fold's training indices, and share the result with both that fold's
    train and test TactileDatasetV3 -- never compute from test objects
    (leakage).

    cache_dir: optional path to precompute_features.py's cache -- when
    given, raw per-trial features are read from there instead of being
    recomputed from HDF5+ResNet+MFCC (falls back to the original path per
    trial if a cache entry is missing, see _load_trial_features_cached).

    Placeholder/dropout trials are excluded from the relevant modality's
    stats (they're exact-zero and would bias mean/std toward zero
    otherwise): IMU trials with no clean de-interleaved cycles, image
    trials with has_image=False (tap/slide + missing/blank-frame trials),
    audio trials with has_audio=False (full sensor dropout).

    pressure_ap has no dropout/placeholder case in the current dataset
    (extract_pressure_ap_features always returns real resampled data
    given >=2 raw timesteps, and the full dataset's minimum trial length
    is 2) so every trial's full (224, 50) contributes.

    image_resnet is special: dimensions with training std <=
    RESNET_DEAD_CHANNEL_STD_THRESH (dead across every training image) are
    DROPPED, not just floored -- 'image_resnet' below only covers the
    surviving dims, and 'image_resnet_alive_dims' gives their original
    (0..2047) indices, needed by apply_norm_stats to select the matching
    dims from the raw 2048-dim ResNet features. This means the final
    normalized image_resnet width (and therefore the whole normalized
    'image' vector, image_resnet + 256 GLCM) is FOLD-DEPENDENT -- the set
    (and count) of dead dims can vary fold to fold since each fold trains
    on a different ~68-object subset. Read the actual width off
    len(stats['image_resnet_alive_dims']) + GLCM_FEATURE_DIM rather than
    assuming a fixed 2304 (or 2048) downstream in model.py.

    Returns {modality: (mean, std)} with:
      'imu':                (10,), (10,)
      'pressure_ap':         (50,), (50,)
      'image_resnet':        (n_alive,), (n_alive,)   -- n_alive <= 2048, fold-dependent
      'image_resnet_alive_dims': (n_alive,) int64 array of original dim indices (not a (mean,std) pair)
      'image_glcm':          (256,), (256,)
      'audio':               (2, 64), (2, 64)
    """
    sums, sqs, cnts = {}, {}, {}
    npz_cache = {}

    def accumulate(key, rows):
        # rows: (n_samples, n_channels)
        s = rows.sum(axis=0)
        sq = (rows.astype(np.float64) ** 2).sum(axis=0)
        n = rows.shape[0]
        if key not in sums:
            sums[key], sqs[key], cnts[key] = s.astype(np.float64), sq, n
        else:
            sums[key] += s
            sqs[key] += sq
            cnts[key] += n

    for obj_idx in train_object_ids:
        obj_id = str(object_ids[obj_idx])
        h5_path = os.path.join(hdf5_dir, f'{obj_id}.h5')
        if not os.path.isfile(h5_path):
            continue

        with h5py.File(h5_path, 'r') as f:
            trial_list = [(action, trial) for action in ACTIONS if action in f
                          for trial in sorted(f[action].keys())]

        for action, trial in trial_list:
            feats = _load_trial_features_cached(hdf5_dir, cropped_dir, obj_id, action, trial,
                                                 device=device, cache_dir=cache_dir,
                                                 npz_cache=npz_cache)

            if feats['imu_is_real']:
                accumulate('imu', feats['imu'].reshape(1, -1))

            accumulate('pressure_ap', feats['pressure_ap'])  # (224, 50) -> 224 rows

            if feats['has_image']:
                accumulate('image_resnet', feats['image'][:RESNET_FEATURE_DIM].reshape(1, -1))
                accumulate('image_glcm', feats['image'][RESNET_FEATURE_DIM:].reshape(1, -1))

            if feats['has_audio']:
                audio = feats['audio']                          # (2, 64, T)
                flat = audio.transpose(2, 0, 1).reshape(audio.shape[2], -1)  # (T, 2*64)
                accumulate('audio', flat)

    stats = {}
    for key in sums:
        mean = sums[key] / cnts[key]
        var = np.maximum(sqs[key] / cnts[key] - mean ** 2, 0.0)
        std = np.sqrt(var)

        if key == 'image_resnet':
            alive = std > RESNET_DEAD_CHANNEL_STD_THRESH
            stats['image_resnet_alive_dims'] = np.where(alive)[0].astype(np.int64)
            mean = mean[alive]
            std = np.maximum(std[alive], 1e-6)
            stats[key] = (mean.astype(np.float32), std.astype(np.float32))
            continue

        std = np.maximum(std, 1e-6)
        if key == 'audio':
            mean = mean.reshape(2, 64)
            std = std.reshape(2, 64)
        stats[key] = (mean.astype(np.float32), std.astype(np.float32))
    return stats


def apply_norm_stats(features: dict, stats: dict) -> dict:
    """
    features: dict as returned by _load_raw_trial_features (raw,
    unnormalized arrays + imu_is_real/has_image/has_audio flags).
    stats: dict from compute_norm_stats.

    Returns a new dict with the same array keys, z-scored per modality.
    Placeholder/dropout arrays (imu when not imu_is_real, audio when not
    has_audio) are left exactly as-is (all zero) rather than normalized --
    normalizing a zero placeholder would shift it to a nonzero, meaningless
    value (-mean/std); the has_*/imu_is_real flags are how the model is
    meant to gate on these, so the placeholder should stay recognizably
    zero. pressure_ap has no such placeholder case and is always
    normalized.

    image is a partial exception: real images are first reduced to the
    fold's surviving (non-dead) ResNet dims via
    stats['image_resnet_alive_dims'] before normalizing (see
    compute_norm_stats), so the output width is
    len(alive_dims) + GLCM_FEATURE_DIM, not the raw 2304. The has_image=0
    placeholder is built at THIS reduced width (all zero), not the raw
    2304-wide placeholder from features['image'] -- otherwise real and
    placeholder 'image' vectors would have different lengths within the
    same fold and couldn't be batched together.
    """
    out = dict(features)

    if features['imu_is_real']:
        mean, std = stats['imu']
        out['imu'] = (features['imu'] - mean) / std

    mean, std = stats['pressure_ap']
    out['pressure_ap'] = (features['pressure_ap'] - mean) / std

    alive_dims = stats['image_resnet_alive_dims']
    if features['has_image']:
        img = features['image']
        mean_r, std_r = stats['image_resnet']
        mean_g, std_g = stats['image_glcm']
        resnet_norm = (img[:RESNET_FEATURE_DIM][alive_dims] - mean_r) / std_r
        glcm_norm = (img[RESNET_FEATURE_DIM:] - mean_g) / std_g
        out['image'] = np.concatenate([resnet_norm, glcm_norm]).astype(np.float32)
    else:
        out['image'] = np.zeros(len(alive_dims) + GLCM_FEATURE_DIM, dtype=np.float32)

    if features['has_audio']:
        mean, std = stats['audio']
        out['audio'] = (features['audio'] - mean[:, :, None]) / std[:, :, None]

    return out


# ── Dataset ──────────────────────────────────────────────────────────────

class TactileDatasetV3(Dataset):
    def __init__(self, object_indices, object_ids, labels_dist, labels_mean, norm_stats,
                 hdf5_dir, cropped_dir, device='cpu', cache_dir=None):
        """
        Args:
            object_indices: list of int indices into labels_dist/labels_mean/object_ids
                             for this split (train or test set of a fold).
            object_ids:     (85,) array of object id strings.
            labels_dist:    (85, 18, 7) float32.
            labels_mean:    (85, 18) float32.
            norm_stats:     dict from compute_norm_stats -- computed once from a
                             fold's TRAINING indices, shared by that fold's train
                             and test datasets.
            hdf5_dir:       path to DATA_Final/hdf5 (pressure_ap/imu_raw/audio).
            cropped_dir:    path to DATA_Final/hdf5_cropped_images (image only).
            device:         device for the frozen ResNet-50 forward pass.
            cache_dir:      optional path to precompute_features.py's cache. When
                             given, __getitem__ reads raw features from there
                             instead of recomputing them, falling back to the
                             original HDF5+ResNet+MFCC path per-trial if a cache
                             entry is missing (see _load_trial_features_cached).
        """
        self.object_ids = object_ids
        self.labels_dist = labels_dist
        self.labels_mean = labels_mean
        self.norm_stats = norm_stats
        self.hdf5_dir = hdf5_dir
        self.cropped_dir = cropped_dir
        self.device = device
        self.cache_dir = cache_dir
        self._npz_cache = {}
        self.samples = []
        self._build_sample_list(object_indices)

    def _build_sample_list(self, object_indices):
        for obj_idx in object_indices:
            obj_id = str(self.object_ids[obj_idx])
            h5_path = os.path.join(self.hdf5_dir, f'{obj_id}.h5')
            if not os.path.isfile(h5_path):
                print(f"[TactileDatasetV3] Warning: not found: {h5_path}")
                continue
            with h5py.File(h5_path, 'r') as f:
                for action in ACTIONS:
                    if action not in f:
                        continue
                    for trial in sorted(f[action].keys()):
                        self.samples.append((obj_idx, action, trial))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obj_idx, action, trial = self.samples[idx]
        obj_id = str(self.object_ids[obj_idx])

        raw = _load_trial_features_cached(self.hdf5_dir, self.cropped_dir, obj_id, action, trial,
                                           device=self.device, cache_dir=self.cache_dir,
                                           npz_cache=self._npz_cache)
        norm = apply_norm_stats(raw, self.norm_stats)

        label_dist = self.labels_dist[obj_idx].astype(np.float32)
        label_mean = self.labels_mean[obj_idx].astype(np.float32)

        return {
            'imu': torch.from_numpy(np.ascontiguousarray(norm['imu'])),
            'pressure_ap': torch.from_numpy(np.ascontiguousarray(norm['pressure_ap'])),
            'image': torch.from_numpy(np.ascontiguousarray(norm['image'])),
            'audio': torch.from_numpy(np.ascontiguousarray(norm['audio'])),
            'has_image': torch.tensor(float(raw['has_image']), dtype=torch.float32),
            'has_audio': torch.tensor(float(raw['has_audio']), dtype=torch.float32),
            'label_dist': torch.from_numpy(label_dist),
            'label_mean': torch.from_numpy(label_mean),
            'obj_idx': torch.tensor(obj_idx, dtype=torch.long),
        }
