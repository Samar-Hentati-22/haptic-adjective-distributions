"""
Pipeline v3-sparshimg — unified dataset class. Copied from
Training_Phase2/pipeline_v3/src/dataset.py with ONE change: the image
branch is now precomputed Sparsh-X embeddings (768-dim, fixed) instead of
frozen ResNet-50 + GLCM (2304-dim raw, fold-dependent after dead-channel
dropping). IMU, pressure_ap, and audio are untouched -- same modules
(imu_features.py, pressure_ap_features.py, audio_ght_features.py, imported
from model/src/), same precompute_features.py cache reuse (from
TACTILE_CACHE_DIR -- imu/pressure_ap/audio entries in that cache are
still valid since those extractors didn't change; only the image/has_image
fields from that cache are ignored and always recomputed from the Sparsh-X
feature files instead).

Imports imu/pressure_ap/audio feature modules and paths.py from model/src/;
reads TACTILE_CACHE_DIR/*.npz (read-only) and SPARSHX_FEATURES_DIR/*.npy
(read-only) -- both external caches, not code.

Sparsh-X embeddings: precomputed by precompute_sparshx_features.py,
one (768,) float32 .npy per {obj_id}_{action}_{trial}.npy, for action in
{squeeze, hold} and trial in {trial_000..trial_004} (that script's scope --
see its ALL_OBJECTS/PROCEDURES/TRIALS constants). tap/slide trials have no
Sparsh-X embedding at all and always get has_image=False, matching
pipeline_v3's original tap/slide-always-has_image=False convention; a
squeeze/hold trial gets has_image=False only if its specific .npy is
missing (e.g. that trial failed extraction -- see precompute_all.py's
per-trial try/except and [FAILED] log lines).

Per-trial output feature dims: imu (10), pressure_ap (224, 50), image
(768, FIXED -- unlike pipeline_v3's fold-dependent ResNet width, Sparsh-X's
embedding is already LayerNorm'd and has no dead/constant channels to
prune, so no per-fold alive-dims logic is needed here), audio (2, 64, T --
T variable, unchanged from pipeline_v3).
`pressure` and `ght` are intentionally NOT loaded anywhere in this module,
matching pipeline_v3's scope.

EXCLUDE_OBJECTS is NOT a fixed frozenset here (unlike pipeline_v3, which
currently excludes nothing) -- call get_incomplete_objects(object_ids) to
compute it dynamically from sparsh-x-explore/features_cache/'s actual
contents. An object is excluded only if it has ZERO usable Sparsh-X image
trials at all (no image signal whatsoever); objects with SOME but not all
trials cached are kept and simply get has_image=False for whichever
individual trials are missing -- train.py is expected to pass
get_incomplete_objects's result as make_kfold_splits's `exclude`.
"""
import os
import sys

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402
from imu_features import deinterleave_imu, extract_imu_features  # noqa: E402  (model/src)
from pressure_ap_features import extract_pressure_ap_features  # noqa: E402  (model/src)
from audio_ght_features import extract_audio_features  # noqa: E402  (model/src)

ACTIONS = ['tap', 'squeeze', 'slide', 'hold']
# Ported from pipeline_v3/src/dataset.py: only squeeze/hold get a real
# camera frame at all in this dataset.
IMAGE_ACTIONS = frozenset({'squeeze', 'hold'})
# Sparsh-X features were precomputed for all 5 trials of squeeze/hold --
# see sparsh-x-explore/precompute_all.py's TRIALS constant.
IMAGE_TRIALS = [f'trial_{i:03d}' for i in range(5)]

IMU_FEATURE_DIM = 10
IMAGE_FEATURE_DIM = 768  # Sparsh-X img-only base checkpoint embedding width, fixed (not fold-dependent)
PRESSURE_AP_SHAPE = (224, 50)

# Sparsh-X features_cache written by precompute_sparshx_features.py (SPARSHX_FEATURES_DIR, see paths.py).
SPARSHX_FEATURES_DIR = paths.sparshx_features_dir()

# No objects excluded by default (matches pipeline_v3's current
# EXCLUDE_OBJECTS = frozenset()) -- train.py is expected to compute and
# pass the real exclusion set via get_incomplete_objects() instead, since
# it depends on features_cache/'s contents, not a fixed list.
EXCLUDE_OBJECTS = frozenset()


# ── K-fold splitting (ported from pipeline_v3/src/dataset.py, unchanged) ───

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


# ── Object-level Sparsh-X feature completeness (new) ────────────────────

def _sparshx_path(obj_id, action, trial):
    return os.path.join(SPARSHX_FEATURES_DIR, f'{obj_id}_{action}_{trial}.npy')


def get_incomplete_objects(object_ids, hdf5_dir=None):
    """
    Returns the set of object_id strings that have ZERO usable Sparsh-X
    image trials on disk in SPARSHX_FEATURES_DIR (0 of up to 10:
    squeeze/hold x trial_000..trial_004) -- i.e. the exact objects a
    caller should pass to make_kfold_splits's `exclude` to drop objects
    that would otherwise train/eval with has_image=False on every single
    trial (no image signal at all, not just a partially incomplete one).

    Objects with SOME but not all trials cached are NOT excluded here --
    they're trained/evaluated on whichever trials actually have a Sparsh-X
    embedding; any specific missing trial just resolves to has_image=False
    for that trial via _load_sparshx_image_features, same placeholder
    convention already used for tap/slide and any other missing/failed
    trial elsewhere in this module.

    hdf5_dir: if given, the candidate trial list per object/action is read
    from DATA_Final/hdf5/{obj_id}.h5 directly (robust to an object having
    fewer than 5 trials of an action). If omitted, falls back to assuming
    the standard 5 trials (IMAGE_TRIALS) for both squeeze and hold, which
    holds for every object in the current dataset.
    """
    incomplete = set()
    for oid in object_ids:
        oid = str(oid)

        if hdf5_dir is not None:
            h5_path = os.path.join(hdf5_dir, f'{oid}.h5')
            if not os.path.isfile(h5_path):
                incomplete.add(oid)
                continue
            with h5py.File(h5_path, 'r') as f:
                candidates = [(action, trial) for action in IMAGE_ACTIONS if action in f
                              for trial in sorted(f[action].keys())]
        else:
            candidates = [(action, trial) for action in IMAGE_ACTIONS for trial in IMAGE_TRIALS]

        if not any(os.path.isfile(_sparshx_path(oid, action, trial)) for action, trial in candidates):
            incomplete.add(oid)
    return incomplete


# ── Per-trial raw feature loading (shared by compute_norm_stats and __getitem__) ─

def _load_sparshx_image_features(obj_id, action, trial):
    """
    Returns (features (768,) float32, has_image: bool).

    has_image is False (all-zero placeholder) whenever action isn't
    squeeze/hold, OR trial isn't one of the 5 Sparsh-X was precomputed for
    (IMAGE_TRIALS), OR the .npy file is missing (e.g. that trial failed
    extraction in precompute_all.py -- see its per-trial try/except and
    [FAILED] log lines).
    """
    if action not in IMAGE_ACTIONS or trial not in IMAGE_TRIALS:
        return np.zeros(IMAGE_FEATURE_DIM, dtype=np.float32), False

    path = _sparshx_path(obj_id, action, trial)
    if not os.path.isfile(path):
        return np.zeros(IMAGE_FEATURE_DIM, dtype=np.float32), False

    feats = np.load(path).astype(np.float32)
    if feats.shape != (IMAGE_FEATURE_DIM,):
        return np.zeros(IMAGE_FEATURE_DIM, dtype=np.float32), False
    return feats, True


def _load_raw_trial_features(hdf5_dir, cropped_dir, obj_id, action, trial, device='cpu'):
    """
    Loads one trial and runs every modality through its (already validated)
    feature extractor. Returns a dict of RAW (unnormalized) features plus
    'imu_is_real'/'has_image'/'has_audio' flags, for either direct use by
    compute_norm_stats (accumulating stats) or as input to apply_norm_stats
    (for __getitem__).

    cropped_dir is accepted (unused) only to keep this function's call
    signature identical to pipeline_v3's -- the image branch no longer
    reads cropped HDF5 images, it reads precomputed Sparsh-X embeddings.
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

    image_feat, has_image = _load_sparshx_image_features(obj_id, action, trial)

    audio_feat, has_audio = extract_audio_features(audio_arr)

    return {
        'imu': imu_feat, 'imu_is_real': imu_is_real,
        'pressure_ap': pap_feat,
        'image': image_feat, 'has_image': has_image,
        'audio': audio_feat, 'has_audio': has_audio,
    }


def _load_object_npz(cache_dir, obj_id):
    """Open one object's precomputed-feature .npz (see
    pipeline_v3/src/precompute_features.py -- reused as-is here for
    imu/pressure_ap/audio; its image/has_image entries are ignored, see
    _load_trial_features_cached), or None if it doesn't exist."""
    path = os.path.join(cache_dir, f'{obj_id}.npz')
    if not os.path.isfile(path):
        return None
    return np.load(path)


def _load_trial_features_cached(hdf5_dir, cropped_dir, obj_id, action, trial, device='cpu',
                                  cache_dir=None, npz_cache=None):
    """
    Like _load_raw_trial_features, but reads imu/pressure_ap/audio from
    pipeline_v3/src/precompute_features.py's cache when available (falling
    back to the original HDF5 extraction path per-trial if a cache entry
    is missing, matching pipeline_v3's original fallback behavior) --
    EXCEPT image/has_image, which are ALWAYS loaded fresh from the
    Sparsh-X features_cache (that cache's own image entries, if any, are
    ResNet-based and not applicable here).

    npz_cache (optional dict, keyed by obj_id) lets a caller that loads
    many trials of the same object (compute_norm_stats, or a
    TactileDatasetV3 instance across its lifetime) avoid reopening the
    same .npz file per trial.
    """
    image, has_image = _load_sparshx_image_features(obj_id, action, trial)

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
                    'image': image, 'has_image': has_image,
                    'audio': npz[f'{prefix}__audio'],
                    'has_audio': bool(npz[f'{prefix}__has_audio']),
                }

    raw = _load_raw_trial_features(hdf5_dir, cropped_dir, obj_id, action, trial, device=device)
    raw['image'] = image
    raw['has_image'] = has_image
    return raw


# ── Normalization ────────────────────────────────────────────────────────

def compute_norm_stats(train_object_ids, object_ids, hdf5_dir, cropped_dir, device='cpu',
                        cache_dir=None):
    """
    Scans every trial of the given TRAINING object indices only and returns
    per-modality mean/std for z-scoring. Call once per fold with that
    fold's training indices, and share the result with both that fold's
    train and test TactileDatasetV3 -- never compute from test objects
    (leakage).

    cache_dir: optional path to pipeline_v3/src/precompute_features.py's
    cache (imu/pressure_ap/audio only -- see _load_trial_features_cached).

    Placeholder/dropout trials are excluded from the relevant modality's
    stats (they're exact-zero and would bias mean/std toward zero
    otherwise): IMU trials with no clean de-interleaved cycles, image
    trials with has_image=False (tap/slide, or a squeeze/hold trial whose
    Sparsh-X extraction failed), audio trials with has_audio=False (full
    sensor dropout).

    pressure_ap has no dropout/placeholder case in the current dataset
    (extract_pressure_ap_features always returns real resampled data
    given >=2 raw timesteps, and the full dataset's minimum trial length
    is 2) so every trial's full (224, 50) contributes.

    image is a plain per-dim z-score over the fixed 768 Sparsh-X
    dimensions (unlike pipeline_v3's ResNet branch, no dead-channel
    dropping is done here -- Sparsh-X's embedding is already LayerNorm'd
    at extraction time and isn't expected to have exact-zero dead units
    the way a frozen ImageNet ResNet's post-ReLU features did).

    Returns {modality: (mean, std)} with:
      'imu':          (10,), (10,)
      'pressure_ap':  (50,), (50,)
      'image':        (768,), (768,)
      'audio':        (2, 64), (2, 64)
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
                accumulate('image', feats['image'].reshape(1, -1))

            if feats['has_audio']:
                audio = feats['audio']                          # (2, 64, T)
                flat = audio.transpose(2, 0, 1).reshape(audio.shape[2], -1)  # (T, 2*64)
                accumulate('audio', flat)

    stats = {}
    for key in sums:
        mean = sums[key] / cnts[key]
        var = np.maximum(sqs[key] / cnts[key] - mean ** 2, 0.0)
        std = np.sqrt(var)
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
    Placeholder/dropout arrays (imu when not imu_is_real, image when not
    has_image, audio when not has_audio) are left exactly as-is (all zero)
    rather than normalized -- normalizing a zero placeholder would shift it
    to a nonzero, meaningless value (-mean/std); the has_*/imu_is_real
    flags are how the model is meant to gate on these, so the placeholder
    should stay recognizably zero. pressure_ap has no such placeholder
    case and is always normalized.
    """
    out = dict(features)

    if features['imu_is_real']:
        mean, std = stats['imu']
        out['imu'] = (features['imu'] - mean) / std

    mean, std = stats['pressure_ap']
    out['pressure_ap'] = (features['pressure_ap'] - mean) / std

    if features['has_image']:
        mean, std = stats['image']
        out['image'] = ((features['image'] - mean) / std).astype(np.float32)
    else:
        out['image'] = np.zeros(IMAGE_FEATURE_DIM, dtype=np.float32)

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
            cropped_dir:    accepted for signature parity with pipeline_v3, unused
                             (image now comes from precomputed Sparsh-X embeddings).
            device:         unused (kept for signature parity with pipeline_v3).
            cache_dir:      optional path to pipeline_v3/src/precompute_features.py's
                             cache (imu/pressure_ap/audio only). Falls back to the
                             original HDF5 path per-trial if a cache entry is missing.
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
