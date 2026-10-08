# Data layout

**Included:** the aggregate label files in `data/labels/` (see [Included label files](#included-label-files-datalabels)) and, outside this folder, the per-object predictions in `results/*/predictions_vs_truth.csv`.

**Not included:** raw recordings, the HDF5 files made from them, and participant-level ratings. This page describes the layout the code expects, so that you can point it at your own copy.

Every value below is quoted with the file and line it comes from.

## Locations (environment variables)

| Variable | Contents | Defined in |
|---|---|---|
| `TACTILE_DATA_ROOT` | `hdf5/` and `hdf5_cropped_images/` | `model/src/paths.py:40-50` |
| `TACTILE_LABELS_DIR` | `labels_phase2.npy`, `labels_mean_phase2.npy`, `object_ids_phase2.npy` | `model/src/paths.py:53-56` |
| `TACTILE_CACHE_DIR` | optional feature cache, default `<repo>/cache` | `model/src/paths.py:59-60` |

```
$TACTILE_DATA_ROOT/
├── hdf5/
│   ├── obj_01.h5          one file per object, written by data_collection/mcap_to_hdf5.py
│   ├── ...
│   └── obj_85.h5
└── hdf5_cropped_images/
    ├── obj_01.h5          cropped images only, written by data_prep/crop_images.py
    └── ...

$TACTILE_LABELS_DIR/
├── labels_phase2.npy
├── labels_mean_phase2.npy
└── object_ids_phase2.npy
```

## `hdf5/obj_XX.h5`

Written by `data_collection/mcap_to_hdf5.py`. Output folder: `<output-root>/hdf5/` (line 340), file name `<obj>.h5` (line 342).

```
obj_XX.h5
├── metadata/object_name          string "obj_XX"
└── <action>/                     tap, squeeze, slide, hold
    └── trial_NNN/                trial_000, trial_001, ...
        attrs: image_valid (bool)
        ├── image                 (224, 224, 3)  float32
        ├── imu_raw               (N, 3)         float32
        ├── pressure_ap           (N, 50)        float32
        ├── pressure              (N, 2)         float32
        ├── ght                   (4,)           float32
        └── audio                 (2, 64, T)     float32
```

| Item | Value | Source (`data_collection/mcap_to_hdf5.py`) |
|---|---|---|
| `metadata/object_name` | object folder name | line 345 |
| Action groups | `tap`, `squeeze`, `slide`, `hold` | lines 51, 358 |
| Trial groups | `trial_{idx:03d}`, one per recording | line 304 |
| `image_valid` attribute | True if an image frame was found | lines 306-307 |
| `image` | (224, 224, 3) float32, RGB in [0, 1], zeros if no frame | lines 149-165, 308 |
| `image` frame choice | the frame closest to the middle of the recording | lines 157-158 |
| `image` pre-crop | threshold 10, largest contour, inscribed square, resized to 224×224 | lines 125-144 |
| `imu_raw` | (N, 3) float32, message fields x, y, z | lines 168-172, 309 |
| `pressure_ap` | (N, 50) float32: `ch_a` and `ch_b`, each padded/cut to 25 values | lines 175-191, 310 |
| `pressure` | (N, 2) float32: pressure, temperature | lines 194-198, 311 |
| `ght` | (4,) float32: mean of temperature, humidity, gas, pressure | lines 201-209, 312 |
| `audio` | (2, 64, T) float32, one log-mel spectrogram per microphone (mic_0, mic_1), trimmed to the shorter T | lines 271-282, 320 |
| `audio` minimum length | zero-padded to at least 100 frames | lines 49, 316-319 |

ROS topics read (lines 54-62): `/image_raw/index_0`, `/imu_raw_topic/index_0`, `/pressure_ap_topic/index_0`, `/pressure_topic/index_0`, `/ght_topic/index_0`, `/mic_0/index_0`, `/mic_1/index_0`.

### Mel spectrogram parameters

| Parameter | Value | Source (`data_collection/mcap_to_hdf5.py`) |
|---|---|---|
| Sample rate | 48,000 Hz (a constant in the code, not read from the messages) | line 48 |
| n_mels | 64 | line 46 |
| hop_length | 512 | line 47 |
| n_fft | 2048 | line 212 |
| Window | `np.hanning(2048)` | line 254 |
| Frames | start at sample 0, step hop_length, no centre padding | line 256 |
| Spectrum | power, \|rfft\|² | lines 262-263 |
| Mel filterbank | triangular, HTK mel formula, fmin 0, fmax sr/2, no area normalisation | lines 214-238 |
| dB scaling | 10·log10(mel / max(mel)), floor 1e-10 | lines 266-267 |

The model only reads `imu_raw`, `pressure_ap` and `audio` from these files (`model/src/dataset.py:127-131`). `pressure` and `ght` are stored but not used (`model/src/dataset.py:20-22`).

## `hdf5_cropped_images/obj_XX.h5`

Written by `data_prep/crop_images.py`. Only trials with `image_valid = True` are written (lines 55-58).

```
obj_XX.h5
└── <action>/<trial_NNN>/
    attrs: image_valid, crop_radius, crop_n_fragments
    └── image                     (224, 224, 3)  float32
```

| Item | Value | Source |
|---|---|---|
| Group | `<action>/<trial_NNN>` | `data_prep/crop_images.py:73` |
| `image` | (224, 224, 3) float32 | `data_prep/crop_images.py:74`, `data_prep/image_crop.py:55-77` (size 224, `cv2.INTER_AREA`) |
| Attributes | `image_valid`, `crop_radius`, `crop_n_fragments` | `data_prep/crop_images.py:75-77` |

The model reads only `squeeze` and `hold` images. Tap and slide trials, and any trial missing from this file, get `has_image = False` (`model/src/dataset.py:49, 104-112`).

## Label files (`$TACTILE_LABELS_DIR`)

Written by `human_study/build_labels.py --out-dir <folder>` from `labels_distribution.csv`.

| File | Shape | dtype | Contents | Source (`human_study/build_labels.py`) |
|---|---|---|---|---|
| `labels_phase2.npy` | (85, 18, 7) | float32 | per object and adjective, the fraction of ratings at each level 1-7 | lines 59, 63-65, 69 |
| `labels_mean_phase2.npy` | (85, 18) | float32 | expected rating, sum over k of k·p_k (k = 1…7) | lines 39, 60, 66, 70 |
| `object_ids_phase2.npy` | (85,) | object array of strings | `obj_01` … `obj_85`, sorted by number | lines 53, 57, 71 |

Adjective order (axis 1): hard, soft, rough, smooth, bumpy, flat, sticky, slippery, cold, warm, wet, dry, compressible, firm, fuzzy, scratchy, pleasant, irritating (`human_study/build_labels.py` `ADJECTIVES`).

`labels_distribution.csv` (85 rows × 127 columns: `object_id` plus `<adjective>_<level>`) comes from `human_study/build_label_distributions.py`. For each object and adjective it counts the ratings at each level, divides by the number of non-missing ratings, and rounds to 6 decimals (lines 50-58).

## Included label files (`data/labels/`)

These four files are in the repository. They contain per-object aggregates only, with no participant IDs. To use them for training or analysis, set `TACTILE_LABELS_DIR=<repo>/data/labels`.

| File | Shape | dtype | Contents |
|---|---|---|---|
| `labels_distribution.csv` | 85 rows × 127 columns | text | `object_id`, then `<adjective>_1` … `<adjective>_7` for the 18 adjectives: the fraction of ratings at each level (columns written by `human_study/build_label_distributions.py:40, 58`) |
| `labels_phase2.npy` | (85, 18, 7) | float32 | the same distributions as an array (`human_study/build_labels.py:59, 65`) |
| `labels_mean_phase2.npy` | (85, 18) | float32 | expected rating per object and adjective, on the 1–7 scale (`human_study/build_labels.py:60, 66`) |
| `object_ids_phase2.npy` | (85,) | object (strings) | `obj_01` … `obj_85`, the row order of both arrays (`human_study/build_labels.py:53, 57`) |

The three `.npy` files were built from `labels_distribution.csv` by `build_labels.py`. Rebuilding them from this CSV gives identical arrays. Load `object_ids_phase2.npy` with `np.load(..., allow_pickle=True)`.

## Optional caches

- **Feature cache** (`TACTILE_CACHE_DIR`, written by `model/src/precompute_features.py`): one `<obj_id>.npz` per object (line 66). Keys are `<action>__<trial>__imu`, `__imu_is_real`, `__pressure_ap`, `__image`, `__has_image`, `__audio` and `__has_audio` (lines 56-63). These are raw, un-normalised features. The cache is not invalidated automatically, so delete it after you change a feature extractor.
- **Sparsh-X features** (`SPARSHX_FEATURES_DIR`, written by `experiments/sparsh_x/precompute_sparshx_features.py`): one `<obj_id>_<action>_<trial>.npy` per squeeze/hold trial (lines 292, 304). Each is a 768-dim embedding (`experiments/sparsh_x/dataset_sparshimg.py:68, 111-112`).
