# Predicting Human Perceptual Distributions of Haptic Adjectives

This repository holds the code for the Master's thesis "Predicting Human Perceptual Distributions of Haptic Adjectives" by Samar (INSAT, Tunisia, and TU Dresden, LASR Lab). We recorded 85 everyday objects with a Meta DIGIT 360 tactile sensor. Twenty people rated the same objects on 18 haptic adjectives. A multimodal neural network takes the sensor recordings and predicts, for each object and adjective, the full distribution of human ratings over the 7-point scale, not only the mean. The code covers data collection, label building, preprocessing, training, evaluation and one negative-result experiment with Sparsh-X image features.

## Overview

- **Objects:** 85, in two sessions (`obj_01`–`obj_45` and `obj_46`–`obj_85`).
- **Participants:** 20.
- **Labels:** 18 adjectives rated on a 7-point scale, turned into one 7-bin distribution per object and adjective (18 × 7 per object).
- **Sensor:** Meta DIGIT 360 with ROS 2 Jazzy.
- **Procedures:** tap, squeeze, slide and hold, with 5 trials of about 4 s per procedure.
- **Model inputs:** four modalities: IMU, pressure_ap, audio and tactile image.

Details: [docs/methodology.md](docs/methodology.md).

## Method

- **Image:** frozen ResNet-50 features plus GLCM texture features, then an MLP.
- **IMU:** DFT321 on the accelerometer, then MFCCs plus summary statistics, then an MLP.
- **pressure_ap:** resampled to (224, 50), then a 2D CNN.
- **Audio:** stereo log-mel spectrogram, then a CNN.
- **Fusion:** the four branch outputs are concatenated, and an MLP head with a softmax gives an 18 × 7 distribution.

`pressure` and `ght` are recorded but not used.

Evaluation uses object-level 5-fold cross-validation, with EMD as the main metric, plus MAE and the mean per-adjective Pearson r.

Branch sizes, feature parameters, training settings, metric definitions and third-party credits: [docs/methodology.md](docs/methodology.md).

## Results

| Run | EMD | MAE | r |
|---|---|---|---|
| Main run (seed 42) | 0.1037 | 0.6837 | 0.506 |
| 5 seeds (42, 7, 99, 123, 2024), mean ± SD | 0.1047 ± 0.0011 | 0.6897 ± 0.0084 | 0.497 ± 0.014 |
| Adjective-weighted EMD (ablation) | 0.1196 | 0.7945 | 0.508 |
| Regularized, dropout 0.3, weight decay 5e-4 (ablation) | 0.1296 | 0.8733 | 0.014 |
| Sparsh-X image features (negative result, 75 objects only) | 0.1189 | 0.7934 | 0.286 |

r is low for five adjectives (sticky, cold, warm, wet and dry), whose ratings cluster at one end of the scale. Per-adjective details are in [docs/methodology.md](docs/methodology.md#results-per-adjective). The Sparsh-X experiment is described in [docs/sparsh_x.md](docs/sparsh_x.md).

Human mean ratings and model predictions for session 1 (main run, out-of-fold), and predicted vs. true expected rating per adjective:

![Human mean ratings, session 1](docs/figures/heatmap_human_session1.png)
![Model predictions, session 1](docs/figures/heatmap_predicted_session1.png)
![Predicted vs. true expected rating per adjective](docs/figures/predicted_vs_true_grid.png)

All figures, including session 2 and the difference heatmaps: [docs/methodology.md](docs/methodology.md#all-result-figures).

## Repository structure

```
data_collection/        recording (ROS 2) and .mcap -> HDF5
human_study/            questionnaires -> label files
  plots/                human-rating figures
data_prep/              image cropping
model/
  src/                  features, dataset, model, training, paths.py
  slurm/                SLURM template
experiments/sparsh_x/   Sparsh-X experiment and upstream patch
analysis/
  preprocessing/        preprocessing checks
  figures/              example figures
  results/              result plots and metrics
tests/                  diagnostic checks (need the data)
results/                metrics and predictions of each reported run
data/labels/            aggregate label files
data/README.md          data layout
docs/                   methodology, reproducibility, Sparsh-X, figures
```

Scripts write their outputs to `outputs/<script_name>/`. Training runs write to `outputs/train/results/` by default.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Package versions are not pinned yet. `data_collection/` also needs ROS 2 Jazzy (for `rclpy` and `rosidl_runtime_py`). The Sparsh-X experiment needs a separate environment, see `experiments/sparsh_x/requirements.txt`.

All data paths come from environment variables (`TACTILE_DATA_ROOT`, `TACTILE_LABELS_DIR`, and others). The full list is in [docs/reproducibility.md](docs/reproducibility.md#environment-variables).

## Running the project

1. Record: `bash data_collection/record2.sh obj_01 A`
2. Convert to HDF5: `python3 data_collection/mcap_to_hdf5.py --data-root ~/tactile_recordings/A --output-root $TACTILE_DATA_ROOT`
3. Crop images: `python data_prep/crop_images.py`
4. Build labels: `python human_study/build_labels.py --out-dir $TACTILE_LABELS_DIR`
5. Precompute features (optional): `python model/src/precompute_features.py`
6. Train: `python model/src/train.py`
7. Evaluate: `python analysis/results/mean_per_adjective_r.py --csv results/main_seed42/predictions_vs_truth.csv`

All options, the steps before `build_labels.py`, the SLURM template and every analysis script: [docs/reproducibility.md](docs/reproducibility.md#running-the-pipeline-in-full).

## Data

Included:

- Aggregate labels in `data/labels/`: the per-object rating distributions and expected ratings for all 85 objects. You can set `TACTILE_LABELS_DIR` to this folder.
- Per-object predictions of each reported run in `results/*/predictions_vs_truth.csv`.

Not included: participant-level ratings and raw sensor recordings (and the HDF5 files made from them). `data/README.md` describes the data layout.

## Limitations & reproducibility

The held-out test folds were also used to select the best epoch, trigger early stopping and drive the learning-rate schedule, so the reported scores are optimistically biased; the size of this bias was not measured. The main run came from an earlier revision of `train.py`, bit-exact reproduction was not verified, and no environment versions were recorded. r is low for sticky, cold, warm, wet and dry. The full list, open items and verification status are in [docs/reproducibility.md](docs/reproducibility.md).

## License

This work (code, patch files, data and figures) is licensed under CC BY-NC 4.0. See `LICENSE`, which also lists the third-party material.

## Acknowledgements

Thanks to Prof. Roberto Calandra and Prof. Merle Fairhurst, and the members of the LASR Lab at TU Dresden for their support. Thanks to Niklas Leukroth for the lab's ROS 2 adaptation of the DIGIT 360 driver, and to the 20 participants who took part in the perception study.
