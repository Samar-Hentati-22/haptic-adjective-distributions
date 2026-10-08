# Methodology

Details of the study, the recordings, the features, the model, the training and the metrics. The short version is in the [README](../README.md). Reproduction notes are in [reproducibility.md](reproducibility.md), and the data format is in [data/README.md](../data/README.md).

## Human study

- 85 objects. Session 1 covers `obj_01`–`obj_45`, session 2 covers `obj_46`–`obj_85`.
- 20 participants.
- 18 adjectives: hard, soft, rough, smooth, bumpy, flat, sticky, slippery, cold, warm, wet, dry, compressible, firm, fuzzy, scratchy, pleasant, irritating.
- Ratings are on a 7-point unipolar scale (1–7).
- For each object and adjective, the ratings are turned into a 7-bin distribution: the fraction of participants who gave each rating.
- The label tensor has shape (85, 18, 7). A second file holds the expected rating per object and adjective, shape (85, 18).

The scripts in `human_study/` turn the questionnaire exports into these files. See [data/README.md](../data/README.md) for the exact format.

## Recording protocol

- Sensor: Meta DIGIT 360, recorded with ROS 2 Jazzy.
- Driver: the lab adaptation `digit360-ros2` (maintained by Niklas Leukroth, LASR) of `facebookresearch/digit360`. The driver was not modified.
- Each object was recorded with `data_collection/record2.sh` in four procedures: tap, squeeze, slide and hold.
- Each procedure has 5 trials, so there are 20 trials per object and 1,700 in total.
- Trial length: trials are about 4 s long (median 4.3–4.5 s across imu_raw, pressure_ap and pressure, computed from sample counts at the code's assumed rates; `trial_duration_summary.csv`, written by `analysis/preprocessing/03_trial_duration_by_procedure.py`). `record2.sh` sets `TRIAL_DURATION=2` but stops the recorder 2 s later.
- `record2.sh` records all topics. The recordings contain pressure, pressure_ap, imu_raw/euler/quat, ght, image_raw, mic_0 and mic_1.
- `data_collection/mcap_to_hdf5.py` converts the recordings into one HDF5 file per object. `data_prep/crop_images.py` then crops the tactile images.

## Features and model

The model works on single trials. At test time the trial predictions of an object are averaged.

It has four input branches. All features are z-scored with statistics from the training objects of the current fold only.

- **IMU.** The raw IMU log mixes four sensor streams. The code separates them and keeps the accelerometer stream. Its three axes are combined into one signal with DFT321 and high-pass filtered (4th-order Butterworth, 20 Hz, zero-phase). From this signal the code takes 5 MFCCs over a fixed 4 s window plus mean, standard deviation, minimum, maximum and RMS. That gives 10 values. An MLP maps them to 32 values.
- **pressure_ap.** The 50-channel pressure_ap array is cleaned of a known readout glitch (blocks of channels that read exactly 0). It is then resampled by linear interpolation to 224 time steps, giving shape (224, 50). A three-layer 2D CNN maps it to 64 values.
- **Audio.** A stereo log-mel spectrogram of shape (2, 64, T). It is computed with sample rate 48,000 Hz, 64 mel bands, n_fft 2048, hop length 512 and a Hann window (`np.hanning`); see [data/README.md](../data/README.md). A three-layer 2D CNN with global average pooling maps it to 64 values, so T can vary.
- **Tactile image.** Only squeeze and hold trials have an image. The cropped 224×224 image goes through a frozen ResNet-50 (ImageNet weights) for 2048 features, plus a 16-level grey-level co-occurrence matrix (distance 1, angle 0) for 256 features. ResNet dimensions that are constant over the training images of a fold are dropped, so the input width depends on the fold. A four-layer MLP (128, 256, 256, 128) maps them to 128 values.

The 288 branch outputs are concatenated. The image and audio outputs are set to zero when a trial has no image or no audio. An MLP head (128, 64) outputs 18 × 7 values, and a softmax over the 7 bins gives one distribution per adjective.

`pressure` and `ght` are recorded but deliberately not used.

## Training

- Object-level 5-fold cross-validation: each object is in exactly one test fold (17 objects per fold).
- Loss: EMD between predicted and human distributions (see below).
- Optimizer: Adam, learning rate 1e-4, weight decay 1e-4.
- Batch size 16, at most 100 epochs, gradient clipping at norm 1.0.
- Scheduler: ReduceLROnPlateau on the test-fold EMD, factor 0.5, patience 7.
- Early stopping: patience 15 on the test-fold EMD. The best epoch's checkpoint is used for the final scores.
- Default seed 42.

## Evaluation and metrics

All metrics are computed per object (trial predictions averaged), then averaged. EMD and MAE are averaged over the objects of a fold and then over the 5 folds.

- **EMD.** For each adjective: the mean, over the 7 bins, of the absolute difference between the predicted and the human cumulative distributions. Then averaged over the 18 adjectives. Implemented in `model/src/train.py`.
- **MAE.** The absolute difference between predicted and human expected ratings. The expected rating is the sum of k·p_k over the bins k = 1…7, so it lies on the 1–7 scale.
- **Pearson r.** For each adjective, the Pearson correlation between the true and predicted expected ratings over the 85 objects' out-of-fold predictions. The reported r is the mean of these 18 values (`analysis/results/mean_per_adjective_r.py`). A pooled r over all 1,530 object–adjective pairs is not used.

In the README results table, SD is the sample standard deviation over the 5 seeds. EMD and MAE come from `results/*/final_results.json`. The r values come from the per-object predictions in `results/*/predictions_vs_truth.csv`; [reproducibility.md](reproducibility.md#evaluation-and-analysis-scripts) shows how to recompute them.

## Results per adjective

**Main run.** r is between about 0.57 and 0.78 for 13 adjectives. It is low for sticky, cold, warm, wet and dry (between −0.15 and 0.26). These five adjectives also have the lowest MAE. Their ratings cluster at one end of the scale, so a nearly constant prediction gets a low error without ranking the objects correctly.

## All result figures

All figures show the main run (seed 42, out-of-fold predictions). The heatmaps show one value per object and adjective.

**Session 1 (obj_01–obj_45).** Human mean ratings, model predictions, and prediction minus human mean:

![Human mean ratings, session 1](figures/heatmap_human_session1.png)
![Model predictions, session 1](figures/heatmap_predicted_session1.png)
![Prediction minus human mean, session 1](figures/heatmap_diff_session1.png)

**Session 2 (obj_46–obj_85).** Human mean ratings, model predictions, and prediction minus human mean:

![Human mean ratings, session 2](figures/heatmap_human_session2.png)
![Model predictions, session 2](figures/heatmap_predicted_session2.png)
![Prediction minus human mean, session 2](figures/heatmap_diff_session2.png)

**Predicted vs. true expected rating**, one panel per adjective, one dot per object:

![Predicted vs. true expected rating per adjective](figures/predicted_vs_true_grid.png)

## Third-party code and credits

- **DFT321** (combining three accelerometer axes into one signal): Landin, Romano, McMahan, Kuchenbecker. [complete reference]
- **Image branch** (frozen ResNet + GLCM features, and the 128-256-256-128 MLP): following Awan & Jeon 2025. [complete reference]
- **ResNet-50 ImageNet weights** (`IMAGENET1K_V2`) via torchvision.
- **Sparsh-X:** `facebookresearch/sparsh-multisensory-touch`, commit `219ee2f30c52da6eba1071c0a8e71b24ec33a697`, CC BY-NC 4.0. The two local patches are in `experiments/sparsh_x/patches/sparsh_d360_compat.patch`. The checkpoints come from upstream and are not included.
- **DIGIT 360 driver:** adapted from `facebookresearch/digit360` (CC BY-NC 4.0). Lab adaptation `digit360-ros2` maintained by Niklas Leukroth (LASR); URL to be added.
