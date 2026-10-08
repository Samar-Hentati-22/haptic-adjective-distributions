# Reproducibility

Limitations, open items, configuration and every script you can run. The method itself is described in [methodology.md](methodology.md), the data format in [data/README.md](../data/README.md) and the Sparsh-X experiment in [sparsh_x.md](sparsh_x.md).

## Limitations

1. **No validation split.** The held-out test fold selects the best epoch, triggers early stopping (patience 15), drives the learning-rate scheduler and produces the reported metrics. The reported scores are therefore optimistically biased. The size of this bias was not measured.
2. **Reproduction of the main run.** The main run in `results/` came from an earlier revision of `train.py`, before the `--adjective-weighting` option (off by default) was added. A re-run matched to 4 decimals at fold 1, epoch 2. Full bit-exact reproduction was not verified. No environment versions or hashes were recorded.
3. **Determinism is not guaranteed.** The code sets seeds and `cudnn.deterministic`, but does not call `torch.use_deterministic_algorithms`, and it uses 8 DataLoader workers.
4. **The feature cache does not invalidate itself.** Delete it after changing a feature extractor.
5. **The early CNN+LSTM baseline is not in this repository.** It is described in the thesis only.
6. **Five adjectives have low r:** sticky, cold, warm, wet and dry (see [methodology.md](methodology.md#results-per-adjective)).

## Open items (unverified)

- Package versions are not pinned. Cluster runs used Python 3.9.
- Sampling rates are assumed in the code, not read from the data: IMU about 438 Hz raw (109.5 Hz per stream after separation), pressure_ap 248 Hz, audio 48,000 Hz. These are unverified.

## Verification status

- The EMD and MAE values in the README results table match `results/*/final_results.json`.
- The r values in the table were recomputed with `analysis/results/mean_per_adjective_r.py` from `results/*/predictions_vs_truth.csv` and match.
- `analysis/results/plot_heatmap_predicted.py` checks that the true ratings in `predictions_vs_truth.csv` equal `data/labels/labels_mean_phase2.npy`. For the main run the largest difference is 9.5e-07.
- The full pipeline (recording to training) has not been re-run end to end in this repository. Most scripts need the raw data, which is not included.

## Known issues

- `data_collection/read_data.py`: the IMU section uses the names `accels` and `gyros` without defining them. It catches the error and prints an error message for that section instead of the IMU summary. The other sections are not affected.

## Environment variables

All paths come from environment variables, read in [`model/src/paths.py`](../model/src/paths.py). A missing required variable stops the script with an error that names it.

| Variable | Needed by | Meaning |
|---|---|---|
| `TACTILE_DATA_ROOT` | most scripts | folder with `hdf5/` and `hdf5_cropped_images/` |
| `TACTILE_LABELS_DIR` | training, analysis | folder with `labels_phase2.npy`, `labels_mean_phase2.npy`, `object_ids_phase2.npy` |
| `TACTILE_CACHE_DIR` | optional | feature cache, default `<repo>/cache` |
| `HUMAN_DATA_DIR` | `human_study/` | folder with the participant-level CSV files |
| `SPARSH_REPO` | Sparsh-X | patched clone of sparsh-multisensory-touch |
| `SPARSHX_FEATURES_DIR` | Sparsh-X, optional | default `<cache>/sparshx_features` |
| `SPARSHX_PICKLE_DIR` | Sparsh-X, optional | default `<cache>/sparshx_pickle` |
| `RAW_MCAP_FULL_DIR` | `precompute_sparshx_features.py` | raw .mcap recordings |
| `RAW_MCAP_TEST_DIR` | Sparsh-X validation scripts | raw .mcap recordings used for validation |
| `DATA_DIR`, `ROS_WS` | recording scripts | output folder (default `~/tactile_recordings`) and ROS 2 workspace (default `~/ros2_ws`) |

## Running the pipeline in full

Run the steps in this order. Paths inside the scripts are computed from the script location, so you can run them from any folder.

1. **Record** (on the sensor machine, once per sensor session, then once per object):
   ```bash
   DATA_DIR=~/tactile_recordings bash data_collection/record_idle.sh A
   DATA_DIR=~/tactile_recordings bash data_collection/record2.sh obj_01 A
   ```
2. **Convert to HDF5** (in a ROS 2 Jazzy environment):
   ```bash
   python3 data_collection/mcap_to_hdf5.py --data-root ~/tactile_recordings/A --output-root $TACTILE_DATA_ROOT [--obj obj_01] [--action tap] [--verify]
   ```
   `python3 data_collection/read_data.py <bag_folder>` prints a summary of one recording (see Known issues).
3. **Crop the tactile images:**
   ```bash
   python data_prep/crop_images.py
   ```
4. **Build the labels** (needs `HUMAN_DATA_DIR`):
   ```bash
   python human_study/reorganize_tactile_data.py
   python human_study/reorganize_session2_data.py
   python human_study/build_label_distributions.py
   python human_study/build_labels.py --out-dir $TACTILE_LABELS_DIR [--csv path/to/labels_distribution.csv]
   ```
5. **Precompute features** (optional, makes training much faster):
   ```bash
   python model/src/precompute_features.py
   ```
6. **Train:**
   ```bash
   python model/src/train.py [--seed 42] [--results-dir DIR] [--smoke-test] [--adjective-weighting]
   python model/src/train_regularized.py [--dropout 0.3] [--weight-decay 5e-4] [same flags as train.py]
   ```
7. **Evaluate:** see [Evaluation and analysis scripts](#evaluation-and-analysis-scripts).

## SLURM template

On a SLURM cluster, use [`model/slurm/train.sbatch`](../model/slurm/train.sbatch). The comments in that file list the command for each reported run.

- Variables, passed with `sbatch --export=ALL,...`: `SCRIPT` (`train.py` or `train_regularized.py`, default `train.py`), `SEED` (default 42), `RESULTS_DIR` (required) and `EXTRA_ARGS` (for example `--adjective-weighting`).
- Replace the placeholders `<PARTITION>`, `<ACCOUNT>`, `<PROJECT_DIR>` and `<VENV_DIR>` in the file before use.
- Set `TACTILE_DATA_ROOT` and `TACTILE_LABELS_DIR` in the submitting shell; `--export=ALL` forwards them.
- Example: `sbatch --job-name=tactile_v3_seed7 --export=ALL,SEED=7,RESULTS_DIR=outputs/seed7 model/slurm/train.sbatch`
- Job logs are written to the submission directory as `slurm_<jobname>_<jobid>.out` and `.err`.

The Sparsh-X jobs use [`experiments/sparsh_x/precompute.sbatch`](../experiments/sparsh_x/precompute.sbatch) and [`experiments/sparsh_x/train_sparshimg.sbatch`](../experiments/sparsh_x/train_sparshimg.sbatch), see [sparsh_x.md](sparsh_x.md).

## Evaluation and analysis scripts

Scripts write their outputs to `outputs/<script_name>/`.

### Results (`analysis/results/`)

Recompute the mean per-adjective r of a reported run:

```bash
python analysis/results/mean_per_adjective_r.py --csv results/main_seed42/predictions_vs_truth.csv
```

For a new training run (default output folder):

```bash
python analysis/results/mean_per_adjective_r.py --csv outputs/train/results/predictions_vs_truth.csv
python analysis/results/plot_heatmap_predicted.py [--pred-csv FILE] [--labels-dir DIR] [--out-dir DIR] [--run-label TEXT] [--max-true-diff 0.01]
python analysis/results/plot_predicted_vs_true_grid.py
python analysis/results/plot_actual_vs_predicted_bars.py
python analysis/results/diagnose_train_test_gap.py
python analysis/results/single_object_distributions.py
```

`plot_predicted_vs_true_grid.py` and `plot_actual_vs_predicted_bars.py` read `outputs/train/results/predictions_vs_truth.csv`. `diagnose_train_test_gap.py` and `single_object_distributions.py` read the checkpoints in `outputs/train/results/`.

### Human-study figures (`human_study/plots/`, need `HUMAN_DATA_DIR`)

```bash
python human_study/plots/plot_heatmap_named.py
python human_study/plots/plot_session2_named.py
python human_study/plots/plot_strip_per_object_named.py
python human_study/plots/plot_strip_per_adjective_named.py
python human_study/plots/plot_strip_sorted_named.py
python human_study/plots/distribution_plots_named.py
python human_study/plots/radar_plots_named.py
python human_study/plots/per_property_comparison_named.py
```

### Example figures (`analysis/figures/`, need `TACTILE_DATA_ROOT`)

```bash
python analysis/figures/audio_mel_spectrogram_example_obj35.py
python analysis/figures/mel_spectrogram_example.py
python analysis/figures/dft321_real_example.py
python analysis/figures/dft321_real_example_obj82.py
python analysis/figures/glcm_example.py
python analysis/figures/glitch_before_after.py
```

### Preprocessing checks (`analysis/preprocessing/`, need the data)

```bash
python analysis/preprocessing/01_visualize_raw_signals.py
python analysis/preprocessing/02_imu_dft321_psd.py
python analysis/preprocessing/03_trial_duration_by_procedure.py
python analysis/preprocessing/04_pressure_ap_contact_timing.py
python analysis/preprocessing/05_tap_structure_check.py
python analysis/preprocessing/06_extract_pressure_ap_features.py
python analysis/preprocessing/07_pressure_ap_correlation_by_procedure.py [--k K | --compare-k K1 K2 ... | --csv FILE]
python analysis/preprocessing/12_imu_feature_stability_check.py
python analysis/preprocessing/13_imu_mfcc_stability_ratio_plot.py
python analysis/preprocessing/14_pressure_ap_glitch_prevalence_check.py
```

Run `06_*` before `07_*`; 07 reads the CSV files that 06 writes. `pressure_ap_event_features.py` in the same folder is a helper module used by 06 and 07, not a script.

### Diagnostic tests (`tests/`, need the data)

```bash
python tests/08_image_crop_test.py
python tests/10_image_crop_merged_spotcheck.py
python tests/11_imu_features_test.py
python tests/13b_pressure_ap_resample_test.py
python tests/15_image_features_test.py
python tests/16_audio_ght_features_test.py
python tests/17_dataset_v3_test.py
python tests/18_dataset_v3_resnet_fix_verify.py
python tests/19_dataset_v3_ght_removal_verify.py
python tests/20_model_v3_test.py
```

Run `tests/17_dataset_v3_test.py` before 18–20, because they reuse its cached statistics. `tests/10_*` reads the log written by `data_prep/crop_images.py`.
