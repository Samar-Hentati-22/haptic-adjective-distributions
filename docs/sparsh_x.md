# Sparsh-X experiment

This experiment replaces the image branch with Sparsh-X image embeddings (768 values per trial from the upstream image-only base checkpoint). The other branches and all training settings are unchanged (see [methodology.md](methodology.md)).

Result (negative): EMD 0.1189, MAE 0.7934, r 0.286, on 75 objects only. The full results table is in the [README](../README.md#results).

## Reproduction

1. Set up the upstream repository and environment as described in [`experiments/sparsh_x/requirements.txt`](../experiments/sparsh_x/requirements.txt). This includes applying [`patches/sparsh_d360_compat.patch`](../experiments/sparsh_x/patches/sparsh_d360_compat.patch), which holds two local compatibility patches.
2. Compute the embeddings: `python experiments/sparsh_x/precompute_sparshx_features.py` (or [`experiments/sparsh_x/precompute.sbatch`](../experiments/sparsh_x/precompute.sbatch)).
3. Train: `python experiments/sparsh_x/train_sparshimg.py` with the same flags as `train.py` (or [`experiments/sparsh_x/train_sparshimg.sbatch`](../experiments/sparsh_x/train_sparshimg.sbatch)).

The environment variables this experiment needs (`SPARSH_REPO`, `SPARSHX_FEATURES_DIR`, `SPARSHX_PICKLE_DIR`, `RAW_MCAP_FULL_DIR`, `RAW_MCAP_TEST_DIR`) are listed in [reproducibility.md](reproducibility.md#environment-variables).

## Caveats

Only 75 objects are used. Objects 09, 10, 11, 26 and 57–62 have no usable Sparsh-X image trials, because the microphone topics that the Sparsh-X loader needs are missing in those recordings. The cross-validation folds therefore differ from the main model's, so the comparison is not fold-matched.

The scripts in [`experiments/sparsh_x/validation/`](../experiments/sparsh_x/validation/) were used to check the Sparsh-X pipeline.
