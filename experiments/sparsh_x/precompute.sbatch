#!/bin/bash
# Sparsh-X image-embedding precompute (precompute_sparshx_features.py).
# Runs in the separate Sparsh-X venv. Set before submitting (forwarded by
# --export=ALL): SPARSH_REPO, RAW_MCAP_FULL_DIR, and optionally
# SPARSHX_FEATURES_DIR / SPARSHX_PICKLE_DIR / TACTILE_CACHE_DIR (see model/src/paths.py).
#
#   sbatch --export=ALL experiments/sparsh_x/precompute.sbatch
#
# .out/.err go to the submission directory.
#SBATCH -N 1
#SBATCH --job-name=sparsh_precompute
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=precompute_%j.out
#SBATCH --error=precompute_%j.err

cd "<PROJECT_DIR>/experiments/sparsh_x"
source "<VENV_DIR>/bin/activate"   # the Sparsh-X venv
python3 precompute_sparshx_features.py
