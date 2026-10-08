#!/bin/bash
# Training run with Sparsh-X image embeddings (train_sparshimg.py).
# Set before submitting (forwarded by --export=ALL): TACTILE_DATA_ROOT,
# TACTILE_LABELS_DIR, SPARSHX_FEATURES_DIR (or its default), and optionally
# TACTILE_CACHE_DIR (see model/src/paths.py).
#
#   sbatch --export=ALL experiments/sparsh_x/train_sparshimg.sbatch
#
# Results go to train_sparshimg.py's default, <repo>/outputs/train_sparshimg/results/.
# SLURM cannot expand variables in #SBATCH lines, so .out/.err go to the
# submission directory instead of the results folder as in the original.
#SBATCH --job-name=tactile_phase2_v3_sparshimg
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

set -euo pipefail

BASE_DIR="<PROJECT_DIR>"   # root of this repository on the cluster
VENV_DIR="<VENV_DIR>"

cd "$BASE_DIR"
mkdir -p outputs/train_sparshimg/results

source "$VENV_DIR/bin/activate"

python experiments/sparsh_x/train_sparshimg.py
