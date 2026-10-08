#!/bin/bash
# One template for all pipeline_v3 training runs (replaces the 7 original
# slurm_job*.sh files, which differed only in job name, script, args and
# output dir). Variables are passed with sbatch --export:
#
#   SCRIPT       train.py or train_regularized.py (in model/src/)   default: train.py
#   SEED         --seed value                                       default: 42
#   RESULTS_DIR  --results-dir value                                required
#   EXTRA_ARGS   extra CLI args, e.g. "--adjective-weighting"       default: empty
#
# TACTILE_DATA_ROOT, TACTILE_LABELS_DIR (and optionally TACTILE_CACHE_DIR)
# must be set in the submitting shell; --export=ALL forwards them.
#
# Original runs (RESULTS_DIR names match results/ in this repo):
#   sbatch --job-name=tactile_v3          --export=ALL,SEED=42,RESULTS_DIR=outputs/main_seed42             model/slurm/train.sbatch
#   sbatch --job-name=tactile_v3_seed7    --export=ALL,SEED=7,RESULTS_DIR=outputs/seed7                    model/slurm/train.sbatch
#   sbatch --job-name=tactile_v3_seed99   --export=ALL,SEED=99,RESULTS_DIR=outputs/seed99                  model/slurm/train.sbatch
#   sbatch --job-name=tactile_v3_seed123  --export=ALL,SEED=123,RESULTS_DIR=outputs/seed123                model/slurm/train.sbatch
#   sbatch --job-name=tactile_v3_seed2024 --export=ALL,SEED=2024,RESULTS_DIR=outputs/seed2024              model/slurm/train.sbatch
#   sbatch --job-name=tactile_v3_weighted --export=ALL,SEED=42,RESULTS_DIR=outputs/adj_weighted,EXTRA_ARGS=--adjective-weighting model/slurm/train.sbatch
#   sbatch --job-name=tactile_v3_regularized --export=ALL,SCRIPT=train_regularized.py,SEED=42,RESULTS_DIR=outputs/regularized model/slurm/train.sbatch
# (SEED=42 is the scripts' default, which the original main/weighted/
# regularized jobs used implicitly.)
#
# SLURM cannot expand variables in #SBATCH lines, so the job's .out/.err go
# to the submission directory as slurm_<jobname>_<jobid>.* instead of into
# RESULTS_DIR as in the originals.
#SBATCH --job-name=tactile_v3
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

SCRIPT="${SCRIPT:-train.py}"
SEED="${SEED:-42}"
RESULTS_DIR="${RESULTS_DIR:?set RESULTS_DIR via sbatch --export}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$BASE_DIR"
mkdir -p "$RESULTS_DIR"

source "$VENV_DIR/bin/activate"

# EXTRA_ARGS is intentionally unquoted so it can hold several arguments.
python "model/src/$SCRIPT" --seed "$SEED" --results-dir "$RESULTS_DIR" $EXTRA_ARGS
