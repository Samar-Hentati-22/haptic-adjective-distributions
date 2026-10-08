"""
Central path configuration for the haptic-adjective-distributions repository.

Every data location is read from an environment variable, so no script
contains a machine-specific path:

  TACTILE_DATA_ROOT     folder containing hdf5/ and hdf5_cropped_images/  (required)
  TACTILE_LABELS_DIR    folder containing labels_phase2.npy,
                        labels_mean_phase2.npy, object_ids_phase2.npy      (required)
  TACTILE_CACHE_DIR     precomputed feature cache (optional, default <repo>/cache)
  HUMAN_DATA_DIR        folder with the participant-level CSVs
                        (required by human_study/ scripts only)

Sparsh-X experiments (experiments/sparsh_x/) additionally use:

  SPARSH_REPO           local clone of sparsh-multisensory-touch           (required there)
  SPARSHX_FEATURES_DIR  Sparsh-X features_cache (optional, default <repo>/cache/sparshx_features)
  SPARSHX_PICKLE_DIR    processed_pickle folder (optional, default <repo>/cache/sparshx_pickle)
  RAW_MCAP_FULL_DIR     raw_mcap_full folder (required by precompute_sparshx_features.py)
  RAW_MCAP_TEST_DIR     raw_mcap_test folder (required by the validation scripts that use it)

Accessors for required variables raise a RuntimeError naming the missing
variable. Scripts write their outputs under <repo>/outputs/<script_name>/.
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUTS_ROOT = os.path.join(REPO_ROOT, 'outputs')


def _require_env(name, description):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Environment variable {name} is not set. It must point to {description}. "
            f"Example: export {name}=/path/to/folder")
    return value


def data_root():
    return _require_env('TACTILE_DATA_ROOT',
                        'the folder containing hdf5/ and hdf5_cropped_images/')


def hdf5_dir():
    return os.path.join(data_root(), 'hdf5')


def cropped_dir():
    return os.path.join(data_root(), 'hdf5_cropped_images')


def labels_dir():
    return _require_env('TACTILE_LABELS_DIR',
                        'the folder containing labels_phase2.npy, labels_mean_phase2.npy '
                        'and object_ids_phase2.npy')


def cache_dir():
    return os.environ.get('TACTILE_CACHE_DIR') or os.path.join(REPO_ROOT, 'cache')


def human_data_dir():
    return _require_env('HUMAN_DATA_DIR',
                        'the folder containing the participant-level study CSVs')


def sparsh_repo():
    return _require_env('SPARSH_REPO',
                        'a local clone of sparsh-multisensory-touch '
                        '(with experiments/sparsh_x/patches/sparsh_d360_compat.patch applied)')


def sparshx_features_dir():
    return os.environ.get('SPARSHX_FEATURES_DIR') or os.path.join(cache_dir(), 'sparshx_features')


def sparshx_pickle_dir():
    return os.environ.get('SPARSHX_PICKLE_DIR') or os.path.join(cache_dir(), 'sparshx_pickle')


def raw_mcap_full_dir():
    return _require_env('RAW_MCAP_FULL_DIR', 'the raw_mcap_full folder of recorded .mcap trials')


def raw_mcap_test_dir():
    return _require_env('RAW_MCAP_TEST_DIR', 'the raw_mcap_test folder of recorded .mcap trials')


def output_dir(script_file, create=True):
    """<repo>/outputs/<script_name>/ for the script at `script_file`."""
    out = os.path.join(OUTPUTS_ROOT, os.path.splitext(os.path.basename(script_file))[0])
    if create:
        os.makedirs(out, exist_ok=True)
    return out
