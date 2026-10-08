"""
Pipeline v3 — standalone audio and GHT (environmental sensor) passthrough
functions.

Self-contained: no imports from outside pipeline_v3/.

Both functions here are deliberately thin: they validate/pass through
per-trial data and document the intended normalization, but do NOT apply
normalization themselves. Per-channel z-score using train-only, per-fold
statistics happens later in dataset.py (see compute_norm_stats in
Training_Phase2/dataset.py), the same convention used for imu_raw/
pressure/pressure_ap elsewhere in this pipeline -- that step needs
full-dataset training-fold statistics a single-trial function doesn't
have access to.

NOTE on mcap_to_hdf5.py: this module's docstrings were supposed to cite
mcap_to_hdf5.py directly (for the audio zero-pad/sparse-floor convention
and the GHT field order), but that file does not exist anywhere in this
codebase -- searched the full project tree, not found. Where this
matters (the GHT field order below), the claim is instead verified
empirically against real DATA_Final/hdf5/ data; flagged inline.
"""
import numpy as np

# ── Audio ────────────────────────────────────────────────────────────────
# mel_spectrogram: (2, 64, T) -- 2 mics, 64 mel bins, T time frames. T
# varies per trial; per the existing audio CNN encoder (model.py) it is
# consumed at variable length, not resampled here. Per user-provided
# context, T is already zero-padded/floored upstream (sparse-audio floor
# at 100 frames) by whatever produced DATA_Final/hdf5/ -- not independently
# verified against mcap_to_hdf5.py since that file could not be located.
#
# Normalization (not applied here): per mel-bin channel z-score (mean/std
# over the mel-value distribution, computed per (mic, mel-bin) channel,
# from training-fold trials only) -- to be applied in dataset.py, matching
# compute_norm_stats's existing per-channel treatment of imu_raw/pressure.


def extract_audio_features(mel_spectrogram: np.ndarray):
    """
    mel_spectrogram: (2, 64, T) raw mel spectrogram for one trial.

    Passthrough only -- does not resample or normalize (see module
    docstring). Returns (mel_spectrogram as float32, has_audio), where
    has_audio is False iff the entire array is exactly zero (full sensor
    dropout for this trial), reusing the same all-zero convention already
    used in Training_Phase2/dataset.py (`has_audio = 0.0 if
    np.all(audio_arr == 0) else 1.0`).
    """
    mel_spectrogram = np.asarray(mel_spectrogram, dtype=np.float32)
    has_audio = not bool(np.all(mel_spectrogram == 0))
    return mel_spectrogram, has_audio


# ── GHT (environmental: gas/humidity/temperature/pressure) ─────────────────
# ght: (4,) static per-trial vector. Field order could not be confirmed
# from mcap_to_hdf5.py (file not found in this codebase). Verified
# instead by inspecting real non-zero GHT values across 15 objects
# (300 trials, 174 non-zero) in DATA_Final/hdf5/ and matching each
# column's range against known BME680-style environmental sensor
# behavior:
#   col 0: ~37.7-71.3      -> temperature, deg C (matches the same
#                             order-of-magnitude self-heating drift
#                             already documented for the `pressure`
#                             modality elsewhere in this pipeline, see
#                             tactile-pipeline-bugs-v2fix memory: 40-58C)
#   col 1: ~3.6-33.9        -> relative humidity, % (0-100 range)
#   col 2: ~702,814-938,158 -> gas sensor resistance, Ohm (BME680 gas
#                             resistance is commonly hundreds of
#                             kOhm to ~1 MOhm)
#   col 3: ~99,545-99,976   -> barometric pressure, Pa (i.e. ~995-1000
#                             hPa, correct ambient atmospheric range)
# This matches the user-specified order (temperature, humidity, gas,
# pressure) -- confirmed empirically, not from source documentation.
GHT_FIELD_NAMES = ('temperature_c', 'humidity_pct', 'gas_resistance_ohm', 'pressure_pa')

# Normalization (not applied here): z-score per field, train-only stats
# per fold, same convention as audio/imu_raw/pressure -- to be applied in
# dataset.py.


def extract_ght_features(ght: np.ndarray) -> np.ndarray:
    """
    ght: (4,) raw static GHT vector, field order
    [temperature_c, humidity_pct, gas_resistance_ohm, pressure_pa]
    (see module docstring for how this order was verified).

    Passthrough + shape validation only -- does not normalize (see module
    docstring).
    """
    ght = np.asarray(ght, dtype=np.float32)
    if ght.shape != (4,):
        raise ValueError(f"expected ght shape (4,), got {ght.shape}")
    return ght
