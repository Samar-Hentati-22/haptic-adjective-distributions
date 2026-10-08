"""
Pipeline v3 — standalone pressure_ap preprocessing.

Self-contained: no imports from outside pipeline_v3/.

Approach (agreed 2026-07-19): pressure_ap is a 50-channel spatial array,
not a 1D vibration signal like IMU -- no DFT321/MFCC/dimensionality
reduction here. Each trial is resampled along time to a fixed length so a
downstream CNN encoder (model.py, not yet implemented) gets a fixed-shape
(target_len, 50) input per trial; all 50 channels stay intact.

This module only resamples. It does NOT normalize -- per-channel z-score
using train-only stats happens later in dataset.py, same convention as
imu_raw/pressure/audio elsewhere in this pipeline (see
tactile-pipeline-bugs-v2fix memory: compute_norm_stats operates on
already-preprocessed per-modality arrays at the full-dataset level).

Resampling method: linear interpolation (scipy.interpolate.interp1d),
not scipy.signal.resample (FFT-based). Reasoning: pressure_ap trials are
not periodic -- they start near a baseline and end mid-contact/release,
with different start/end values. FFT-based resample implicitly assumes
the signal is one period of a periodic waveform, which introduces
wrap-around (Gibbs) ringing at trial boundaries whenever start != end --
exactly the shape of a real contact trial. Linear interpolation makes no
such assumption, preserving trial-boundary shape at the cost of no
explicit anti-aliasing filter. Given the resampling ratios seen in this
dataset are modest (target_len=224 vs full-dataset trial lengths: median
~1078 samples/~4.3s, p95 ~1712/~6.9s, max 2049/~8.3s, min 2/~0.01s, at
fs=248Hz -- so mostly ~4-9x downsampling, rarely more), plain linear
interpolation is an acceptable trade for artifact-free trial edges.
"""
import numpy as np
from scipy.interpolate import interp1d

DEFAULT_FS = 248.0
DEFAULT_TARGET_LEN = 224   # matches Sparsh-X tokenization length
N_CHANNELS = 50

# ── Glitch cleaning ──────────────────────────────────────────────────────────
#
# Full-dataset prevalence audit (2026-07-19, see
# preprocessing_investigation/pressure_ap_glitch_events.csv) found a sensor
# readout glitch in 544/1700 trials (32%, all 85 objects affected, evenly
# spread across tap/squeeze/slide/hold): a single timestep where a
# contiguous block of channels reads exactly 0.0 -- never a legitimate
# baseline value, baseline sits around 2000-2200 raw counts -- while the
# immediately preceding/following timestep (and/or the other channels at
# that same timestep) show normal baseline. Run length clusters at fixed
# widths (25 channels in 56% of events, then 31 and 39), consistent with a
# dropped partial readout frame rather than random sensor noise. Left
# uncleaned, this both distorts the fixed-length resampling below (a single
# extreme glitch sample can dominate the interpolation grid) and likely
# contaminates the contact-event peak detection in pressure_ap_features.py's
# older event-based method.
GLITCH_MIN_RUN = 20            # contiguous zero channels to flag as a glitch
GLITCH_BASELINE_THRESH = 500.0  # real baseline is ~2000-2200; exact 0 is never legitimate


def _find_zero_runs(row: np.ndarray, min_run: int):
    """Return [(start, end_exclusive), ...] for contiguous exact-zero runs >= min_run."""
    runs = []
    n = len(row)
    i = 0
    while i < n:
        if row[i] == 0.0:
            j = i
            while j < n and row[j] == 0.0:
                j += 1
            if j - i >= min_run:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _is_glitch_run(pressure_ap: np.ndarray, t: int, a: int, b: int, n_t: int) -> bool:
    """A zero-run at timestep t, channels [a,b), counts as a glitch if the rest
    of that timestep looks like normal baseline, and/or the same channel
    range one timestep before/after does -- i.e. an isolated blip, not a
    genuinely quiet/blank trial."""
    row = pressure_ap[t]
    other = np.concatenate([row[:a], row[b:]])
    same_t_other_ok = other.size > 0 and other.mean() > GLITCH_BASELINE_THRESH

    neighbor_ok = False
    for t_nb in (t - 1, t + 1):
        if 0 <= t_nb < n_t and pressure_ap[t_nb, a:b].mean() > GLITCH_BASELINE_THRESH:
            neighbor_ok = True
            break

    return same_t_other_ok or neighbor_ok


def _nearest_clean_timestep(pressure_ap: np.ndarray, t: int, a: int, b: int,
                              n_t: int, direction: int):
    """Search outward from t in the given direction (+1/-1) for the nearest
    timestep whose [a,b) channel range is NOT itself near-zero (i.e. not
    also glitched), so we have real data to interpolate from."""
    t_search = t + direction
    while 0 <= t_search < n_t:
        if pressure_ap[t_search, a:b].mean() > GLITCH_BASELINE_THRESH:
            return t_search
        t_search += direction
    return None


def clean_pressure_ap_glitches(pressure_ap: np.ndarray) -> np.ndarray:
    """
    pressure_ap: (N, 50) raw trial.

    Detects glitch frames (see module note above) and repairs each one by
    linearly interpolating the affected channels between the nearest clean
    timesteps before and after. If there's no clean timestep on one side
    (glitch at the very start/end of a trial), holds the other side's value
    instead. Returns a same-shape float32 copy; does not modify the input
    in place.
    """
    pressure_ap = np.array(pressure_ap, dtype=np.float64, copy=True)
    n_t = pressure_ap.shape[0]
    if n_t < 2:
        return pressure_ap.astype(np.float32)

    for t in range(n_t):
        for (a, b) in _find_zero_runs(pressure_ap[t], GLITCH_MIN_RUN):
            if not _is_glitch_run(pressure_ap, t, a, b, n_t):
                continue
            t_prev = _nearest_clean_timestep(pressure_ap, t, a, b, n_t, direction=-1)
            t_next = _nearest_clean_timestep(pressure_ap, t, a, b, n_t, direction=+1)
            if t_prev is None and t_next is None:
                continue
            elif t_prev is None:
                pressure_ap[t, a:b] = pressure_ap[t_next, a:b]
            elif t_next is None:
                pressure_ap[t, a:b] = pressure_ap[t_prev, a:b]
            else:
                frac = (t - t_prev) / (t_next - t_prev)
                pressure_ap[t, a:b] = (
                    (1 - frac) * pressure_ap[t_prev, a:b] + frac * pressure_ap[t_next, a:b]
                )

    return pressure_ap.astype(np.float32)


def extract_pressure_ap_features(pressure_ap: np.ndarray, fs: float = DEFAULT_FS,
                                   target_len: int = DEFAULT_TARGET_LEN) -> np.ndarray:
    """
    pressure_ap: (N, 50) raw trial, N varies per trial.

    Cleans the sensor readout glitch (clean_pressure_ap_glitches, see module
    note above) then resamples along the time axis to (target_len, 50) via
    linear interpolation, applied identically to all 50 channels at once
    (interp1d's axis=0 vectorizes over channels, not a per-channel loop,
    but the result is the same as resampling each channel independently).

    Returns all-zeros (target_len, 50) if pressure_ap has fewer than 2
    timesteps -- interpolation needs at least 2 points. No trial in the
    full 1700-trial dataset actually hits this (min observed length is 2
    samples), but it's guarded defensively since fewer than 2 could occur
    on unseen data.
    """
    pressure_ap = np.asarray(pressure_ap, dtype=np.float64)
    n = pressure_ap.shape[0]
    if n < 2:
        return np.zeros((target_len, N_CHANNELS), dtype=np.float32)

    cleaned = clean_pressure_ap_glitches(pressure_ap)

    t_orig = np.linspace(0.0, 1.0, n)
    t_new = np.linspace(0.0, 1.0, target_len)
    interp = interp1d(t_orig, cleaned, kind='linear', axis=0)
    resampled = interp(t_new)
    return resampled.astype(np.float32)
