"""
Shared pressure_ap contact-event feature extraction for pipeline_v3.

Diagnostic/feature-engineering only: no changes to dataset.py / model.py /
train.py / any existing pipeline.

Methodology (approved by user 2026-07-17, see preprocessing_investigation
analyses): contact_intensity(t) = mean absolute deviation from an
early-trial baseline across the 50 pressure_ap channels. Discrete contact
"events" are peaks in this signal detected via scipy.signal.find_peaks
with a robust (median + k*MAD) height/prominence threshold and a minimum
inter-peak distance, replacing the naive single-global-peak approach from
the earlier contact-timing analysis (which mostly found single-sample
noise spikes).
"""
import numpy as np
from scipy.signal import find_peaks

PRESSURE_AP_RATE_HZ = 248.0
MIN_PEAK_DISTANCE_SAMPLES = 4  # ~16ms at 248Hz
MAD_SCALE = 1.4826  # scales MAD to be a consistent estimator of std under normality

FEATURE_NAMES = [
    'n_events', 'peak_amplitude', 'mean_event_amplitude', 'total_event_energy',
    'event_timing_std', 'dominant_event_location', 'spatial_spread',
]


def contact_intensity(pap):
    """pap: (N, 50) -> (intensity (N,), baseline (50,))"""
    n_base = max(3, min(10, pap.shape[0] // 10))
    baseline = pap[:n_base, :].mean(axis=0)
    intensity = np.abs(pap - baseline[None, :]).mean(axis=1)
    return intensity, baseline


def _half_max_window(intensity, peak_idx, peak_val):
    thresh = 0.5 * peak_val
    start = peak_idx
    while start > 0 and intensity[start - 1] >= thresh:
        start -= 1
    end = peak_idx
    while end < len(intensity) - 1 and intensity[end + 1] >= thresh:
        end += 1
    return start, end


def extract_features(pap, k):
    """pap: (N, 50) raw pressure_ap array. k: robust-threshold multiplier.

    Returns a dict with FEATURE_NAMES keys, or None if pap is too short to
    compute a baseline/intensity signal at all (< 10 timesteps).
    """
    if pap.shape[0] < 10:
        return None

    intensity, baseline = contact_intensity(pap)
    median = np.median(intensity)
    mad = np.median(np.abs(intensity - median)) * MAD_SCALE
    thresh = median + k * max(mad, 1e-9)

    peaks, _ = find_peaks(intensity, height=thresh, prominence=thresh,
                           distance=MIN_PEAK_DISTANCE_SAMPLES)

    n_events = len(peaks)
    if n_events == 0:
        return {
            'n_events': 0, 'peak_amplitude': np.nan, 'mean_event_amplitude': np.nan,
            'total_event_energy': 0.0, 'event_timing_std': np.nan,
            'dominant_event_location': np.nan, 'spatial_spread': np.nan,
        }

    peak_vals = intensity[peaks]
    dominant_idx = peaks[np.argmax(peak_vals)]
    dominant_val = intensity[dominant_idx]

    total_energy = 0.0
    for p in peaks:
        s, e = _half_max_window(intensity, p, intensity[p])
        total_energy += intensity[s:e + 1].sum()

    event_timing_std = float(np.std(peaks / max(1, len(intensity) - 1))) if n_events >= 2 else np.nan

    channel_dev = np.abs(pap[dominant_idx, :] - baseline)
    w_sum = channel_dev.sum()
    if w_sum > 0:
        idx = np.arange(pap.shape[1])
        mean_ch = (idx * channel_dev).sum() / w_sum
        spatial_spread = float(np.sqrt((channel_dev * (idx - mean_ch) ** 2).sum() / w_sum))
    else:
        spatial_spread = np.nan

    return {
        'n_events': n_events,
        'peak_amplitude': float(dominant_val),
        'mean_event_amplitude': float(peak_vals.mean()),
        'total_event_energy': float(total_energy),
        'event_timing_std': event_timing_std,
        'dominant_event_location': float(dominant_idx / max(1, len(intensity) - 1)),
        'spatial_spread': spatial_spread,
    }
