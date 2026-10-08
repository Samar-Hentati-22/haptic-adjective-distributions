"""
Pipeline v3 — standalone IMU feature extraction.

Self-contained: no imports from outside pipeline_v3/. The de-interleaving
logic is ported (not imported) from the old Training_Phase2/dataset.py,
where it was validated against the d360_msgs/ImuRawD360 spec (see
tactile-pipeline-bugs-v2fix memory / prior session notes) — kept identical
here, just copied so this module has no runtime dependency on the old file.

Pipeline: raw imu_raw (T, 3) -> deinterleave_imu -> (N, 12), take the ACC
sub-stream (columns 0:3) -> extract_imu_features -> 10-dim vector
(5 MFCC + 5 time-domain stats).

fs=109.5 Hz is the per-stream rate after de-interleaving 4 streams out of
the raw ~438 Hz imu_raw log.
"""
import numpy as np
from scipy.signal import butter, filtfilt
import librosa

# ── De-interleaving (ported from Training_Phase2/dataset.py) ───────────────
IMU_ACC_NORM_THRESH = 2500.0   # counts; 1 g ~ 4300, other streams << 2500
IMU_DEINT_CHANNELS = 12
IMU_MIN_YIELD = 0.5


def deinterleave_imu(imu: np.ndarray) -> np.ndarray:
    """(T, 3) interleaved imu_raw -> (N, 12) de-interleaved clean cycles.

    Columns 0:3 are the ACC stream (the anchor); 3:6, 6:9, 9:12 are the
    other three interleaved streams (GYRO/MAG/LINACC per the d360 spec,
    exact non-ACC identity not required for this feature extraction).
    Returns (0, 12) if fewer than 50% of samples form clean period-4
    cycles (sensor dropout / period-2 recording mode).
    """
    if imu.shape[0] < 8:
        return np.zeros((0, IMU_DEINT_CHANNELS), dtype=np.float32)
    norms = np.linalg.norm(imu.astype(np.float64), axis=1)
    anchors = np.where(norms > IMU_ACC_NORM_THRESH)[0]
    if len(anchors) < 2:
        return np.zeros((0, IMU_DEINT_CHANNELS), dtype=np.float32)
    good = anchors[:-1][np.diff(anchors) == 4]
    if len(good) == 0 or 4 * len(good) < IMU_MIN_YIELD * imu.shape[0]:
        return np.zeros((0, IMU_DEINT_CHANNELS), dtype=np.float32)
    out = np.concatenate(
        [imu[good], imu[good + 1], imu[good + 2], imu[good + 3]], axis=1
    )
    return np.ascontiguousarray(out, dtype=np.float32)


# ── DFT321 (Landin/Romano/McMahan/Kuchenbecker) — verbatim, do not modify ──
def dft321(u):
    """
    Combine 3-axis vibration into 1-axis. Landin/Romano/McMahan/Kuchenbecker.
    u: (N, 3) -> returns (N,)
    """
    N_sampl = u.shape[0]
    is_odd = N_sampl % 2 == 1
    U = np.fft.fft(u, axis=0)
    N_half = (N_sampl + 1) // 2 if is_odd else N_sampl // 2 + 1
    absY = np.sqrt(np.sum((U[:N_half] * np.conj(U[:N_half])).real, axis=1))
    phase = np.angle(np.sum(U[:N_half], axis=1))
    Y_half = absY * np.exp(1j * phase)
    if is_odd:
        Y = np.concatenate([Y_half, np.conj(Y_half[-1:0:-1])])
    else:
        Y = np.concatenate([Y_half, np.conj(Y_half[-2:0:-1])])
    return np.real(np.fft.ifft(Y))


# ── Highpass filter ─────────────────────────────────────────────────────────
HIGHPASS_CUTOFF_HZ = 20.0
HIGHPASS_ORDER = 4


def _highpass(x: np.ndarray, fs: float) -> np.ndarray:
    """4th-order Butterworth highpass @ 20Hz, zero-phase (filtfilt).

    filtfilt needs more samples than its default pad length (3x filter
    order); real IMU trials are long enough (median ~400 samples at
    fs~109.5Hz) but a few dropout-affected trials are very short. For
    those, fall back to a padlen capped at len(x)-1, and skip filtering
    entirely (return x unfiltered) if there aren't even 2 samples.
    """
    if x.shape[0] < 2:
        return x
    b, a = butter(HIGHPASS_ORDER, HIGHPASS_CUTOFF_HZ, btype='highpass', fs=fs)
    default_padlen = 3 * (max(len(a), len(b)) - 1)
    padlen = min(default_padlen, x.shape[0] - 1)
    return filtfilt(b, a, x, padlen=padlen)


# ── MFCC (Awan & Jeon recipe, adapted for IMU sample rate) ─────────────────
#
# The Awan & Jeon spec calls for 25ms frames at 50% overlap within each
# 0.5s segment. That is an audio-rate recipe: at fs~109.5Hz, 25ms is
# ~2.7 samples — far too short to run an FFT / build a mel filterbank
# with 13 usable coefficients (you'd need n_fft on the order of a few
# samples, which can't support 13 mel bands with any real Nyquist
# resolution). The frame/overlap numbers do NOT translate cleanly at
# this rate, so we don't apply them literally.
#
# First adaptation tried (0.5s segments, single Hann frame per segment,
# n_fft=55 at fs=109.5Hz) was checked against real data and rejected: it
# gives only 28 rfft bins mapped into 26 mel bands (~1:1, no real
# smoothing), and empirically the resulting MFCC coefficients c5-c12 had
# segment-to-segment std up to 13x their own mean within a single trial
# -- i.e. those coefficients were noise, not reproducible spectral shape.
# Only c0-c4 were stable.
#
# Current adaptation: a single Hann-windowed FFT/MFCC pass over a fixed
# ~4s window per trial (pad short trials with zeros, truncate long ones),
# instead of repeated short 0.5s segments. This gives every trial the same
# frequency resolution (~0.25 Hz/bin at fs=109.5Hz vs ~2 Hz/bin before)
# and the same physical meaning per MFCC coefficient regardless of trial
# duration -- variable per-trial n_fft (e.g. from using the raw trial
# length) was considered and rejected because it would make longer trials
# systematically get finer/smoother features than shorter ones, confounding
# any downstream model. 4s covers the median ACC-substream duration (full
# dataset: median ~409 samples / ~3.7s, p75 ~555 / ~5.1s) without excessive
# padding; longer trials lose their tail, shorter ones get zero-padded.
#
# Even with this finer resolution, a follow-up stability check (repeat
# trials of the same object/procedure, 4 objects x 5 squeeze trials each)
# showed c0-c4 are tight and reproducible within an object (std/mean
# ratio ~0.03-1.0, clearly object-separated) while c5-c12 are not
# (std/mean ratio 1.6-5.9x -- the across-trial spread exceeds the
# coefficient's own mean, and trials of the same object don't cluster any
# tighter than trials of different objects in the stability plot). The
# mel-band energy spectrum confirms this isn't an FFT-bin artifact: real,
# object-differentiated structure appears below ~mel band 10, but all
# objects' spectra converge into an overlapping, jagged plateau above
# that -- consistent with broadband accelerometer noise, not signal, at
# the higher frequencies c5-c12 summarize. So only c0-c4 are kept.
N_MFCC = 5
FIXED_ANALYSIS_SEC = 4.0


def _mfcc_features(x: np.ndarray, fs: float, n_mfcc: int = N_MFCC) -> np.ndarray:
    fixed_len = max(8, int(round(FIXED_ANALYSIS_SEC * fs)))
    n = x.shape[0]
    if n >= fixed_len:
        x = x[:fixed_len]
    else:
        x = np.pad(x, (0, fixed_len - n))

    n_freq_bins = fixed_len // 2 + 1
    n_mels = min(26, n_freq_bins - 1)
    n_mels = max(n_mels, n_mfcc)

    mfcc = librosa.feature.mfcc(
        y=x.astype(np.float64), sr=fs, n_mfcc=n_mfcc,
        n_fft=fixed_len, hop_length=fixed_len, win_length=fixed_len,
        window='hann', center=False, n_mels=n_mels,
    )  # (n_mfcc, 1) -- exactly one frame, covering the whole fixed window
    return mfcc[:, 0]


# ── Public API ───────────────────────────────────────────────────────────
DEFAULT_FS = 109.5


def extract_imu_features(acc_substream: np.ndarray, fs: float = DEFAULT_FS) -> np.ndarray:
    """
    acc_substream: (N, 3) de-interleaved ACC sub-stream, i.e.
    deinterleave_imu(imu_raw)[:, 0:3].

    Returns a 10-dim float32 vector: 5 MFCC coefficients (c0-c4; single
    FFT pass over a fixed-length window, see _mfcc_features -- c5-c12
    were dropped, see the note above _mfcc_features) followed by
    [mean, std, min, max, rms] of the highpass-filtered DFT321 signal.
    Returns all-zeros if acc_substream is empty (sensor dropout trial with
    no clean de-interleaved cycles — deinterleave_imu returns (0, 12) for
    those; ~14/1700 trials in the full dataset per the IMU yield audit).
    """
    acc_substream = np.asarray(acc_substream, dtype=np.float64)
    if acc_substream.shape[0] == 0:
        return np.zeros(N_MFCC + 5, dtype=np.float32)
    combined = dft321(acc_substream)
    filtered = _highpass(combined, fs=fs)

    mfcc_feats = _mfcc_features(filtered, fs=fs)
    stats = np.array([
        filtered.mean(), filtered.std(), filtered.min(), filtered.max(),
        np.sqrt(np.mean(filtered ** 2)),
    ])

    return np.concatenate([mfcc_feats, stats]).astype(np.float32)
