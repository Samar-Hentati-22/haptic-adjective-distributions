#!/usr/bin/env python3
"""
Convert Digit360 ROS2 .mcap bag files to a structured ML training dataset,
stored as one HDF5 file per object instead of PNG + npy files on disk.

Expected input layout:
    <data_root>/obj_XX/<action>/*.mcap   (all .mcap files in one folder = one recording)
    or nested: <data_root>/obj_XX/<action>/trial_NNN/*.mcap  (one recording per trial)

Output layout:
    <output_root>/hdf5/obj_XX.h5
        metadata/object_name          — string "obj_XX"
        <action>/trial_NNN/image       — float32 (224, 224, 3) normalized [0,1], zeros if no frames
        <action>/trial_NNN/imu_raw     — float32 (N, 3)
        <action>/trial_NNN/pressure_ap — float32 (N, 50)
        <action>/trial_NNN/pressure    — float32 (N, 2)
        <action>/trial_NNN/ght         — float32 (4,)
        <action>/trial_NNN/audio       — float32 (2, 64, T)
    each trial_NNN group also carries a boolean attribute "image_valid".

Each raw recording (one .mcap file set) becomes exactly one HDF5 trial,
covering the recording's full time span — no sub-windowing. Per-modality
extraction logic (image cropping, mel-spectrogram, GHT mean, …) is copied
verbatim from mcap_to_dataset.py — only the output sink changes from
files-on-disk to HDF5 datasets, and each modality is computed over the
whole recording instead of a 1.3 s slice.

Usage (inside ROS2 Jazzy Docker container):
    python3 mcap_to_hdf5.py --data-root PATH --output-root PATH

    --data-root   folder containing obj_XX/ recordings (e.g. $DATA_DIR/A from record2.sh)
    --output-root folder that will receive hdf5/ (use as TACTILE_DATA_ROOT)
"""

import argparse
import math
import os
import glob
import numpy as np
import cv2
import h5py
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

N_MELS     = 64
HOP_LENGTH = 512
SAMPLE_RATE = 48_000
MIN_AUDIO_FRAMES = 100   # pad audio with zeros if a trial produces fewer mel frames than this

ACTIONS = ["tap", "squeeze", "slide", "hold"]

# topic → (ros_topic_string, ros_type_string)
TOPICS = {
    "image":       ("/image_raw/index_0",             "sensor_msgs/msg/Image"),
    "imu":         ("/imu_raw_topic/index_0",         "d360_msgs/msg/ImuRawD360"),
    "pressure_ap": ("/pressure_ap_topic/index_0",     "d360_msgs/msg/PressureAPD360"),
    "pressure":    ("/pressure_topic/index_0",        "d360_msgs/msg/PressureD360"),
    "ght":         ("/ght_topic/index_0",             "d360_msgs/msg/GasHTData"),
    "audio_0":     ("/mic_0/index_0",                 "d360_msgs/msg/AudioDataD360"),
    "audio_1":     ("/mic_1/index_0",                 "d360_msgs/msg/AudioDataD360"),
}


# ── MCAP loading ──────────────────────────────────────────────────────────────

def find_mcap_files(action_path):
    """
    Return a list of (mcap_file_list, label) pairs for each recording found
    under action_path.  Supports:
      - flat layout:  action_path/*.mcap  → one recording
      - nested layout: action_path/subdir/*.mcap  → one recording per subdir
    Windows from all recordings are numbered consecutively in the output.
    """
    direct = sorted(glob.glob(os.path.join(action_path, "*.mcap")))
    if direct:
        return [(direct, os.path.basename(action_path))]

    recordings = []
    for entry in sorted(os.listdir(action_path)):
        sub = os.path.join(action_path, entry)
        if os.path.isdir(sub):
            files = sorted(glob.glob(os.path.join(sub, "*.mcap")))
            if files:
                recordings.append((files, entry))
    return recordings


def load_messages(mcap_files):
    """
    Read all relevant topics from a set of .mcap files.
    Returns dict key → sorted list of (log_time_ns, deserialized_msg).
    """
    msg_types   = {k: get_message(ts) for k, (_, ts) in TOPICS.items()}
    topic_to_key = {tp: k for k, (tp, _) in TOPICS.items()}
    data = {k: [] for k in TOPICS}

    for path in mcap_files:
        with open(path, "rb") as f:
            reader = make_reader(f)
            for _, channel, message in reader.iter_messages(topics=list(topic_to_key)):
                key = topic_to_key[channel.topic]
                msg = deserialize_message(message.data, msg_types[key])
                data[key].append((message.log_time, msg))

    for key in data:
        data[key].sort(key=lambda x: x[0])

    return data


# ── Window helpers ────────────────────────────────────────────────────────────

def in_window(msgs, t0, t1):
    """Binary-search slice: messages with log_time in [t0, t1)."""
    import bisect
    ts_only = [t for t, _ in msgs]
    lo = bisect.bisect_left(ts_only, t0)
    hi = bisect.bisect_left(ts_only, t1)
    return msgs[lo:hi]


# ── Tactile image preprocessing ───────────────────────────────────────────────

def _detect_circle_crop(img_bgr: np.ndarray) -> np.ndarray:
    """
    Auto-detect the circular contact region, crop the largest inscribed
    square (no black corners), and resize to 224×224. Returns BGR uint8.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)

    (cx, cy), r = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
    half = r / math.sqrt(2)
    h, w = img_bgr.shape[:2]
    x0 = max(0, int(math.ceil(cx - half)))
    y0 = max(0, int(math.ceil(cy - half)))
    x1 = min(w, int(math.floor(cx + half)))
    y1 = min(h, int(math.floor(cy + half)))

    return cv2.resize(img_bgr[y0:y1, x0:x1], (224, 224), interpolation=cv2.INTER_LINEAR)


# ── Per-modality extractors (return arrays instead of writing files) ─────────

def extract_image(msgs, t0, t1):
    """
    Returns (found, arr) where arr is float32 (224, 224, 3) RGB in [0, 1],
    zeros if no frame was found in the window.
    """
    window = in_window(msgs, t0, t1)
    if not window:
        return False, np.zeros((224, 224, 3), dtype=np.float32)
    t_mid = (t0 + t1) // 2
    _, msg = min(window, key=lambda x: abs(x[0] - t_mid))
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    channels = msg.step // msg.width
    img = buf.reshape(msg.height, msg.step)[:, :msg.width * channels] \
             .reshape(msg.height, msg.width, channels)
    cropped_bgr = _detect_circle_crop(img)
    rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    return True, (rgb.astype(np.float32) / 255.0)


def extract_imu(msgs, t0, t1):
    window = in_window(msgs, t0, t1)
    if not window:
        return np.zeros((0, 3), dtype=np.float32)
    return np.array([[m.x, m.y, m.z] for _, m in window], dtype=np.float32)


def _pad_channel(arr, n=25):
    """Convert a list/array to a fixed-length float32 array, zero-padding if short."""
    out = np.zeros(n, dtype=np.float32)
    data = np.asarray(arr, dtype=np.float32)
    out[:min(len(data), n)] = data[:n]
    return out


def extract_pressure_ap(msgs, t0, t1):
    window = in_window(msgs, t0, t1)
    if not window:
        return np.zeros((0, 50), dtype=np.float32)
    rows = [
        np.concatenate([_pad_channel(m.ch_a), _pad_channel(m.ch_b)])
        for _, m in window
    ]
    return np.array(rows, dtype=np.float32)


def extract_pressure(msgs, t0, t1):
    window = in_window(msgs, t0, t1)
    if not window:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array([[m.pressure, m.temperature] for _, m in window], dtype=np.float32)


def extract_ght(msgs, t0, t1, fallback):
    window = in_window(msgs, t0, t1)
    if window:
        arr  = np.array([[m.temperature, m.humidity, m.gas, m.pressure]
                         for _, m in window], dtype=np.float32)
        return arr.mean(axis=0)
    if fallback is not None:
        return fallback.astype(np.float32)
    return np.zeros(4, dtype=np.float32)   # [temperature, humidity, gas, pressure]


_MEL_N_FFT = 2048

def _build_mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    """Triangular mel filterbank, matching librosa's default (HTK scale)."""
    if fmax is None:
        fmax = sr / 2.0

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)

    mel_pts  = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts   = mel_to_hz(mel_pts)
    bin_pts  = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    n_bins   = n_fft // 2 + 1
    fb       = np.zeros((n_mels, n_bins), dtype=np.float32)

    for m in range(n_mels):
        lo, ctr, hi = bin_pts[m], bin_pts[m + 1], bin_pts[m + 2]
        if ctr > lo:
            fb[m, lo:ctr] = (np.arange(lo, ctr) - lo) / (ctr - lo)
        if hi > ctr:
            fb[m, ctr:hi] = (hi - np.arange(ctr, hi)) / (hi - ctr)

    return fb


_MEL_FB = None  # cached filterbank (built once per process)


def _melspectrogram_db(samples):
    """
    Mel spectrogram in dB. Matches librosa defaults:
      n_fft=2048, hop_length=HOP_LENGTH, window=hann, n_mels=N_MELS, sr=SAMPLE_RATE
    Returns float32 array of shape (N_MELS, T).
    """
    global _MEL_FB
    if _MEL_FB is None:
        _MEL_FB = _build_mel_filterbank(SAMPLE_RATE, _MEL_N_FFT, N_MELS)

    win    = np.hanning(_MEL_N_FFT).astype(np.float32)
    n      = len(samples)
    starts = np.arange(0, n - _MEL_N_FFT + 1, HOP_LENGTH)

    # STFT power spectrum (n_fft//2+1, T)
    S = np.zeros((_MEL_N_FFT // 2 + 1, len(starts)), dtype=np.float32)
    for i, s in enumerate(starts):
        frame    = samples[s:s + _MEL_N_FFT] * win
        spec     = np.fft.rfft(frame)
        S[:, i]  = (spec.real ** 2 + spec.imag ** 2)

    mel   = _MEL_FB @ S                          # (N_MELS, T)
    mel   = np.maximum(mel, 1e-10)
    mel_db = 10.0 * np.log10(mel / mel.max())    # power_to_db, ref=max
    return mel_db.astype(np.float32)


def extract_audio(msgs0, msgs1, t0, t1):
    spectrograms = []
    for msgs in (msgs0, msgs1):
        window = in_window(msgs, t0, t1)
        if not window:
            return np.zeros((2, N_MELS, 0), dtype=np.float32)
        samples = np.concatenate([np.array(m.data, dtype=np.float32) for _, m in window])
        spectrograms.append(_melspectrogram_db(samples))

    # Trim both mics to the same T in case of minor length differences
    T = min(s.shape[1] for s in spectrograms)
    return np.stack([s[:, :T] for s in spectrograms]).astype(np.float32)  # (2, 64, T)


# ── Recording processor ───────────────────────────────────────────────────────

def process_recording(mcap_files, action_group, trial_idx, label_prefix=""):
    """
    Extract one recording's full time span as a single trial_NNN group
    inside action_group (no sub-windowing — one raw recording = one trial).
    """
    print(f"      Loading {len(mcap_files)} file(s) …", flush=True)
    data = load_messages(mcap_files)

    all_ts = [t for key in data for t, _ in data[key]]
    if all_ts:
        w0 = min(all_ts)
        w1 = max(all_ts) + 1   # +1 ns so in_window's [t0, t1) includes the last message
        print(f"      {(w1 - w0)/1e9:.2f} s  →  trial_{trial_idx:03d}")
    else:
        w0 = w1 = 0
        print(f"      No messages found  →  trial_{trial_idx:03d} (all zeros)")

    trial_grp = action_group.create_group(f"trial_{trial_idx:03d}")

    image_valid, image_arr = extract_image(data["image"], w0, w1)
    trial_grp.attrs["image_valid"] = image_valid
    trial_grp.create_dataset("image",       data=image_arr)
    trial_grp.create_dataset("imu_raw",     data=extract_imu(data["imu"], w0, w1))
    trial_grp.create_dataset("pressure_ap", data=extract_pressure_ap(data["pressure_ap"], w0, w1))
    trial_grp.create_dataset("pressure",    data=extract_pressure(data["pressure"], w0, w1))
    trial_grp.create_dataset("ght",         data=extract_ght(data["ght"], w0, w1, None))

    audio_arr = extract_audio(data["audio_0"], data["audio_1"], w0, w1)
    n_frames = audio_arr.shape[-1]
    if n_frames < MIN_AUDIO_FRAMES:
        print(f"      WARNING: sparse audio at {label_prefix}/trial_{trial_idx:03d} "
              f"— {n_frames} frame(s) < {MIN_AUDIO_FRAMES}, padding with zeros")
        audio_arr = np.pad(audio_arr, ((0, 0), (0, 0), (0, MIN_AUDIO_FRAMES - n_frames)))
    trial_grp.create_dataset("audio", data=audio_arr)


# ── Main ──────────────────────────────────────────────────────────────────────

def verify_output(h5_path):
    """Print all keys and shapes/dtypes/attrs found in an HDF5 file."""
    print(f"    [verify] {h5_path}")
    with h5py.File(h5_path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"      {name}: shape={obj.shape}  dtype={obj.dtype}")
            else:
                attrs = dict(obj.attrs)
                suffix = f"  attrs={attrs}" if attrs else ""
                print(f"      {name}/{suffix}")
        f.visititems(visitor)


def process_object(obj, data_root, output_root, actions, verify=False):
    hdf5_dir = os.path.join(output_root, "hdf5")
    os.makedirs(hdf5_dir, exist_ok=True)
    h5_path = os.path.join(hdf5_dir, f"{obj}.h5")

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("metadata/object_name", data=obj)

        for action in actions:
            action_path = os.path.join(data_root, obj, action)
            if not os.path.isdir(action_path):
                continue

            recordings = find_mcap_files(action_path)
            if not recordings:
                print(f"  {obj}/{action}: no .mcap files — skipped")
                continue

            print(f"\n  {obj}/{action}  ({len(recordings)} recording(s))")
            action_group = f.create_group(action)

            for idx, (mcap_files, label) in enumerate(recordings):
                print(f"    [{idx+1}/{len(recordings)}] {label}")
                process_recording(mcap_files, action_group, idx, label_prefix=f"{obj}/{action}")

    if verify:
        verify_output(h5_path)

    return h5_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert Digit360 .mcap bags to an HDF5 ML dataset (one file per object)."
    )
    parser.add_argument("--data-root",   required=True,
                        help="Root dir containing obj_XX folders")
    parser.add_argument("--output-root", required=True,
                        help="Output root directory (HDF5 files go to <output-root>/hdf5/)")
    parser.add_argument("--obj",    default=None,
                        help="Process only this object, e.g. obj_02")
    parser.add_argument("--action", default=None,
                        help="Process only this action, e.g. tap")
    parser.add_argument("--verify", action="store_true",
                        help="After writing each object's file, print all keys/shapes")
    args = parser.parse_args()

    data_root   = os.path.expanduser(args.data_root)
    output_root = os.path.expanduser(args.output_root)

    if not os.path.isdir(data_root):
        print(f"ERROR: data-root not found: {data_root}")
        return 1

    obj_dirs = sorted(
        d for d in os.listdir(data_root)
        if d.startswith("obj_") and os.path.isdir(os.path.join(data_root, d))
    )
    if args.obj:
        obj_dirs = [d for d in obj_dirs if d == args.obj]
        if not obj_dirs:
            print(f"ERROR: --obj {args.obj!r} not found in {data_root}")
            return 1

    actions = [args.action] if args.action else ACTIONS

    print(f"Found {len(obj_dirs)} object(s) in {data_root}")

    for obj in obj_dirs:
        print(f"\n=== {obj} ===")
        process_object(obj, data_root, output_root, actions, verify=args.verify)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
