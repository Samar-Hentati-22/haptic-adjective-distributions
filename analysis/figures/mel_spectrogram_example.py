"""
mel_spectrogram_example.py

Mel-spectrogram visualization of one DIGIT 360 tactile-audio trial, for a
presentation slide.

IMPORTANT DATA NOTE: the per-trial HDF5 files under DATA_Final/hdf5/ do NOT
contain the raw stereo waveform - only the already-computed log-mel
spectrogram is stored, under the 'audio' dataset, with exactly the shape
the thesis pipeline feeds into its 2D CNN encoder: (2, 64, T) = (stereo
channels, mel bins, time frames), in dB (log) scale (values run ~-80..0).
No raw .wav/waveform audio exists anywhere else in the project tree either
(checked). So rather than fabricate a re-run of librosa.feature.melspectrogram
on data that isn't available, this script loads that precomputed log-mel
array directly and plots it - which is the exact representation
librosa.feature.melspectrogram(...) -> librosa.power_to_db(...) would have
produced, since that is what generated it upstream.

  1. Scan slide/squeeze trials (continuous contact, not a single tap
     impulse) on a curated list of rough/textured objects (steel wool,
     abrasive pad, corrugated cardboard, etc.) for the one with the most
     sustained, time-varying energy - i.e. a real friction/texture sound
     rather than one brief spike against silence.
  2. Load its (2, 64, T) log-mel array.
  3. Plot both stereo channels side by side, mel bin on y, time frame on x,
     using a custom blue-toned presentation colormap.
"""
import glob
import os
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── Presentation palette ─────────────────────────────────────────────────────────
COLOR_BG   = "#FFFFFF"
COLOR_TEXT = "#26215C"
CUSTOM_CMAP = LinearSegmentedColormap.from_list(
    "custom", ["#EAF2FF", "#BDCCFF", "#7B9AE8", "#3D5FC4", "#1B2A6B"]
)

# ── Config ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402
HDF5_DIR = paths.hdf5_dir()
ACTIONS_TO_SCAN = ["slide", "squeeze"]  # continuous-contact procedures, not "tap"
SAMPLE_RATE = 48_000      # Hz, per thesis DIGIT 360 mic spec
N_MELS = 64
MIN_T = 60                 # skip very short trials - want a sustained interaction
ACTIVE_DB_THRESHOLD = -35  # a frame counts as "active" above this dB

# Rough/textured objects likely to produce a continuous, texture-rich
# friction sound under slide/squeeze (vs. something smooth like glass/silicone)
ROUGH_OBJECT_IDS = [
    "obj_10", "obj_17", "obj_20", "obj_22", "obj_23", "obj_27", "obj_31",
    "obj_35", "obj_39", "obj_50", "obj_56", "obj_67", "obj_71", "obj_81",
    "obj_04", "obj_54", "obj_73",
]

OBJECT_NAMES = {
    "obj_01": "Wooden spoon", "obj_02": "Plush toy", "obj_03": "Latex cleaning glove",
    "obj_04": "Hand-woven palm leaf mat", "obj_05": "Microfiber cloth",
    "obj_06": "Plastic ruler", "obj_07": "Earmuffs", "obj_08": "Satin scarf",
    "obj_09": "Melamine foam sponge", "obj_10": "Abrasive foam pad",
    "obj_17": "Corrugated cardboard", "obj_20": "Crocheted lace fabric",
    "obj_22": "Kraft paper", "obj_23": "Cellulose sponge rough side",
    "obj_27": "Cellulose sponge soft side", "obj_31": "Rinsing sponge",
    "obj_35": "Steel wool pad", "obj_39": "Pink woven fabric",
    "obj_50": "Textured glass", "obj_54": "Denim fabric", "obj_56": "Bubble wrap",
    "obj_67": "Aluminium foil", "obj_71": "Cleaning brush",
    "obj_73": "Burlap tote bag", "obj_81": "Wood-bristle brush",
}

OUT_DIR = paths.output_dir(__file__)
OUT_PATH = os.path.join(OUT_DIR, "mel_spectrogram_example.png")


def find_representative_trial():
    """Pick the slide/squeeze trial on a rough/textured object with the most
    sustained, time-varying energy - i.e. a continuous friction/texture
    sound rather than one brief tap spike against silence. Scored by
    (fraction of frames with real energy) x (frame-to-frame spectral
    change, averaged over mel bins), both computed on the mono mix."""
    best = None
    for obj_id in ROUGH_OBJECT_IDS:
        fp = os.path.join(HDF5_DIR, f"{obj_id}.h5")
        if not os.path.isfile(fp):
            continue
        with h5py.File(fp, "r") as f:
            for action in ACTIONS_TO_SCAN:
                if action not in f:
                    continue
                for trial_name in f[action]:
                    grp = f[action][trial_name]
                    if "audio" not in grp:
                        continue
                    mel_db = grp["audio"][()]  # (2, 64, T), dB
                    T = mel_db.shape[-1]
                    if T < MIN_T:
                        continue
                    mono = mel_db.mean(axis=0)  # (64, T)
                    active_frac = float((mono.max(axis=0) > ACTIVE_DB_THRESHOLD).mean())
                    flux = float(np.abs(np.diff(mono, axis=1)).mean())
                    score = active_frac * flux
                    if best is None or score > best["score"]:
                        best = {
                            "obj_id": obj_id, "action": action, "trial": trial_name,
                            "mel_db": mel_db, "score": score,
                            "active_frac": active_frac, "flux": flux,
                        }
    if best is None:
        raise RuntimeError("no matching slide/squeeze trial with 'audio' found")
    return best


# ── Locate and load ──────────────────────────────────────────────────────────────
info = find_representative_trial()
mel_db = info["mel_db"]                 # (2, 64, T)
obj_id, action, trial = info["obj_id"], info["action"], info["trial"]
label = OBJECT_NAMES.get(obj_id, obj_id)
n_channels, n_mels, T = mel_db.shape

print(f"Trial used: {obj_id} ({label}) / {action} / {trial}")
print(f"Sample rate: {SAMPLE_RATE} Hz (thesis DIGIT 360 mic spec)")
print(f"Mel-spectrogram shape: {mel_db.shape}  ->  T = {T} time frames, "
      f"{n_mels} mel bins, {n_channels} stereo channels")
print(f"dB range shown: [{mel_db.min():.1f}, {mel_db.max():.1f}] dB")
print(f"Selection: {info['active_frac']*100:.0f}% of frames active (>{ACTIVE_DB_THRESHOLD} dB), "
      f"mean frame-to-frame spectral flux = {info['flux']:.2f} dB")
print("Note: loaded the precomputed log-mel array already stored per trial "
      "(no raw waveform is present in this dataset copy to re-run librosa on).")

# ── Plot ─────────────────────────────────────────────────────────────────────────
# The raw [min, max] dB range is dominated by a long, near-silent noise floor
# (most values sit below -40 dB, with only the brief contact transient near
# 0 dB), which washes out detail under a linear scale. Clip to a robust
# percentile range instead so the actual texture/transient contrast pops.
vmin = float(np.percentile(mel_db, 5))
vmax = float(np.percentile(mel_db, 99.5))
channel_names = ["Left channel", "Right channel"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=COLOR_BG)

for ch, (ax, name) in enumerate(zip(axes, channel_names)):
    ax.set_facecolor(COLOR_BG)
    im = ax.imshow(
        mel_db[ch], aspect="auto", origin="lower", cmap=CUSTOM_CMAP,
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    ax.set_title(name, fontsize=15, fontweight="bold", color=COLOR_TEXT, pad=8)
    ax.set_xlabel("Time frame", fontsize=12, color=COLOR_TEXT, labelpad=8)
    ax.tick_params(colors=COLOR_TEXT, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(COLOR_TEXT)

axes[0].set_ylabel("Mel frequency bin", fontsize=12, color=COLOR_TEXT, labelpad=8)

cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
cbar.set_label("Log-mel power (dB)", fontsize=12, color=COLOR_TEXT)
cbar.ax.yaxis.set_tick_params(color=COLOR_TEXT, labelsize=10)
plt.setp(cbar.ax.get_yticklabels(), color=COLOR_TEXT)
cbar.outline.set_edgecolor(COLOR_TEXT)

fig.suptitle(
    f"Mel-Spectrogram — {label} ({obj_id}), {action}/{trial}\n"
    f"{N_MELS} mel bins, {T} time frames, {SAMPLE_RATE/1000:.0f} kHz stereo audio",
    fontsize=15, fontweight="bold", color=COLOR_TEXT, y=1.04,
)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor=COLOR_BG)
plt.close()
print(f"\nSaved -> {OUT_PATH}")
