"""
glitch_before_after.py

Before/after visualization of the DIGIT 360 pressure-array hardware glitch
for a presentation slide: a contiguous block of channels reads exactly 0.0
for a single timestep, then is fixed by interpolating from the neighboring
timesteps (thesis fix).

  1. Scan pressure_ap arrays (T, 50) across the HDF5 dataset for a timestep
     where a contiguous run of ~25-39 channels is exactly 0.0 while the
     immediately neighboring timesteps are not (i.e. a real glitch, not a
     start/end-of-trial padding artifact).
  2. Take a short window of timesteps around the glitch.
  3. 'Before': raw windowed pressure_ap heatmap (channels x time).
  4. 'After': the same window with the affected channels at the glitch
     timestep linearly interpolated between the previous and next timestep
     (matches the thesis fix).
  5. Both heatmaps side by side, same colormap + scale, in the
     presentation's purple/lavender palette.
"""
import glob
import os
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── Presentation palette ─────────────────────────────────────────────────────────
COLOR_BG        = "#FFFFFF"
COLOR_TEXT      = "#26215C"
CUSTOM_CMAP = LinearSegmentedColormap.from_list(
    "custom", ["#F0E2F8", "#BDCCFF", "#BEA4D2", "#7165B2", "#26215C"]
)

# ── Config ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402
HDF5_DIR = paths.hdf5_dir()
ACTIONS = ["tap", "squeeze", "slide", "hold"]
MIN_GLITCH_CHANNELS = 25
MAX_GLITCH_CHANNELS = 39
WINDOW_HALF = 12  # timesteps shown on each side of the glitch

OBJECT_NAMES = {
    "obj_01": "Wooden spoon", "obj_02": "Plush toy", "obj_03": "Latex cleaning glove",
    "obj_04": "Hand-woven palm leaf mat", "obj_05": "Microfiber cloth",
    "obj_06": "Plastic ruler", "obj_07": "Earmuffs", "obj_08": "Satin scarf",
    "obj_09": "Melamine foam sponge", "obj_10": "Abrasive foam pad",
}  # trimmed to the range actually scanned - only used for a friendlier label

OUT_DIR = paths.output_dir(__file__)
OUT_PATH = os.path.join(OUT_DIR, "pressure_glitch_before_after.png")


def find_glitch(prefer_widest=True):
    """Scan the dataset for real glitch trials/timesteps. Returns the widest
    one found (most affected channels) if prefer_widest, else the first."""
    files = sorted(glob.glob(os.path.join(HDF5_DIR, "obj_*.h5")))
    candidates = []
    for fp in files:
        obj_id = os.path.basename(fp).replace(".h5", "")
        with h5py.File(fp, "r") as f:
            for action in ACTIONS:
                if action not in f:
                    continue
                for trial_name in f[action]:
                    grp = f[action][trial_name]
                    if "pressure_ap" not in grp:
                        continue
                    arr = grp["pressure_ap"][()]  # (T, 50)
                    zero_mask = (arr == 0.0)
                    for t in range(1, arr.shape[0] - 1):
                        row = zero_mask[t]
                        n_zero = row.sum()
                        if n_zero < MIN_GLITCH_CHANNELS or n_zero > MAX_GLITCH_CHANNELS:
                            continue
                        idx = np.where(row)[0]
                        if idx.max() - idx.min() + 1 != idx.size:
                            continue  # not a contiguous block
                        # must be a real glitch: neighbors aren't zero there too
                        if zero_mask[t - 1][idx].sum() > 5 or zero_mask[t + 1][idx].sum() > 5:
                            continue
                        if t - WINDOW_HALF < 0 or t + WINDOW_HALF >= arr.shape[0]:
                            continue  # keep it simple: need a full window
                        candidates.append({
                            "obj_id": obj_id, "action": action, "trial": trial_name,
                            "t": t, "chan_lo": int(idx.min()), "chan_hi": int(idx.max()),
                            "n_channels": int(idx.size), "arr": arr,
                        })
                        break  # one hit per trial is enough
        if len(candidates) >= 8:
            break  # plenty to choose from without scanning all 85 objects

    if not candidates:
        raise RuntimeError("no glitch matching the criteria was found in the dataset")
    if prefer_widest:
        candidates.sort(key=lambda c: c["n_channels"], reverse=True)
    return candidates[0]


def interpolate_glitch(window, t_local, chan_lo, chan_hi):
    """window: (W, 50) copy. Linearly interpolate the glitched channels at
    t_local from the immediately surrounding rows (t_local-1, t_local+1)."""
    fixed = window.copy()
    prev_row = window[t_local - 1, chan_lo:chan_hi + 1]
    next_row = window[t_local + 1, chan_lo:chan_hi + 1]
    fixed[t_local, chan_lo:chan_hi + 1] = 0.5 * (prev_row + next_row)
    return fixed


# ── Locate the glitch ────────────────────────────────────────────────────────────
info = find_glitch()
arr = info["arr"]
t = info["t"]
chan_lo, chan_hi = info["chan_lo"], info["chan_hi"]
label = OBJECT_NAMES.get(info["obj_id"], info["obj_id"])

t0, t1 = t - WINDOW_HALF, t + WINDOW_HALF
window = arr[t0:t1 + 1]              # (W, 50)
t_local = t - t0

fixed_window = interpolate_glitch(window, t_local, chan_lo, chan_hi)

print(f"Glitch found: {info['obj_id']} ({label}) / {info['action']} / {info['trial']}")
print(f"  Timestep index (within trial): {t}  (~{t / 266.0:.3f} s @ ~266 Hz)")
print(f"  Affected channels: {chan_lo}-{chan_hi}  ({info['n_channels']} channels)")
print(f"  Window shown: timesteps {t0}-{t1} ({window.shape[0]} steps)")

# ── Shared color scale, based on the interpolated (post-fix) data so both panels
# ── are directly comparable. Pressure_ap has a few very high-magnitude contact
# ── spikes that would otherwise saturate the scale and wash out the baseline
# ── texture where the glitch actually shows up, so we clip to a robust range.
vmin = float(np.percentile(fixed_window, 1))
vmax = float(np.percentile(fixed_window, 99))

# ── Plot ─────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), facecolor=COLOR_BG)

time_extent = [t0, t1, 50, 0]  # channels 0 (top) .. 49 (bottom), time on x

for ax, data, title in zip(
    axes, [window, fixed_window], ["Before (glitch)", "After (interpolated)"]
):
    ax.set_facecolor(COLOR_BG)
    im = ax.imshow(
        data.T, aspect="auto", cmap=CUSTOM_CMAP, vmin=vmin, vmax=vmax,
        extent=time_extent, interpolation="nearest",
    )
    ax.axvline(t, color=COLOR_TEXT, linestyle="--", linewidth=1.3, alpha=0.7)
    ax.set_title(title, fontsize=17, fontweight="bold", color=COLOR_TEXT, pad=10)
    ax.set_xlabel("Timestep", fontsize=13, color=COLOR_TEXT, labelpad=8)
    ax.tick_params(colors=COLOR_TEXT, labelsize=11)
    for spine in ax.spines.values():
        spine.set_color(COLOR_TEXT)

axes[0].set_ylabel("Pressure channel", fontsize=13, color=COLOR_TEXT, labelpad=8)

cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
cbar.set_label("Pressure (raw units)", fontsize=12, color=COLOR_TEXT)
cbar.ax.yaxis.set_tick_params(color=COLOR_TEXT, labelsize=10)
plt.setp(cbar.ax.get_yticklabels(), color=COLOR_TEXT)
cbar.outline.set_edgecolor(COLOR_TEXT)

fig.suptitle(
    f"Pressure Sensor Hardware Glitch — {label} ({info['obj_id']}), "
    f"{info['action']}/{info['trial']}, t={t}\n"
    f"{info['n_channels']} channels (ch. {chan_lo}\u2013{chan_hi}) read exactly 0.0 "
    f"for a single timestep",
    fontsize=13.5, color=COLOR_TEXT, y=1.03,
)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor=COLOR_BG)
plt.close()
print(f"\nSaved -> {OUT_PATH}")
