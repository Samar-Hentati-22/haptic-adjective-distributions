"""
glcm_example.py

Concrete GLCM texture-feature example for two tactile objects, for a
presentation slide.

Pipeline (matches the thesis preprocessing in
data_prep/image_crop.py, using skimage's Otsu
implementation as requested):
  1. Load the raw DIGIT 360 tactile image (224x224x3 float32, [0,1]) from
     the HDF5 dataset, one 'squeeze' trial per object.
  2. Grayscale it, then run skimage.filters.threshold_otsu to isolate the
     active sensing region from the background.
  3. Detect the region's contour (cv2.findContours on the Otsu mask),
     take the largest contour's bounding box, and crop to it.
  4. Resize the crop back to 224x224x3.
  5. Convert the preprocessed crop to grayscale and compute GLCM texture
     features (contrast, homogeneity, correlation, energy) with
     skimage.feature.graycomatrix / graycoprops, averaged over the 4
     standard offset angles (0/45/90/135 deg) at distance 1 for
     rotation-robust values.

Outputs one PNG: preprocessed image pair + GLCM feature table, side by side.
"""
import os
import sys

import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray

# ── Config ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402
HDF5_DIR = paths.hdf5_dir()
PROCEDURE = "squeeze"
TRIAL = "trial_000"

OBJECTS = [
    ("obj_50", "Textured glass"),
    ("obj_46", "Wool beanie"),
]

OUT_DIR = paths.output_dir(__file__)
OUT_PATH = os.path.join(OUT_DIR, "glcm_example.png")

GLCM_DISTANCES = [1]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPS = ["contrast", "homogeneity", "correlation", "energy"]
GLCM_LEVELS = 256


def load_raw_image(obj_id):
    path = os.path.join(HDF5_DIR, f"{obj_id}.h5")
    with h5py.File(path, "r") as f:
        img = f[PROCEDURE][TRIAL]["image"][()].astype(np.float32)  # (224,224,3) in [0,1]
    return img


def otsu_crop_resize(img_float, size=224):
    """Otsu-threshold -> contour -> bounding-box crop -> resize to size x size x 3."""
    gray = rgb2gray(img_float)                     # (H,W) float in [0,1]
    gray_u8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)

    thresh = threshold_otsu(gray_u8)
    mask = (gray_u8 > thresh).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("no foreground contours found after Otsu thresholding")
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    crop = img_float[y:y + h, x:x + w]
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return resized, mask


def compute_glcm_features(img_float):
    """img_float: (H,W,3) in [0,1]. Returns dict of the 4 GLCM props."""
    gray = rgb2gray(img_float)
    gray_u8 = (np.clip(gray, 0, 1) * (GLCM_LEVELS - 1)).astype(np.uint8)

    glcm = graycomatrix(
        gray_u8, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
        levels=GLCM_LEVELS, symmetric=True, normed=True,
    )
    feats = {}
    for prop in GLCM_PROPS:
        # average over the 4 angles for a rotation-robust value
        feats[prop] = graycoprops(glcm, prop).mean()
    return feats


# ── Run pipeline for both objects ───────────────────────────────────────────────
results = []
for obj_id, label in OBJECTS:
    raw = load_raw_image(obj_id)
    processed, mask = otsu_crop_resize(raw)
    feats = compute_glcm_features(processed)
    results.append({"obj_id": obj_id, "label": label, "raw": raw,
                     "processed": processed, "mask": mask, "feats": feats})
    print(f"{label} ({obj_id}): {feats}")

# ── Figure: preprocessed images + feature table ─────────────────────────────────
fig = plt.figure(figsize=(11, 6.8))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.3], hspace=0.35, wspace=0.15,
                       top=0.85, bottom=0.05)

for i, r in enumerate(results):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(np.clip(r["processed"], 0, 1))
    ax.set_title(f"{r['label']}\n({r['obj_id']}, {PROCEDURE}/{TRIAL})",
                 fontsize=13, fontweight="bold", pad=8)
    ax.axis("off")

ax_table = fig.add_subplot(gs[1, :])
ax_table.axis("off")

col_labels = [r["label"] for r in results]
row_labels = [p.capitalize() for p in GLCM_PROPS]
cell_text = [
    [f"{results[0]['feats'][p]:.4f}", f"{results[1]['feats'][p]:.4f}"]
    for p in GLCM_PROPS
]

table = ax_table.table(
    cellText=cell_text,
    rowLabels=row_labels,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.0)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(fontweight="bold")
        cell.set_facecolor("#dbe9f6")
    if col == -1:
        cell.set_text_props(fontweight="bold")

fig.suptitle("GLCM Texture Features — Preprocessed Tactile Images",
             fontsize=16, fontweight="bold", y=0.99)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", pad_inches=0.3)
plt.close()
print(f"\nSaved -> {OUT_PATH}")
