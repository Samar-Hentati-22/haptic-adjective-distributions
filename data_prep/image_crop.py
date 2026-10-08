"""
Shared circular active-region crop detection for tactile images, used by
both the test-sample script (08_image_crop_test.py) and the full-dataset
run (09_apply_crop_full_dataset.py). Validated on a 9-image test sample
before being applied at scale — see preprocessing_investigation/
image_crop_test.png and image_crop_overlay.png.

Method: Otsu-threshold grayscale to separate the bright active region
from the (not actually black, but dimmer) background, morphological
opening to remove noise specks, merge all contour fragments >= 5% of the
largest one's area (the active region is sometimes split into 2+ pieces
by an internal seam/reflection artifact - using only the single largest
contour under-crops in that case), fit a minimum enclosing circle over
the union, pad, crop a square box (clipped to image bounds), resize to
224x224.
"""
import cv2
import numpy as np

PADDING_FRAC = 0.08           # extra radius padding, fraction of detected radius
MIN_CONTOUR_AREA_FRAC = 0.05  # keep contours >= this fraction of the largest
MIN_GRAY_STD = 2.0             # below this, treat as a blank/stuck frame - no real content
MAX_FOREGROUND_FRAC = 0.90     # above this, Otsu likely degenerated (no real background)


def detect_circle(gray_u8):
    """gray_u8: (H, W) uint8. Returns (cx, cy, r, n_fragments). Raises
    RuntimeError with a specific reason if detection isn't usable."""
    if gray_u8.std() < MIN_GRAY_STD:
        raise RuntimeError(f'blank/stuck frame (gray std={gray_u8.std():.2f} < {MIN_GRAY_STD})')

    _, mask = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_frac = (mask > 0).mean()
    if fg_frac > MAX_FOREGROUND_FRAC:
        raise RuntimeError(f'Otsu threshold degenerated (foreground fraction={fg_frac:.2f} '
                            f'> {MAX_FOREGROUND_FRAC})')

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError('no foreground contours found after morphological opening')

    areas = [cv2.contourArea(c) for c in contours]
    max_area = max(areas)
    if max_area <= 0:
        raise RuntimeError('all detected contours have zero area')
    kept = [c for c, a in zip(contours, areas) if a >= MIN_CONTOUR_AREA_FRAC * max_area]
    pts = np.vstack(kept)
    (cx, cy), r = cv2.minEnclosingCircle(pts)
    return cx, cy, r, len(kept)


def crop_and_resize(img_float, cx, cy, r, size=224):
    """img_float: (H, W, 3) float. Returns (resized (size,size,3), was_clipped: bool)."""
    h, w = img_float.shape[:2]
    r_pad = r * (1.0 + PADDING_FRAC)
    x0 = int(round(cx - r_pad))
    x1 = int(round(cx + r_pad))
    y0 = int(round(cy - r_pad))
    y1 = int(round(cy + r_pad))
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(w, x1), min(h, y1)
    was_clipped = (x0, y0, x1, y1) != (x0c, y0c, x1c, y1c)
    crop = img_float[y0c:y1c, x0c:x1c]
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return resized, was_clipped


def detect_and_crop(img_float, size=224):
    """img_float: (H, W, 3) float in [0,1]. Returns a result dict or raises."""
    gray_u8 = (np.clip(img_float.mean(axis=2), 0, 1) * 255).astype(np.uint8)
    cx, cy, r, n_fragments = detect_circle(gray_u8)
    resized, was_clipped = crop_and_resize(img_float, cx, cy, r, size=size)
    return {
        'cropped': resized, 'cx': cx, 'cy': cy, 'r': r,
        'n_fragments': n_fragments, 'was_clipped': was_clipped,
    }
