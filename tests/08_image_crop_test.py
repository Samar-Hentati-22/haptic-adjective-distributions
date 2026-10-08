"""
Pipeline v3 — Step 1: test circular active-region crop detection on a
sample of tactile images.

Test/analysis only: does NOT modify DATA_Final/hdf5/ and does not process
the full dataset. Purely diagnostic, saved to <repo>/outputs/08_image_crop_test/.

Detection method:
  1. Grayscale (mean over RGB), Otsu threshold to separate the bright
     active region from background.
  2. Morphological opening to remove small noise specks.
  3. Keep all contours with area >= 5% of the largest (the active region
     is sometimes split into 2+ fragments by an internal seam/reflection
     artifact visible in some objects' images - using only the single
     largest contour under-crops in that case, see report).
  4. minEnclosingCircle over the union of the surviving contours' points.
  5. Pad the radius, crop a square bounding box around the circle
     (clipped to image bounds), resize to 224x224.
"""
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_prep'))  # <repo>/data_prep

HDF5_DIR = paths.hdf5_dir()
OUTPUT_DIR = paths.output_dir(__file__)

from image_crop import detect_and_crop, PADDING_FRAC  # noqa: E402

TEST_OBJECT_NUMS = [1, 10, 20, 30, 40, 50, 60, 70, 80]


def load_valid_image(h5_path):
    with h5py.File(h5_path, 'r') as f:
        for action in ('squeeze', 'hold'):
            if action not in f:
                continue
            for trial_name in sorted(f[action].keys()):
                img = f[action][trial_name]['image'][()].astype(np.float32)
                if not np.all(img == 0):
                    return action, trial_name, img
    return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    for obj_num in TEST_OBJECT_NUMS:
        obj_id = f'obj_{obj_num:02d}'
        h5_path = os.path.join(HDF5_DIR, f'{obj_id}.h5')
        if not os.path.isfile(h5_path):
            continue
        loaded = load_valid_image(h5_path)
        if loaded is None:
            print(f"  WARNING: no valid image found for {obj_id}")
            continue
        action, trial_name, img = loaded

        try:
            result = detect_and_crop(img)
        except RuntimeError as e:
            print(f"  WARNING: circle detection failed for {obj_id}/{action}/{trial_name}: {e}")
            results.append({'obj_id': obj_id, 'action': action, 'trial_name': trial_name,
                             'img': img, 'ok': False})
            continue

        results.append({
            'obj_id': obj_id, 'action': action, 'trial_name': trial_name, 'img': img,
            'cx': result['cx'], 'cy': result['cy'], 'r': result['r'],
            'n_fragments': result['n_fragments'], 'cropped': result['cropped'],
            'was_clipped': result['was_clipped'], 'ok': True,
        })
        print(f"{obj_id}/{action}/{trial_name}: center=({result['cx']:.1f},{result['cy']:.1f}) "
              f"r={result['r']:.1f} fragments={result['n_fragments']} "
              f"edge_clipped={result['was_clipped']}")

    # ── Comparison figure: original vs cropped-and-resized ─────────────
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.6))
    for col, res in enumerate(results):
        ax = axes[0, col]
        ax.imshow(np.clip(res['img'], 0, 1))
        ax.set_title(f"{res['obj_id']}\n(orig)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[1, col]
        if res['ok']:
            ax.imshow(np.clip(res['cropped'], 0, 1))
            ax.set_title('cropped+resized', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'DETECTION\nFAILED', ha='center', va='center', color='red')
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Tactile image circular-region crop: original (top) vs cropped+resized 224x224 (bottom)')
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'image_crop_test.png')
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")

    # ── Overlay figure: detected boundary drawn on original image ──────
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3))
    for col, res in enumerate(results):
        ax = axes[col]
        ax.imshow(np.clip(res['img'], 0, 1))
        if res['ok']:
            circle = plt.Circle((res['cx'], res['cy']), res['r'], color='lime',
                                 fill=False, linewidth=1.5)
            ax.add_patch(circle)
            r_pad = res['r'] * (1.0 + PADDING_FRAC)
            rect = plt.Rectangle((res['cx'] - r_pad, res['cy'] - r_pad),
                                  2 * r_pad, 2 * r_pad, edgecolor='yellow',
                                  fill=False, linewidth=1.2, linestyle='--')
            ax.add_patch(rect)
        ax.set_title(f"{res['obj_id']}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Detected active-region circle (green) and padded crop box (yellow, dashed)')
    fig.tight_layout()
    overlay_path = os.path.join(OUTPUT_DIR, 'image_crop_overlay.png')
    fig.savefig(overlay_path, dpi=150)
    print(f"Saved {overlay_path}")


if __name__ == '__main__':
    main()
