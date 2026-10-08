"""
Pipeline v3 — apply the validated circular-region crop (image_crop.py) to
every image_valid=True trial across the full dataset (85 objects x 4
procedures x 5 trials), writing a new, image-only HDF5 tree.

Additive only: TACTILE_DATA_ROOT/hdf5/ (original files) are opened read-only and
never modified. Output goes to a separate TACTILE_DATA_ROOT/hdf5_cropped_images/
tree containing only the `image` dataset per trial - pressure_ap/imu_raw/
pressure/audio/ght stay referenced from the original files.
"""
import csv
import os
import sys

import h5py
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402

SRC_HDF5_DIR = paths.hdf5_dir()
DST_HDF5_DIR = paths.cropped_dir()
LOG_DIR = paths.output_dir(__file__)
LOG_PATH = os.path.join(LOG_DIR, 'image_crop_full_run_log.csv')

sys.path.insert(0, BASE_DIR)
from image_crop import detect_and_crop  # noqa: E402

ACTIONS = ['tap', 'squeeze', 'slide', 'hold']


def main():
    os.makedirs(DST_HDF5_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    log_rows = []
    n_success = 0
    n_failed = 0
    n_skipped_invalid = 0

    for obj_num in range(1, 86):
        obj_id = f'obj_{obj_num:02d}'
        src_path = os.path.join(SRC_HDF5_DIR, f'{obj_id}.h5')
        if not os.path.isfile(src_path):
            continue
        dst_path = os.path.join(DST_HDF5_DIR, f'{obj_id}.h5')

        with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
            for action in ACTIONS:
                if action not in src:
                    continue
                for trial_name in src[action]:
                    trial = src[action][trial_name]
                    image_valid = bool(trial.attrs.get('image_valid', False))
                    if not image_valid:
                        n_skipped_invalid += 1
                        continue

                    img = trial['image'][()].astype(np.float32)
                    try:
                        result = detect_and_crop(img)
                    except RuntimeError as e:
                        n_failed += 1
                        log_rows.append({
                            'object_id': obj_id, 'procedure': action, 'trial': trial_name,
                            'status': 'failed', 'radius': '', 'fragment_merged': '',
                            'n_fragments': '', 'edge_clipped': '', 'error': str(e),
                        })
                        print(f"  FAILED: {obj_id}/{action}/{trial_name}: {e}")
                        continue

                    grp = dst.require_group(f'{action}/{trial_name}')
                    grp.create_dataset('image', data=result['cropped'].astype(np.float32))
                    grp.attrs['image_valid'] = True
                    grp.attrs['crop_radius'] = result['r']
                    grp.attrs['crop_n_fragments'] = result['n_fragments']

                    n_success += 1
                    log_rows.append({
                        'object_id': obj_id, 'procedure': action, 'trial': trial_name,
                        'status': 'success', 'radius': f"{result['r']:.3f}",
                        'fragment_merged': result['n_fragments'] > 1,
                        'n_fragments': result['n_fragments'],
                        'edge_clipped': result['was_clipped'], 'error': '',
                    })

        print(f"{obj_id}: done")

    with open(LOG_PATH, 'w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=[
            'object_id', 'procedure', 'trial', 'status', 'radius',
            'fragment_merged', 'n_fragments', 'edge_clipped', 'error'])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nSaved log to {LOG_PATH}")
    print(f"Success: {n_success}, Failed: {n_failed}, "
          f"Skipped (image_valid=False): {n_skipped_invalid}")

    # ── Radius distribution / fragment-merge report ─────────────────────
    radii = np.array([float(r['radius']) for r in log_rows if r['status'] == 'success'])
    merged = np.array([r['fragment_merged'] for r in log_rows if r['status'] == 'success'])

    print(f"\nRadius distribution (n={len(radii)}):")
    print(f"  min={radii.min():.2f}  p5={np.percentile(radii,5):.2f}  "
          f"median={np.median(radii):.2f}  p95={np.percentile(radii,95):.2f}  "
          f"max={radii.max():.2f}")

    p5 = np.percentile(radii, 5)
    below_p5 = [r for r in log_rows if r['status'] == 'success' and float(r['radius']) < p5]
    print(f"\n{len(below_p5)} trials below p5 radius ({p5:.2f}px) - flagged for review:")
    for r in sorted(below_p5, key=lambda x: float(x['radius']))[:20]:
        print(f"  {r['object_id']}/{r['procedure']}/{r['trial']}: r={r['radius']}")

    n_merged = int(merged.sum())
    print(f"\nFragment-merging triggered: {n_merged}/{len(radii)} trials "
          f"({100*n_merged/len(radii):.1f}%)")
    if n_merged > 0:
        merged_radii = radii[merged]
        unmerged_radii = radii[~merged]
        print(f"  merged trials radius:   mean={merged_radii.mean():.2f} "
              f"median={np.median(merged_radii):.2f}")
        print(f"  unmerged trials radius: mean={unmerged_radii.mean():.2f} "
              f"median={np.median(unmerged_radii):.2f}")

    edge_clipped = [r for r in log_rows if r['status'] == 'success' and r['edge_clipped'] == 'True']
    print(f"\nEdge-clipped crops (padded box extended past image bounds): {len(edge_clipped)}")


if __name__ == '__main__':
    main()
