"""
Pipeline v3 — diagnostic: train-set vs. test-set correlation for fold 1,
to check whether the model is overfitting or underfitting.

Loads fold 1's best checkpoint, re-normalizes both fold 1's training and
test objects with that fold's norm_stats (rebuilt from the feature cache,
so this is fast), runs inference on both, computes per-object expected
ratings, and reports Pearson r on train (68 objects) vs. test (17
objects) per adjective.

Diagnostic only -- does not modify train.py, model.py, dataset.py, or any
existing results. Investigation script, stays outside src/.
"""
import os
import sys
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402

HDF5_DIR = paths.hdf5_dir()
CROPPED_DIR = paths.cropped_dir()
LABELS_DIR = paths.labels_dir()
# Training run folder written by model/src/train.py (default --results-dir).
RESULTS_DIR = os.path.join(paths.OUTPUTS_ROOT, 'train', 'results')
CACHE_DIR = paths.cache_dir()

from dataset import compute_norm_stats, TactileDatasetV3  # noqa: E402
from model import TactilePerceptionModelV3, collate_batch  # noqa: E402

ADJECTIVES = [
    'hard', 'soft', 'rough', 'smooth', 'bumpy', 'flat',
    'sticky', 'slippery', 'cold', 'warm', 'wet', 'dry',
    'compressible', 'firm', 'fuzzy', 'scratchy', 'pleasant', 'irritating',
]
WEAK_ADJECTIVES = {'wet', 'dry', 'sticky', 'cold', 'warm'}

RATINGS = torch.arange(1, 8, dtype=torch.float32)


def expected_rating(dist: torch.Tensor) -> torch.Tensor:
    return (dist * RATINGS.view(*([1] * (dist.dim() - 1)), 7)).sum(dim=-1)


def get_object_expected_ratings(model, loader, device):
    """Returns (obj_indices sorted list, pred_expected (n_obj,18), true_expected (n_obj,18))."""
    model.eval()
    obj_preds = {}
    obj_true = {}
    with torch.no_grad():
        for batch in loader:
            imu = batch['imu'].to(device)
            pressure_ap = batch['pressure_ap'].to(device)
            image = batch['image'].to(device)
            audio = batch['audio'].to(device)
            has_image = batch['has_image'].to(device)
            has_audio = batch['has_audio'].to(device)
            obj_indices = batch['obj_idx']

            pred = model(imu, pressure_ap, image, audio, has_image, has_audio)
            for i in range(pred.size(0)):
                oi = obj_indices[i].item()
                obj_preds.setdefault(oi, []).append(pred[i].cpu())
                if oi not in obj_true:
                    obj_true[oi] = batch['label_dist'][i]

    obj_ids_sorted = sorted(obj_preds.keys())
    pred_expected = []
    true_expected = []
    for oi in obj_ids_sorted:
        avg_pred_dist = torch.stack(obj_preds[oi]).mean(dim=0)  # (18,7)
        pred_expected.append(expected_rating(avg_pred_dist))
        true_expected.append(expected_rating(obj_true[oi]))

    return obj_ids_sorted, torch.stack(pred_expected).numpy(), torch.stack(true_expected).numpy()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    object_ids = np.load(os.path.join(LABELS_DIR, 'object_ids_phase2.npy'), allow_pickle=True)
    labels_dist = np.load(os.path.join(LABELS_DIR, 'labels_phase2.npy'))
    labels_mean = np.load(os.path.join(LABELS_DIR, 'labels_mean_phase2.npy'))

    with open(os.path.join(RESULTS_DIR, 'fold_1_results.json')) as f:
        fold1_results = json.load(f)
    train_indices = fold1_results['train_objects']
    test_indices = fold1_results['test_objects']
    image_input_dim = fold1_results['image_input_dim']
    print(f"Fold 1: {len(train_indices)} train objects, {len(test_indices)} test objects, "
          f"image_input_dim={image_input_dim}")

    cache_dir = CACHE_DIR if os.path.isdir(CACHE_DIR) else None
    print("Recomputing fold 1's norm_stats (from training objects only, same as during training)...")
    norm_stats = compute_norm_stats(train_indices, object_ids, HDF5_DIR, CROPPED_DIR,
                                     device=str(device), cache_dir=cache_dir)
    actual_image_dim = len(norm_stats['image_resnet_alive_dims']) + 256
    assert actual_image_dim == image_input_dim, (
        f"norm_stats image dim {actual_image_dim} != checkpoint's recorded {image_input_dim} "
        f"-- cache or fold split may have changed since training.")

    train_ds = TactileDatasetV3(train_indices, object_ids, labels_dist, labels_mean, norm_stats,
                                 hdf5_dir=HDF5_DIR, cropped_dir=CROPPED_DIR, cache_dir=cache_dir)
    test_ds = TactileDatasetV3(test_indices, object_ids, labels_dist, labels_mean, norm_stats,
                                hdf5_dir=HDF5_DIR, cropped_dir=CROPPED_DIR, cache_dir=cache_dir)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_batch)

    model = TactilePerceptionModelV3(image_input_dim=image_input_dim).to(device)
    ckpt_path = os.path.join(RESULTS_DIR, 'fold_1_best.pt')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded {ckpt_path} (best epoch {ckpt['epoch']}, val_emd={ckpt['val_emd']:.4f})")

    print("\nRunning inference on train and test sets...")
    _, train_pred, train_true = get_object_expected_ratings(model, train_loader, device)
    _, test_pred, test_true = get_object_expected_ratings(model, test_loader, device)
    print(f"  train_pred/true shape: {train_pred.shape}  |  test_pred/true shape: {test_pred.shape}")

    rows = []
    for adj_idx, adj in enumerate(ADJECTIVES):
        train_r = np.corrcoef(train_true[:, adj_idx], train_pred[:, adj_idx])[0, 1]
        test_r = np.corrcoef(test_true[:, adj_idx], test_pred[:, adj_idx])[0, 1]
        gap = train_r - test_r
        rows.append((adj, train_r, test_r, gap))

    rows.sort(key=lambda r: -r[3])

    print(f"\n{'='*70}")
    print(f"  Train (n={len(train_indices)}) vs. Test (n={len(test_indices)}) Pearson r -- Fold 1")
    print(f"{'='*70}")
    print(f"  {'adjective':14s} {'train_r':>9s} {'test_r':>9s} {'gap':>8s}   {'weak?':5s}")
    print(f"  {'-'*60}")
    for adj, train_r, test_r, gap in rows:
        flag = 'weak' if adj in WEAK_ADJECTIVES else ''
        print(f"  {adj:14s} {train_r:9.3f} {test_r:9.3f} {gap:8.3f}   {flag:5s}")

    mean_train_r = np.mean([r[1] for r in rows if r[0] not in WEAK_ADJECTIVES])
    mean_test_r = np.mean([r[2] for r in rows if r[0] not in WEAK_ADJECTIVES])
    mean_gap = np.mean([r[3] for r in rows if r[0] not in WEAK_ADJECTIVES])
    print(f"  {'-'*60}")
    print(f"  {'MEAN (13 non-weak)':14s} {mean_train_r:9.3f} {mean_test_r:9.3f} {mean_gap:8.3f}")

    print(f"""
{'='*70}
  Interpretation guide
{'='*70}
  gap = train_r - test_r

  - Large POSITIVE gap (train_r >> test_r): the model fits training
    objects' rank order much better than held-out objects -- classic
    OVERFITTING signature. More regularization, more data, or a smaller
    model would likely help more than more capacity/training time.
  - Small gap (train_r ~ test_r), both r reasonably high: model
    generalizes about as well on held-out objects as on seen ones --
    healthy, not much overfitting for this adjective.
  - Small or negative gap with LOW absolute r on both: the model isn't
    fitting the training data well either -- UNDERFITTING signature
    (headroom for more capacity/training/better features), not a
    generalization problem.
""")


if __name__ == '__main__':
    main()
