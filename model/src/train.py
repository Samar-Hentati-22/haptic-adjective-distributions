"""
Pipeline v3 — training script: 5-fold cross-validation tying together
dataset.py and model.py.

Self-contained: no imports from outside model/src/ (dataset.py, model.py,
paths.py are siblings in src/; the EMD loss and expected-rating helpers below are
ported, not imported, from the old Training_Phase2/train.py and
predict_vs_truth.py).

Results go to <repo>/outputs/train/results/ by default (override with
--results-dir).

Per fold: compute_norm_stats on that fold's training objects only (this is
also where the fold's image_input_dim is determined, since dead ResNet
dims are dropped per fold -- see dataset.py), build train/test
TactileDatasetV3 sharing those stats, train a fresh TactilePerceptionModelV3
sized to that fold's image_input_dim, save the best checkpoint + a per-fold
results JSON, and append that fold's test-set predictions to a single
predictions_vs_truth.csv covering all 5 folds by the end. After all folds,
aggregate into final_results.json.
"""
import os
import csv
import json
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import make_kfold_splits, compute_norm_stats, TactileDatasetV3, GLCM_FEATURE_DIM
from model import TactilePerceptionModelV3, collate_batch
import paths

# ── Paths ────────────────────────────────────────────────────────────────
# All locations come from environment variables -- see paths.py.
HDF5_DIR = paths.hdf5_dir()
CROPPED_DIR = paths.cropped_dir()
LABELS_DIR = paths.labels_dir()
RESULTS_DIR_DEFAULT = os.path.join(paths.OUTPUTS_ROOT, 'train', 'results')
# Used only when --adjective-weighting is passed AND --results-dir is not
# explicitly overridden -- keeps the weighted run's outputs fully separate
# from the existing, already-verified results/ directory (never written to
# by a weighted run unless the user explicitly points --results-dir at it).
RESULTS_DIR_WEIGHTED_DEFAULT = os.path.join(paths.OUTPUTS_ROOT, 'train', 'results_weighted')
# precompute_features.py's cache -- see dataset.py's _load_trial_features_cached.
# Falls back to the original HDF5+ResNet+MFCC path automatically if this
# doesn't exist yet (e.g. before running precompute_features.py once).
CACHE_DIR = paths.cache_dir()

# ── Constants ────────────────────────────────────────────────────────────
ADJECTIVES = [
    'hard', 'soft', 'rough', 'smooth', 'bumpy', 'flat',
    'sticky', 'slippery', 'cold', 'warm', 'wet', 'dry',
    'compressible', 'firm', 'fuzzy', 'scratchy', 'pleasant', 'irritating',
]

NUM_OBJECTS = 85
BATCH_SIZE = 16
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
EARLY_STOP_PATIENCE = 15
SCHED_PATIENCE = 7
SCHED_FACTOR = 0.5
SEED = 42
NUM_FOLDS = 5
GRAD_CLIP = 1.0
# CPU-only feature extraction (ResNet-50/MFCC/glitch-cleaning) happens
# inside TactileDatasetV3.__getitem__ with no caching between epochs (see
# the smoke-test report) -- multiple DataLoader workers parallelize that
# real per-trial cost across processes. Not part of the old pipeline's
# hyperparameters (doesn't affect training dynamics/results, only speed),
# so free to pick independently of "same hyperparameters" fidelity.
NUM_WORKERS = 8

RATINGS = torch.arange(1, 8, dtype=torch.float32)  # (7,) bin values 1..7

# ── Class-imbalance-aware loss weighting (opt-in, --adjective-weighting) ──
# Purely additive: when the flag is off (default), train_fold's training
# loss calls the original, unmodified earth_mover_distance exactly as
# before -- same function, same code path, same numbers. These 5
# adjectives skew heavily toward one end of the 1-7 scale in this dataset
# (most objects are "not wet"/"not sticky"/room-temperature), so their
# rare high-rating objects contribute little gradient signal under a flat
# mean; inverse-frequency weighting by rating bin counteracts that for
# just these 5 columns.
TARGET_ADJECTIVES_WEIGHTED = ['wet', 'dry', 'sticky', 'cold', 'warm']
N_WEIGHT_BINS = 5  # equal-width bins across [1,7], within the requested 4-5 range


def compute_adjective_weights(train_object_ids, labels_mean, target_adjectives=None):
    """
    train_object_ids: list/array of int object indices for this fold's
    TRAINING objects only (bin edges and inverse-frequency weights are
    derived ONLY from these -- no leakage from test objects).
    labels_mean: (85, 18) float32 true mean ratings [1-7].
    target_adjectives: adjectives to compute weights for (default
    TARGET_ADJECTIVES_WEIGHTED); every other adjective is untouched by
    this function entirely.

    For each target adjective: bin training objects' true mean ratings
    into N_WEIGHT_BINS equal-width bins over [1,7], weight each bin by
    inverse frequency (rarer bin -> higher weight; a bin with zero
    training objects gets weight 0 -- no support to base a weight on),
    then rescale so the per-object weights average to 1.0 over the
    TRAINING objects (keeps this adjective's contribution to the overall
    loss on the same scale as before weighting was added, rather than
    inflating or shrinking the total loss magnitude).

    Returns {adjective: (85,) float32 array}, one weight per object index
    (0..84) -- computed for all 85 so a caller can look up any object's
    weight uniformly, but only ever meaningfully used for objects that
    appear in a training batch (test objects aren't used in the training
    loss this feeds).
    """
    if target_adjectives is None:
        target_adjectives = TARGET_ADJECTIVES_WEIGHTED

    train_object_ids = np.asarray(train_object_ids)
    bin_edges = np.linspace(1.0, 7.0, N_WEIGHT_BINS + 1)

    weights = {}
    for adj in target_adjectives:
        adj_idx = ADJECTIVES.index(adj)
        train_vals = labels_mean[train_object_ids, adj_idx]  # (n_train,)

        bin_idx_train = np.clip(np.digitize(train_vals, bin_edges[1:-1]), 0, N_WEIGHT_BINS - 1)
        bin_counts = np.bincount(bin_idx_train, minlength=N_WEIGHT_BINS)

        inv_freq = np.zeros(N_WEIGHT_BINS, dtype=np.float64)
        nonzero = bin_counts > 0
        inv_freq[nonzero] = 1.0 / bin_counts[nonzero]

        mean_w = inv_freq[bin_idx_train].mean()
        if mean_w > 0:
            inv_freq = inv_freq / mean_w

        all_vals = labels_mean[:, adj_idx]  # (85,)
        bin_idx_all = np.clip(np.digitize(all_vals, bin_edges[1:-1]), 0, N_WEIGHT_BINS - 1)
        weights[adj] = inv_freq[bin_idx_all].astype(np.float32)  # (85,)

    return weights


def build_sample_weight_matrix(obj_indices, adjective_weights, device):
    """
    obj_indices: (B,) tensor of object indices for one batch.
    adjective_weights: dict from compute_adjective_weights.

    Returns (B, 18) float32 tensor: exactly 1.0 for every non-target
    adjective column (so multiplying by it is an exact no-op, see
    weighted_earth_mover_distance), and that adjective's per-object weight
    in the target-adjective columns.
    """
    B = obj_indices.shape[0]
    w = torch.ones(B, len(ADJECTIVES), dtype=torch.float32)
    obj_indices_np = obj_indices.cpu().numpy()
    for adj, weights_arr in adjective_weights.items():
        col = ADJECTIVES.index(adj)
        w[:, col] = torch.from_numpy(weights_arr[obj_indices_np])
    return w.to(device)


def weighted_earth_mover_distance(pred: torch.Tensor, target: torch.Tensor,
                                    sample_weights: torch.Tensor) -> torch.Tensor:
    """
    Like earth_mover_distance, but each (sample, adjective)'s EMD is
    multiplied by sample_weights (B, 18) before the final mean.

    The per-(sample, adjective) EMD values themselves (before this
    function's final .mean()) are bit-identical to what earth_mover_distance
    computes internally -- multiplying a non-target column by exactly 1.0
    is an exact IEEE-754 no-op, so weighting 5 columns never touches the
    other 13's own values (verified in the smoke-test report). The final
    SCALAR this function returns is NOT guaranteed bit-identical to
    earth_mover_distance's scalar even when all weights are 1.0, though:
    this does a two-step reduction (mean over 7 bins, then mean over B*18),
    while earth_mover_distance does one flat mean over all B*18*7 elements
    -- mathematically equal, but floating-point addition is not associative,
    so the two reduction orders differ by ~1e-8. This is irrelevant in
    practice: train_fold only calls this function when --adjective-weighting
    is on; with the flag off it calls the untouched earth_mover_distance
    directly, so the default path's reproducibility is unaffected.
    """
    cdf_pred = torch.cumsum(pred, dim=2)
    cdf_target = torch.cumsum(target, dim=2)
    per_sample_adj_emd = torch.abs(cdf_pred - cdf_target).mean(dim=2)  # (B, 18)
    return (per_sample_adj_emd * sample_weights).mean()


# ── Utilities ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=SEED,
                         help=f'Random seed controlling the k-fold split and per-fold model '
                              f'init (default: {SEED})')
    parser.add_argument('--results-dir', type=str, default=None,
                         help=f'Directory to write checkpoints/results to (default: '
                              f'{RESULTS_DIR_DEFAULT}, or {RESULTS_DIR_WEIGHTED_DEFAULT} if '
                              f'--adjective-weighting is set)')
    parser.add_argument('--smoke-test', action='store_true',
                         help='Run just 2 epochs on fold 1 only, writing to '
                              '<results-dir>/smoke_test/, to verify the full loop works '
                              'end-to-end before committing to a full 5-fold run.')
    parser.add_argument('--adjective-weighting', action='store_true',
                         help=f'Opt-in: apply inverse-frequency class weighting to the '
                              f'training loss for {TARGET_ADJECTIVES_WEIGHTED} only (see '
                              f'compute_adjective_weights). Off by default -- the default '
                              f'behavior is unchanged and uses the original unweighted '
                              f'earth_mover_distance exactly as before this flag existed.')
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def earth_mover_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean |CDF_pred - CDF_target| over all adjectives, ratings, and batch.
    Ported verbatim from Training_Phase2/train.py."""
    cdf_pred = torch.cumsum(pred, dim=2)
    cdf_target = torch.cumsum(target, dim=2)
    return torch.mean(torch.abs(cdf_pred - cdf_target))


def expected_rating(dist: torch.Tensor) -> torch.Tensor:
    """(..., 7) probability distribution -> (...,) probability-weighted mean
    rating [1-7]. Ported from Training_Phase2/predict_vs_truth.py."""
    return (dist * RATINGS.view(*([1] * (dist.dim() - 1)), 7)).sum(dim=-1)


# ── Evaluation (object-level aggregation) ───────────────────────────────

def evaluate(model, loader, device):
    """
    Run inference, average predictions per object (a trial-level model is
    evaluated at the object level, matching the labels), compute metrics.
    Returns: (mean_emd, mean_mae, per_adj_emd, per_adj_mae)
    """
    model.eval()

    obj_preds = {}    # obj_idx -> [tensor(18,7), ...]
    obj_labels = {}   # obj_idx -> (label_dist(18,7), label_mean(18,))

    with torch.no_grad():
        for batch in loader:
            imu = batch['imu'].to(device)
            pressure_ap = batch['pressure_ap'].to(device)
            image = batch['image'].to(device)
            audio = batch['audio'].to(device)
            has_image = batch['has_image'].to(device)
            has_audio = batch['has_audio'].to(device)
            obj_indices = batch['obj_idx']

            pred = model(imu, pressure_ap, image, audio, has_image, has_audio)  # (B, 18, 7)

            for i in range(pred.size(0)):
                oi = obj_indices[i].item()
                if oi not in obj_preds:
                    obj_preds[oi] = []
                    obj_labels[oi] = (batch['label_dist'][i], batch['label_mean'][i])
                obj_preds[oi].append(pred[i].cpu())

    emd_per_obj = []
    mae_per_obj = []

    for oi in sorted(obj_preds.keys()):
        avg_pred = torch.stack(obj_preds[oi]).mean(dim=0, keepdim=True)  # (1, 18, 7)
        true_dist = obj_labels[oi][0].unsqueeze(0)                       # (1, 18, 7)
        true_mean = obj_labels[oi][1]                                    # (18,)

        cdf_p = torch.cumsum(avg_pred, dim=2)
        cdf_t = torch.cumsum(true_dist, dim=2)
        emd_adj = torch.abs(cdf_p - cdf_t).mean(dim=2).squeeze(0)  # (18,)

        pred_m = expected_rating(avg_pred.squeeze(0))  # (18,)
        mae_adj = torch.abs(pred_m - true_mean)         # (18,)

        emd_per_obj.append(emd_adj)
        mae_per_obj.append(mae_adj)

    emd_all = torch.stack(emd_per_obj).mean(dim=0)  # (18,)
    mae_all = torch.stack(mae_per_obj).mean(dim=0)  # (18,)

    return (emd_all.mean().item(), mae_all.mean().item(), emd_all.tolist(), mae_all.tolist())


def write_predictions_csv(model, loader, object_ids, labels_dist, fold_num, results_dir, device):
    """
    Per-object, per-adjective true vs. predicted expected rating for this
    fold's test set, APPENDED to results_dir/predictions_vs_truth.csv
    (header written only if the file doesn't exist yet) -- so the CSV
    accumulates across all 5 folds by the end of the run.
    """
    model.eval()
    obj_preds = {}
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

    rows = []
    for oi in sorted(obj_preds.keys()):
        avg_pred_dist = torch.stack(obj_preds[oi]).mean(dim=0)              # (18, 7)
        true_dist = torch.from_numpy(labels_dist[oi].astype(np.float32))    # (18, 7)

        pred_expected = expected_rating(avg_pred_dist)  # (18,)
        true_expected = expected_rating(true_dist)      # (18,)

        obj_id = str(object_ids[oi])
        for adj_idx, adj in enumerate(ADJECTIVES):
            rows.append({
                'true_expected_rating': true_expected[adj_idx].item(),
                'predicted_expected_rating': pred_expected[adj_idx].item(),
                'adjective': adj,
                'object_id': obj_id,
                'fold': fold_num,
            })

    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'predictions_vs_truth.csv')
    file_exists = os.path.isfile(csv_path)
    fieldnames = ['true_expected_rating', 'predicted_expected_rating', 'adjective',
                  'object_id', 'fold']

    with open(csv_path, 'a' if file_exists else 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    print(f"  Appended {len(rows)} rows -> {csv_path}")


# ── Per-fold training ────────────────────────────────────────────────────

def train_fold(fold_idx, train_indices, test_indices, object_ids, labels_dist, labels_mean,
               results_dir, device, num_epochs=NUM_EPOCHS, adjective_weighting=False):
    print(f"\n{'=' * 65}")
    print(f"  FOLD {fold_idx + 1}/{NUM_FOLDS}  |  "
          f"train objects: {len(train_indices)}  |  test objects: {len(test_indices)}")
    print(f"{'=' * 65}")

    cache_dir = CACHE_DIR if os.path.isdir(CACHE_DIR) else None
    if cache_dir:
        print(f"  Feature cache: using {cache_dir}")
    else:
        print("  Feature cache: NOT FOUND, falling back to live HDF5+ResNet+MFCC extraction "
              "(run precompute_features.py once to speed this up)")

    print("  Computing per-channel normalization stats from training objects...")
    t_stats0 = time.time()
    norm_stats = compute_norm_stats(train_indices, object_ids, HDF5_DIR, CROPPED_DIR,
                                     device=str(device), cache_dir=cache_dir)
    print(f"  norm_stats computed in {time.time() - t_stats0:.1f}s")

    image_input_dim = len(norm_stats['image_resnet_alive_dims']) + GLCM_FEATURE_DIM
    print(f"  image_input_dim for this fold: {image_input_dim}")

    adjective_weights = None
    if adjective_weighting:
        adjective_weights = compute_adjective_weights(train_indices, labels_mean)
        print(f"  Adjective weighting ON for {TARGET_ADJECTIVES_WEIGHTED} "
              f"({N_WEIGHT_BINS} bins, computed from this fold's training objects only)")

    train_ds = TactileDatasetV3(train_indices, object_ids, labels_dist, labels_mean, norm_stats,
                                 hdf5_dir=HDF5_DIR, cropped_dir=CROPPED_DIR, device=str(device),
                                 cache_dir=cache_dir)
    test_ds = TactileDatasetV3(test_indices, object_ids, labels_dist, labels_mean, norm_stats,
                                hdf5_dir=HDF5_DIR, cropped_dir=CROPPED_DIR, device=str(device),
                                cache_dir=cache_dir)

    print(f"  Train trials: {len(train_ds):4d}  |  Test trials: {len(test_ds):4d}")
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError(f"Fold {fold_idx + 1}: empty dataset -- check paths.")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, collate_fn=collate_batch)

    model = TactilePerceptionModelV3(image_input_dim=image_input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=SCHED_PATIENCE, factor=SCHED_FACTOR
    )

    best_val_emd = float('inf')
    best_epoch = 0
    patience_count = 0
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(results_dir, f'fold_{fold_idx + 1}_best.pt')

    fold_t0 = time.time()

    for epoch in range(1, num_epochs + 1):
        ep_t0 = time.time()

        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            imu = batch['imu'].to(device)
            pressure_ap = batch['pressure_ap'].to(device)
            image = batch['image'].to(device)
            audio = batch['audio'].to(device)
            has_image = batch['has_image'].to(device)
            has_audio = batch['has_audio'].to(device)
            label_dist = batch['label_dist'].to(device)

            optimizer.zero_grad()
            pred = model(imu, pressure_ap, image, audio, has_image, has_audio)
            if adjective_weighting:
                sample_weights = build_sample_weight_matrix(batch['obj_idx'], adjective_weights,
                                                              device)
                loss = weighted_earth_mover_distance(pred, label_dist, sample_weights)
            else:
                loss = earth_mover_distance(pred, label_dist)  # unchanged from before this flag existed
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches

        val_emd, val_mae, _, _ = evaluate(model, test_loader, device)
        scheduler.step(val_emd)

        ep_elapsed = time.time() - ep_t0
        fold_elapsed = time.time() - fold_t0
        lr_now = optimizer.param_groups[0]['lr']

        print(
            f"  Fold {fold_idx + 1} | Epoch {epoch:3d}/{num_epochs} | "
            f"Train EMD: {train_loss:.4f} | Val EMD: {val_emd:.4f} | "
            f"Val MAE: {val_mae:.4f} | LR: {lr_now:.1e} | "
            f"Ep: {ep_elapsed:.1f}s | Total: {fold_elapsed / 60:.1f}min"
        )

        if val_emd < best_val_emd:
            best_val_emd = val_emd
            best_epoch = epoch
            patience_count = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_emd': val_emd,
                'fold': fold_idx + 1,
                'image_input_dim': image_input_dim,
            }, ckpt_path)
        else:
            patience_count += 1
            if patience_count >= EARLY_STOP_PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}. "
                      f"Best epoch: {best_epoch}  (val EMD {best_val_emd:.4f})")
                break

    # ── Final evaluation with best checkpoint ────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    final_emd, final_mae, per_adj_emd, per_adj_mae = evaluate(model, test_loader, device)

    print(f"\n  Fold {fold_idx + 1} final  (best epoch {best_epoch}):")
    print(f"  Mean EMD: {final_emd:.4f}   Mean MAE: {final_mae:.4f}")
    print(f"\n  {'Adjective':15s}  {'EMD':>7s}  {'MAE':>7s}")
    print(f"  {'-' * 33}")
    for i, adj in enumerate(ADJECTIVES):
        print(f"  {adj:15s}  {per_adj_emd[i]:7.4f}  {per_adj_mae[i]:7.4f}")

    results = {
        'fold': fold_idx + 1,
        'best_epoch': best_epoch,
        'best_val_emd': best_val_emd,
        'mean_emd': final_emd,
        'mean_mae': final_mae,
        'per_adjective_emd': {adj: per_adj_emd[i] for i, adj in enumerate(ADJECTIVES)},
        'per_adjective_mae': {adj: per_adj_mae[i] for i, adj in enumerate(ADJECTIVES)},
        'train_objects': train_indices,
        'test_objects': test_indices,
        'image_input_dim': image_input_dim,
        'adjective_weighting': adjective_weighting,
    }

    out_path = os.path.join(results_dir, f'fold_{fold_idx + 1}_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results -> {out_path}")

    write_predictions_csv(model, test_loader, object_ids, labels_dist, fold_idx + 1,
                           results_dir, device)

    return results


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    seed = args.seed

    if args.results_dir is not None:
        results_dir = args.results_dir
    else:
        results_dir = RESULTS_DIR_WEIGHTED_DEFAULT if args.adjective_weighting else RESULTS_DIR_DEFAULT
    if args.smoke_test:
        results_dir = os.path.join(results_dir, 'smoke_test')

    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'=' * 65}")
    print(f"  Pipeline v3 training{'  [SMOKE TEST]' if args.smoke_test else ''}"
          f"{'  [ADJECTIVE WEIGHTING]' if args.adjective_weighting else ''}")
    print(f"  Device      : {device}")
    print(f"  Seed        : {seed}")
    print(f"  Results dir : {results_dir}")
    print(f"  Epochs/fold : {2 if args.smoke_test else NUM_EPOCHS}")
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  Folds       : {1 if args.smoke_test else NUM_FOLDS}")
    print(f"  Adj weight  : {args.adjective_weighting}"
          f"{'  (' + ', '.join(TARGET_ADJECTIVES_WEIGHTED) + ')' if args.adjective_weighting else ''}")
    print(f"{'=' * 65}")

    print("\nLoading labels...")
    labels_dist = np.load(os.path.join(LABELS_DIR, 'labels_phase2.npy'))       # (85,18,7)
    labels_mean = np.load(os.path.join(LABELS_DIR, 'labels_mean_phase2.npy'))  # (85,18)
    object_ids = np.load(os.path.join(LABELS_DIR, 'object_ids_phase2.npy'),
                          allow_pickle=True)                                   # (85,)
    print(f"  labels_dist : {labels_dist.shape}  {labels_dist.dtype}")
    print(f"  labels_mean : {labels_mean.shape}  {labels_mean.dtype}")
    print(f"  object_ids  : {object_ids.shape}   first={object_ids[0]}")

    splits = make_kfold_splits(object_ids=object_ids, n_splits=NUM_FOLDS, seed=seed)

    all_results = []
    global_t0 = time.time()

    n_folds_to_run = 1 if args.smoke_test else NUM_FOLDS
    n_epochs_this_run = 2 if args.smoke_test else NUM_EPOCHS

    for fold_idx, (train_indices, test_indices) in enumerate(splits[:n_folds_to_run]):
        set_seed(seed + fold_idx)   # different seed per fold for model init
        results = train_fold(
            fold_idx, train_indices, test_indices, object_ids, labels_dist, labels_mean,
            results_dir, device, num_epochs=n_epochs_this_run,
            adjective_weighting=args.adjective_weighting,
        )
        all_results.append(results)

    if args.smoke_test:
        print(f"\n{'=' * 65}")
        print(f"  SMOKE TEST COMPLETE  —  total time {(time.time() - global_t0) / 60:.1f} min")
        print(f"{'=' * 65}")
        return

    # ── Aggregate across folds ────────────────────────────────────────────
    total_time = (time.time() - global_t0) / 60
    print(f"\n{'=' * 65}")
    print(f"  FINAL RESULTS  (5-fold average)  —  total time {total_time:.1f} min")
    print(f"{'=' * 65}")

    mean_emd_cv = np.mean([r['mean_emd'] for r in all_results])
    mean_mae_cv = np.mean([r['mean_mae'] for r in all_results])
    print(f"  Mean EMD: {mean_emd_cv:.4f}   Mean MAE: {mean_mae_cv:.4f}\n")

    print(f"  {'Adjective':15s}  {'EMD':>7s}  {'MAE':>7s}")
    print(f"  {'-' * 33}")
    per_adj_emd_cv, per_adj_mae_cv = {}, {}
    for adj in ADJECTIVES:
        emd_v = float(np.mean([r['per_adjective_emd'][adj] for r in all_results]))
        mae_v = float(np.mean([r['per_adjective_mae'][adj] for r in all_results]))
        per_adj_emd_cv[adj] = emd_v
        per_adj_mae_cv[adj] = mae_v
        print(f"  {adj:15s}  {emd_v:7.4f}  {mae_v:7.4f}")

    final = {
        'mean_emd': float(mean_emd_cv),
        'mean_mae': float(mean_mae_cv),
        'per_adjective_emd': per_adj_emd_cv,
        'per_adjective_mae': per_adj_mae_cv,
        'fold_results': all_results,
    }
    os.makedirs(results_dir, exist_ok=True)
    final_path = os.path.join(results_dir, 'final_results.json')
    with open(final_path, 'w') as f:
        json.dump(final, f, indent=2)
    print(f"\n  Final results -> {final_path}")


if __name__ == '__main__':
    main()
