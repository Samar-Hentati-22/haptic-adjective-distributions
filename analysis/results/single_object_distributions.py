"""
Pipeline v3 — illustrate that the model outputs full (18, 7) predicted
probability distributions per object, not point estimates.

Picks one object (obj_48, fold 1 test set -- mid-range true ratings on
hard/soft/rough/smooth/bumpy with low prediction error, per
results/predictions_vs_truth.csv), reruns that fold's best checkpoint on
just this object's trials (same methodology as
Training_Phase2/predict_vs_truth.py / src/train.py's evaluate(): average
predicted (18,7) dist across the object's trials), and plots:

  Figure 1: predicted distribution only, one bar-chart subplot per adjective.
  Figure 2: predicted vs. true distribution, grouped bars per adjective.
"""
import json
import os

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model', 'src'))  # <repo>/model/src
from dataset import compute_norm_stats, TactileDatasetV3  # noqa: E402
from model import TactilePerceptionModelV3, collate_batch  # noqa: E402
import paths  # noqa: E402


HDF5_DIR = paths.hdf5_dir()
CROPPED_DIR = paths.cropped_dir()
LABELS_DIR = paths.labels_dir()
# Training run folder written by model/src/train.py (default --results-dir).
RESULTS_DIR = os.path.join(paths.OUTPUTS_ROOT, 'train', 'results')
CACHE_DIR = paths.cache_dir()

ADJECTIVES = [
    'hard', 'soft', 'rough', 'smooth', 'bumpy', 'flat',
    'sticky', 'slippery', 'cold', 'warm', 'wet', 'dry',
    'compressible', 'firm', 'fuzzy', 'scratchy', 'pleasant', 'irritating',
]

OBJECT_ID = 'obj_48'
FOLD_NUM = 1
PLOT_ADJECTIVES = ['hard', 'soft', 'rough', 'smooth', 'wet']

FIG1_OUT = os.path.join(paths.output_dir(__file__), 'single_object_predicted_only.png')
FIG2_OUT = os.path.join(paths.output_dir(__file__), 'single_object_predicted_vs_true.png')

PRED_COLOR = '#2a78d6'
TRUE_COLOR = '#4d4d4d'
INK = '#0b0b0b'
MUTED = '#898781'
GRID_COLOR = '#e1e0d9'
SURFACE = '#fcfcfb'
BINS = np.arange(1, 8)


def load_fold_and_object():
    object_ids = np.load(os.path.join(LABELS_DIR, 'object_ids_phase2.npy'), allow_pickle=True)
    labels_dist = np.load(os.path.join(LABELS_DIR, 'labels_phase2.npy'))       # (85,18,7)
    labels_mean = np.load(os.path.join(LABELS_DIR, 'labels_mean_phase2.npy'))  # (85,18)

    obj_idx = int(np.where(object_ids.astype(str) == OBJECT_ID)[0][0])

    with open(os.path.join(RESULTS_DIR, f'fold_{FOLD_NUM}_results.json')) as f:
        fold_info = json.load(f)
    assert obj_idx in fold_info['test_objects'], (
        f"{OBJECT_ID} (idx {obj_idx}) not in fold {FOLD_NUM}'s test_objects")

    return object_ids, labels_dist, labels_mean, obj_idx, fold_info


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    object_ids, labels_dist, labels_mean, obj_idx, fold_info = load_fold_and_object()

    cache_dir = CACHE_DIR if os.path.isdir(CACHE_DIR) else None
    print(f"Computing norm_stats from fold {FOLD_NUM}'s {len(fold_info['train_objects'])} "
          f"training objects...")
    norm_stats = compute_norm_stats(fold_info['train_objects'], object_ids, HDF5_DIR, CROPPED_DIR,
                                     device=str(device), cache_dir=cache_dir)

    ds = TactileDatasetV3([obj_idx], object_ids, labels_dist, labels_mean, norm_stats,
                           hdf5_dir=HDF5_DIR, cropped_dir=CROPPED_DIR, device=str(device),
                           cache_dir=cache_dir)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_batch)

    image_input_dim = fold_info['image_input_dim']
    model = TactilePerceptionModelV3(image_input_dim=image_input_dim).to(device)
    ckpt = torch.load(os.path.join(RESULTS_DIR, f'fold_{FOLD_NUM}_best.pt'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    preds = []
    with torch.no_grad():
        for batch in loader:
            imu = batch['imu'].to(device)
            pressure_ap = batch['pressure_ap'].to(device)
            image = batch['image'].to(device)
            audio = batch['audio'].to(device)
            has_image = batch['has_image'].to(device)
            has_audio = batch['has_audio'].to(device)
            pred = model(imu, pressure_ap, image, audio, has_image, has_audio)  # (B, 18, 7)
            preds.append(pred.cpu())
    pred_dist = torch.cat(preds, dim=0).mean(dim=0).numpy()  # (18, 7), averaged over trials
    true_dist = labels_dist[obj_idx]                          # (18, 7)

    print(f"Object {OBJECT_ID} (idx {obj_idx}), fold {FOLD_NUM}, "
          f"{len(ds)} trials averaged.")

    # ── Figure 1: predicted distribution only ──────────────────────────────
    n = len(PLOT_ADJECTIVES)
    fig1, axes1 = plt.subplots(1, n, figsize=(3.2 * n, 4), dpi=150, sharey=True)
    fig1.patch.set_facecolor(SURFACE)
    for ax, adj in zip(axes1, PLOT_ADJECTIVES):
        adj_idx = ADJECTIVES.index(adj)
        ax.set_facecolor(SURFACE)
        ax.bar(BINS, pred_dist[adj_idx], color=PRED_COLOR, width=0.7, zorder=2)
        ax.set_xticks(BINS)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Rating bin')
        ax.set_title(adj, fontsize=12, color=INK)
        ax.grid(True, axis='y', color=GRID_COLOR, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.tick_params(colors=MUTED)
    axes1[0].set_ylabel('Predicted probability')
    fig1.suptitle(f'Predicted rating distribution — {OBJECT_ID}', fontsize=14, color=INK)
    fig1.tight_layout(rect=[0, 0, 1, 0.94])
    fig1.savefig(FIG1_OUT, dpi=150, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig1)

    # ── Figure 2: predicted vs. true, grouped bars ─────────────────────────
    fig2, axes2 = plt.subplots(1, n, figsize=(3.2 * n, 4), dpi=150, sharey=True)
    fig2.patch.set_facecolor(SURFACE)
    width = 0.38
    for ax, adj in zip(axes2, PLOT_ADJECTIVES):
        adj_idx = ADJECTIVES.index(adj)
        ax.set_facecolor(SURFACE)
        ax.bar(BINS - width / 2, true_dist[adj_idx], width=width, color=TRUE_COLOR,
               label='True', zorder=2)
        ax.bar(BINS + width / 2, pred_dist[adj_idx], width=width, color=PRED_COLOR,
               label='Predicted', zorder=2)
        ax.set_xticks(BINS)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Rating bin')
        ax.set_title(adj, fontsize=12, color=INK)
        ax.grid(True, axis='y', color=GRID_COLOR, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.tick_params(colors=MUTED)
    axes2[0].set_ylabel('Probability')
    axes2[-1].legend(loc='upper right', fontsize=9)
    fig2.suptitle(f'Predicted vs. true rating distribution — {OBJECT_ID}', fontsize=14, color=INK)
    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    fig2.savefig(FIG2_OUT, dpi=150, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig2)

    print(f"Saved {FIG1_OUT}")
    print(f"Saved {FIG2_OUT}")


if __name__ == '__main__':
    main()
