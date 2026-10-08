"""
Pipeline v3 — 18 per-adjective scatter plots (predicted vs. true expected
rating) as a single 4x5 grid, matching the visual format of
Training_Phase2/plot_predicted_vs_true_grid.py's combined grid, but for
pipeline_v3's results and adding Spearman rho alongside Pearson r (the old
grid's per-subplot annotation used EMD, not r/rho, since it had an 'emd'
column pipeline_v3's predictions_vs_truth.csv doesn't include).

Reads <repo>/outputs/train/results/predictions_vs_truth.csv (all
5 folds appended by train.py, one row per object x adjective across the
85 objects). Writes predicted_vs_true_grid.png.
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402
# Training run folder written by model/src/train.py (default --results-dir).
RESULTS_DIR = os.path.join(paths.OUTPUTS_ROOT, 'train', 'results')
CSV_PATH = os.path.join(RESULTS_DIR, 'predictions_vs_truth.csv')
GRID_OUT = os.path.join(paths.output_dir(__file__), 'predicted_vs_true_grid.png')

ADJECTIVES = [
    'hard', 'soft', 'rough', 'smooth', 'bumpy', 'flat',
    'sticky', 'slippery', 'cold', 'warm', 'wet', 'dry',
    'compressible', 'firm', 'fuzzy', 'scratchy', 'pleasant', 'irritating',
]

N_ROWS, N_COLS = 4, 5   # 20 slots, 18 used, 2 empty

POINT_COLOR = '#2a78d6'
INK = '#0b0b0b'
SECONDARY = '#52514e'
MUTED = '#898781'
GRID_COLOR = '#e1e0d9'
DIAGONAL_COL = '#898781'
SURFACE = '#fcfcfb'


def load_data():
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    by_adj = defaultdict(lambda: {'true': [], 'pred': []})
    for r in rows:
        d = by_adj[r['adjective']]
        d['true'].append(float(r['true_expected_rating']))
        d['pred'].append(float(r['predicted_expected_rating']))
    return by_adj


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.set_xlim(1, 7)
    ax.set_ylim(1, 7)
    ax.set_xticks(range(1, 8))
    ax.set_yticks(range(1, 8))
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def draw_adjective(ax, true_vals, pred_vals, adj, title_fontsize=11):
    style_axes(ax)
    ax.plot([1, 7], [1, 7], linestyle='--', linewidth=1.2, color=DIAGONAL_COL, zorder=1)
    ax.scatter(true_vals, pred_vals, s=22, alpha=0.6, color=POINT_COLOR,
               edgecolors='none', zorder=2)

    r = np.corrcoef(true_vals, pred_vals)[0, 1]
    rho = spearmanr(true_vals, pred_vals).statistic
    annotation = f"r = {r:.2f}\nρ = {rho:.2f}"
    ax.text(
        0.04, 0.96, annotation,
        transform=ax.transAxes, fontsize=8.5, color=SECONDARY,
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.25', facecolor=SURFACE,
                  edgecolor=GRID_COLOR, linewidth=0.8),
    )

    ax.set_title(adj, fontsize=title_fontsize, color=INK, pad=6)
    return r, rho


def main():
    by_adj = load_data()

    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(20, 16), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    axes_flat = axes.flatten()

    for i, adj in enumerate(ADJECTIVES):
        d = by_adj[adj]
        true_vals = np.array(d['true'])
        pred_vals = np.array(d['pred'])
        r, rho = draw_adjective(axes_flat[i], true_vals, pred_vals, adj)
        print(f"  {adj:15s}  r = {r:.3f}  rho = {rho:.3f}  (n={len(true_vals)})")

    for j in range(len(ADJECTIVES), N_ROWS * N_COLS):
        axes_flat[j].axis('off')

    fig.suptitle(
        'Predicted vs. True Expected Ratings — pipeline_v3, 85 Objects',
        fontsize=16, color=INK, y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(GRID_OUT, dpi=150, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved grid -> {GRID_OUT}")


if __name__ == '__main__':
    main()
