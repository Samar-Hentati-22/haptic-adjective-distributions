"""
Pipeline v3 — grouped bar charts (actual vs. predicted expected rating) per
adjective, similar in spirit to Awan & Jeon's Figure 8, adapted for unipolar
1-7 ratings (their figure is bipolar). One black/colored bar pair per
object, objects sorted by true rating ascending.

Reads <repo>/outputs/train/results/predictions_vs_truth.csv (the
unweighted, primary result). 85 objects x 2 bars = 170 bars per adjective
panel -- too dense for an 18-panel grid, so this splits into 4 figures of
4-5 adjectives each, one full-width subplot per adjective (not a compact
grid), so each panel has enough horizontal room for 85 object-pairs.
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402
# Training run folder written by model/src/train.py (default --results-dir).
RESULTS_DIR = os.path.join(paths.OUTPUTS_ROOT, 'train', 'results')
CSV_PATH = os.path.join(RESULTS_DIR, 'predictions_vs_truth.csv')
OUT_PREFIX = os.path.join(paths.output_dir(__file__), 'actual_vs_predicted_bars')

ADJECTIVES = [
    'hard', 'soft', 'rough', 'smooth', 'bumpy', 'flat',
    'sticky', 'slippery', 'cold', 'warm', 'wet', 'dry',
    'compressible', 'firm', 'fuzzy', 'scratchy', 'pleasant', 'irritating',
]

# 18 adjectives split into groups of 5 -- last group has 3 to total 18.
GROUPS = [ADJECTIVES[i:i + 5] for i in range(0, len(ADJECTIVES), 5)]

TRUE_COLOR = '#0b0b0b'      # primary ink -- "actual" is the reference, not a data series
PRED_COLOR = '#2a78d6'      # categorical slot 1 (matches this pipeline's other plots)
INK = '#0b0b0b'
SECONDARY = '#52514e'
MUTED = '#898781'
GRID_COLOR = '#e1e0d9'
BASELINE_COLOR = '#c3c2b7'
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


def draw_adjective(ax, true_vals, pred_vals, adj):
    order = np.argsort(true_vals)
    true_sorted = np.asarray(true_vals)[order]
    pred_sorted = np.asarray(pred_vals)[order]
    n = len(true_sorted)
    x = np.arange(n)
    w = 0.42

    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.grid(True, axis='y', color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.axhline(1.0, color=BASELINE_COLOR, linewidth=0.8, zorder=1)

    ax.bar(x - w / 2, true_sorted, width=w, color=TRUE_COLOR, zorder=2)
    ax.bar(x + w / 2, pred_sorted, width=w, color=PRED_COLOR, zorder=2)

    ax.set_xlim(-1, n)
    ax.set_ylim(1, 7)
    ax.set_yticks(range(1, 8))
    ax.tick_params(axis='y', colors=MUTED, labelsize=8)
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    r = np.corrcoef(true_sorted, pred_sorted)[0, 1]
    ax.text(0.005, 1.02, adj, transform=ax.transAxes, fontsize=12, color=INK,
            va='bottom', ha='left', fontweight='medium')
    ax.text(0.995, 1.02, f"r = {r:.2f}", transform=ax.transAxes, fontsize=9,
            color=SECONDARY, va='bottom', ha='right')


def main():
    by_adj = load_data()
    os.makedirs(os.path.dirname(OUT_PREFIX), exist_ok=True)

    legend_handles = [
        Patch(facecolor=TRUE_COLOR, label='Actual (true expected rating)'),
        Patch(facecolor=PRED_COLOR, label='Predicted (expected rating)'),
    ]

    saved_paths = []
    for gi, group in enumerate(GROUPS, start=1):
        n_rows = len(group)
        fig, axes = plt.subplots(n_rows, 1, figsize=(20, 2.6 * n_rows), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        if n_rows == 1:
            axes = [axes]

        for ax, adj in zip(axes, group):
            d = by_adj[adj]
            draw_adjective(ax, d['true'], d['pred'], adj)

        axes[-1].set_xlabel('Objects, sorted by true expected rating (ascending) →',
                             fontsize=10, color=SECONDARY)

        fig.legend(handles=legend_handles, loc='upper center', ncol=2,
                   bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=10,
                   labelcolor=INK)
        fig.suptitle(f'Actual vs. Predicted Expected Ratings — pipeline_v3 '
                     f'(group {gi}/{len(GROUPS)})',
                     fontsize=14, color=INK, y=1.04)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = f'{OUT_PREFIX}_group{gi}.png'
        fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches='tight')
        plt.close(fig)
        saved_paths.append(out_path)
        print(f"Saved {out_path}  ({', '.join(group)})")

    print(f"\n{len(saved_paths)} files saved (18 adjectives split into groups of "
          f"{[len(g) for g in GROUPS]}).")


if __name__ == '__main__':
    main()
