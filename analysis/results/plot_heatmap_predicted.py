"""
Pipeline v3 — model-prediction heatmaps that mirror the human-rating heatmaps
in human_study/plots/:

  Session 1 (obj_01-obj_45): plot_heatmap_named.py  -> heatmap_mean_ratings_named.png
  Session 2 (obj_46-obj_85): plot_session2_named.py -> heatmap_mean_ratings_session2_named.png

Each session's figures copy THAT session's human-heatmap style (the two
human scripts differ: session 1 uses YlOrRd at 20x18 / dpi 150, session 2 a
lightened BuPu at 22x17 / dpi 200 with larger bold annotations). Row order
(objects by numeric id), column order (ADJECTIVES), object names, fixed 1-7
scale and 1-decimal rounding of the plotted values are identical to the
human scripts. The style code is copied, not imported (the human scripts
have hardcoded paths and run on import).

Reads predictions_vs_truth.csv written by model/src/train.py: one row per
(object, adjective) with the out-of-fold expected rating predicted by the
fold model that did not train on that object. Objects are identified by id
string ('obj_01'...), matched against object_ids_phase2.npy. Session
membership is by object number (session 1 = obj_01-obj_45, session 2 =
obj_46-obj_85), the same split as session1_final.csv / session2_final.csv.

Writes, per session s in {1, 2}:
  heatmap_predicted_session{s}.png  predicted expected rating (1-7)
  heatmap_diff_session{s}.png       predicted minus human mean, diverging
                                    colormap centred on 0, the same symmetric
                                    limits for both sessions (max |diff| over
                                    all 85 objects, rounded up to 0.5)

Before plotting, the CSV's true_expected_rating is checked against
labels_mean_phase2.npy (TACTILE_LABELS_DIR, aggregated per-object means --
no participant-level files needed); the script aborts if any cell differs
by more than --max-true-diff.
"""
import argparse
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402

# Object-number range of each human-study session (session1_final.csv /
# session2_final.csv contain exactly these objects).
SESSION_OBJECT_RANGES = {1: (1, 45), 2: (46, 85)}

ADJECTIVES = [
    "hard", "soft", "rough", "smooth", "bumpy", "flat",
    "sticky", "slippery", "cold", "warm", "wet", "dry",
    "compressible", "firm", "fuzzy", "scratchy", "pleasant", "irritating",
]

# Copied verbatim from the human *_named.py scripts.
OBJECT_NAMES = {
    "obj_01": "Wooden spoon",
    "obj_02": "Plush toy",
    "obj_03": "Latex cleaning glove",
    "obj_04": "Hand-woven palm leaf mat",
    "obj_05": "Microfiber cloth",
    "obj_06": "Plastic ruler",
    "obj_07": "Earmuffs",
    "obj_08": "Satin scarf",
    "obj_09": "Melamine foam sponge",
    "obj_10": "Abrasive foam pad",
    "obj_11": "Terry cloth",
    "obj_12": "Ceramic tile",
    "obj_13": "Latex makeup sponge",
    "obj_14": "Silicone phone case",
    "obj_15": "Decorative garland",
    "obj_16": "Leather belt",
    "obj_17": "Corrugated cardboard",
    "obj_18": "Hook-and-loop fastener soft side",
    "obj_19": "Wet rigid polymer foam sponge",
    "obj_20": "Crocheted lace fabric",
    "obj_21": "Metal spoon",
    "obj_22": "Kraft paper",
    "obj_23": "Cellulose sponge rough side",
    "obj_24": "Smooth dish-washing glove",
    "obj_25": "Wool yarn ball",
    "obj_26": "Paper towel",
    "obj_27": "Cellulose sponge soft side",
    "obj_28": "Plastic container",
    "obj_29": "Textured dish-washing glove",
    "obj_30": "Microfiber headband",
    "obj_31": "Rinsing sponge",
    "obj_32": "Glass jar",
    "obj_33": "Textured stress ball",
    "obj_34": "Muslin fabric",
    "obj_35": "Steel wool pad",
    "obj_36": "Satin scrunchie",
    "obj_37": "Metallic feather",
    "obj_38": "Glass straw",
    "obj_39": "Pink woven fabric",
    "obj_40": "Hook-and-loop fastener rough side",
    "obj_41": "Velvet headband",
    "obj_42": "Stress ball",
    "obj_43": "Cork coaster",
    "obj_44": "Soap bar",
    "obj_45": "Rigid polymer foam sponge",
    "obj_46": "Wool beanie",
    "obj_47": "Wet wipe",
    "obj_48": "Mouse pad",
    "obj_49": "Pillow cover",
    "obj_50": "Textured glass",
    "obj_51": "Black adhesive tape adhesive side",
    "obj_52": "Polystyrene foam",
    "obj_53": "Wet sponge soft side",
    "obj_54": "Denim fabric",
    "obj_55": "Wooden heart",
    "obj_56": "Bubble wrap",
    "obj_57": "Cotton jersey fabric",
    "obj_58": "Glittered craft flower",
    "obj_59": "Microfiber floor mop pad",
    "obj_60": "Toothbrush",
    "obj_61": "Egg carton cardboard",
    "obj_62": "Hair roller",
    "obj_63": "Baking parchment paper",
    "obj_64": "Wool scarf",
    "obj_65": "Wet sponge rough side",
    "obj_66": "Craft foam sheet",
    "obj_67": "Aluminium foil",
    "obj_68": "Watch",
    "obj_69": "Cotton makeup pad",
    "obj_70": "Wet paper towel",
    "obj_71": "Cleaning brush",
    "obj_72": "Cotton ball",
    "obj_73": "Burlap tote bag",
    "obj_74": "Slimey stress ball",
    "obj_75": "Muslin fabric",
    "obj_76": "Black adhesive tape non-adhesive side",
    "obj_77": "Gift bag",
    "obj_78": "Muffin paper cup",
    "obj_79": "Sticky note non-adhesive side",
    "obj_80": "Textured silicone mold",
    "obj_81": "Wood-bristle brush",
    "obj_82": "Pencil",
    "obj_83": "Measuring tape",
    "obj_84": "Adhesive tape",
    "obj_85": "Plastic earbuds case",
}


def get_label(obj_id):
    return OBJECT_NAMES.get(obj_id, obj_id)


# Lightened BuPu, copied from plot_session2_named.py.
_bupu = plt.get_cmap("BuPu")
BUPU_LIGHT = mcolors.LinearSegmentedColormap.from_list(
    "BuPu_light", _bupu(np.linspace(0.08, 0.72, 256))
)

# Per-session styling, copied from the corresponding human script.
SESSION_STYLE = {
    1: {
        'figsize': (20, 18),
        'cmap': 'YlOrRd',
        'annot_kws': {"size": 7.5},
        'title_suffix': '',
        'title_kws': {'fontsize': 16, 'fontweight': 'bold', 'pad': 14},
        'axis_label_kws': {'fontsize': 12, 'labelpad': 8},
        'xtick_fontsize': 10,
        'ytick_fontsize': 9,
        'cbar_tick_size': None,     # session 1 leaves colorbar fonts at default
        'cbar_label_kws': None,
        'dpi': 150,
    },
    2: {
        'figsize': (22, 17),
        'cmap': BUPU_LIGHT,
        'annot_kws': {"size": 11, "weight": "bold", "color": "#1a1a1a"},
        'title_suffix': ' — Session 2',
        'title_kws': {'fontsize': 22, 'fontweight': 'bold', 'pad': 20},
        'axis_label_kws': {'fontsize': 17, 'labelpad': 12},
        'xtick_fontsize': 13,
        'ytick_fontsize': 13,
        'cbar_tick_size': 12,
        'cbar_label_kws': {'fontsize': 14, 'labelpad': 10},
        'dpi': 200,
    },
}

# Both predicted heatmaps share one palette (session 2's lightened BuPu),
# overriding session 1's YlOrRd so the two sessions read on the same colours.
PRED_CMAP = BUPU_LIGHT
DIFF_CMAP = 'RdBu_r'   # red = model over-predicts, blue = under-predicts


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--pred-csv', default=os.path.join(paths.OUTPUTS_ROOT, 'train', 'results',
                                                      'predictions_vs_truth.csv'))
    p.add_argument('--labels-dir', default=None,
                   help='Folder holding labels_mean_phase2.npy and object_ids_phase2.npy '
                        '(default: TACTILE_LABELS_DIR)')
    p.add_argument('--out-dir', default=None,
                   help='Output folder (default: <repo>/outputs/plot_heatmap_predicted/)')
    p.add_argument('--run-label', default='out-of-fold, 5-fold CV, seed 42',
                   help='Shown in the prediction title as "Model prediction (<run-label>)"')
    p.add_argument('--max-true-diff', type=float, default=0.01,
                   help='Abort if any CSV true rating differs from the human mean by more than this')
    return p.parse_args()


def load_predictions(pred_csv, object_ids_path):
    pred = pd.read_csv(pred_csv)
    if pred.duplicated(['object_id', 'adjective']).any():
        raise SystemExit(f'{pred_csv}: duplicate (object_id, adjective) rows')

    object_ids = {str(o) for o in np.load(object_ids_path, allow_pickle=True)}
    csv_ids = set(pred['object_id'].astype(str))
    if csv_ids != object_ids:
        raise SystemExit(f'Object ids in {pred_csv} do not match {object_ids_path}: '
                         f'only in CSV={sorted(csv_ids - object_ids)}, '
                         f'only in npy={sorted(object_ids - csv_ids)}')

    predicted = pred.pivot(index='object_id', columns='adjective',
                           values='predicted_expected_rating')[ADJECTIVES]
    true = pred.pivot(index='object_id', columns='adjective',
                      values='true_expected_rating')[ADJECTIVES]
    return predicted, true


def load_label_means(labels_dir):
    """(85, 18) per-object mean rating from labels_mean_phase2.npy, indexed by object_id."""
    labels_mean = np.load(os.path.join(labels_dir, 'labels_mean_phase2.npy'))
    object_ids = [str(o) for o in np.load(os.path.join(labels_dir, 'object_ids_phase2.npy'),
                                          allow_pickle=True)]
    return pd.DataFrame(labels_mean, index=object_ids, columns=ADJECTIVES)


def session_object_ids(object_ids, session):
    """Object ids of one session, in the human scripts' order (numeric id)."""
    lo, hi = SESSION_OBJECT_RANGES[session]
    objects = [o for o in object_ids if lo <= int(o.split("_")[1]) <= hi]
    return sorted(objects, key=lambda x: int(x.split("_")[1]))


def draw_heatmap(values, session, out_path, title, cbar_label, cmap, vmin, vmax, fmt, center=None):
    style = SESSION_STYLE[session]
    values = values.copy()
    values.index = [get_label(obj_id) for obj_id in values.index]

    fig, ax = plt.subplots(figsize=style['figsize'])
    sns.heatmap(
        values,
        ax=ax,
        vmin=vmin, vmax=vmax, center=center,
        cmap=cmap,
        annot=True,
        fmt=fmt,
        annot_kws=style['annot_kws'],
        linewidths=0.4,
        linecolor="#e0e0e0",
        cbar_kws={"label": cbar_label, "shrink": 0.6},
    )

    if style['cbar_tick_size'] is not None:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=style['cbar_tick_size'])
        cbar.set_label(cbar_label, **style['cbar_label_kws'])

    ax.set_title(title + style['title_suffix'], **style['title_kws'])
    ax.set_xlabel("Adjective", **style['axis_label_kws'])
    ax.set_ylabel("Object", **style['axis_label_kws'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=style['xtick_fontsize'])
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=style['ytick_fontsize'])

    plt.tight_layout()
    plt.savefig(out_path, dpi=style['dpi'], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    args = parse_args()
    labels_dir = args.labels_dir or paths.labels_dir()
    if args.out_dir is None:
        args.out_dir = paths.output_dir(__file__)
    os.makedirs(args.out_dir, exist_ok=True)

    predicted, true = load_predictions(args.pred_csv, os.path.join(labels_dir, 'object_ids_phase2.npy'))

    label_means = load_label_means(labels_dir)
    session_objects = {s: session_object_ids(label_means.index, s) for s in SESSION_STYLE}
    all_session_objects = session_objects[1] + session_objects[2]
    if len(set(all_session_objects)) != len(all_session_objects) or \
            set(all_session_objects) != set(predicted.index):
        raise SystemExit('Session object lists do not partition the 85 model objects')

    # Sanity check: the model CSV's true ratings must be the label means.
    for s in SESSION_STYLE:
        objs = session_objects[s]
        abs_diff = (true.loc[objs] - label_means.loc[objs]).abs()
        print(f"Session {s}: |CSV true - labels_mean_phase2|  max={abs_diff.values.max():.2e}  "
              f"mean={abs_diff.values.mean():.2e}  ({len(session_objects[s])} objects)")
        if abs_diff.values.max() > args.max_true_diff:
            raise SystemExit(f'Session {s}: true ratings differ from labels_mean_phase2 by more than '
                             f'{args.max_true_diff}; not plotting.')

    diff = predicted - true
    max_abs = float(diff.abs().values.max())
    lim = math.ceil(max_abs / 0.5) * 0.5
    print(f"Diff colour limits: +/-{lim} (max |pred - true| over all 85 objects = {max_abs:.3f})")

    for s in SESSION_STYLE:
        objs = session_objects[s]
        # Rounded to 1 decimal before plotting, as the human scripts do.
        draw_heatmap(
            predicted.loc[objs].round(1), s,
            os.path.join(args.out_dir, f'heatmap_predicted_session{s}.png'),
            title=f'Model prediction ({args.run_label})',
            cbar_label='Predicted rating (1–7)',
            cmap=PRED_CMAP, vmin=1, vmax=7, fmt='.1f',
        )
        draw_heatmap(
            diff.loc[objs].round(1) + 0.0, s,   # + 0.0 turns -0.0 into 0.0 so cells don't read "-0.0"
            os.path.join(args.out_dir, f'heatmap_diff_session{s}.png'),
            title='Prediction minus human mean',
            cbar_label='Predicted − human mean rating',
            cmap=DIFF_CMAP, vmin=-lim, vmax=lim, fmt='+.1f', center=0,
        )

        top = diff.loc[objs].stack().abs().sort_values(ascending=False).head(3)
        print(f"Session {s}: largest |pred - true| cells:")
        for (obj, adj), _ in top.items():
            print(f"  {obj} ({get_label(obj)}) / {adj}: {diff.loc[obj, adj]:+.2f}")


if __name__ == '__main__':
    main()
