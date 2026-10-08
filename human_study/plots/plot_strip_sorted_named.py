"""
plot_strip_sorted.py

Generates mean-sorted strip plots for Session 1 and Session 2.

Session 1:
  Per-object  (45 plots) → <repo>/outputs/plot_strip_sorted_named/per_object_sorted_named/
    X axis: 18 adjectives sorted low→high by their mean rating for that object
  Per-adjective (18 plots) → <repo>/outputs/plot_strip_sorted_named/per_adjective_sorted_named/
    X axis: 45 objects sorted low→high by their mean rating on that adjective

Session 2:
  Per-object  (45 plots) → <repo>/outputs/plot_strip_sorted_named/session2/per_object_sorted_named/
  Per-adjective (18 plots) → <repo>/outputs/plot_strip_sorted_named/session2/per_adjective_sorted_named/

Style is identical to the existing unsorted plots.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402
BASE_DIR = paths.human_data_dir()
OUT_DIR = paths.output_dir(__file__)

ADJECTIVES = [
    "hard", "soft", "rough", "smooth", "bumpy", "flat",
    "sticky", "slippery", "cold", "warm", "wet", "dry",
    "compressible", "firm", "fuzzy", "scratchy", "pleasant", "irritating",
]

SESSIONS = [
    {
        "data_file": os.path.join(BASE_DIR, "session1_final.csv"),
        "dir_obj":   os.path.join(OUT_DIR, "per_object_sorted_named"),
        "dir_adj":   os.path.join(OUT_DIR, "per_adjective_sorted_named"),
        "label":     "Session 1",
    },
    {
        "data_file": os.path.join(BASE_DIR, "session2_final.csv"),
        "dir_obj":   os.path.join(OUT_DIR, "session2", "per_object_sorted_named"),
        "dir_adj":   os.path.join(OUT_DIR, "session2", "per_adjective_sorted_named"),
        "label":     "Session 2",
    },
]

JITTER_WIDTH = 0.25

# Optional object name mapping
# Set to None to use object IDs, or fill in the dict to use readable names
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
# Set to None to use object IDs instead
# OBJECT_NAMES = None


def get_label(obj_id):
    if OBJECT_NAMES is not None:
        return OBJECT_NAMES.get(obj_id, obj_id)
    return obj_id


def legend_handles(colors, participants):
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=colors[pid], markersize=7, label=pid)
        for pid in participants
    ]
    handles.append(
        Line2D([0], [0], color="#1a1a1a", linewidth=2.2, label="Mean")
    )
    return handles


def plot_per_object_sorted(df, objects, participants, colors, plots_dir):
    """One plot per object; adjectives sorted by their mean for that object."""
    x_pos = np.arange(len(ADJECTIVES))

    for obj_id in objects:
        obj_df = df[df["object_id"] == obj_id].copy()

        # Compute per-adjective mean for this object and sort low→high
        adj_means = obj_df[ADJECTIVES].mean()
        sorted_adjs = adj_means.sort_values().index.tolist()

        fig, ax = plt.subplots(figsize=(16, 5.5))
        fig.patch.set_facecolor("#FAFAFA")
        ax.set_facecolor("#FAFAFA")

        rng = np.random.default_rng(seed=42)

        for xi, adj in enumerate(sorted_adjs):
            ratings = obj_df[adj].values
            pids    = obj_df["participant_id"].values
            jitter  = rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=len(ratings))

            for pid, rating, jit in zip(pids, ratings, jitter):
                ax.scatter(
                    xi + jit, rating,
                    color=colors[pid],
                    s=55, alpha=0.85, linewidths=0.4,
                    edgecolors="white", zorder=3,
                )

            mean_val = np.nanmean(ratings)
            ax.hlines(
                mean_val, xi - 0.32, xi + 0.32,
                colors="#1a1a1a", linewidths=2.2, zorder=4,
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(sorted_adjs, rotation=40, ha="right", fontsize=9.5)
        ax.set_yticks(range(1, 8))
        ax.set_ylim(0.4, 7.6)
        ax.set_xlim(-0.6, len(ADJECTIVES) - 0.4)
        ax.set_ylabel("Rating (1 – 7)", fontsize=10)
        ax.set_title(get_label(obj_id), fontsize=15, fontweight="bold", pad=10)
        ax.grid(axis="y", color="#cccccc", linewidth=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

        ax.legend(
            handles=legend_handles(colors, participants),
            title="Participant",
            bbox_to_anchor=(1.01, 1), loc="upper left",
            fontsize=7.5, title_fontsize=8.5,
            framealpha=0.9, edgecolor="#cccccc",
            ncol=1,
        )

        plt.tight_layout()
        out_path = os.path.join(plots_dir, f"{obj_id}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {obj_id}.png")

    print(f"Per-object sorted done. {len(objects)} plots → {plots_dir}\n")


def plot_per_adjective_sorted(df, objects, participants, colors, plots_dir):
    """One plot per adjective; objects sorted by their mean on that adjective."""
    for adj in ADJECTIVES:
        # Compute per-object mean for this adjective and sort low→high
        obj_means = (
            df.groupby("object_id")[adj]
            .mean()
            .reindex(objects)
        )
        sorted_objects = obj_means.sort_values().index.tolist()
        x_pos = np.arange(len(sorted_objects))

        fig, ax = plt.subplots(figsize=(22, 5.5))
        fig.patch.set_facecolor("#FAFAFA")
        ax.set_facecolor("#FAFAFA")

        rng = np.random.default_rng(seed=42)

        for xi, obj_id in enumerate(sorted_objects):
            obj_df  = df[df["object_id"] == obj_id]
            ratings = obj_df[adj].values
            pids    = obj_df["participant_id"].values
            jitter  = rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=len(ratings))

            for pid, rating, jit in zip(pids, ratings, jitter):
                ax.scatter(
                    xi + jit, rating,
                    color=colors[pid],
                    s=45, alpha=0.85, linewidths=0.4,
                    edgecolors="white", zorder=3,
                )

            mean_val = np.nanmean(ratings)
            ax.hlines(
                mean_val, xi - 0.32, xi + 0.32,
                colors="#1a1a1a", linewidths=2.2, zorder=4,
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels([get_label(o) for o in sorted_objects], rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(1, 8))
        ax.set_ylim(0.4, 7.6)
        ax.set_xlim(-0.6, len(sorted_objects) - 0.4)
        ax.set_ylabel("Rating (1 – 7)", fontsize=10)
        ax.set_title(adj, fontsize=15, fontweight="bold", pad=10)
        ax.grid(axis="y", color="#cccccc", linewidth=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

        ax.legend(
            handles=legend_handles(colors, participants),
            title="Participant",
            bbox_to_anchor=(1.01, 1), loc="upper left",
            fontsize=7.5, title_fontsize=8.5,
            framealpha=0.9, edgecolor="#cccccc",
            ncol=1,
        )

        plt.tight_layout()
        out_path = os.path.join(plots_dir, f"{adj}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {adj}.png")

    print(f"Per-adjective sorted done. {len(ADJECTIVES)} plots → {plots_dir}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

for session in SESSIONS:
    data_file = session["data_file"]
    if not os.path.exists(data_file):
        print(f"[SKIP] {data_file} not found.")
        continue

    print(f"\n{'=' * 60}")
    print(f"  {session['label']}  —  {data_file}")
    print(f"{'=' * 60}")

    os.makedirs(session["dir_obj"], exist_ok=True)
    os.makedirs(session["dir_adj"], exist_ok=True)

    df = pd.read_csv(data_file)
    df[ADJECTIVES] = df[ADJECTIVES].apply(pd.to_numeric, errors="coerce")

    participants = sorted(df["participant_id"].unique(), key=lambda x: int(x[1:]))
    n_p = len(participants)

    tab20 = plt.get_cmap("tab20")
    colors = {pid: tab20(i / n_p) for i, pid in enumerate(participants)}

    objects = sorted(df["object_id"].unique(), key=lambda x: int(x.split("_")[1]))

    print(f"\n-- Per-object sorted ({len(objects)} plots) --")
    plot_per_object_sorted(df, objects, participants, colors, session["dir_obj"])

    print(f"-- Per-adjective sorted ({len(ADJECTIVES)} plots) --")
    plot_per_adjective_sorted(df, objects, participants, colors, session["dir_adj"])

print("All done.")
