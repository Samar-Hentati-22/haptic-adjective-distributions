"""
plot_strip_per_object.py

Generates one strip plot per object (45 total):
  - X axis: 18 adjectives
  - Y axis: rating 1–7
  - One dot per participant, colored uniquely
  - Horizontal dash showing mean per adjective
  - Saved to <repo>/outputs/plot_strip_per_object_named/per_object_named/<obj_id>.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

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


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402
BASE_DIR = paths.human_data_dir()
DATA_FILE = os.path.join(BASE_DIR, "session1_final.csv")
PLOTS_DIR = os.path.join(paths.output_dir(__file__), "per_object_named")
os.makedirs(PLOTS_DIR, exist_ok=True)

ADJECTIVES = [
    "hard", "soft", "rough", "smooth", "bumpy", "flat",
    "sticky", "slippery", "cold", "warm", "wet", "dry",
    "compressible", "firm", "fuzzy", "scratchy", "pleasant", "irritating",
]

df = pd.read_csv(DATA_FILE)
df[ADJECTIVES] = df[ADJECTIVES].apply(pd.to_numeric, errors="coerce")

participants = sorted(df["participant_id"].unique(), key=lambda x: int(x[1:]))
n_p = len(participants)

# 20 visually distinct colors from tab20
tab20 = plt.get_cmap("tab20")
colors = {pid: tab20(i / n_p) for i, pid in enumerate(participants)}

x_pos = np.arange(len(ADJECTIVES))
jitter_width = 0.25

objects = sorted(df["object_id"].unique(), key=lambda x: int(x.split("_")[1]))

for obj_id in objects:
    obj_df = df[df["object_id"] == obj_id].copy()

    fig, ax = plt.subplots(figsize=(16, 5.5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    rng = np.random.default_rng(seed=42)

    for xi, adj in enumerate(ADJECTIVES):
        ratings = obj_df[adj].values
        pids    = obj_df["participant_id"].values

        # jitter x positions, one per participant
        jitter = rng.uniform(-jitter_width, jitter_width, size=len(ratings))

        for pid, rating, jit in zip(pids, ratings, jitter):
            ax.scatter(
                xi + jit, rating,
                color=colors[pid],
                s=55, alpha=0.85, linewidths=0.4,
                edgecolors="white", zorder=3,
            )

        # mean dash
        mean_val = np.nanmean(ratings)
        ax.hlines(
            mean_val, xi - 0.32, xi + 0.32,
            colors="#1a1a1a", linewidths=2.2, zorder=4,
        )

    # axes formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ADJECTIVES, rotation=40, ha="right", fontsize=9.5)
    ax.set_yticks(range(1, 8))
    ax.set_ylim(0.4, 7.6)
    ax.set_xlim(-0.6, len(ADJECTIVES) - 0.4)
    ax.set_ylabel("Rating (1 – 7)", fontsize=10)
    ax.set_title(get_label(obj_id), fontsize=15, fontweight="bold", pad=10)

    ax.grid(axis="y", color="#cccccc", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # legend: participants + mean indicator
    p_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=colors[pid], markersize=7,
               label=pid)
        for pid in participants
    ]
    mean_handle = Line2D([0], [0.5], color="#1a1a1a", linewidth=2.2,
                         label="Mean", linestyle="-")
    ax.legend(
        handles=p_handles + [mean_handle],
        title="Participant",
        bbox_to_anchor=(1.01, 1), loc="upper left",
        fontsize=7.5, title_fontsize=8.5,
        framealpha=0.9, edgecolor="#cccccc",
        ncol=1,
    )

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f"{obj_id}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {obj_id}.png")

print(f"\nDone. {len(objects)} plots saved to:\n  {PLOTS_DIR}")
