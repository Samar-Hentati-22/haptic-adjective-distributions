"""
distribution_plots.py

Generates one distribution plot per object:
  - Grid of small bar charts, one per adjective
  - X axis of each bar chart: rating levels 1–7
  - Y axis: count of participants who gave that rating
  - Grid layout adapts automatically to number of adjectives (6 columns)
  - Saved to <repo>/outputs/distribution_plots_named/OUTPUT_FOLDER/distributions_named/<object_id>.png

To run on a different dataset, change INPUT_FILE and OUTPUT_FOLDER only.
"""

import os
import sys
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402
INPUT_FILE         = os.path.join(paths.human_data_dir(), "session1_final.csv")
OUTPUT_FOLDER      = "session1"
OBJECT_ID_COL      = "object_id"
PARTICIPANT_ID_COL = "participant_id"
# adjectives inferred automatically as all columns except the two ID columns
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
# ───────────────────────────────────────────────────────────────────────────────


def get_label(obj_id):
    if OBJECT_NAMES is not None:
        return OBJECT_NAMES.get(obj_id, obj_id)
    return obj_id


BASE_DIR  = paths.output_dir(__file__)  # plots go to <repo>/outputs/<script_name>/
PLOTS_DIR = os.path.join(BASE_DIR, OUTPUT_FOLDER, "distributions_named")
os.makedirs(PLOTS_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)
adjectives = [c for c in df.columns if c not in (OBJECT_ID_COL, PARTICIPANT_ID_COL)]
df[adjectives] = df[adjectives].apply(pd.to_numeric, errors="coerce")

def _num_key(s):
    nums = re.findall(r"\d+", str(s))
    return int(nums[-1]) if nums else s

objects    = sorted(df[OBJECT_ID_COL].unique(), key=_num_key)
n_p        = df[PARTICIPANT_ID_COL].nunique()
n_adj      = len(adjectives)
LEVELS     = list(range(1, 8))

NCOLS = min(n_adj, 6)
NROWS = math.ceil(n_adj / NCOLS)

for obj_id in objects:
    obj_df = df[df[OBJECT_ID_COL] == obj_id]

    fig, axes = plt.subplots(
        NROWS, NCOLS,
        figsize=(NCOLS * 2.6, NROWS * 2.4),
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle(get_label(obj_id), fontsize=14, fontweight="bold", y=1.01)

    axes_flat = np.array(axes).flatten()

    for i, adj in enumerate(adjectives):
        ax = axes_flat[i]
        ax.set_facecolor("#FAFAFA")

        ratings = obj_df[adj].dropna().values
        counts  = [(ratings == lvl).sum() for lvl in LEVELS]

        ax.bar(
            LEVELS, counts,
            color="#4C72B0", edgecolor="white", linewidth=0.5,
            width=0.75, zorder=3,
        )
        ax.set_title(adj, fontsize=8, fontweight="bold", pad=3)
        ax.set_xticks(LEVELS)
        ax.set_xlim(0.4, 7.6)
        ax.set_ylim(0, n_p)
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel("Rating", fontsize=6.5, labelpad=2)
        ax.set_ylabel("Count",  fontsize=6.5, labelpad=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#eeeeee", linewidth=0.5, zorder=0)

    # Hide any unused subplots
    for j in range(n_adj, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.savefig(os.path.join(PLOTS_DIR, f"{obj_id}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {obj_id}.png")

print(f"\nDone. {len(objects)} distribution plots → {PLOTS_DIR}")
