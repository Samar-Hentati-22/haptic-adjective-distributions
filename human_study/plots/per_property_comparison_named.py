"""
per_property_comparison.py

Generates one violin plot per adjective comparing objects:
  - X axis: objects sorted low→high by mean rating on that adjective
  - Y axis: rating 1–7
  - Violin body shows full rating distribution per object
  - Orange dot overlaid at the mean
  - Saved to <repo>/outputs/per_property_comparison_named/OUTPUT_FOLDER/per_property_named/<adjective>.png
    (or .../per_property_named/OUTPUT_SUBDIR/<adjective>.png if OUTPUT_SUBDIR is set)

To run on a different dataset, change INPUT_FILE and OUTPUT_FOLDER only.

To regenerate only some adjectives, and/or show only the top-N highest-rated
objects (sorted ascending, keeping the last N) instead of all of them, set
ADJECTIVES_TO_PLOT / TOP_N / OUTPUT_SUBDIR below. This is handy for pulling a
few report-ready figures without touching the full original set — e.g. for
Master Thesis "Session 2" report examples:
    ADJECTIVES_TO_PLOT = ["hard", "warm"]
    TOP_N = 20
    OUTPUT_SUBDIR = "report_examples"
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model', 'src'))  # <repo>/model/src
import paths  # noqa: E402
INPUT_FILE          = os.path.join(paths.human_data_dir(), "session2_final.csv")
OUTPUT_FOLDER       = "session2"
OBJECT_ID_COL       = "object_id"
PARTICIPANT_ID_COL  = "participant_id"
# adjectives inferred automatically as all columns except the two ID columns

# Which adjectives to (re)generate. None = all adjectives in the data.
ADJECTIVES_TO_PLOT = ["hard", "warm"]

# Keep only the top-N highest-rated objects for each adjective (after sorting
# ascending by mean rating), instead of all objects. None = keep all objects.
TOP_N = 20

# Save under PLOTS_DIR/<OUTPUT_SUBDIR>/ instead of directly under PLOTS_DIR,
# so a filtered/subset run never overwrites the full original plots.
# "" (empty string) = save directly under PLOTS_DIR as before.
OUTPUT_SUBDIR = "report_examples"
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
    "obj_76": "Black tape",
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
PLOTS_DIR = os.path.join(BASE_DIR, OUTPUT_FOLDER, "per_property_named")
if OUTPUT_SUBDIR:
    PLOTS_DIR = os.path.join(PLOTS_DIR, OUTPUT_SUBDIR)
os.makedirs(PLOTS_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)
adjectives = [c for c in df.columns if c not in (OBJECT_ID_COL, PARTICIPANT_ID_COL)]
if ADJECTIVES_TO_PLOT is not None:
    adjectives = [a for a in adjectives if a in ADJECTIVES_TO_PLOT]
df[adjectives] = df[adjectives].apply(pd.to_numeric, errors="coerce")

def _num_key(s):
    nums = re.findall(r"\d+", str(s))
    return int(nums[-1]) if nums else s

objects = sorted(df[OBJECT_ID_COL].unique(), key=_num_key)

for adj in adjectives:
    # Sort objects by mean rating on this adjective (low → high), independently
    # per adjective, then optionally keep only the top-N highest-rated ones
    # (i.e. the tail of the ascending sort).
    means = (
        df.groupby(OBJECT_ID_COL)[adj]
        .mean()
        .reindex(objects)
    )
    sorted_objects = means.sort_values().index.tolist()
    if TOP_N is not None:
        sorted_objects = sorted_objects[-TOP_N:]

    n_obj = len(sorted_objects)
    FIG_W = max(18, n_obj * 0.9)

    data_list = [
        df[df[OBJECT_ID_COL] == obj][adj].dropna().values
        for obj in sorted_objects
    ]
    mean_vals = [arr.mean() for arr in data_list]

    x_pos = np.arange(1, n_obj + 1)

    fig, ax = plt.subplots(figsize=(FIG_W, 7.5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    vp = ax.violinplot(
        data_list,
        positions=x_pos,
        widths=0.7,
        showmeans=False,
        showextrema=True,
        showmedians=False,
    )

    # Style violin bodies
    for body in vp["bodies"]:
        body.set_facecolor("#6EA8D5")
        body.set_edgecolor("#3B78A8")
        body.set_alpha(0.7)
    for part in ("cmaxes", "cmins", "cbars"):
        vp[part].set_edgecolor("#3B78A8")
        vp[part].set_linewidth(1.0)

    # Mean dot overlay
    ax.scatter(
        x_pos, mean_vals,
        color="#E8720C", s=55, zorder=4,
        edgecolors="white", linewidths=0.8,
        label="Mean",
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [get_label(o) for o in sorted_objects],
        rotation=45, ha="right", rotation_mode="anchor", fontsize=13,
    )
    ax.set_yticks(range(1, 8))
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(0.4, 7.6)
    ax.set_xlim(0.3, n_obj + 0.7)
    ax.set_ylabel("Rating (1 – 7)", fontsize=17, labelpad=10)
    ax.set_xlabel("Object", fontsize=17, labelpad=12)
    ax.set_title(adj, fontsize=22, fontweight="bold", pad=16)
    ax.grid(axis="y", color="#cccccc", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # Mean value labelled above each dot for readability at print scale
    for xi, mv in zip(x_pos, mean_vals):
        ax.annotate(
            f"{mv:.1f}", (xi, mv), xytext=(0, 9), textcoords="offset points",
            ha="center", fontsize=10.5, fontweight="bold", color="#8a3d00",
        )

    violin_patch = mpatches.Patch(facecolor="#6EA8D5", edgecolor="#3B78A8",
                                  alpha=0.7, label="Distribution")
    from matplotlib.lines import Line2D
    mean_handle = Line2D([0], [0], marker="o", color="w",
                         markerfacecolor="#E8720C", markersize=9,
                         markeredgecolor="white", label="Mean")
    ax.legend(
        handles=[violin_patch, mean_handle],
        bbox_to_anchor=(1.01, 1), loc="upper left",
        fontsize=12, framealpha=0.9, edgecolor="#cccccc",
    )

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{adj}.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {adj}.png")

print(f"\nDone. {len(adjectives)} per-property plots → {PLOTS_DIR}")
