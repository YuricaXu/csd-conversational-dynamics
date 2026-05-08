"""
build_fig3_potential_curves.py

Figure 3 — Per-character emotional potential curves U_c(SEP) = -log p_c(SEP).

A 2x3 grid: one panel per main character.  Each panel overlays the
character's empirical potential against the group-average baseline,
with TP (gold star) and FP (cross) segments placed at their mean SEP
position.  Per-character precision is shown in each panel title.

Phoebe Buffay's 0/3 result is the sharpest illustration of the structural
problem: her three flagged segments sit in the dynamically right place on
her own potential surface, but the scenes belong to other characters.

Output: figures/fig3_potential_curves.png
"""

import os, sys

HERE        = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(HERE, "..", "model")
FIGURES_DIR = os.path.join(HERE, "..", "..", "figures")
sys.path.insert(0, MODEL_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from data_loader import load_and_sort, filter_speakers
from vad_engine import load_vad_lexicon, build_trajectories
from csd_detector import detect_csd_collapse
from config import JSON_PATHS, VAD_LEXICON_PATH, MAIN_CHARACTERS
from config_csd import EWS_WINDOW, K_SIGMA, LEAD_LAG, AC1_THRESHOLD_QUANTILE, VAR_THRESHOLD_QUANTILE, MIN_SEGMENT_LEN

os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Manual-review verdicts on the 20 detected segments ────────────────────────
TP_FP = {
    "SEG01":(0,"Chandler Bing","s01_e13_c01_u008","s01_e13_c05_u008"),
    "SEG02":(0,"Chandler Bing","s03_e07_c11_u014","s03_e07_c13_u002"),
    "SEG03":(1,"Chandler Bing","s04_e03_c01_u008","s04_e03_c03_u006"),
    "SEG04":(0,"Chandler Bing","s04_e22_c04_u009","s04_e22_c04_u013"),
    "SEG05":(0,"Ross Geller",  "s01_e08_c04_u012","s01_e08_c05_u005"),
    "SEG06":(0,"Ross Geller",  "s01_e15_c07_u010","s01_e15_c12_u005"),
    "SEG07":(1,"Ross Geller",  "s02_e21_c01_u014","s02_e21_c02_u014"),
    "SEG08":(0,"Ross Geller",  "s03_e21_c09_u013","s03_e21_c09_u017"),
    "SEG09":(0,"Monica Geller","s02_e15_c03_u015","s02_e15_c05_u004"),
    "SEG10":(1,"Monica Geller","s02_e24_c02_u011","s02_e24_c04_u004"),
    "SEG11":(0,"Monica Geller","s04_e01_c07_u008","s04_e01_c12_u024"),
    "SEG12":(1,"Monica Geller","s04_e05_c10_u019","s04_e05_c12_u013"),
    "SEG13":(0,"Joey Tribbiani","s02_e12_c03_u006","s02_e12_c03_u011"),
    "SEG14":(1,"Joey Tribbiani","s02_e18_c08_u006","s02_e18_c13_u014"),
    "SEG15":(0,"Joey Tribbiani","s04_e16_c01_u013","s04_e16_c07_u002"),
    "SEG16":(0,"Rachel Green",  "s01_e21_c11_u004","s01_e21_c11_u012"),
    "SEG17":(1,"Rachel Green",  "s03_e15_c10_u016","s03_e15_c12_u006"),
    "SEG18":(0,"Phoebe Buffay", "s03_e06_c13_u005","s03_e06_c13_u011"),
    "SEG19":(0,"Phoebe Buffay", "s04_e18_c10_u012","s04_e18_c12_u010"),
    "SEG20":(0,"Phoebe Buffay", "s04_e23_c08_u004","s04_e23_c08_u008"),
}
PRECISION = {
    "Phoebe Buffay":(0,3),"Chandler Bing":(1,4),"Joey Tribbiani":(1,3),
    "Ross Geller":(1,4),"Rachel Green":(1,2),"Monica Geller":(2,4),
}
ACCENTS = {
    "Phoebe Buffay":  "#9b59b6", "Chandler Bing":  "#2980b9",
    "Joey Tribbiani": "#e67e22", "Ross Geller":    "#16a085",
    "Rachel Green":   "#c0392b", "Monica Geller":  "#d35400",
}


# ── Load + run detector for all six characters ───────────────────────────────
print("Loading...")
all_utts = load_and_sort(JSON_PATHS); all_utts = filter_speakers(all_utts)
vad_lex, mwe_lex = load_vad_lexicon(VAD_LEXICON_PATH)
trajs = build_trajectories(all_utts, vad_lex, mwe_lex, MAIN_CHARACTERS)
char_data = {}
for char in MAIN_CHARACTERS:
    rows = trajs[char]
    _, info = detect_csd_collapse(rows, k=EWS_WINDOW, k_sigma=K_SIGMA,
                                   ac1_q=AC1_THRESHOLD_QUANTILE, var_q=VAR_THRESHOLD_QUANTILE,
                                   lead_lag=LEAD_LAG, min_len=MIN_SEGMENT_LEN)
    char_data[char] = {"rows": rows, "info": info}


def find_idx(rows, uid):
    for i, r in enumerate(rows):
        if r["utterance_id"] == uid: return i
    return None


def segment_mean_sep(char, start_uid, end_uid):
    rows = char_data[char]["rows"]
    info = char_data[char]["info"]
    idx_s = find_idx(rows, start_uid); idx_e = find_idx(rows, end_uid)
    return float(np.mean(info["sep_seq"][idx_s:idx_e+1]))


# ── Compute U(SEP) per character + group average ─────────────────────────────
SEP_GRID = np.linspace(-1.0, 1.0, 400)
char_U = {}
all_sep = []
for char in MAIN_CHARACTERS:
    sep = np.array(char_data[char]["info"]["sep_seq"])
    sep = sep[~np.isnan(sep)]
    kde = gaussian_kde(sep, bw_method=0.18)
    p = np.maximum(kde(SEP_GRID), 1e-4)
    U = -np.log(p)
    U = U - U.min()
    char_U[char] = {
        "U": U,
        "min_sep": float(SEP_GRID[np.argmin(U)]),
        "mu":      char_data[char]["info"]["eq"]["mu_sep"],
        "sigma":   char_data[char]["info"]["eq"]["sigma_sep"],
    }
    all_sep.extend(sep)
all_sep = np.array(all_sep)
kde_avg = gaussian_kde(all_sep, bw_method=0.18)
p_avg = np.maximum(kde_avg(SEP_GRID), 1e-4)
U_avg = -np.log(p_avg)
U_avg = U_avg - U_avg.min()


# ── Faceted figure: 2x3 grid, one panel per character ────────────────────────
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"]   = "white"

fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
axes_flat = axes.flatten()

# Order: low precision to high (story arc)
char_order = sorted(MAIN_CHARACTERS, key=lambda c: PRECISION[c][0]/PRECISION[c][1])
y_max = max(L["U"].max() for L in char_U.values()) * 1.05

for ax, char in zip(axes_flat, char_order):
    L = char_U[char]
    color = ACCENTS[char]
    tp, total = PRECISION[char]
    p_pct = int(tp/total*100)

    # Layer-1 territory shading (character-specific threshold)
    thr = L["mu"] - 0.5 * L["sigma"]
    ax.axvspan(-1.0, thr, color="#e74c3c", alpha=0.06)

    # Group-average curve (light grey baseline)
    ax.plot(SEP_GRID, U_avg, color="#bdc3c7", linewidth=2.0,
             linestyle="--", alpha=0.85, label="Group average",
             zorder=2)

    # Character's own curve in their accent colour
    ax.plot(SEP_GRID, L["U"], color=color, linewidth=2.5, alpha=0.95,
             label=char, zorder=4)

    # Basin minimum marker
    ax.scatter([L["min_sep"]], [L["U"][np.argmin(L["U"])]],
                s=180, marker="v", color=color, edgecolor="black",
                linewidth=1.2, zorder=8)

    # TP / FP markers on this character's curve
    char_segs = [(sid, v, a, b) for sid, (v, c2, a, b) in TP_FP.items() if c2 == char]

    # Pre-compute (sep, u) and apply horizontal jitter for near-overlapping markers
    seg_data = []
    for sid, v, a, b in char_segs:
        sep_mean = segment_mean_sep(char, a, b)
        u_val    = float(np.interp(sep_mean, SEP_GRID, L["U"]))
        seg_data.append({"sid": sid, "v": v,
                         "sep": sep_mean, "u": u_val,
                         "x_off": 0.0})

    JITTER = 0.045
    seg_data.sort(key=lambda d: d["sep"])
    for i in range(len(seg_data)):
        for j in range(i + 1, len(seg_data)):
            if abs(seg_data[i]["sep"] - seg_data[j]["sep"]) < 0.05:
                seg_data[i]["x_off"] -= JITTER / 2
                seg_data[j]["x_off"] += JITTER / 2

    for d in seg_data:
        x_plot = d["sep"] + d["x_off"]
        u_plot = float(np.interp(x_plot, SEP_GRID, L["U"]))
        if d["v"] == 1:
            ax.scatter([x_plot], [u_plot], s=320, marker="*",
                        color="gold", edgecolor=color, linewidth=1.8, zorder=10)
        else:
            ax.scatter([x_plot], [u_plot], s=130, marker="X",
                        color=color, edgecolor="black", linewidth=1.0,
                        alpha=0.85, zorder=10)
        ax.annotate(d["sid"],
                    xy=(x_plot, u_plot),
                    xytext=(0, -13), textcoords="offset points",
                    fontsize=7.2, color=color, ha="center", va="top",
                    fontweight="bold", alpha=0.95, zorder=11)

    ax.axvline(L["mu"], color="gray", linestyle=":", linewidth=0.7, alpha=0.5)

    ax.set_title(f"{char}    [precision {tp}/{total} = {p_pct}%]",
                  fontsize=11, fontweight="bold", color=color, pad=6)
    ax.grid(True, alpha=0.25)

for ax in axes[1, :]:
    ax.set_xlabel("SEP $= V + 0.5\\cdot D$", fontsize=11)
for ax in axes[:, 0]:
    ax.set_ylabel("$U(\\mathrm{SEP}) = -\\log p(\\mathrm{SEP})$", fontsize=10)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], color="#bdc3c7", linewidth=2, linestyle="--",
           label="Group average $\\bar{U}(\\mathrm{SEP})$ (reference)"),
    Line2D([0],[0], marker="*", color="w", markerfacecolor="gold",
           markeredgecolor="black", markersize=12,
           label="True-positive segment"),
    Line2D([0],[0], marker="X", color="w", markerfacecolor="#7f8c8d",
           markeredgecolor="black", markersize=8,
           label="False-positive segment"),
    Line2D([0],[0], marker="v", color="w", markerfacecolor="#7f8c8d",
           markeredgecolor="black", markersize=10,
           label="Basin minimum"),
]
fig.legend(handles=legend_elements, loc="upper center", ncol=4,
            bbox_to_anchor=(0.5, 0.02), fontsize=9, frameon=False)

fig.suptitle(
    "Per-character emotional potential curves $U_c(\\mathrm{SEP}) = -\\log p_c(\\mathrm{SEP})$",
    fontsize=13, fontweight="bold", y=0.99, color="#2c3e50")
fig.text(0.5, 0.95,
          "Each panel shows one character's potential against the same group-average baseline (dashed grey). "
          "Stars = validated tipping segments; crosses = false positives.",
          ha="center", fontsize=10, fontstyle="italic", color="#586e75")

plt.tight_layout(rect=[0, 0.04, 1, 0.93])
out = os.path.join(FIGURES_DIR, "fig3_potential_curves.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
plt.rcdefaults()
print(f"Saved: {out}")
