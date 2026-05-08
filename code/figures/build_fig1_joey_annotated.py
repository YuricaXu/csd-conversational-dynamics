"""
build_fig1_joey_annotated.py

Figure 1 — Joey Tribbiani's full Seasons 1–4 SEP and lag-1 autocorrelation
trajectory, with the three pipeline-flagged segments coloured by manual-
review verdict (one true positive, two false positives).  An inset zooms
on the SEG14 true positive (Dr Ramoray killed off, S02E18) and shows the
textbook CSD signature: r_1 and variance rise before the SEP nadir.

Output: figures/fig1_joey_annotated.png
"""

import os, sys

# Find the engine modules at code/model/
HERE       = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(HERE, "..", "model")
FIGURES_DIR = os.path.join(HERE, "..", "..", "figures")
sys.path.insert(0, MODEL_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_and_sort, filter_speakers
from vad_engine import load_vad_lexicon, build_trajectories
from csd_detector import detect_csd_collapse
from config import JSON_PATHS, VAD_LEXICON_PATH, MAIN_CHARACTERS
from config_csd import EWS_WINDOW, K_SIGMA, LEAD_LAG, AC1_THRESHOLD_QUANTILE, VAR_THRESHOLD_QUANTILE, MIN_SEGMENT_LEN

os.makedirs(FIGURES_DIR, exist_ok=True)


# Joey's three segments and their TP/FP labels
JOEY_SEGS = [
    ("SEG13", 0, "s02_e12_c03_u006", "s02_e12_c03_u011", "Erika the stalker fan"),
    ("SEG14", 1, "s02_e18_c08_u006", "s02_e18_c13_u014", "Dr Ramoray killed off"),
    ("SEG15", 0, "s04_e16_c01_u013", "s04_e16_c07_u002", "Fake Party scene"),
]

print("Loading...")
all_utts = load_and_sort(JSON_PATHS); all_utts = filter_speakers(all_utts)
vad_lex, mwe_lex = load_vad_lexicon(VAD_LEXICON_PATH)
trajs = build_trajectories(all_utts, vad_lex, mwe_lex, MAIN_CHARACTERS)
rows = trajs["Joey Tribbiani"]
_, info = detect_csd_collapse(rows, k=EWS_WINDOW, k_sigma=K_SIGMA,
                               ac1_q=AC1_THRESHOLD_QUANTILE, var_q=VAR_THRESHOLD_QUANTILE,
                               lead_lag=LEAD_LAG, min_len=MIN_SEGMENT_LEN)
eq = info["eq"]


def find_idx(uid):
    for i, r in enumerate(rows):
        if r["utterance_id"] == uid: return i
    return None


# ── Figure ────────────────────────────────────────────────────────────────
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"]   = "white"

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[3, 1.4],
                       hspace=0.30, wspace=0.18)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
# Inset zoom panels for SEG14
ax_zoom_sep = fig.add_subplot(gs[0, 1])
ax_zoom_ac1 = fig.add_subplot(gs[1, 1], sharex=ax_zoom_sep)

n = len(rows)
x = np.arange(n)
sep = info["sep_seq"]
roll_delta = info["roll_delta"]
ac1 = info["ac1_seq"]
mu = eq["mu_sep"]; sigma = eq["sigma_sep"]
threshold_sep = mu - 0.5 * sigma

# ── Top panel: SEP ──
ax1.plot(x, sep, color="#2c3e50", linewidth=0.7, alpha=0.85, label="SEP")
ax1.plot(x, [mu + r for r in roll_delta], color="#e67e22", linewidth=0.9,
         alpha=0.7, label=r"$\bar{\Delta}$ (smoothed)")
ax1.axhline(mu, color="gray", linestyle="--", linewidth=0.7,
             label=f"$\\mu_{{SEP}}$ = {mu:+.3f}")
ax1.axhline(threshold_sep, color="#e74c3c", linestyle="--", linewidth=0.8,
             alpha=0.7, label=f"Layer 1 threshold = {threshold_sep:+.3f}")

# Shade segments by TP/FP and annotate
for sid, is_tp, a, b, name in JOEY_SEGS:
    i_s = find_idx(a); i_e = find_idx(b)
    if i_s is None or i_e is None: continue
    color = "#27ae60" if is_tp else "#c0392b"
    ax1.axvspan(i_s, i_e, color=color, alpha=0.22)
    midx = (i_s + i_e) / 2
    badge = "TP" if is_tp else "FP"
    ax1.annotate(f"{sid} ({badge})\n{name}",
                  xy=(midx, 1.1), xytext=(midx, 1.18),
                  fontsize=9, ha="center", fontweight="bold",
                  color=color,
                  arrowprops=dict(arrowstyle="-", color=color, linewidth=0.8))

# Episode boundaries
for t in range(1, n):
    if rows[t]["_episode_id"] != rows[t - 1]["_episode_id"]:
        ax1.axvline(t, color="lightgray", linewidth=0.2, alpha=0.4)
        ax2.axvline(t, color="lightgray", linewidth=0.2, alpha=0.4)

ax1.set_ylabel("SEP $= V + 0.5\\cdot D$", fontsize=11)
ax1.set_ylim(-1.2, 1.45)
ax1.legend(loc="lower right", fontsize=8, ncol=2)
ax1.set_title(
    "Joey Tribbiani --- Seasons 1--4 emotional trajectory with detector output\n"
    "Three pipeline-flagged segments: 1 true positive (Dr Ramoray killed off, S02E18) and 2 false positives",
    fontsize=12, fontweight="bold")
ax1.grid(True, alpha=0.2)

# ── Bottom panel: AC(1) ──
ax2.plot(x, ac1, color="#e74c3c", linewidth=0.7, alpha=0.85)
ax2.axhline(info["ac1_threshold"], color="#e74c3c", linestyle="--",
             linewidth=0.7, label=f"$Q_{{75}}(r_1)$ = {info['ac1_threshold']:.3f}")

for sid, is_tp, a, b, name in JOEY_SEGS:
    i_s = find_idx(a); i_e = find_idx(b)
    if i_s is None or i_e is None: continue
    color = "#27ae60" if is_tp else "#c0392b"
    ax2.axvspan(i_s, i_e, color=color, alpha=0.22)

ax2.set_ylabel("Lag-1 autocorrelation $r_1$", fontsize=11)
ax2.set_xlabel("Utterance index (chronological, S01--S04)", fontsize=11)
ax2.legend(loc="lower right", fontsize=8)
ax2.grid(True, alpha=0.2)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#27ae60", alpha=0.4, label="True-positive segment (manual review)"),
    Patch(facecolor="#c0392b", alpha=0.4, label="False-positive segment (manual review)"),
]
fig.legend(handles=legend_elements, loc="upper left",
            bbox_to_anchor=(0.06, 0.97), fontsize=9, framealpha=0.95)


# ── Inset zoom: SEG14 (TP) close-up showing AC1 rise BEFORE SEP nadir ──
seg14 = JOEY_SEGS[1]
i_s = find_idx(seg14[2]); i_e = find_idx(seg14[3])
eid = rows[i_s]["_episode_id"]
zlo = max(0, i_s - 10); zhi = min(len(rows)-1, i_e + 10)
while zlo < i_s and rows[zlo]["_episode_id"] != eid: zlo += 1
while zhi > i_e and rows[zhi]["_episode_id"] != eid: zhi -= 1
zoom_x = list(range(zlo, zhi+1))
zoom_sep = [info["sep_seq"][i] for i in zoom_x]
zoom_ac1 = [info["ac1_seq"][i] for i in zoom_x]
zoom_var = [info["var_seq"][i] for i in zoom_x]

seg_seps = [info["sep_seq"][i] for i in range(i_s, i_e+1)]
nadir_offset = seg_seps.index(min(seg_seps))
nadir_idx = i_s + nadir_offset

ax_zoom_sep.plot(zoom_x, zoom_sep, "-o", color="#27ae60", linewidth=1.6,
                  markersize=4, alpha=0.9)
ax_zoom_sep.axhline(mu, color="gray", linestyle="--", linewidth=0.6, alpha=0.7)
ax_zoom_sep.axhline(threshold_sep, color="#e74c3c", linestyle="--",
                     linewidth=0.7, alpha=0.6)
ax_zoom_sep.axvspan(i_s, i_e, color="#27ae60", alpha=0.18)
ax_zoom_sep.scatter([nadir_idx], [info["sep_seq"][nadir_idx]], s=120,
                     marker="v", color="#16a085", edgecolor="black",
                     linewidth=1.2, zorder=10, label="SEP nadir")
ax_zoom_sep.set_title("SEG14 zoom: Dr Ramoray killed off (TP)\n"
                       "$r_1$ rises BEFORE the SEP nadir",
                       fontsize=10, fontweight="bold", color="#16a085")
ax_zoom_sep.set_ylabel("SEP", fontsize=9)
ax_zoom_sep.legend(loc="upper right", fontsize=7)
ax_zoom_sep.grid(True, alpha=0.25)
ax_zoom_sep.tick_params(labelsize=7)

ax_zoom_ac1.plot(zoom_x, zoom_ac1, "-o", color="#e74c3c",
                  linewidth=1.4, markersize=4, alpha=0.9, label=r"$r_1$")
ax_zoom_ac1.axhline(info["ac1_threshold"], color="#e74c3c",
                     linestyle="--", linewidth=0.7, alpha=0.6,
                     label=f"$Q_{{75}}$={info['ac1_threshold']:.2f}")
ax_zoom_ac1.axvspan(i_s, i_e, color="#27ae60", alpha=0.18)
ax_zoom_ac1.axvline(nadir_idx, color="#16a085", linestyle=":",
                     linewidth=1.4, alpha=0.85,
                     label="SEP nadir")
crossing_idx = None
for i in zoom_x:
    if info["ac1_seq"][i] > info["ac1_threshold"]:
        crossing_idx = i; break
if crossing_idx is not None and crossing_idx < nadir_idx:
    ax_zoom_ac1.axvline(crossing_idx, color="#c0392b",
                         linestyle=":", linewidth=1.2, alpha=0.7)
    lead = nadir_idx - crossing_idx
    ax_zoom_ac1.annotate(f"$r_1$ rises\n{lead} steps\nbefore nadir",
                          xy=(crossing_idx, 0.7),
                          xytext=(crossing_idx - 4, 0.95),
                          fontsize=7, color="#c0392b", fontweight="bold",
                          arrowprops=dict(arrowstyle="->", color="#c0392b",
                                          linewidth=0.8))

ax_zoom_var = ax_zoom_ac1.twinx()
ax_zoom_var.plot(zoom_x, zoom_var, "-s", color="#9b59b6",
                  linewidth=1.2, markersize=3, alpha=0.7, label="Var")
ax_zoom_var.set_ylabel("Variance", fontsize=8, color="#9b59b6")
ax_zoom_var.tick_params(labelsize=7, colors="#9b59b6")

ax_zoom_ac1.set_xlabel("utterance index", fontsize=9)
ax_zoom_ac1.set_ylabel("$r_1$", fontsize=9)
ax_zoom_ac1.legend(loc="upper left", fontsize=7)
ax_zoom_ac1.grid(True, alpha=0.25)
ax_zoom_ac1.tick_params(labelsize=7)
ax_zoom_ac1.set_ylim(-1.1, 1.1)

plt.tight_layout()
out = os.path.join(FIGURES_DIR, "fig1_joey_annotated.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
plt.rcdefaults()
print(f"Saved: {out}")
