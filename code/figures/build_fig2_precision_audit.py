"""
build_fig2_precision_audit.py

Figure 2 — Manual-validation audit of all 20 detected segments.
Left  panel: True positives (6) vs false positives (14) — precision = 30%.
Right panel: Failure-mode taxonomy across the 14 false positives, naming
each segment under its dominant mode.

The verdicts and failure modes encoded here are the output of manual
review against the codebook in METHOD.md (Beck, Nolen-Hoeksema, Rude,
Al-Mosaiwi, Bowlby).  Some segments exhibit two failure modes
simultaneously, so the per-mode counts on the right panel sum to more
than 14.

Output: figures/fig2_precision_audit.png
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

os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Manual-review verdicts on all 20 detected segments ───────────────────────
# (verdict, character, episode, dominant failure mode if FP)
SEG_VERDICTS = {
    "SEG01": (0, "Chandler Bing", "S01E13", "defensive humor"),
    "SEG02": (0, "Chandler Bing", "S03E07", "cross-scene contamination"),
    "SEG03": (1, "Chandler Bing", "S04E03", None),
    "SEG04": (0, "Chandler Bing", "S04E22", "defensive humor"),
    "SEG05": (0, "Ross Geller",   "S01E08", "normal trajectory"),
    "SEG06": (0, "Ross Geller",   "S01E15", "lexicon error"),
    "SEG07": (1, "Ross Geller",   "S02E21", None),
    "SEG08": (0, "Ross Geller",   "S03E21", "normal trajectory"),
    "SEG09": (0, "Monica Geller", "S02E15", "lexicon error"),
    "SEG10": (1, "Monica Geller", "S02E24", None),
    "SEG11": (0, "Monica Geller", "S04E01", "cross-scene contamination"),
    "SEG12": (1, "Monica Geller", "S04E05", None),
    "SEG13": (0, "Joey Tribbiani","S02E12", "statistical noise"),
    "SEG14": (1, "Joey Tribbiani","S02E18", None),
    "SEG15": (0, "Joey Tribbiani","S04E16", "statistical noise"),
    "SEG16": (0, "Rachel Green",  "S01E21", "emotion attribution"),
    "SEG17": (1, "Rachel Green",  "S03E15", None),
    "SEG18": (0, "Phoebe Buffay", "S03E06", "emotion attribution"),
    "SEG19": (0, "Phoebe Buffay", "S04E18", "cross-scene contamination"),
    "SEG20": (0, "Phoebe Buffay", "S04E23", "emotion attribution"),
}

# Failure-mode taxonomy.  Some segments appear under two modes
# (cross-scene + lexicon for SEG06, attribution + noise for SEG15);
# the right panel reports per-mode counts using the full taxonomy.
FAILURE_TAXONOMY = {
    "Cross-scene\ncontamination":   ["SEG02", "SEG06", "SEG11", "SEG19"],
    "Emotion\nattribution":         ["SEG15", "SEG16", "SEG18", "SEG20"],
    "Defensive\nhumor":             ["SEG01", "SEG04"],
    "Lexicon\nerror":               ["SEG06", "SEG09"],
    "Normal\ntrajectory":           ["SEG05", "SEG08"],
    "Statistical\nnoise":           ["SEG13", "SEG15"],
}

n_TP = sum(1 for v, *_ in SEG_VERDICTS.values() if v == 1)
n_FP = sum(1 for v, *_ in SEG_VERDICTS.values() if v == 0)
precision = n_TP / 20

print(f"True Positives:  {n_TP}/20 = {precision*100:.0f}%")
print(f"False Positives: {n_FP}/20 = {(1-precision)*100:.0f}%")


# ── Figure ────────────────────────────────────────────────────────────────────
fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(13, 5),
    gridspec_kw={"width_ratios": [1, 1.6]},
)

# ── Left panel — TP vs FP bar chart ──────────────────────────────────────────
ax_titles = ["True\nPositives\n(real collapse)", "False\nPositives\n(model wrong)"]
counts    = [n_TP, n_FP]
colors    = ["#27ae60", "#e74c3c"]
bars = axA.bar(ax_titles, counts, color=colors, alpha=0.85,
               edgecolor="black", linewidth=1.2)
for bar, c in zip(bars, counts):
    axA.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.3,
             f"{c}", ha="center", va="bottom",
             fontsize=22, fontweight="bold")
axA.set_ylabel("Number of segments out of 20", fontsize=11)
axA.set_ylim(0, 18)
axA.set_title(f"(A)  Pipeline output audit\nPrecision = {precision:.2f}  ({n_TP}/20)",
              fontsize=12, fontweight="bold")
axA.grid(True, axis="y", alpha=0.2)

# ── Right panel — failure-mode taxonomy ──────────────────────────────────────
modes_sorted = sorted(FAILURE_TAXONOMY.items(), key=lambda kv: -len(kv[1]))
labels = [m[0] for m in modes_sorted]
sizes  = [len(m[1]) for m in modes_sorted]
mode_colors = ["#c0392b", "#d35400", "#f39c12", "#2980b9", "#8e44ad", "#7f8c8d"]
ypos = np.arange(len(labels))
b = axB.barh(ypos, sizes, color=mode_colors, alpha=0.85,
             edgecolor="black", linewidth=0.6)
for bar, s, segs in zip(b, sizes, [m[1] for m in modes_sorted]):
    axB.text(bar.get_width()+0.05, bar.get_y()+bar.get_height()/2,
             f"{s} cases — {', '.join(segs)}",
             va="center", fontsize=8.5)
axB.set_yticks(ypos)
axB.set_yticklabels(labels, fontsize=10)
axB.invert_yaxis()
axB.set_xlim(0, 6)
axB.set_xlabel("Number of false-positive segments\n"
               "(some seg counted in 2 modes — total events ≥ 14)",
               fontsize=10)
axB.set_title("(B)  Failure-mode taxonomy across 14 false positives",
              fontsize=12, fontweight="bold")
axB.grid(True, axis="x", alpha=0.2)

plt.tight_layout()
out = os.path.join(FIGURES_DIR, "fig2_precision_audit.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
