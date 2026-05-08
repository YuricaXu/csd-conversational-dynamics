"""
build_fig4_what_behind_the_words.py

Figure 4 — Case study of Friends s02e18, Scene 9 ("The One Where Dr Ramoray
Dies"), the comfort scene immediately after Joey learns his soap-opera
character has been killed off.

Two co-flowing rivers (comforters & Joey) plus per-utterance VAD-lexicon
readings.  Joey's river uses a teal collapse gradient that lightens where
the surrounding comforter SEP is most positive — visualising the failed
emotional hedge: the comforters' words are above zero while Joey's
trajectory stays flat at -0.6.  The lexicon's VAD reading (white crosses)
clusters near the centre, unable to separate sufferer from comforter.

Output: figures/fig4_what_behind_the_words.png
"""

import os, sys, json

HERE        = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(HERE, "..", "model")
FIGURES_DIR = os.path.join(HERE, "..", "..", "figures")
sys.path.insert(0, MODEL_DIR)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb
from scipy.interpolate import make_interp_spline

from vad_engine import load_vad_lexicon, utterance_vad
from config import VAD_LEXICON_PATH, JSON_PATHS

os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Load lexicon and the s02e18 scene 9 utterances ───────────────────────────
vad_uni, vad_mwe = load_vad_lexicon(VAD_LEXICON_PATH)

# Find the train-split file that contains s02e18 (Emory NLP places it there)
TRAIN_PATH = next(p for p in JSON_PATHS if p.endswith("trn.json"))
with open(TRAIN_PATH) as f:
    raw = json.load(f)

scene_utts = []
for ep in raw["episodes"]:
    if ep.get("episode_id") == "s02_e18":
        for u in ep["scenes"][9]["utterances"]:
            spk    = u.get("speakers", ["?"])[0]
            emo    = u.get("emotion", "?")
            tokens = u.get("tokens", [])
            txt    = u.get("transcript", "")
            vad, _ = utterance_vad(tokens, vad_uni, vad_mwe)
            sep = (vad[0] + 0.5 * vad[2]) if vad else None
            scene_utts.append({"spk": spk, "emo": emo, "txt": txt, "sep": sep})
        break


# ── Constants ────────────────────────────────────────────────────────────────
CHAR_COLOR = {
    "Joey Tribbiani": "#3a607a",
    "Ross Geller":    "#74add1",
    "Rachel Green":   "#f46d43",
    "Chandler Bing":  "#fdae61",
    "Monica Geller":  "#a6d96a",
    "Phoebe Buffay":  "#d9a8e0",
}
LABEL_SEP = {
    "Sad": -0.62, "Mad": -0.42, "Scared": -0.30,
    "Neutral": 0.0,
    "Peaceful": +0.50, "Powerful": +0.62, "Joyful": +0.70,
}

# Joey collapse palette — deep teal-blue, same brightness layer as the
# comforter colours so the two rivers feel co-planar.  Hue stays cool
# (depression / cold-water register) and is distinct from Ross's light blue.
JOEY_DEEP   = "#3a607a"   # alone in grief (origin)
JOEY_BASE   = "#4d7a92"   # standard collapse
JOEY_HEDGED = "#7d96a8"   # comfort pressure visible

# Short quote per utterance (1–3 lines max for compactness)
QUOTES = {
    1:  "C'mon",
    2:  "Joey",
    3:  "Open up",
    4:  "don't feel\nlike talkin",
    5:  "we care\nabout you",
    6:  "we're\nworried",
    7:  "have\nto pee",
    8:  "Sorry\nJoey",
    9:  "Hey",
    10: "sorry about\nyour death",
    11: "came over\nas soon",
    12: "how could\nyou not\ntell us?",
    13: "kinda hopin'\nno one\nwould find out",
    14: "maybe they\ncan bring\nyou back",
    15: "brain was\nsmashed",
    16: "you're\ngonna\nbe fine",
    17: "greatest\nthing\never happened",
    18: "I was going\nto\nincorporate",
    19: "straightened\nyour shower\ncurtain",
    20: "It's\ngonna\nbe ok",
    21: "work your\nwhole life",
    22: "I'm\nsorry man",
    23: "Joey honey",
    24: "means\nnothin\nto me",
}

n  = len(scene_utts)
xs = np.arange(1, n + 1)


# ── Helper: smooth gradient ribbon ───────────────────────────────────────────
def draw_gradient_ribbon(ax, x_pts, y_pts, rgb_pts, intensity_pts,
                         base_width=0.05, max_width=0.12,
                         n_per_segment=80, alpha=0.85, zorder=4):
    """Smooth ribbon connecting (x, y) points with per-vertex colour and width."""
    if len(x_pts) < 4:
        return
    x_arr = np.array(x_pts, dtype=float)
    y_arr = np.array(y_pts, dtype=float)
    spline = make_interp_spline(x_arr, y_arr, k=3)
    rgb_arr = np.array([to_rgb(c) for c in rgb_pts])
    int_arr = np.array(intensity_pts, dtype=float)
    n_total = n_per_segment * (len(x_arr) - 1)
    sm_x = np.linspace(x_arr[0], x_arr[-1], n_total)
    sm_y = spline(sm_x)
    sm_rgb = np.zeros((len(sm_x), 3))
    sm_int = np.zeros(len(sm_x))
    for i, x in enumerate(sm_x):
        idx = np.searchsorted(x_arr, x) - 1
        idx = max(0, min(idx, len(x_arr) - 2))
        x0, x1 = x_arr[idx], x_arr[idx + 1]
        t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        t = t * t * (3 - 2 * t)        # smoothstep
        sm_rgb[i] = rgb_arr[idx] * (1 - t) + rgb_arr[idx + 1] * t
        sm_int[i] = int_arr[idx] * (1 - t) + int_arr[idx + 1] * t
    sm_int_norm = (sm_int - sm_int.min()) / max(1e-9, sm_int.max() - sm_int.min())
    half_w = base_width + (max_width - base_width) * sm_int_norm
    for i in range(len(sm_x) - 1):
        ax.fill_between(
            [sm_x[i], sm_x[i + 1]],
            [sm_y[i] - half_w[i],     sm_y[i + 1] - half_w[i + 1]],
            [sm_y[i] + half_w[i],     sm_y[i + 1] + half_w[i + 1]],
            color=tuple(sm_rgb[i]), alpha=alpha,
            edgecolor="none", linewidth=0, zorder=zorder)


# ── Build river data ─────────────────────────────────────────────────────────
joey_pts, joey_int = [], []
friend_pts, friend_colors, friend_int = [], [], []

for i, u in enumerate(scene_utts):
    x = xs[i]
    y = LABEL_SEP[u["emo"]]
    if u["spk"] == "Joey Tribbiani":
        joey_pts.append((x, y))
        joey_int.append(abs(y))
    else:
        friend_pts.append((x, y))
        friend_colors.append(CHAR_COLOR[u["spk"]])
        friend_int.append(abs(y))

joey_x   = [p[0] for p in joey_pts]
joey_y   = [p[1] for p in joey_pts]
friend_x = [p[0] for p in friend_pts]
friend_y = [p[1] for p in friend_pts]


# ── Joey hedge gradient ──────────────────────────────────────────────────────
def hedge_exposure(joey_xi, comf_x, comf_y, window=3.5):
    """Local positive comforter pressure around Joey utterance index joey_xi."""
    weights = np.exp(-((np.array(comf_x) - joey_xi) / window) ** 2)
    pos     = np.maximum(0.0, np.array(comf_y))
    return float(np.sum(pos * weights))

j_hedge = np.array([hedge_exposure(jx, friend_x, friend_y) for jx in joey_x])
j_hedge_norm = (j_hedge - j_hedge.min()) / max(1e-9, j_hedge.max() - j_hedge.min())

deep   = np.array(to_rgb(JOEY_DEEP))
base   = np.array(to_rgb(JOEY_BASE))
hedged = np.array(to_rgb(JOEY_HEDGED))

joey_ribbon_colors = []
for h in j_hedge_norm:
    if h < 0.5:
        t = h / 0.5
        rgb = deep * (1 - t) + base * t
    else:
        t = (h - 0.5) / 0.5
        rgb = base * (1 - t) + hedged * t
    joey_ribbon_colors.append(
        "#{:02x}{:02x}{:02x}".format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
    )

all_vad_x = [xs[i] for i, u in enumerate(scene_utts) if u["sep"] is not None]
all_vad_y = [u["sep"] for u in scene_utts if u["sep"] is not None]


# ── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(26, 12), facecolor="#0d1117")
ax  = fig.add_axes([0.05, 0.08, 0.93, 0.82])
ax.set_facecolor("#0d1117")

ax.axhspan(+0.30, +1.00, alpha=0.05, color="#f46d43", zorder=1)
ax.axhspan(-1.00, -0.30, alpha=0.05, color="#444444", zorder=1)
ax.axhspan(-0.18, +0.18, alpha=0.04, color="#999999", zorder=1)
ax.axhline(0, color="#222", lw=0.6, zorder=2)

ax.text(0.005, +0.65, "COMFORTERS  RIVER",
        transform=ax.get_yaxis_transform(),
        color="#f46d43", fontsize=10.5, alpha=0.7, va="center", ha="left",
        fontweight="bold")
ax.text(0.005, -0.55, "JOEY'S  RIVER",
        transform=ax.get_yaxis_transform(),
        color="#888", fontsize=10.5, alpha=0.85, va="center", ha="left",
        fontweight="bold")
ax.text(0.005, 0.00, "AI MIRAGE  ·  where VAD readings cluster",
        transform=ax.get_yaxis_transform(),
        color="#999", fontsize=8.5, alpha=0.65, va="center", ha="left",
        fontstyle="italic")


# ── Comforters' gradient ribbon (shadow + main) ─────────────────────────────
draw_gradient_ribbon(ax, friend_x, friend_y, friend_colors, friend_int,
                     base_width=0.10, max_width=0.20, alpha=0.18, zorder=3)
draw_gradient_ribbon(ax, friend_x, friend_y, friend_colors, friend_int,
                     base_width=0.05, max_width=0.11, alpha=0.85, zorder=4)


# ── Joey's collapse-hedge gradient ribbon ───────────────────────────────────
draw_gradient_ribbon(ax, joey_x, joey_y, joey_ribbon_colors, joey_int,
                     base_width=0.10, max_width=0.18, alpha=0.22, zorder=3)
draw_gradient_ribbon(ax, joey_x, joey_y, joey_ribbon_colors, joey_int,
                     base_width=0.06, max_width=0.10, alpha=0.96, zorder=5)


# ── Per-character thread (faint background weaving) ─────────────────────────
char_threads = {}
for i, u in enumerate(scene_utts):
    if u["spk"] != "Joey Tribbiani":
        char_threads.setdefault(u["spk"], []).append((xs[i], LABEL_SEP[u["emo"]]))

for spk, pts in char_threads.items():
    if len(pts) >= 2:
        col = CHAR_COLOR[spk]
        xx = [p[0] for p in pts]; yy = [p[1] for p in pts]
        if len(xx) >= 4:
            sp = make_interp_spline(xx, yy, k=3)
            sx = np.linspace(min(xx), max(xx), 80)
            sy = sp(sx)
            ax.plot(sx, sy, color=col, lw=1.0, alpha=0.30, zorder=4)
        else:
            ax.plot(xx, yy, color=col, lw=1.0, alpha=0.30, zorder=4)


# ── Markers ──────────────────────────────────────────────────────────────────
for i, u in enumerate(scene_utts):
    if u["spk"] == "Joey Tribbiani":
        continue
    x = xs[i]; y = LABEL_SEP[u["emo"]]
    ax.scatter([x], [y], s=190, color=CHAR_COLOR[u["spk"]], zorder=9,
               edgecolors="white", linewidths=1.4)

for i, u in enumerate(scene_utts):
    if u["spk"] != "Joey Tribbiani":
        continue
    x = xs[i]; y = LABEL_SEP[u["emo"]]
    ax.scatter([x], [y], s=260, color=JOEY_DEEP, zorder=10,
               edgecolors="white", linewidths=1.6)


# ── Quote snippets ───────────────────────────────────────────────────────────
for i, u in enumerate(scene_utts):
    x = xs[i]; y = LABEL_SEP[u["emo"]]
    quote = QUOTES.get(i + 1, "")
    if not quote:
        continue
    is_joey = (u["spk"] == "Joey Tribbiani")
    col = CHAR_COLOR[u["spk"]]
    if is_joey:
        ax.text(x, y - 0.16, quote,
                fontsize=11.0, color="#e5e5e5", ha="center", va="top",
                fontstyle="italic", alpha=0.98,
                bbox=dict(boxstyle="round,pad=0.32",
                          facecolor="#0d1117", edgecolor="#666",
                          alpha=0.90, linewidth=0.8))
    else:
        ax.text(x, y + 0.16, quote,
                fontsize=10.5, color=col, ha="center", va="bottom",
                alpha=0.98,
                bbox=dict(boxstyle="round,pad=0.26",
                          facecolor="#0d1117", edgecolor=col,
                          alpha=0.70, linewidth=0.7))


# ── VAD readings (the AI mirage) ─────────────────────────────────────────────
ax.scatter(all_vad_x, all_vad_y, marker="x", s=55,
           color="#dddddd", alpha=0.75, zorder=8, linewidths=1.4)
ax.plot(all_vad_x, all_vad_y, color="#bbb", lw=0.8,
        ls=":", alpha=0.45, zorder=6)


# ── Axes ─────────────────────────────────────────────────────────────────────
ax.set_xlim(0.3, n + 0.7)
ax.set_ylim(-1.0, 1.0)
ax.set_xticks(xs)
ax.set_xticklabels([str(i) for i in xs], fontsize=7.5, color="#666")
ax.set_xlabel("Utterance sequence in scene  →", color="#aaa", fontsize=10)
ax.set_ylabel("SEP   ↑ healthy   ↓ collapse", color="#aaa", fontsize=10.5)
ax.tick_params(colors="#666")
for sp in ax.spines.values():
    sp.set_color("#222")

ax.set_title(
    "What Behind the Words Reaches Joey?",
    color="white", fontsize=15, pad=14, fontweight="bold")

legend_patches = [
    mpatches.Patch(color=CHAR_COLOR["Rachel Green"],   label="Rachel"),
    mpatches.Patch(color=CHAR_COLOR["Chandler Bing"],  label="Chandler"),
    mpatches.Patch(color=CHAR_COLOR["Monica Geller"],  label="Monica"),
    mpatches.Patch(color=CHAR_COLOR["Phoebe Buffay"],  label="Phoebe"),
    mpatches.Patch(color=CHAR_COLOR["Ross Geller"],    label="Ross"),
    mpatches.Patch(color=JOEY_DEEP,                    label="Joey  (deep grief)"),
    mpatches.Patch(color=JOEY_HEDGED,                  label="Joey  (hedged)"),
    mpatches.Patch(color="#dddddd",                    label="✕  VAD reading"),
]
ax.legend(handles=legend_patches, loc="upper right", fontsize=8.5,
          facecolor="#1a1f2e", edgecolor="#333",
          labelcolor="#ccc", framealpha=0.92, ncol=4)

out = os.path.join(FIGURES_DIR, "fig4_what_behind_the_words.png")
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Saved: {out}")
