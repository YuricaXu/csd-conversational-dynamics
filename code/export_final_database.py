"""
export_final_database.py

Generates the project's two CSV deliverables in `final_database/`:

  1. friends_main_chars_vad_trajectory.csv
        Full per-utterance VAD + SEP trajectory for the six main characters
        across Friends Seasons 1-4 (~9,776 rows).  This is THE database
        produced by the pipeline — every downstream analysis runs on it.

  2. scene9_s02e18_with_vad.csv
        The 24 utterances of "The One Where Dr. Ramoray Dies" (S02E18)
        Scene 9 — Joey's friends gather to comfort him after his
        soap-opera character is killed off.  This is the case-study
        subset that drives Figure 4.

Run from the repo root or from anywhere — paths self-resolve.
"""

import os, sys, csv, json

HERE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
sys.path.insert(0, MODEL_DIR)

from data_loader import load_and_sort, filter_speakers
from vad_engine import load_vad_lexicon, build_trajectories, utterance_vad
from config import JSON_PATHS, VAD_LEXICON_PATH, MAIN_CHARACTERS, DATABASE_DIR

os.makedirs(DATABASE_DIR, exist_ok=True)


# ── Shared lexicon load ──────────────────────────────────────────────────────
print("Loading lexicon and corpus...")
vad_lex, mwe_lex = load_vad_lexicon(VAD_LEXICON_PATH)
all_utts = filter_speakers(load_and_sort(JSON_PATHS))
trajs    = build_trajectories(all_utts, vad_lex, mwe_lex, MAIN_CHARACTERS)


# ── Export 1: full main-character VAD trajectory ─────────────────────────────
out1 = os.path.join(DATABASE_DIR, "friends_main_chars_vad_trajectory.csv")
fields = [
    "utterance_id", "speaker", "episode_id", "scene_id",
    "transcript", "emotion_label",
    "valence", "arousal", "dominance", "sep",
    "vad_coverage",
]
n_rows = 0
with open(out1, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for char in MAIN_CHARACTERS:
        for r in trajs[char]:
            v, a, d = r["vad"]
            sep     = v + 0.5 * d
            w.writerow({
                "utterance_id":  r["utterance_id"],
                "speaker":       char,
                "episode_id":    r["_episode_id"],
                "scene_id":      r.get("_scene_id", ""),
                "transcript":    r.get("transcript", ""),
                "emotion_label": r.get("emotion", ""),
                "valence":       f"{v:.4f}",
                "arousal":       f"{a:.4f}",
                "dominance":     f"{d:.4f}",
                "sep":           f"{sep:.4f}",
                "vad_coverage":  f"{r.get('vad_cov', 0.0):.3f}",
            })
            n_rows += 1
print(f"  → {out1}  ({n_rows:,} rows)")


# ── Export 2: scene 9 of s02e18 with VAD ─────────────────────────────────────
out2 = os.path.join(DATABASE_DIR, "scene9_s02e18_with_vad.csv")

# Locate the train-split JSON that contains s02e18
TRAIN_PATH = next(p for p in JSON_PATHS if p.endswith("trn.json"))
with open(TRAIN_PATH) as f:
    raw = json.load(f)

scene_rows = []
for ep in raw["episodes"]:
    if ep.get("episode_id") == "s02_e18":
        scene = ep["scenes"][9]
        for i, u in enumerate(scene["utterances"], start=1):
            spk    = u.get("speakers", ["?"])[0]
            emo    = u.get("emotion", "?")
            tokens = u.get("tokens", [])
            txt    = u.get("transcript", "")
            vad, cov = utterance_vad(tokens, vad_lex, mwe_lex)
            if vad is not None:
                v, a, d = vad
                sep     = v + 0.5 * d
            else:
                v = a = d = sep = float("nan")
            scene_rows.append({
                "scene_position": i,
                "utterance_id":   u.get("utterance_id", ""),
                "speaker":        spk,
                "emotion_label":  emo,
                "transcript":     txt,
                "valence":        f"{v:.4f}" if vad else "",
                "arousal":        f"{a:.4f}" if vad else "",
                "dominance":      f"{d:.4f}" if vad else "",
                "sep":            f"{sep:.4f}" if vad else "",
                "vad_coverage":   f"{cov:.3f}",
            })
        break

with open(out2, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(scene_rows[0].keys()))
    w.writeheader()
    w.writerows(scene_rows)
print(f"  → {out2}  ({len(scene_rows)} rows)")

print("\nDone.")
