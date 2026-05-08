"""
config.py — Project paths and global constants.

Paths are resolved relative to this file so the pipeline works regardless of
the current working directory.  Layout assumed:

    <repo_root>/
        code/model/      ← this file
        code/figures/
        data/            ← Emory NLP JSON + NRC VAD lexicon
"""

import os

# ── Path resolution ───────────────────────────────────────────────────────────
HERE         = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_DIR     = os.path.join(REPO_ROOT, "data")
FIGURES_DIR  = os.path.join(REPO_ROOT, "figures")
DATABASE_DIR = os.path.join(REPO_ROOT, "final_database")

JSON_PATHS = [
    os.path.join(DATA_DIR, "emotion-detection-trn.json"),
    os.path.join(DATA_DIR, "emotion-detection-dev.json"),
    os.path.join(DATA_DIR, "emotion-detection-tst.json"),
]
VAD_LEXICON_PATH = os.path.join(DATA_DIR, "NRC-VAD-Lexicon-v2.1.txt")

# Make sure output directories exist
os.makedirs(FIGURES_DIR,  exist_ok=True)
os.makedirs(DATABASE_DIR, exist_ok=True)

# Backwards-compatibility: some legacy code references OUTPUT_DIR
OUTPUT_DIR = FIGURES_DIR


# ── Six main characters ───────────────────────────────────────────────────────
MAIN_CHARACTERS = [
    "Chandler Bing",
    "Ross Geller",
    "Monica Geller",
    "Joey Tribbiani",
    "Rachel Green",
    "Phoebe Buffay",
]


# ── VAD / Preprocessing ───────────────────────────────────────────────────────
# MWE matching uses bigrams (greedy left-to-right, bigram-first)
MWE_MAX_N = 2


# ── Interpersonal coupling (used by dynamics.py) ─────────────────────────────
COUPLING_WINDOW = 6   # number of prior global utterances to search
