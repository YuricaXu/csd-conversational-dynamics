"""
config_csd.py  --  Configuration for Critical Slowing Down (CSD) analysis.

Core shift: Collapse = Loss of Resilience, detected via simultaneous
elevation of Lag-1 Autocorrelation and Variance (Early Warning Signals).
"""

import os
import sys

# Allow imports from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    JSON_PATHS,
    VAD_LEXICON_PATH,
    MAIN_CHARACTERS,
    COUPLING_WINDOW,
    MWE_MAX_N,
)

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── SEP formula (inherited from R4) ─────────────────────────────────────────
# SEP_t = V_t + ALPHA * D_t
ALPHA = 0.5

# ── Rolling window for EWS computation ───────────────────────────────────────
# Default window. Sensitivity analysis tests K_SENSITIVITY_LIST.
EWS_WINDOW = 5

# ── Sensitivity analysis: window sizes to test ──────────────────────────────
K_SENSITIVITY_LIST = [3, 5, 7, 10]

# ── SEP Deviation Precondition ──────────────────────────────────────────────
# Collapse requires the system to FIRST be in the negative subspace.
# Δ̄_t < -K_SIGMA * sigma_SEP  (sustained negative deviation from equilibrium)
# Lowered from 0.8 to 0.5: Layer 1 captures the "seedling" of distress,
# Layer 2 (CSD) determines whether it is genuine collapse vs transient dip.
K_SIGMA = 0.5

# ── Lead-lag: CSD signal can precede the emotional trough ───────────────────
# In physical systems, AC(1) rises BEFORE the state variable reaches its
# nadir. Allow Layer 2 to fire up to LEAD_LAG steps ahead of Layer 1.
# Detection: Layer1(t) AND (Layer2(t) OR Layer2(t-1) OR ... OR Layer2(t-LEAD_LAG))
LEAD_LAG = 1

# ── CSD Collapse Thresholds ─────────────────────────────────────────────────
# WITHIN the negative subspace, collapse is flagged when BOTH AC(1) and
# Variance exceed their Q75 (computed over the FULL trajectory, not just
# the negative subspace, to preserve the distributional meaning of Q75).
AC1_THRESHOLD_QUANTILE = 0.75
VAR_THRESHOLD_QUANTILE = 0.75

# Minimum contiguous segment length to qualify as a collapse region
MIN_SEGMENT_LEN = 3

# ── Recovery Time ────────────────────────────────────────────────────────────
# After a perturbation (deviation below baseline), count steps to return
# to within RECOVERY_BAND * sigma of the character's equilibrium.
RECOVERY_BAND = 0.5  # within 0.5 sigma of mu_SEP = "recovered"

# ── Visualization ────────────────────────────────────────────────────────────
FIGURE_DPI = 150
FIGURE_FORMAT = "png"
