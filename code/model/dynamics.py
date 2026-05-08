"""
dynamics.py — Lightweight utilities used by the CSD detector.

Slimmed down from a larger version that also implemented earlier R1/R2/R3
collapse pipelines.  Only the helpers needed by `csd_detector.py` are kept
in this release: a quantile estimator and an episode-boundary check.
"""

from __future__ import annotations
from typing import Any, Dict, List


# ── Quantile estimator (linear interpolation) ────────────────────────────────

def _quantile(values: List[float], q: float) -> float:
    """
    Return the q-th quantile (0 < q < 1) of `values` via linear interpolation
    between the two surrounding order statistics.

    Returns NaN if `values` is empty.  Input is not modified.
    """
    arr = sorted(values)
    n   = len(arr)
    if n == 0:
        return float("nan")
    idx = q * (n - 1)
    lo  = int(idx)
    hi  = min(lo + 1, n - 1)
    return arr[lo] + (arr[hi] - arr[lo]) * (idx - lo)


# ── Episode-boundary check ────────────────────────────────────────────────────

def is_valid_transition(rows: List[Dict[str, Any]], t: int) -> bool:
    """
    True iff the step rows[t] → rows[t+1] stays within the same episode.

    Cross-episode steps span narrative gaps (sometimes days or weeks of
    in-show time), so emotional-continuity assumptions are unjustifiable
    across them.  The CSD detector uses this to refuse rolling-window
    computations that would straddle two episodes.

    The point itself is retained — only the step is excluded from any
    rolling computation that depends on continuity.
    """
    if t + 1 >= len(rows):
        return False
    return rows[t]["_episode_id"] == rows[t + 1]["_episode_id"]
