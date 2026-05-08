"""
csd_detector.py  --  Critical Slowing Down (CSD) Collapse Detection (v2)

Corrected definition:
  A collapse is the LOSS OF RESILIENCE that occurs when the system is
  already in a negative emotional state.  Two layers:

  Layer 1 — State precondition (SEP Deviation):
    The system must be in sustained negative deviation from equilibrium:
      Δ̄_t < -k_σ * σ_SEP
    This ensures we are looking at genuinely distressed periods,
    excluding high-volatility positive scenes.

  Layer 2 — Dynamical CSD signal:
    Within the negative subspace, AC(1) and Variance must both be elevated
    (> Q75), indicating the system has lost its ability to self-recover.
      AC(1) ↑ : system is "sticky", current state depends on previous state
      Var   ↑ : system is "flickering", oscillating before collapse

  Validation — Recovery Time (τ):
    τ = steps for SEP to return within 0.5σ of equilibrium.
    Collapse regions must show τ_inside >> τ_outside.

Reference:
  van de Leemput et al. (2014). Critical slowing down as early warning
  for the onset and termination of depression. PNAS, 111(1), 87-92.
"""

from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config_csd import (
    ALPHA,
    EWS_WINDOW,
    K_SIGMA,
    LEAD_LAG,
    AC1_THRESHOLD_QUANTILE,
    VAR_THRESHOLD_QUANTILE,
    MIN_SEGMENT_LEN,
    RECOVERY_BAND,
    K_SENSITIVITY_LIST,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dynamics import is_valid_transition, _quantile


# ═══════════════════════════════════════════════════════════════════════════
# Step A: SEP trajectory & equilibrium
# ═══════════════════════════════════════════════════════════════════════════

def compute_equilibrium(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    valid = [r for r in rows if r.get("vad_cov", 0) > 0]
    if not valid:
        valid = rows
    Vs = [r["vad"][0] for r in valid]
    Ds = [r["vad"][2] for r in valid]
    mu_sep = mean([v + ALPHA * d for v, d in zip(Vs, Ds)])
    all_sep = [r["vad"][0] + ALPHA * r["vad"][2] for r in rows]
    sigma_sep = stdev(all_sep) if len(all_sep) > 1 else 0.01
    return {
        "mu_v": mean(Vs),
        "mu_a": mean([r["vad"][1] for r in valid]),
        "mu_d": mean(Ds),
        "mu_sep": mu_sep,
        "sigma_sep": sigma_sep,
    }


def compute_sep_trajectory(
    rows: List[Dict[str, Any]], eq: Dict[str, float],
) -> Tuple[List[float], List[float]]:
    mu = eq["mu_sep"]
    sep_seq = [r["vad"][0] + ALPHA * r["vad"][2] for r in rows]
    delta_seq = [s - mu for s in sep_seq]
    return sep_seq, delta_seq


# ═══════════════════════════════════════════════════════════════════════════
# Step B: Rolling EWS — AC(1) and Moving Variance
# ═══════════════════════════════════════════════════════════════════════════

def _get_episode_window(t: int, rows: List[Dict[str, Any]], k: int) -> List[int]:
    ep = rows[t]["_episode_id"]
    return [j for j in range(max(0, t - k + 1), t + 1)
            if rows[j]["_episode_id"] == ep]


def compute_lag1_autocorrelation(
    values: List[float], rows: List[Dict[str, Any]], k: int,
) -> List[float]:
    ac1_seq = []
    for t in range(len(values)):
        win_idx = _get_episode_window(t, rows, k)
        win_vals = [values[j] for j in win_idx]
        if len(win_vals) < 3:
            ac1_seq.append(0.0)
            continue
        x, y = win_vals[:-1], win_vals[1:]
        mx, my = mean(x), mean(y)
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        den_x = math.sqrt(sum((xi - mx) ** 2 for xi in x))
        den_y = math.sqrt(sum((yi - my) ** 2 for yi in y))
        if den_x < 1e-12 or den_y < 1e-12:
            ac1_seq.append(0.0)
        else:
            ac1_seq.append(num / (den_x * den_y))
    return ac1_seq


def compute_moving_variance(
    values: List[float], rows: List[Dict[str, Any]], k: int,
) -> List[float]:
    var_seq = []
    for t in range(len(values)):
        win_idx = _get_episode_window(t, rows, k)
        win_vals = [values[j] for j in win_idx]
        if len(win_vals) < 2:
            var_seq.append(0.0)
            continue
        mu = mean(win_vals)
        v = sum((x - mu) ** 2 for x in win_vals) / (len(win_vals) - 1)
        var_seq.append(v)
    return var_seq


def _rolling_mean_episode(
    values: List[float], rows: List[Dict[str, Any]], k: int,
) -> List[float]:
    result = []
    for t in range(len(values)):
        win_idx = _get_episode_window(t, rows, k)
        win_vals = [values[j] for j in win_idx]
        result.append(mean(win_vals) if win_vals else values[t])
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Step C & D: Two-layer CSD Collapse Detection
# ═══════════════════════════════════════════════════════════════════════════

def contiguous_true_segments(
    mask: List[bool], rows: List[Dict[str, Any]], min_len: int = MIN_SEGMENT_LEN,
) -> List[Tuple[int, int]]:
    segments = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while (j + 1 < n and mask[j + 1]
                   and rows[j + 1]["_episode_id"] == rows[j]["_episode_id"]):
                j += 1
            if j - i + 1 >= min_len:
                segments.append((i, j))
            i = j + 1
        else:
            i += 1
    return segments


def detect_csd_collapse(
    rows: List[Dict[str, Any]],
    k: int = EWS_WINDOW,
    k_sigma: float = K_SIGMA,
    lead_lag: int = LEAD_LAG,
    ac1_q: float = AC1_THRESHOLD_QUANTILE,
    var_q: float = VAR_THRESHOLD_QUANTILE,
    min_len: int = MIN_SEGMENT_LEN,
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    """
    Two-layer CSD collapse detection with lead-lag.

    Layer 1 (State precondition):
      Δ̄_t < -k_σ * σ_SEP   (sustained negative deviation)
      k_σ = 0.5 — captures the "seedling" of distress.

    Layer 2 (Dynamical CSD signal):
      AC(1)_t > Q75  AND  Var_t > Q75
      In physical systems AC(1) rises BEFORE the state variable reaches
      its nadir.  We allow Layer 2 to fire up to `lead_lag` steps ahead.

    Collapse = Layer1(t) AND (Layer2(t) OR Layer2(t-1) OR ... OR Layer2(t-lead_lag))
    with the constraint that the lead-lag lookback stays within the same episode.
    """
    n = len(rows)

    # Step A: equilibrium & SEP
    eq = compute_equilibrium(rows)
    sep_seq, delta_seq = compute_sep_trajectory(rows, eq)

    # Step B: EWS on SEP trajectory
    ac1_seq = compute_lag1_autocorrelation(sep_seq, rows, k)
    var_seq = compute_moving_variance(sep_seq, rows, k)

    # Smoothed deviation for Layer 1
    roll_delta = _rolling_mean_episode(delta_seq, rows, k)

    # Thresholds
    sigma = eq["sigma_sep"]
    delta_threshold = -k_sigma * sigma
    ac1_threshold = _quantile(ac1_seq, ac1_q)
    var_threshold = _quantile(var_seq, var_q)

    # Layer 1: sustained negative deviation
    layer1_mask = [roll_delta[t] < delta_threshold for t in range(n)]

    # Layer 2 (raw): CSD signals elevated at time t
    layer2_raw = [
        ac1_seq[t] > ac1_threshold and var_seq[t] > var_threshold
        for t in range(n)
    ]

    # Layer 2 (with lead-lag): True if CSD fired at t or any of
    # t-1 .. t-lead_lag (same episode only)
    layer2_mask = [False] * n
    for t in range(n):
        ep = rows[t]["_episode_id"]
        for lag in range(lead_lag + 1):  # lag = 0, 1, ..., lead_lag
            j = t - lag
            if j >= 0 and rows[j]["_episode_id"] == ep and layer2_raw[j]:
                layer2_mask[t] = True
                break

    # Combined: Layer 1 AND Layer 2 (with lead-lag)
    collapse_mask = [layer1_mask[t] and layer2_mask[t] for t in range(n)]

    segments = contiguous_true_segments(collapse_mask, rows, min_len)

    info = {
        "eq": eq,
        "sep_seq": sep_seq,
        "delta_seq": delta_seq,
        "roll_delta": roll_delta,
        "ac1_seq": ac1_seq,
        "var_seq": var_seq,
        "delta_threshold": delta_threshold,
        "ac1_threshold": ac1_threshold,
        "var_threshold": var_threshold,
        "layer1_mask": layer1_mask,
        "layer2_raw": layer2_raw,
        "layer2_mask": layer2_mask,
        "csd_mask": collapse_mask,
        "n_layer1": sum(layer1_mask),
        "n_layer2_raw": sum(layer2_raw),
        "n_layer2": sum(layer2_mask),
        "n_csd_flagged": sum(collapse_mask),
        "k_used": k,
        "lead_lag": lead_lag,
    }

    return segments, info


# ═══════════════════════════════════════════════════════════════════════════
# Step E: Recovery Time (τ)
# ═══════════════════════════════════════════════════════════════════════════

def compute_recovery_times(
    rows: List[Dict[str, Any]],
    sep_seq: List[float],
    eq: Dict[str, float],
    band: float = RECOVERY_BAND,
) -> List[Optional[int]]:
    mu = eq["mu_sep"]
    sigma = eq["sigma_sep"]
    n = len(rows)
    perturbed_threshold = mu - band * sigma
    recovery_lower = mu - band * sigma
    recovery_upper = mu + band * sigma

    tau_seq: List[Optional[int]] = [None] * n
    for t in range(n):
        if sep_seq[t] >= perturbed_threshold:
            continue
        ep = rows[t]["_episode_id"]
        steps = 0
        for t2 in range(t + 1, n):
            if rows[t2]["_episode_id"] != ep:
                break
            steps += 1
            if recovery_lower <= sep_seq[t2] <= recovery_upper:
                break
        if steps > 0:
            tau_seq[t] = steps
    return tau_seq


def recovery_time_validation(
    rows: List[Dict[str, Any]],
    segments: List[Tuple[int, int]],
    info: Dict[str, Any],
    band: float = RECOVERY_BAND,
) -> Dict[str, Any]:
    sep_seq = info["sep_seq"]
    eq = info["eq"]
    tau_seq = compute_recovery_times(rows, sep_seq, eq, band)

    inside_indices = set()
    for s, e in segments:
        inside_indices.update(range(s, e + 1))

    tau_inside, tau_outside = [], []
    for t in range(len(rows)):
        if tau_seq[t] is None:
            continue
        if t in inside_indices:
            tau_inside.append(tau_seq[t])
        else:
            tau_outside.append(tau_seq[t])

    result = {
        "tau_seq": tau_seq,
        "tau_inside_values": tau_inside,
        "tau_outside_values": tau_outside,
        "tau_inside": mean(tau_inside) if tau_inside else None,
        "tau_outside": mean(tau_outside) if tau_outside else None,
        "n_perturbations_inside": len(tau_inside),
        "n_perturbations_outside": len(tau_outside),
    }

    if result["tau_inside"] is not None and result["tau_outside"] is not None:
        result["tau_ratio"] = (result["tau_inside"] / result["tau_outside"]
                               if result["tau_outside"] > 0 else float("inf"))
        if result["tau_ratio"] > 1.5:
            result["p_description"] = (
                "VALIDATED: tau_inside significantly exceeds tau_outside. "
                "Recovery mechanism is impaired in detected regions.")
        elif result["tau_ratio"] > 1.0:
            result["p_description"] = (
                "Mild support: tau_inside slightly exceeds tau_outside.")
        else:
            result["p_description"] = (
                "NOT validated: recovery time is not elevated in detected regions.")
    else:
        result["tau_ratio"] = None
        result["p_description"] = "Insufficient perturbation data for comparison."

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Supplementary dynamical metrics
# ═══════════════════════════════════════════════════════════════════════════

def csd_dynamical_metrics(
    rows: List[Dict[str, Any]],
    info: Dict[str, Any],
    segments: List[Tuple[int, int]],
) -> Dict[str, Optional[float]]:
    mask = info["csd_mask"]
    delta_seq = info["delta_seq"]
    drift_vals, escape_vals = [], []
    for t in range(len(mask) - 1):
        if not is_valid_transition(rows, t):
            continue
        if mask[t]:
            drift_vals.append(delta_seq[t + 1] - delta_seq[t])
            escape_vals.append(0.0 if mask[t + 1] else 1.0)
    lengths = [e - s + 1 for s, e in segments]
    return {
        "delta_drift": mean(drift_vals) if drift_vals else None,
        "dwell": mean(lengths) if lengths else 0.0,
        "escape": mean(escape_vals) if escape_vals else 1.0,
        "seg_lengths": lengths,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Label enrichment (post-hoc, NOT used for detection)
# ═══════════════════════════════════════════════════════════════════════════

def label_enrichment(
    rows: List[Dict[str, Any]],
    segments: List[Tuple[int, int]],
    target_emotions: set = None,
) -> Dict[str, Any]:
    if target_emotions is None:
        target_emotions = {"Sad", "Mad"}
    n = len(rows)
    overall_rate = sum(1 for r in rows if r.get("emotion", "") in target_emotions) / n if n else 0
    inside_indices = set()
    for s, e in segments:
        inside_indices.update(range(s, e + 1))
    if not inside_indices:
        return {"overall_rate": overall_rate, "inside_rate": None, "enrichment": None}
    inside_rate = sum(1 for t in inside_indices
                      if rows[t].get("emotion", "") in target_emotions) / len(inside_indices)
    return {
        "overall_rate": overall_rate,
        "inside_rate": inside_rate,
        "enrichment": "ELEVATED" if inside_rate > overall_rate + 0.05 else "similar",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Full analysis for one character (single k)
# ═══════════════════════════════════════════════════════════════════════════

def run_csd_analysis(
    character: str,
    rows: List[Dict[str, Any]],
    k: int = EWS_WINDOW,
) -> Dict[str, Any]:
    segments, info = detect_csd_collapse(rows, k=k)
    tau_result = recovery_time_validation(rows, segments, info)
    dyn_metrics = csd_dynamical_metrics(rows, info, segments)
    label_enrich = label_enrichment(rows, segments)
    return {
        "character": character,
        "n_utterances": len(rows),
        "segments": segments,
        "info": info,
        "tau_validation": tau_result,
        "dynamical_metrics": dyn_metrics,
        "label_enrichment": label_enrich,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Sensitivity analysis across window sizes
# ═══════════════════════════════════════════════════════════════════════════

def sensitivity_analysis(
    character: str,
    rows: List[Dict[str, Any]],
    k_list: List[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run CSD detection for multiple window sizes k.
    Returns a list of summary dicts, one per k value.
    """
    if k_list is None:
        k_list = K_SENSITIVITY_LIST

    results = []
    for k in k_list:
        r = run_csd_analysis(character, rows, k=k)
        tau = r["tau_validation"]
        dyn = r["dynamical_metrics"]
        enrich = r["label_enrichment"]
        results.append({
            "k": k,
            "n_segments": len(r["segments"]),
            "n_layer1": r["info"]["n_layer1"],
            "n_layer2_raw": r["info"].get("n_layer2_raw", r["info"]["n_layer2"]),
            "n_layer2": r["info"]["n_layer2"],
            "n_csd": r["info"]["n_csd_flagged"],
            "ac1_threshold": r["info"]["ac1_threshold"],
            "var_threshold": r["info"]["var_threshold"],
            "tau_inside": tau["tau_inside"],
            "tau_outside": tau["tau_outside"],
            "tau_ratio": tau["tau_ratio"],
            "tau_n_inside": tau["n_perturbations_inside"],
            "tau_n_outside": tau["n_perturbations_outside"],
            "delta_drift": dyn["delta_drift"],
            "dwell": dyn["dwell"],
            "escape": dyn["escape"],
            "sad_mad_inside": enrich["inside_rate"],
            "sad_mad_overall": enrich["overall_rate"],
            "full_result": r,
        })
    return results


def format_sensitivity_table(
    character: str,
    sens_results: List[Dict[str, Any]],
) -> str:
    """Format sensitivity analysis as a compact table."""
    lines = []
    lines.append(f"  Sensitivity Analysis: {character}")
    lines.append(f"  Window sizes k = {[r['k'] for r in sens_results]}")
    lines.append("")
    lines.append(
        f"  {'k':>3}  {'segs':>5}  {'L1':>5}  {'L2':>5}  {'CSD':>5}  "
        f"{'AC1_Q75':>8}  {'Var_Q75':>9}  "
        f"{'tau_in':>7}  {'tau_out':>8}  {'ratio':>6}  "
        f"{'n_in':>5}  {'n_out':>6}  "
        f"{'drift':>7}  {'dwell':>6}  {'esc':>5}  "
        f"{'Sad%in':>7}  {'Sad%all':>8}"
    )
    lines.append("  " + "-" * 120)

    for r in sens_results:
        tau_in = f"{r['tau_inside']:.2f}" if r["tau_inside"] is not None else "N/A"
        tau_out = f"{r['tau_outside']:.2f}" if r["tau_outside"] is not None else "N/A"
        ratio = f"{r['tau_ratio']:.2f}" if r["tau_ratio"] is not None else "N/A"
        drift = f"{r['delta_drift']:+.4f}" if r["delta_drift"] is not None else "N/A"
        sad_in = f"{r['sad_mad_inside']:.0%}" if r["sad_mad_inside"] is not None else "N/A"

        lines.append(
            f"  {r['k']:>3}  {r['n_segments']:>5}  {r['n_layer1']:>5}  "
            f"{r['n_layer2']:>5}  {r['n_csd']:>5}  "
            f"{r['ac1_threshold']:>8.4f}  {r['var_threshold']:>9.6f}  "
            f"{tau_in:>7}  {tau_out:>8}  {ratio:>6}  "
            f"{r['tau_n_inside']:>5}  {r['tau_n_outside']:>6}  "
            f"{drift:>7}  {r['dwell']:>6.2f}  {r['escape']:>5.3f}  "
            f"{sad_in:>7}  {r['sad_mad_overall']:>7.0%}"
        )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Formatted report (single k)
# ═══════════════════════════════════════════════════════════════════════════

def format_csd_results(result: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    lines = []
    char = result["character"]
    info = result["info"]
    segments = result["segments"]
    eq = info["eq"]
    tau = result["tau_validation"]
    dyn = result["dynamical_metrics"]
    enrich = result["label_enrichment"]
    k = info["k_used"]

    lines.append("=" * 70)
    lines.append(f"  CSD Analysis: {char}  (k={k})")
    lines.append(f"  Method: SEP Deviation + Critical Slowing Down")
    lines.append("=" * 70)

    lines.append("")
    lines.append(f"  Character Equilibrium:")
    lines.append(f"    mu_SEP = {eq['mu_sep']:+.4f}   sigma_SEP = {eq['sigma_sep']:.4f}")
    lines.append(f"    mu_V   = {eq['mu_v']:+.4f}   mu_A = {eq['mu_a']:+.4f}   mu_D = {eq['mu_d']:+.4f}")

    lines.append("")
    lines.append(f"  Layer 1 — State Precondition:")
    lines.append(f"    Δ̄_t < {info['delta_threshold']:+.4f}  "
                 f"(= -{K_SIGMA} * {eq['sigma_sep']:.4f})")
    lines.append(f"    Flagged: {info['n_layer1']}/{result['n_utterances']} = "
                 f"{info['n_layer1']/result['n_utterances']:.1%}")

    lead_lag = info.get("lead_lag", 0)
    lines.append("")
    lines.append(f"  Layer 2 — CSD Signal (Q75, lead-lag={lead_lag}):")
    lines.append(f"    AC(1) threshold = {info['ac1_threshold']:.4f}")
    lines.append(f"    Var   threshold = {info['var_threshold']:.6f}")
    lines.append(f"    Raw CSD:        {info.get('n_layer2_raw', info['n_layer2'])}/{result['n_utterances']} = "
                 f"{info.get('n_layer2_raw', info['n_layer2'])/result['n_utterances']:.1%}")
    lines.append(f"    With lead-lag:  {info['n_layer2']}/{result['n_utterances']} = "
                 f"{info['n_layer2']/result['n_utterances']:.1%}")

    lines.append("")
    lines.append(f"  Combined (Layer1 AND Layer2):")
    lines.append(f"    CSD collapse: {info['n_csd_flagged']}/{result['n_utterances']} = "
                 f"{info['n_csd_flagged']/result['n_utterances']:.1%}")
    lines.append(f"    Segments (>= {MIN_SEGMENT_LEN} consecutive): {len(segments)}")

    if not segments:
        lines.append("")
        lines.append("  (No CSD collapse segments detected)")
        return "\n".join(lines)

    # Segment details
    lines.append("")
    lines.append(f"  {'#':>3}  {'start_uid':<22} {'end_uid':<22} "
                 f"{'len':>4}  {'mean_Δ':>7}  {'mean_AC1':>8}  {'mean_Var':>8}")
    for i, (s, e) in enumerate(segments):
        s_uid = rows[s]["utterance_id"]
        e_uid = rows[e]["utterance_id"]
        seg_delta = mean(info["delta_seq"][t] for t in range(s, e + 1))
        seg_ac1 = mean(info["ac1_seq"][t] for t in range(s, e + 1))
        seg_var = mean(info["var_seq"][t] for t in range(s, e + 1))
        lines.append(
            f"  {i+1:>3}  {s_uid:<22} {e_uid:<22} "
            f"{e-s+1:>4}  {seg_delta:>+7.3f}  {seg_ac1:>8.4f}  {seg_var:>8.6f}")

    # Segment content
    lines.append("")
    lines.append("  Segment content:")
    for i, (s, e) in enumerate(segments):
        lines.append(f"    --- Segment {i+1}: {rows[s]['utterance_id']} -> "
                     f"{rows[e]['utterance_id']} ---")
        for t in range(s, e + 1):
            text = " ".join(tok for sent in rows[t].get("tokens", []) for tok in sent)
            if len(text) > 100:
                text = text[:97] + "..."
            emo = rows[t].get("emotion", "?")
            lines.append(f"      [{emo:>8}] Δ={info['delta_seq'][t]:+.3f} "
                         f"AC1={info['ac1_seq'][t]:+.3f} "
                         f"Var={info['var_seq'][t]:.4f}  \"{text}\"")

    # Recovery Time Validation
    lines.append("")
    lines.append("  " + "-" * 66)
    lines.append("  Recovery Time Validation (Internal Dynamical Validation)")
    lines.append("  " + "-" * 66)
    lines.append(f"    Perturbations inside  collapse: {tau['n_perturbations_inside']}")
    lines.append(f"    Perturbations outside collapse: {tau['n_perturbations_outside']}")
    if tau["tau_inside"] is not None:
        lines.append(f"    Mean tau (inside):  {tau['tau_inside']:.2f} steps")
    if tau["tau_outside"] is not None:
        lines.append(f"    Mean tau (outside): {tau['tau_outside']:.2f} steps")
    if tau["tau_ratio"] is not None:
        lines.append(f"    tau ratio (inside/outside): {tau['tau_ratio']:.2f}")
    lines.append(f"    Assessment: {tau['p_description']}")

    # Dynamical metrics
    lines.append("")
    lines.append("  Supplementary Dynamical Metrics:")
    if dyn["delta_drift"] is not None:
        sign = "DEEPENING" if dyn["delta_drift"] < 0 else "recovering"
        lines.append(f"    delta-drift = {dyn['delta_drift']:+.4f}  ({sign})")
    lines.append(f"    dwell       = {dyn['dwell']:.2f}")
    lines.append(f"    escape      = {dyn['escape']:.4f}")

    # Label enrichment
    lines.append("")
    lines.append("  Label Enrichment (Sad/Mad, post-hoc only):")
    lines.append(f"    Overall rate: {enrich['overall_rate']:.1%}")
    if enrich["inside_rate"] is not None:
        lines.append(f"    Inside CSD:   {enrich['inside_rate']:.1%}  "
                     f"({enrich['enrichment']})")

    return "\n".join(lines)
