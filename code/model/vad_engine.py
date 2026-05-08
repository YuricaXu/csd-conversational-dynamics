"""
vad_engine.py  —  Steps 2 & 3
  Step 2: Load NRC VAD v2.1 lexicon; map utterance tokens → (v, a, d) with MWE-first matching.
  Step 3: Episode-level imputation for zero-coverage utterances.
"""

from __future__ import annotations
import re
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Optional, Tuple, Any


VADPoint = Tuple[float, float, float]


# ── Load NRC VAD Lexicon v2.1 ─────────────────────────────────────────────────

def load_vad_lexicon(path: str) -> Tuple[Dict[str, VADPoint], Dict[str, VADPoint]]:
    """
    Parse NRC-VAD-Lexicon-v2.1.txt (tab-separated, header row).
    Returns:
      vad_unigram  : {word -> (v, a, d)}         — single-token entries
      vad_mwe      : {"w1 w2" -> (v, a, d)}      — multi-word expression entries (bigrams)

    The lexicon uses space-separated tokens for MWEs (e.g., "a battery", "come on").
    Only bigrams are extracted here (max_n=2) to match the methodology's bigram-first
    greedy strategy.  Longer MWEs are rare in NRC VAD v2.1 and not used.

    All scores are in [-1, +1].
    """
    vad_unigram: Dict[str, VADPoint] = {}
    vad_mwe:     Dict[str, VADPoint] = {}

    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()   # skip "term\tvalence\tarousal\tdominance"
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            term, v_str, a_str, d_str = parts
            try:
                vad = (float(v_str), float(a_str), float(d_str))
            except ValueError:
                continue
            term_lower = term.lower()
            tokens = term_lower.split()
            if len(tokens) == 1:
                vad_unigram[tokens[0]] = vad
            elif len(tokens) == 2:
                vad_mwe[term_lower] = vad
            # Trigrams+ are rare; skip per methodology (bigram-first strategy)

    print(f"[load_vad_lexicon] Loaded {len(vad_unigram):,} unigrams, "
          f"{len(vad_mwe):,} bigram MWEs.")
    return vad_unigram, vad_mwe


# ── Step 2: Utterance → VAD ───────────────────────────────────────────────────

def _mean3(vads: List[VADPoint]) -> VADPoint:
    """Element-wise mean of a list of (v, a, d) tuples."""
    n = len(vads)
    return (
        sum(x[0] for x in vads) / n,
        sum(x[1] for x in vads) / n,
        sum(x[2] for x in vads) / n,
    )


def utterance_vad(
    tokens: List[List[str]],
    vad_unigram: Dict[str, VADPoint],
    vad_mwe: Dict[str, VADPoint],
) -> Tuple[Optional[VADPoint], float]:
    """
    Map one utterance's pre-tokenised sentences to a VAD point in [-1, +1]^3.

    Algorithm (MWE-first greedy, left-to-right within each sentence):
      1. Lowercase and keep only alpha tokens.
      2. At each position i, try bigram lookup (tokens[i] + " " + tokens[i+1]) first.
         If found: record score, advance i by 2.
      3. Else try unigram lookup for tokens[i].
         If found: record score, advance i by 1.
      4. Else: skip tokens[i], advance i by 1.

    Coverage = (alpha tokens consumed by a match) / (total alpha tokens).
    Returns (None, 0.0) when no lexicon entry is matched — caller handles imputation.

    Parameters
    ----------
    tokens      : list[list[str]]   — sentence-segmented token lists from JSON
    vad_unigram : dict str -> (v,a,d)
    vad_mwe     : dict "w1 w2" -> (v,a,d)   (bigram MWEs only)
    """
    scores:          List[VADPoint] = []
    total_alpha:     int = 0
    covered_tokens:  int = 0

    for sent in tokens:
        alpha_tokens = [t.lower() for t in sent if t.isalpha()]
        total_alpha += len(alpha_tokens)
        i = 0
        while i < len(alpha_tokens):
            # Priority 1: bigram MWE
            if i + 1 < len(alpha_tokens):
                bigram = alpha_tokens[i] + " " + alpha_tokens[i + 1]
                if bigram in vad_mwe:
                    scores.append(vad_mwe[bigram])
                    covered_tokens += 2
                    i += 2
                    continue
            # Priority 2: unigram
            if alpha_tokens[i] in vad_unigram:
                scores.append(vad_unigram[alpha_tokens[i]])
                covered_tokens += 1
            i += 1

    if not scores:
        return None, 0.0

    vad_point = _mean3(scores)
    coverage  = covered_tokens / max(1, total_alpha)
    return vad_point, coverage


# ── Step 3: Imputation ────────────────────────────────────────────────────────

def fill_missing_vad(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For utterances where VAD == None (zero lexicon coverage):
      1. Impute with the mean VAD of OTHER utterances by the SAME speaker
         in the SAME episode (episode-level self-mean).
         → vad_cov set to 0.0  (flag: episode-level imputation)
      2. If no valid points exist in that episode, fall back to the
         speaker's global mean across all episodes.
         → vad_cov set to -1.0 (flag: global fallback)

    Rationale: preserves timeline continuity — removing zero-coverage points
    would create artificial gaps in the trajectory.

    Modifies rows in-place; returns the same list.
    """
    # Collect valid (non-None) VAD points grouped by episode
    ep_valid: Dict[str, List[VADPoint]] = defaultdict(list)
    for r in rows:
        if r.get("vad") is not None:
            ep_valid[r["_episode_id"]].append(r["vad"])

    # Speaker global mean as ultimate fallback
    all_valid = [v for vlist in ep_valid.values() for v in vlist]
    global_mean: VADPoint = _mean3(all_valid) if all_valid else (0.0, 0.0, 0.0)

    for r in rows:
        if r.get("vad") is None:
            ep = r["_episode_id"]
            if ep_valid[ep]:
                r["vad"]     = _mean3(ep_valid[ep])
                r["vad_cov"] = 0.0    # episode-level imputation
            else:
                r["vad"]     = global_mean
                r["vad_cov"] = -1.0   # global fallback
    return rows


# ── Step 4: Build per-character trajectories ──────────────────────────────────

def build_trajectories(
    all_utts_sorted: List[Dict[str, Any]],
    vad_unigram: Dict[str, VADPoint],
    vad_mwe: Dict[str, VADPoint],
    target_characters: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Partition filtered utterances by speaker; compute VAD for each.
    Within each speaker's list, order = that character's personal timeline
    (t = 0, 1, 2, … across all episodes and scenes).

    Parameters
    ----------
    all_utts_sorted     : globally sorted list after Step 1 filtering
    vad_unigram / mwe   : lexicon dicts from load_vad_lexicon()
    target_characters   : if given, only build trajectories for these speakers

    Returns
    -------
    by_speaker : {speaker_name: [utterance_dicts with "vad" and "vad_cov"]}
    """
    by_speaker: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for u in all_utts_sorted:
        spk = u["speakers"][0]
        if target_characters and spk not in target_characters:
            continue
        vad, cov = utterance_vad(u["tokens"], vad_unigram, vad_mwe)
        u["vad"]     = vad
        u["vad_cov"] = cov
        by_speaker[spk].append(u)

    # Impute missing VAD within each character's personal trajectory
    for spk in by_speaker:
        by_speaker[spk] = fill_missing_vad(by_speaker[spk])
        n_total   = len(by_speaker[spk])
        n_orig    = sum(1 for r in by_speaker[spk] if r["vad_cov"] > 0)
        n_ep_imp  = sum(1 for r in by_speaker[spk] if r["vad_cov"] == 0.0)
        n_glob    = sum(1 for r in by_speaker[spk] if r["vad_cov"] == -1.0)
        print(f"  {spk:<18} total={n_total:4d}  "
              f"lexicon={n_orig:4d}  ep-imputed={n_ep_imp:3d}  "
              f"global-fallback={n_glob:2d}")

    return dict(by_speaker)
