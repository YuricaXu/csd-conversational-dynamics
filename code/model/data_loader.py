"""
data_loader.py  —  Steps 0 & 1
  Step 0: Load all three JSON splits, attach metadata, sort globally by utterance_id.
  Step 1: Filter out #ALL# and multi-speaker utterances.
"""

import json
import re
from typing import List, Dict, Any


# ── Step 0: parse utterance_id to a sortable tuple ───────────────────────────

def parse_uid(utterance_id: str):
    """
    utterance_id format: s01_e02_c08_u003
    Returns (season, episode, scene, utterance) as an integer tuple
    for correct global narrative ordering.
    """
    m = re.fullmatch(r"s(\d+)_e(\d+)_c(\d+)_u(\d+)", utterance_id)
    if m is None:
        raise ValueError(f"Unexpected utterance_id format: {utterance_id!r}")
    return tuple(int(g) for g in m.groups())   # (s, e, c, u)


def load_and_sort(paths: List[str]) -> List[Dict[str, Any]]:
    """
    Load all JSON splits, attach episode/scene metadata to each utterance,
    then sort the combined list by (season, episode, scene, utterance) index.

    Returns a flat, globally-sorted list of utterance dicts, each augmented with:
      _episode_id  : e.g. "s01_e02"
      _scene_id    : e.g. "s01_e02_c01"
      _sort_key    : (int, int, int, int) for reproducible ordering
    """
    all_utts: List[Dict[str, Any]] = []

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for episode in data["episodes"]:
            ep_id = episode["episode_id"]
            for scene in episode["scenes"]:
                sc_id = scene["scene_id"]
                for utt in scene["utterances"]:
                    utt = dict(utt)                        # shallow copy — don't mutate source
                    utt["_episode_id"] = ep_id
                    utt["_scene_id"]   = sc_id
                    utt["_sort_key"]   = parse_uid(utt["utterance_id"])
                    all_utts.append(utt)

    all_utts.sort(key=lambda x: x["_sort_key"])
    return all_utts


# ── Step 1: filter invalid speakers ──────────────────────────────────────────

def filter_speakers(all_utts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove utterances that cannot be assigned to exactly one speaker:
      - "#ALL#" whole-cast turns
      - Multi-speaker / overlapping turns (len(speakers) != 1)

    These cannot contribute to a single character's energy landscape.
    """
    filtered = []
    n_all    = 0
    n_multi  = 0

    for u in all_utts:
        speakers = u.get("speakers", [])
        if "#ALL#" in speakers:
            n_all += 1
            continue
        if len(speakers) != 1:
            n_multi += 1
            continue
        filtered.append(u)

    print(f"[filter_speakers] Removed {n_all} #ALL# utterances, "
          f"{n_multi} multi-speaker utterances. "
          f"Remaining: {len(filtered):,} utterances.")
    return filtered
