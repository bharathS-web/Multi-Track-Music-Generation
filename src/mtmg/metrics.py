"""Objective metrics on token arrays of shape (K, T) (final vocabulary ids).

These are OUR definitions, inspired by the paper's metric names. They are NOT
guaranteed to be numerically comparable with the paper's tables:

  harmony_analysis   fraction of bars (in which all tracks sound) where the pitch
                     classes of ALL tracks share at least one common pitch class
                     (paper Eq. 27, normalised by the number of bars only).
  chord_labels       one label per bar from template matching on the pitch-class
                     histogram of all tracks (12 major + 12 minor triads, -1 = none).
  chord_match        fraction of bars whose chord label equals the reference's.
  rests_ratio        fraction of steps that are REST/UNK, per track.
  polyphony          fraction of sounding steps with >= 2 simultaneous notes (same track).
  upc                mean number of distinct pitch classes per bar per active track.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .constants import HOLD, NUM_SPECIAL, REST, STEPS_PER_BAR, UNK
from .tokenization import Vocab

_MAJOR = (0, 4, 7)
_MINOR = (0, 3, 7)
_TEMPLATES = np.zeros((24, 12))
for _r in range(12):
    for _i in _MAJOR:
        _TEMPLATES[_r, (_r + _i) % 12] = 1
    for _i in _MINOR:
        _TEMPLATES[12 + _r, (_r + _i) % 12] = 1


def sounding_sets(track: Sequence[int], vocab: Vocab) -> List[tuple]:
    """Per step: tuple of MIDI pitches currently sounding (empty tuple = silence)."""
    out, cur = [], ()
    for tok in track:
        tok = int(tok)
        if tok == HOLD:
            pass
        elif tok >= NUM_SPECIAL:
            cur = vocab.pitches(tok) or ()
        else:                       # REST / UNK / BOS
            cur = ()
        out.append(cur)
    return out


def _bar_pcs(sets: List[tuple]) -> List[set]:
    bars = []
    for b in range(len(sets) // STEPS_PER_BAR):
        pcs = set()
        for s in sets[b * STEPS_PER_BAR:(b + 1) * STEPS_PER_BAR]:
            pcs.update(p % 12 for p in s)
        bars.append(pcs)
    return bars


def harmony_analysis(tokens: np.ndarray, vocab: Vocab) -> float:
    per_track = [_bar_pcs(sounding_sets(t, vocab)) for t in tokens]
    n_bars = min(len(x) for x in per_track)
    hits, total = 0, 0
    for b in range(n_bars):
        sets = [x[b] for x in per_track]
        if all(sets):
            total += 1
            hits += int(len(set.intersection(*sets)) > 0)
    return hits / total if total else float("nan")


def chord_labels(tokens: np.ndarray, vocab: Vocab) -> List[int]:
    sets = [sounding_sets(t, vocab) for t in tokens]
    n_bars = min(len(s) for s in sets) // STEPS_PER_BAR
    labels = []
    for b in range(n_bars):
        hist = np.zeros(12)
        for s in sets:
            for step in s[b * STEPS_PER_BAR:(b + 1) * STEPS_PER_BAR]:
                for p in step:
                    hist[p % 12] += 1
        if hist.sum() == 0:
            labels.append(-1)
            continue
        scores = _TEMPLATES @ hist - 0.5 * ((1 - _TEMPLATES) @ hist)
        labels.append(int(scores.argmax()))
    return labels


def chord_match(pred: np.ndarray, ref: np.ndarray, vocab: Vocab) -> float:
    a, b = chord_labels(pred, vocab), chord_labels(ref, vocab)
    pairs = [(x, y) for x, y in zip(a, b) if y != -1]
    return sum(x == y for x, y in pairs) / len(pairs) if pairs else float("nan")


def rests_ratio(tokens: np.ndarray) -> List[float]:
    return [float(np.isin(t, (REST, UNK)).mean()) for t in tokens]


def polyphony(tokens: np.ndarray, vocab: Vocab) -> float:
    sounding = poly = 0
    for t in tokens:
        for s in sounding_sets(t, vocab):
            if s:
                sounding += 1
                poly += int(len(s) >= 2)
    return poly / sounding if sounding else float("nan")


def upc(tokens: np.ndarray, vocab: Vocab) -> float:
    vals = []
    for t in tokens:
        for pcs in _bar_pcs(sounding_sets(t, vocab)):
            if pcs:
                vals.append(len(pcs))
    return float(np.mean(vals)) if vals else float("nan")


def summarize(samples: Sequence[np.ndarray], vocab: Vocab) -> Dict[str, float]:
    """Mean of each metric over a list of (K, T) token arrays."""
    rows = {"harmony_analysis": [], "polyphony": [], "upc": []}
    rests = []
    for s in samples:
        rows["harmony_analysis"].append(harmony_analysis(s, vocab))
        rows["polyphony"].append(polyphony(s, vocab))
        rows["upc"].append(upc(s, vocab))
        rests.append(rests_ratio(s))
    out = {k: float(np.nanmean(v)) if len(v) else float("nan") for k, v in rows.items()}
    r = np.mean(np.array(rests), axis=0) if rests else np.full(3, np.nan)
    for name, v in zip(("piano", "guitar", "bass"), r):
        out[f"rests_ratio_{name}"] = float(v)
    out["n_samples"] = len(samples)
    return out
