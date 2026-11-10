"""Reward heuristics for PPO fine-tuning (paper Eq. 22):

    R = beta_pitch * R_pitch + beta_repeat * R_repeat + beta_harmony * R_harmony

The paper names the three terms ("penalize implausible pitch range, excessive note repetition, and
poor cross-track chord agreement") but gives no formulas, so the definitions below are MINE:

  R_pitch    in [-1, 0]  minus the fraction of note onsets outside a plausible range for their
                         instrument (PITCH_RANGES, MIDI numbers).
  R_repeat   in [-1, 0]  minus the fraction of onsets that repeat the same pitch-set more than
                         `max_run` times in a row on the same track.
  R_harmony  in [0, 1]   metrics.harmony_analysis: share of bars (where every track sounds) in which
                         all tracks share at least one pitch class. 0 if no such bar exists, so
                         silence is never rewarded.

All rewards are computed on one (K, T) token array, usually only the generated part (not the prompt).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .constants import NUM_SPECIAL, TRACKS
from .metrics import harmony_analysis
from .tokenization import Vocab

# Plausible MIDI ranges per instrument: piano A0-C8, guitar E2-E6, bass E1-G4.
PITCH_RANGES = {"piano": (21, 108), "guitar": (40, 88), "bass": (28, 67)}


@dataclass
class RewardWeights:
    pitch: float = 1.0
    repeat: float = 1.0
    harmony: float = 1.0
    max_run: int = 4


def _onsets(track: np.ndarray, vocab: Vocab):
    return [vocab.pitches(int(t)) for t in track if int(t) >= NUM_SPECIAL and vocab.pitches(int(t))]


def pitch_range_reward(tokens: np.ndarray, vocab: Vocab) -> float:
    bad = total = 0
    for k, name in enumerate(TRACKS):
        lo, hi = PITCH_RANGES[name]
        for ps in _onsets(tokens[k], vocab):
            total += len(ps)
            bad += sum(1 for p in ps if p < lo or p > hi)
    return -bad / total if total else 0.0


def repetition_reward(tokens: np.ndarray, vocab: Vocab, max_run: int = 4) -> float:
    excess = total = 0
    for k in range(len(TRACKS)):
        prev, run = None, 0
        for ps in _onsets(tokens[k], vocab):
            run = run + 1 if ps == prev else 1
            prev = ps
            total += 1
            excess += int(run > max_run)
    return -excess / total if total else 0.0


def harmony_reward(tokens: np.ndarray, vocab: Vocab) -> float:
    h = harmony_analysis(tokens, vocab)
    return 0.0 if np.isnan(h) else float(h)


def reward(tokens: np.ndarray, vocab: Vocab, w: RewardWeights = RewardWeights()) -> Dict[str, float]:
    """All terms plus the weighted total for one (K, T) token array."""
    r = {"pitch": pitch_range_reward(tokens, vocab),
         "repeat": repetition_reward(tokens, vocab, w.max_run),
         "harmony": harmony_reward(tokens, vocab)}
    r["total"] = w.pitch * r["pitch"] + w.repeat * r["repeat"] + w.harmony * r["harmony"]
    return r


def mean_rewards(samples, vocab: Vocab, w: RewardWeights = RewardWeights()) -> Dict[str, float]:
    rows = [reward(s, vocab, w) for s in samples]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {}
