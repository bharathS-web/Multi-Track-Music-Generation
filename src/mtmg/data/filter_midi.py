"""Decide whether a MIDI file is usable and pick the piano / guitar / bass tracks.

Rules (every rejection is counted, so the README can report them):
  * parses without error
  * only 4/4 time signatures
  * stable tempo (max/min tempo ratio <= 1 + tempo_tol)
  * has a piano, a guitar and a bass track with >= min_notes notes each
    (General Midi program ranges, drums ignored; if several instruments
    share a role, the one with the most notes is used)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pretty_midi

from ..constants import ROLE_PROGRAMS, TRACKS


@dataclass
class FilterConfig:
    tempo_tol: float = 0.05
    min_notes: int = 100
    min_bpm: float = 50.0
    max_bpm: float = 220.0


def analyze_midi(path: str | Path, cfg: FilterConfig):
    """Returns (pm_or_None, tracks_or_None, reason). reason == 'ok' when accepted."""
    try:
        pm = pretty_midi.PrettyMIDI(str(path))
    except Exception:
        return None, None, "parse_error"

    for ts in pm.time_signature_changes:
        if (ts.numerator, ts.denominator) != (4, 4):
            return None, None, "not_4_4"

    try:
        _, tempi = pm.get_tempo_changes()
    except Exception:
        return None, None, "parse_error"
    if len(tempi) == 0 or tempi.min() <= 0:
        return None, None, "bad_tempo"
    if tempi.max() / tempi.min() > 1.0 + cfg.tempo_tol:
        return None, None, "unstable_tempo"
    if tempi.min() < cfg.min_bpm or tempi.max() > cfg.max_bpm:
        return None, None, "tempo_out_of_range"

    chosen: Dict[str, pretty_midi.Instrument] = {}
    for role in TRACKS:
        cands = [i for i in pm.instruments
                 if not i.is_drum and i.program in ROLE_PROGRAMS[role]]
        if not cands:
            return None, None, f"missing_{role}"
        best = max(cands, key=lambda i: len(i.notes))
        if len(best.notes) < cfg.min_notes:
            return None, None, f"too_few_notes_{role}"
        chosen[role] = best
    return pm, chosen, "ok"
