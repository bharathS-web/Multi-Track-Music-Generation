"""Grid tokenization for 3-track symbolic music.

Every track gets exactly ONE token per 16th-note step, so the three tracks are
time-aligned (this is what makes cross-track attention easy):

    REST      nothing sounding
    HOLD      no new onset, earlier notes keep sounding
    <pitch-set> one or more notes start on this step (a note or a chord)

A pitch-set is a sorted tuple of at most `max_notes` MIDI pitches. A note lasts
until the next non-HOLD token on the same track (so a track is effectively
"one chord at a time"; this is a deliberate, documented simplification).

During dataset building, pitch-sets are first stored as packed integers
("raw" ids) and later mapped to a compact vocabulary (see `Vocab`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pretty_midi

from .constants import (
    BOS, HOLD, NUM_SPECIAL, NUM_TRACKS, OUT_PROGRAMS, REST, STEPS_PER_BEAT, TRACKS, UNK,
)

RAW_REST = 0
RAW_HOLD = 1
RAW_OFFSET = 4  # raw pitch-set ids are (packed + RAW_OFFSET), always >= 5


# ---------------------------------------------------------------- packing ----
def reduce_pitches(pitches: Sequence[int], max_notes: int = 4) -> Tuple[int, ...]:
    """Sorted unique pitches; if too many keep the lowest + the highest ones."""
    ps = sorted(set(int(p) for p in pitches))
    if len(ps) > max_notes:
        ps = [ps[0]] + ps[-(max_notes - 1):]
    return tuple(ps)


def pack_pitches(pitches: Tuple[int, ...]) -> int:
    """Pack up to 4 pitches (0..127) into one int, one byte per pitch (+1)."""
    assert 1 <= len(pitches) <= 4
    code = 0
    for p in pitches:
        code = (code << 8) | (int(p) + 1)
    return code


def unpack_pitches(code: int) -> Tuple[int, ...]:
    out: List[int] = []
    while code:
        out.append((code & 0xFF) - 1)
        code >>= 8
    return tuple(reversed(out))


# ------------------------------------------------------------ MIDI -> steps ----
def beat_grid(pm: pretty_midi.PrettyMIDI) -> np.ndarray:
    """Beat times in seconds (follows tempo changes)."""
    return np.asarray(pm.get_beats(), dtype=np.float64)


def track_to_raw_steps(notes: Sequence[pretty_midi.Note], beat_times: np.ndarray,
                       max_notes: int = 4) -> np.ndarray:
    """Quantize one track onto the 16th-note grid -> int64 array of raw ids."""
    n_beats = len(beat_times)
    n_steps = (n_beats - 1) * STEPS_PER_BEAT
    out = np.full(max(n_steps, 0), RAW_REST, dtype=np.int64)
    if n_steps <= 0 or not notes:
        return out

    grid = np.arange(n_beats) * STEPS_PER_BEAT
    starts_t = np.array([n.start for n in notes])
    ends_t = np.array([n.end for n in notes])
    pitches = np.array([n.pitch for n in notes])
    valid = (starts_t >= beat_times[0]) & (starts_t < beat_times[-1])
    starts_t, ends_t, pitches = starts_t[valid], ends_t[valid], pitches[valid]
    if len(starts_t) == 0:
        return out

    s = np.rint(np.interp(starts_t, beat_times, grid)).astype(np.int64)
    e = np.rint(np.interp(ends_t, beat_times, grid)).astype(np.int64)
    e = np.maximum(e, s + 1)

    sounding = np.zeros(n_steps, dtype=bool)
    onsets: Dict[int, set] = {}
    for si, ei, p in zip(s, e, pitches):
        if si < 0 or si >= n_steps:
            continue
        onsets.setdefault(int(si), set()).add(int(p))
        sounding[si:min(int(ei), n_steps)] = True

    out[sounding] = RAW_HOLD
    for si, ps in onsets.items():
        out[si] = pack_pitches(reduce_pitches(ps, max_notes)) + RAW_OFFSET
    return out


# ------------------------------------------------------------------ Vocab ----
@dataclass
class Vocab:
    """Final vocabulary. Token id NUM_SPECIAL + i  <->  pitch_sets[i]."""
    pitch_sets: List[Tuple[int, ...]]

    @property
    def size(self) -> int:
        return NUM_SPECIAL + len(self.pitch_sets)

    def pitches(self, token: int) -> Tuple[int, ...] | None:
        i = int(token) - NUM_SPECIAL
        if 0 <= i < len(self.pitch_sets):
            return self.pitch_sets[i]
        return None

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({
            "num_special": NUM_SPECIAL,
            "pitch_sets": [list(p) for p in self.pitch_sets],
        }))

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        d = json.loads(Path(path).read_text())
        return cls([tuple(p) for p in d["pitch_sets"]])


def build_vocab_and_remapper(train_raw: np.ndarray, vocab_size: int):
    """Keep the most frequent pitch-sets of the TRAIN split only.

    Returns (Vocab, remap) where remap(raw_array) -> final-id array (uint16).
    """
    assert vocab_size > NUM_SPECIAL
    n_keep = vocab_size - NUM_SPECIAL
    flat = train_raw.reshape(-1)
    flat = flat[flat >= RAW_OFFSET + 1]
    codes, counts = np.unique(flat, return_counts=True)
    order = np.argsort(-counts, kind="stable")[:n_keep]
    kept_codes = codes[order]                       # rank order -> final ids
    vocab = Vocab([unpack_pitches(int(c) - RAW_OFFSET) for c in kept_codes])

    sorter = np.argsort(kept_codes)
    kept_sorted = kept_codes[sorter]
    rank_of_sorted = sorter                         # position in rank order

    def remap(raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw)
        out = np.full(raw.shape, UNK, dtype=np.int64)
        out[raw == RAW_REST] = REST
        out[raw == RAW_HOLD] = HOLD
        if len(kept_sorted):
            idx = np.searchsorted(kept_sorted, raw)
            idx_c = np.minimum(idx, len(kept_sorted) - 1)
            hit = (raw >= RAW_OFFSET + 1) & (kept_sorted[idx_c] == raw)
            out[hit] = NUM_SPECIAL + rank_of_sorted[idx_c[hit]]
        return out.astype(np.uint16)

    return vocab, remap


# ------------------------------------------------------------ tokens -> MIDI ----
def decode_track(tokens: Sequence[int], vocab: Vocab, step_seconds: float,
                 program: int, velocity: int = 80) -> pretty_midi.Instrument:
    inst = pretty_midi.Instrument(program=program)
    cur = None  # (pitches, start_step)

    def close(end_step: int):
        nonlocal cur
        if cur is not None:
            for p in cur[0]:
                inst.notes.append(pretty_midi.Note(velocity, int(p),
                                                   cur[1] * step_seconds, end_step * step_seconds))
            cur = None

    for t, tok in enumerate(tokens):
        tok = int(tok)
        if tok == HOLD:
            continue
        close(t)
        ps = vocab.pitches(tok) if tok >= NUM_SPECIAL else None
        if ps:
            cur = (ps, t)
    close(len(tokens))
    return inst


def tokens_to_midi(tokens: np.ndarray, vocab: Vocab, bpm: float = 120.0) -> pretty_midi.PrettyMIDI:
    """tokens: (NUM_TRACKS, T) final ids -> PrettyMIDI (fixed tempo, 4/4)."""
    assert tokens.shape[0] == NUM_TRACKS
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    step_seconds = 60.0 / bpm / STEPS_PER_BEAT
    for k, name in enumerate(TRACKS):
        pm.instruments.append(decode_track(tokens[k], vocab, step_seconds, OUT_PROGRAMS[name]))
    return pm
