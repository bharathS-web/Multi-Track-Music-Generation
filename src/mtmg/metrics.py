"""Objective metrics of the paper (Sec. XII-B, Tables I-III) on token arrays of shape (K, T).

Chord labels come from template matching on a pitch-class histogram (12 major + 12 minor triads;
-1 = no notes).

Table I
  chord_matching     Eq. 26: over the chord-playing tracks (piano, guitar), the share of bars in which the
                     chord implied by that track alone equals the bar's chord from all tracks together.
                     Bars where the track or the whole bar is silent are skipped. Adapted from the chord
                     accuracy of PopMAG (Ren et al., ACM MM 2020). Needs no reference piece, so it is computed
                     the same way for generated and real music; real test windows score 0.74, not 1.
  chord_match        teacher-forced diagnostic (evaluate.py): agreement of the all-track chord labels of
                     two pieces, e.g. argmax predictions vs. the real piece.
  harmony_analysis   Eq. 27: share of bars in which the pitch classes of ALL tracks have a common element.
                     Normalised by the number of bars P (a bar where a track is silent counts as 0).
Table II
  nctn               non-chord-tone notes: share of note onsets whose pitch class is not in the bar's chord.
  qr                 qualified rhythm: share of notes whose length is a standard value (16th, 8th, dotted
                     8th, quarter, dotted quarter, half, dotted half, or whole bars).
  upc                used pitch classes: mean number of distinct pitch classes per bar per active track.
  polyphony (PP)     share of sounding steps with >= 2 simultaneous notes on the same track.
Table III
  sequence_similarity  per track: cosine similarity of the per-bar pitch-class histograms of a generated
                     continuation and the real continuation of the same prompt, averaged over bars.
  rests_ratio        per track: share of steps that are REST/UNK.

The paper names NCTN, QR and sequence similarity without formulas; the definitions above are the
ones used for this code's results.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .constants import HOLD, NUM_SPECIAL, REST, STEPS_PER_BAR, TRACKS, UNK
from .tokenization import Vocab

_MAJOR = (0, 4, 7)
_MINOR = (0, 3, 7)
_TEMPLATES = np.zeros((24, 12))
for _r in range(12):
    for _i in _MAJOR:
        _TEMPLATES[_r, (_r + _i) % 12] = 1
    for _i in _MINOR:
        _TEMPLATES[12 + _r, (_r + _i) % 12] = 1

# note lengths in 16th steps that count as "qualified" rhythm (plus any whole number of bars)
QUALIFIED_LENGTHS = {1, 2, 3, 4, 6, 8, 12}


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


def note_events(track: Sequence[int], vocab: Vocab):
    """(start_step, length_in_steps, pitches) for every onset; a note lasts until the next non-HOLD token."""
    events, cur = [], None
    for t, tok in enumerate(track):
        tok = int(tok)
        if tok == HOLD:
            continue
        if cur is not None:
            events.append((cur[0], t - cur[0], cur[1]))
            cur = None
        ps = vocab.pitches(tok) if tok >= NUM_SPECIAL else None
        if ps:
            cur = (t, ps)
    if cur is not None:
        events.append((cur[0], len(track) - cur[0], cur[1]))
    return events


def _bar_pcs(sets: List[tuple]) -> List[set]:
    bars = []
    for b in range(len(sets) // STEPS_PER_BAR):
        pcs = set()
        for s in sets[b * STEPS_PER_BAR:(b + 1) * STEPS_PER_BAR]:
            pcs.update(p % 12 for p in s)
        bars.append(pcs)
    return bars


def harmony_analysis(tokens: np.ndarray, vocab: Vocab) -> float:
    """Paper Eq. 27, normalised by the number of bars P."""
    per_track = [_bar_pcs(sounding_sets(t, vocab)) for t in tokens]
    n_bars = min(len(x) for x in per_track)
    if n_bars == 0:
        return float("nan")
    hits = sum(int(len(set.intersection(*[x[b] for x in per_track])) > 0) for b in range(n_bars))
    return hits / n_bars


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


CHORD_TRACKS = tuple(TRACKS.index(t) for t in ("piano", "guitar"))   # tracks that play chords


def chord_matching(tokens: np.ndarray, vocab: Vocab) -> float:
    """Paper Eq. 26 (Table I): share of (chord track, bar) pairs in which the chord implied by that track
    alone equals the bar's chord from all tracks. Pairs where either is silent are skipped."""
    ref = chord_labels(tokens, vocab)
    hits = total = 0
    for k in CHORD_TRACKS:
        for own, full in zip(chord_labels(tokens[k:k + 1], vocab), ref):
            if own == -1 or full == -1:
                continue
            total += 1
            hits += int(own == full)
    return hits / total if total else float("nan")


def chord_match(pred: np.ndarray, ref: np.ndarray, vocab: Vocab) -> float:
    """Agreement of the all-track chord labels of two pieces (teacher-forced diagnostic)."""
    a, b = chord_labels(pred, vocab), chord_labels(ref, vocab)
    pairs = [(x, y) for x, y in zip(a, b) if y != -1]
    return sum(x == y for x, y in pairs) / len(pairs) if pairs else float("nan")


def nctn(tokens: np.ndarray, vocab: Vocab) -> float:
    """Share of note onsets (all tracks) whose pitch class is not in the bar's chord."""
    labels = chord_labels(tokens, vocab)
    outside = total = 0
    for track in tokens:
        for start, _, ps in note_events(track, vocab):
            b = start // STEPS_PER_BAR
            if b >= len(labels) or labels[b] < 0:
                continue
            chord = _TEMPLATES[labels[b]]
            for p in ps:
                total += 1
                outside += int(chord[p % 12] == 0)
    return outside / total if total else float("nan")


def qualified_rhythm(tokens: np.ndarray, vocab: Vocab) -> float:
    ok = total = 0
    for track in tokens:
        for _, length, _ in note_events(track, vocab):
            total += 1
            ok += int(length in QUALIFIED_LENGTHS or length % STEPS_PER_BAR == 0)
    return ok / total if total else float("nan")


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


def _pc_histograms(track: Sequence[int], vocab: Vocab) -> np.ndarray:
    sets = sounding_sets(track, vocab)
    n_bars = len(sets) // STEPS_PER_BAR
    h = np.zeros((n_bars, 12))
    for b in range(n_bars):
        for s in sets[b * STEPS_PER_BAR:(b + 1) * STEPS_PER_BAR]:
            for p in s:
                h[b, p % 12] += 1
    return h


def sequence_similarity(gen: np.ndarray, ref: np.ndarray, vocab: Vocab) -> List[float]:
    """Per track: mean over bars of the cosine similarity between pitch-class histograms of the
    generated and the reference continuation (bars silent in both are skipped)."""
    out = []
    for k in range(len(TRACKS)):
        hg, hr = _pc_histograms(gen[k], vocab), _pc_histograms(ref[k], vocab)
        sims = []
        for a, b in zip(hg, hr):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na == 0 and nb == 0:
                continue
            sims.append(min(1.0, float(a @ b / (na * nb))) if na and nb else 0.0)
        out.append(float(np.mean(sims)) if sims else float("nan"))
    return out


def summarize(samples: Sequence[np.ndarray], vocab: Vocab) -> Dict[str, float]:
    """Mean of each reference-free metric over a list of (K, T) token arrays (Tables I-III)."""
    fns = {"chord_matching": chord_matching, "harmony_analysis": harmony_analysis, "nctn": nctn,
           "qr": qualified_rhythm, "upc": upc, "polyphony": polyphony}
    rows = {k: [f(s, vocab) for s in samples] for k, f in fns.items()}
    out = {k: float(np.nanmean(v)) if len(v) and not np.all(np.isnan(v)) else float("nan")
           for k, v in rows.items()}
    r = np.mean(np.array([rests_ratio(s) for s in samples]), axis=0) if len(samples) else np.full(3, np.nan)
    for name, v in zip(TRACKS, r):
        out[f"rests_ratio_{name}"] = float(v)
    out["n_samples"] = len(samples)
    return out


def compare_to_reference(samples: Sequence[np.ndarray], refs: Sequence[np.ndarray],
                         vocab: Vocab) -> Dict[str, float]:
    """Metric that needs the real continuation of the same prompt: per-track sequence similarity (Table III)."""
    sim = np.array([sequence_similarity(g, r, vocab) for g, r in zip(samples, refs)])
    out = {}
    for k, name in enumerate(TRACKS):
        col = sim[:, k] if len(sim) else np.array([np.nan])
        out[f"sequence_similarity_{name}"] = float(np.nanmean(col)) if not np.all(np.isnan(col)) else float("nan")
    return out
