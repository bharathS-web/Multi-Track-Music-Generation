"""Synthetic MIDI songs for tests (piano chords, guitar arpeggios, bass roots; 4/4, 120 bpm)."""
from pathlib import Path

import numpy as np
import pretty_midi

PROGRESSION = [(0, "maj"), (5, "maj"), (7, "maj"), (9, "min")]


def make_song(path: Path, bars: int = 24, seed: int = 0, bpm: float = 120.0):
    rng = np.random.default_rng(seed)
    key = int(rng.integers(0, 12))
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    piano = pretty_midi.Instrument(program=0)
    guitar = pretty_midi.Instrument(program=25)
    bass = pretty_midi.Instrument(program=33)
    beat = 60.0 / bpm
    for b in range(bars):
        root, kind = PROGRESSION[(b + int(rng.integers(0, 2))) % 4]
        r = 48 + (root + key) % 12
        third = 3 if kind == "min" else 4
        chord = [r, r + third, r + 7]
        for q in range(4):                                  # piano: chord on every beat
            t = (b * 4 + q) * beat
            for p in chord:
                piano.notes.append(pretty_midi.Note(80, p, t, t + beat * 0.9))
        for e in range(8):                                  # guitar: 8th-note arpeggio
            t = (b * 8 + e) * beat / 2
            p = chord[e % 3] + 12
            guitar.notes.append(pretty_midi.Note(80, p, t, t + beat / 2 * 0.9))
            bass.notes.append(pretty_midi.Note(80, r - 12, t, t + beat / 2 * 0.9))
    pm.instruments += [piano, guitar, bass]
    path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))
    return path


def make_corpus(folder: Path, n_files: int = 30, bars: int = 24):
    for i in range(n_files):
        make_song(folder / f"song_{i:03d}.mid", bars=bars, seed=i)
    return folder
