import numpy as np
from pytest import approx

from mtmg import metrics as M
from mtmg.constants import HOLD, NUM_SPECIAL, REST, STEPS_PER_BAR
from mtmg.tokenization import Vocab

# 4 = C major triad (48, 52, 55), 5 = C2, 6 = C5, 7 = C#2, 8 = D4
VOCAB = Vocab([(48, 52, 55), (36,), (72,), (37,), (62,)])
CHORD, BASS_C, HIGH_C, BASS_CS, D4 = range(NUM_SPECIAL, NUM_SPECIAL + 5)
SILENT_BAR = [REST] * STEPS_PER_BAR


def bar(token, length=4):
    """Onsets every `length` steps, held in between."""
    return ([token] + [HOLD] * (length - 1)) * (STEPS_PER_BAR // length)


def song(*tracks):
    return np.array(tracks, dtype=np.int64)


def test_harmony_analysis_counts_every_bar_eq27():
    good = song(bar(CHORD) * 2, bar(HIGH_C) * 2, bar(BASS_C) * 2)
    assert M.harmony_analysis(good, VOCAB) == 1.0
    half_silent = song(bar(CHORD) * 2, bar(HIGH_C) + SILENT_BAR, bar(BASS_C) * 2)
    assert M.harmony_analysis(half_silent, VOCAB) == 0.5          # the bar with a silent track scores 0


def test_chord_match_and_similarity_against_reference():
    a = song(bar(CHORD) * 2, bar(HIGH_C) * 2, bar(BASS_C) * 2)
    assert M.chord_match(a, a, VOCAB) == 1.0
    assert M.sequence_similarity(a, a, VOCAB) == approx([1.0, 1.0, 1.0])
    b = song(bar(CHORD) * 2, bar(HIGH_C) * 2, bar(BASS_CS) * 2)
    sim = M.sequence_similarity(b, a, VOCAB)
    assert sim[0] == approx(1.0) and sim[2] == 0.0                         # C# shares no pitch class with C


def test_nctn_counts_notes_outside_the_bar_chord():
    inside = song(bar(CHORD), bar(HIGH_C), bar(BASS_C))
    assert M.nctn(inside, VOCAB) == 0.0
    with_d = song(bar(CHORD), bar(D4), bar(BASS_C))                 # D is not in C major
    assert 0.0 < M.nctn(with_d, VOCAB) < 1.0


def test_qualified_rhythm():
    quarters = song(bar(CHORD, 4), bar(HIGH_C, 8), bar(BASS_C, 16))
    assert M.qualified_rhythm(quarters, VOCAB) == 1.0
    odd = song(([CHORD] + [HOLD] * 4) * 3 + [REST], SILENT_BAR, SILENT_BAR)   # 5-step notes
    assert M.qualified_rhythm(odd, VOCAB) == 0.0


def test_note_events_lengths():
    ev = M.note_events([CHORD, HOLD, HOLD, REST, BASS_C, HOLD], VOCAB)
    assert [(s, n) for s, n, _ in ev] == [(0, 3), (4, 2)]


def test_summarize_and_compare_keys():
    a = song(bar(CHORD) * 2, bar(HIGH_C) * 2, bar(BASS_C) * 2)
    s = M.summarize([a, a], VOCAB)
    assert {"chord_matching", "harmony_analysis", "nctn", "qr", "upc", "polyphony", "rests_ratio_bass"} <= set(s)
    c = M.compare_to_reference([a], [a], VOCAB)
    assert c["sequence_similarity_piano"] == approx(1.0)


def test_chord_matching_eq26_scores_each_chord_track_against_the_full_bar():
    # piano plays the C major triad: its own label = the bar's label (C major) -> match.
    # guitar plays only C5: a single pitch class labels as some C-containing triad, which may differ.
    full = song(bar(CHORD) * 2, bar(CHORD) * 2, bar(BASS_C) * 2)
    assert M.chord_matching(full, VOCAB) == 1.0                      # both chord tracks spell C major
    clash = song(bar(CHORD) * 2, bar(BASS_CS) * 2, bar(BASS_C) * 2)  # guitar plays C#: not C major
    assert M.chord_matching(clash, VOCAB) == 0.5
    silent = song(bar(CHORD) * 2, SILENT_BAR * 2, bar(BASS_C) * 2)   # silent guitar bars are skipped
    assert M.chord_matching(silent, VOCAB) == 1.0
