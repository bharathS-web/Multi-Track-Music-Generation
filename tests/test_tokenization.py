import numpy as np
import pretty_midi

from mtmg.constants import HOLD, REST
from mtmg.tokenization import (RAW_HOLD, RAW_OFFSET, RAW_REST, build_vocab_and_remapper,
                               pack_pitches, reduce_pitches, track_to_raw_steps, tokens_to_midi,
                               unpack_pitches)


def test_pack_unpack_roundtrip():
    for ps in [(0,), (127,), (36, 60), (40, 52, 64, 67)]:
        assert unpack_pitches(pack_pitches(ps)) == ps


def test_reduce_pitches_keeps_lowest_and_highest():
    assert reduce_pitches([60, 62, 64, 65, 67, 72], 4) == (60, 65, 67, 72)


def _track(notes):
    return [pretty_midi.Note(80, p, s, e) for p, s, e in notes]


def test_quantization_onsets_holds_rests():
    beats = np.arange(0, 9) * 0.5            # 120 bpm, 8 beats -> 32 steps (0.125 s each)
    # chord (60,64) at step 0 lasting 2 steps, silence, note 67 at step 8 lasting 4 steps
    notes = _track([(60, 0.0, 0.25), (64, 0.0, 0.25), (67, 1.0, 1.5)])
    raw = track_to_raw_steps(notes, beats)
    assert len(raw) == 32
    assert raw[0] == pack_pitches((60, 64)) + RAW_OFFSET
    assert raw[1] == RAW_HOLD and raw[2] == RAW_REST
    assert raw[8] == pack_pitches((67,)) + RAW_OFFSET
    assert (raw[9:12] == RAW_HOLD).all() and raw[12] == RAW_REST


def test_vocab_roundtrip_to_midi():
    beats = np.arange(0, 9) * 0.5
    notes = _track([(60, 0.0, 0.25), (64, 0.0, 0.25), (67, 1.0, 1.5)])
    raw = track_to_raw_steps(notes, beats)
    arr = np.stack([raw, raw, raw])[None]    # (1, 3, 32)
    vocab, remap = build_vocab_and_remapper(arr, vocab_size=16)
    tok = remap(arr)[0]
    assert tok[0, 2] == REST and tok[0, 1] == HOLD
    pm = tokens_to_midi(tok.astype(np.int64), vocab, bpm=120)
    got = sorted((n.pitch, round(n.start, 3), round(n.end, 3)) for n in pm.instruments[0].notes)
    want = sorted([(60, 0.0, 0.25), (64, 0.0, 0.25), (67, 1.0, 1.5)])
    assert got == want


def test_unseen_pitch_set_becomes_unk():
    a = np.array([[[pack_pitches((60,)) + RAW_OFFSET, RAW_REST]] * 3])
    vocab, remap = build_vocab_and_remapper(a, vocab_size=8)
    b = np.array([[[pack_pitches((61,)) + RAW_OFFSET, RAW_REST]] * 3])
    assert (remap(b)[0, :, 0] == 2).all()
