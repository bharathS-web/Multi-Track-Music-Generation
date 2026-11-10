import numpy as np

from mtmg.constants import HOLD, NUM_SPECIAL, REST, STEPS_PER_BAR
from mtmg.rewards import RewardWeights, harmony_reward, pitch_range_reward, repetition_reward, reward
from mtmg.tokenization import Vocab

# pitch-set tokens: 4 = C3 major triad (48,52,55), 5 = C2 (36), 6 = C5 (72), 7 = C#2 (37), 8 = C9 (120)
VOCAB = Vocab([(48, 52, 55), (36,), (72,), (37,), (120,)])
CHORD, BASS_C, HIGH_C, BASS_CS, TOO_HIGH = range(NUM_SPECIAL, NUM_SPECIAL + 5)


def bar(token, n_bars=1):
    """One onset per beat, held in between."""
    beat = [token, HOLD, HOLD, HOLD]
    return beat * (STEPS_PER_BAR // 4) * n_bars


def song(piano, guitar, bass):
    return np.array([piano, guitar, bass], dtype=np.int64)


def test_harmony_rewards_shared_pitch_classes():
    good = song(bar(CHORD, 2), bar(HIGH_C, 2), bar(BASS_C, 2))      # all contain C
    bad = song(bar(CHORD, 2), bar(HIGH_C, 2), bar(BASS_CS, 2))      # C# shares nothing with C major
    assert harmony_reward(good, VOCAB) == 1.0
    assert harmony_reward(bad, VOCAB) == 0.0


def test_silence_is_not_rewarded():
    silent = song([REST] * 32, [REST] * 32, [REST] * 32)
    r = reward(silent, VOCAB)
    assert r["harmony"] == 0.0 and r["total"] == 0.0


def test_pitch_range_penalises_out_of_range_notes():
    ok = song(bar(CHORD), bar(HIGH_C), bar(BASS_C))
    assert pitch_range_reward(ok, VOCAB) == 0.0
    off = song(bar(CHORD), bar(TOO_HIGH), bar(BASS_C))              # guitar note 120 > 88
    assert -1.0 <= pitch_range_reward(off, VOCAB) < 0.0


def test_repetition_penalty_only_beyond_max_run():
    four = song(bar(CHORD), [REST] * 16, [REST] * 16)               # 4 identical onsets
    eight = song(bar(CHORD, 2), [REST] * 16 * 2, [REST] * 16 * 2)    # 8 identical onsets
    assert repetition_reward(four, VOCAB, max_run=4) == 0.0
    assert repetition_reward(eight, VOCAB, max_run=4) == -0.5        # onsets 5..8 of 8 are excess


def test_weights_combine_terms():
    s = song(bar(CHORD, 2), bar(HIGH_C, 2), bar(BASS_C, 2))
    r = reward(s, VOCAB, RewardWeights(pitch=0, repeat=0, harmony=2.0))
    assert r["total"] == 2.0 * r["harmony"]
