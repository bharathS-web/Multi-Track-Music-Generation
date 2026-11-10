"""Shared constants for the 16th-note grid tokenization."""

TRACKS = ("piano", "guitar", "bass")
NUM_TRACKS = len(TRACKS)

# General MIDI program ranges used to map instruments to the three roles.
ROLE_PROGRAMS = {
    "piano": range(0, 8),
    "guitar": range(24, 32),
    "bass": range(32, 40),
}
# Programs used when rendering generated tokens back to MIDI.
OUT_PROGRAMS = {"piano": 0, "guitar": 25, "bass": 33}

STEPS_PER_BEAT = 4          # 16th-note grid
BEATS_PER_BAR = 4           # 4/4 only
STEPS_PER_BAR = STEPS_PER_BEAT * BEATS_PER_BAR

# Token ids in the FINAL vocabulary.
REST = 0    # nothing sounding on this track at this step
HOLD = 1    # no new onset, previous notes keep sounding
UNK = 2     # a pitch-set that was not frequent enough to be in the vocabulary
BOS = 3     # start-of-sequence input token (never a target)
NUM_SPECIAL = 4
# ids >= NUM_SPECIAL are pitch-set tokens (a chord/note onset on that step).

# Order in which the three tracks are generated WITHIN one time step: bass first (harmonic root),
# then piano, then guitar. A track sees the already-generated tokens of the tracks before it.
GEN_ORDER = (2, 0, 1)                                           # track indices, in generation order
SLOT_OF_TRACK = tuple(GEN_ORDER.index(k) for k in range(NUM_TRACKS))   # position of each track in that order
