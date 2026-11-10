"""Sanity-check a built dataset: token statistics + write a few windows back to MIDI to listen to.

    python -m mtmg.inspect_data --data-dir data/processed --split train --n 4 --out-dir outputs/roundtrip

If these MIDI files do not sound like music (wrong rhythm, missing parts, notes stuck on),
fix the tokenization BEFORE training anything.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .constants import HOLD, NUM_SPECIAL, REST, TRACKS, UNK
from .tokenization import Vocab, tokens_to_midi


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=4, help="number of windows to write as MIDI")
    ap.add_argument("--out-dir", default="outputs/roundtrip")
    ap.add_argument("--bpm", type=float, default=110.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    d = Path(args.data_dir)
    x = np.load(d / f"{args.split}.npy", mmap_mode="r")
    vocab = Vocab.load(d / "vocab.json")
    print(f"{args.split}: {x.shape[0]} windows of {x.shape[2]} steps ({x.shape[2] // 16} bars), "
          f"vocab size {vocab.size}")

    sample = np.asarray(x[: min(len(x), 2000)])
    print("\ntoken mix per track (fraction of steps):")
    print(f"{'track':8s} {'REST':>7s} {'HOLD':>7s} {'onset':>7s} {'UNK':>7s}  notes/onset")
    for k, name in enumerate(TRACKS):
        t = sample[:, k, :]
        onset = t >= NUM_SPECIAL
        sizes = [len(vocab.pitches(int(v)) or ()) for v in np.unique(t[onset])] if onset.any() else [0]
        print(f"{name:8s} {(t == REST).mean():7.3f} {(t == HOLD).mean():7.3f} "
              f"{onset.mean():7.3f} {(t == UNK).mean():7.3f}  {np.mean(sizes):.2f}")
    used = np.unique(sample[sample >= NUM_SPECIAL])
    print(f"\ndistinct pitch-set tokens used in this sample: {len(used)} of {len(vocab.pitch_sets)}")
    lows = [min(p) for p in vocab.pitch_sets[:200] if p]
    highs = [max(p) for p in vocab.pitch_sets[:200] if p]
    if lows:
        print(f"pitch range among the 200 most frequent pitch-sets: {min(lows)} .. {max(highs)}")

    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i in rng.choice(len(x), size=min(args.n, len(x)), replace=False):
        pm = tokens_to_midi(np.asarray(x[int(i)], dtype=np.int64), vocab, args.bpm)
        path = out / f"{args.split}_window_{int(i):05d}.mid"
        pm.write(str(path))
        print("wrote", path)


if __name__ == "__main__":
    main()
