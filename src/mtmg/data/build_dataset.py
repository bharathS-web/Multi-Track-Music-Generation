"""Build the training dataset from a folder of MIDI files (e.g. Lakh MIDI).

    python -m mtmg.data.build_dataset --midi-dir D:/lmd_clean --out data/processed

Outputs in --out:
    train.npy / val.npy / test.npy   uint16, shape (N, 3, T)   (T = bars * 16)
    vocab.json                       pitch-set vocabulary (train split only)
    stats.json                       filtering + window statistics

Splits are made by GROUP so that near-duplicate songs cannot leak across train/val/test:
  * --split-by folder : all files in the same folder share a split (clean_midi: one folder per artist;
                        lmd_matched: one folder per matched song). Needs many folders.
  * --split-by file   : each file is assigned on its own (use when folders carry no meaning, e.g.
                        lmd_full, where files sit in 16 hash-prefix folders). Duplicate versions of a
                        song may then land in different splits, which makes test results optimistic.
  * --split-by auto   : "folder" if there are at least 100 distinct folders, otherwise "file".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from ..constants import NUM_TRACKS, REST, STEPS_PER_BAR, TRACKS
from ..tokenization import RAW_REST, beat_grid, build_vocab_and_remapper, track_to_raw_steps
from .filter_midi import FilterConfig, analyze_midi


def long_path(p: str) -> str:
    """Windows: add the extended-path prefix so folder names ending in '.' (e.g. 'Stevie V.')
    and very long paths can be opened. Does nothing on Linux/macOS."""
    if os.name == "nt":
        p = os.path.abspath(p)
        if not p.startswith("\\\\?\\"):
            p = "\\\\?\\" + p
    return p


def find_midis(root: str):
    """Recursively list .mid/.midi files. Returns (sorted file paths, root_used, skipped_dirs)."""
    root_used = long_path(str(root))
    found, skipped = [], []

    def on_error(err):
        skipped.append(str(err.filename))

    for dirpath, _dirs, names in os.walk(root_used, onerror=on_error):
        for n in names:
            if n.lower().endswith((".mid", ".midi")):
                found.append(os.path.join(dirpath, n))
    return sorted(found), root_used, skipped


def assign_split(key: str, val_frac: float, test_frac: float) -> str:
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 10_000 / 10_000
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"


def plan_splits(rels, split_by: str, val_frac: float, test_frac: float, min_groups: int = 100):
    """rels: relative file paths (forward slashes). Returns (splits list, mode used, number of groups)."""
    groups = [os.path.dirname(r) for r in rels]
    n_groups = len(set(groups))
    mode = split_by
    if split_by == "auto":
        mode = "folder" if n_groups >= min_groups else "file"
    keys = groups if mode == "folder" else list(rels)
    return [assign_split(k, val_frac, test_frac) for k in keys], mode, n_groups


def process_file(job):
    """Worker: returns (rel_path, split, reason, windows[int64 (n,3,T)] or None)."""
    path, rel, split, fcfg, bars, stride, min_active, max_notes = job
    pm, tracks, reason = analyze_midi(path, fcfg)
    if reason != "ok":
        return rel, split, reason, None
    try:
        beats = beat_grid(pm)
        if len(beats) < 2:
            return rel, split, "no_beats", None
        arrs = [track_to_raw_steps(tracks[name].notes, beats, max_notes) for name in TRACKS]
        n_steps = min(len(a) for a in arrs)
        n_bars = n_steps // STEPS_PER_BAR
        if n_bars < bars:
            return rel, split, "too_short", None
        full = np.stack([a[:n_steps] for a in arrs])          # (3, n_steps)
        win_len = bars * STEPS_PER_BAR
        wins = []
        for b in range(0, n_bars - bars + 1, stride):
            w = full[:, b * STEPS_PER_BAR: b * STEPS_PER_BAR + win_len]
            active = (w != RAW_REST).mean(axis=1)              # per track
            if (active >= min_active).all():
                wins.append(w)
        if not wins:
            return rel, split, "no_active_windows", None
        return rel, split, "ok", np.stack(wins)
    except Exception:
        return rel, split, "process_error", None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi-dir", required=True)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--bars", type=int, default=16)
    ap.add_argument("--stride-bars", type=int, default=8)
    ap.add_argument("--vocab-size", type=int, default=2048, help="including 4 special tokens")
    ap.add_argument("--max-notes", type=int, default=4, help="max notes per pitch-set token (<=4)")
    ap.add_argument("--min-active", type=float, default=0.25,
                    help="every track must be non-REST for at least this fraction of a window")
    ap.add_argument("--min-notes", type=int, default=100)
    ap.add_argument("--tempo-tol", type=float, default=0.05)
    ap.add_argument("--split-by", choices=["auto", "folder", "file"], default="auto")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--max-files", type=int, default=0, help="0 = all (use a small number to test)")
    ap.add_argument("--workers", type=int, default=0, help="0/1 = no multiprocessing")
    args = ap.parse_args(argv)
    assert 1 <= args.max_notes <= 4

    files, root_used, skipped = find_midis(args.midi_dir)
    if args.max_files:
        files = files[: args.max_files]
    print(f"found {len(files)} MIDI files under {args.midi_dir}"
          + (f" ({len(skipped)} folders could not be read and were skipped)" if skipped else ""))

    fcfg = FilterConfig(tempo_tol=args.tempo_tol, min_notes=args.min_notes)
    rels = [os.path.relpath(p, root_used).replace("\\", "/") for p in files]
    splits, split_mode, n_groups = plan_splits(rels, args.split_by, args.val_frac, args.test_frac)
    print(f"split mode: {split_mode} ({n_groups} distinct folders)")
    if split_mode == "file":
        print("WARNING: splitting per file. Different versions of the same song can end up in "
              "train and test, so test scores may be optimistic.")
    jobs = [(p, r, sp, fcfg, args.bars, args.stride_bars, args.min_active, args.max_notes)
            for p, r, sp in zip(files, rels, splits)]

    reasons = Counter()
    files_ok = Counter()
    windows = {"train": [], "val": [], "test": []}

    def consume(res):
        rel, split, reason, w = res
        reasons[reason] += 1
        if w is not None:
            files_ok[split] += 1
            windows[split].append(w)

    if args.workers and args.workers > 1:
        with Pool(args.workers) as pool:
            for res in tqdm(pool.imap_unordered(process_file, jobs, chunksize=8), total=len(jobs)):
                consume(res)
    else:
        for job in tqdm(jobs):
            consume(process_file(job))

    if not windows["train"]:
        raise SystemExit(f"No usable training windows. Rejection reasons: {dict(reasons)}")

    raw = {k: (np.concatenate(v) if v else np.zeros((0, NUM_TRACKS, args.bars * STEPS_PER_BAR), np.int64))
           for k, v in windows.items()}
    vocab, remap = build_vocab_and_remapper(raw["train"], args.vocab_size)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {"num_files_found": len(files), "skipped_folders": skipped[:50],
             "split_mode": split_mode, "num_groups": n_groups,
             "file_rejections": dict(reasons),
             "files_used": dict(files_ok), "vocab_size": vocab.size, "args": vars(args)}
    for split, arr in raw.items():
        final = remap(arr)
        np.save(out / f"{split}.npy", final)
        unk_rate = float((final == 2).sum() / max(1, (arr >= 5).sum())) if len(arr) else 0.0
        stats[f"{split}_windows"] = int(len(final))
        stats[f"{split}_unk_rate_among_onsets"] = unk_rate
    vocab.save(out / "vocab.json")
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
