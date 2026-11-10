import os

import numpy as np

from mtmg.data.build_dataset import plan_splits


def make_rels(n_artists, songs_each=3):
    return [f"Artist {a}/Song {s}.mid" for a in range(n_artists) for s in range(songs_each)]


def test_folder_mode_keeps_every_folder_in_one_split():
    rels = make_rels(300)
    splits, mode, n_groups = plan_splits(rels, "auto", 0.1, 0.1)
    assert mode == "folder" and n_groups == 300
    by_folder = {}
    for r, s in zip(rels, splits):
        by_folder.setdefault(os.path.dirname(r), set()).add(s)
    assert all(len(v) == 1 for v in by_folder.values())          # no artist spans two splits
    frac = {k: np.mean([s == k for s in splits]) for k in ("train", "val", "test")}
    assert 0.6 < frac["train"] < 0.95 and frac["val"] > 0.02 and frac["test"] > 0.02


def test_auto_falls_back_to_file_mode_with_few_folders():
    rels = [f"{h:x}/{i:04d}.mid" for h in range(16) for i in range(50)]   # hashed folders, like lmd_full
    splits, mode, n_groups = plan_splits(rels, "auto", 0.1, 0.1)
    assert mode == "file" and n_groups == 16
    assert len(set(splits)) == 3


def test_split_is_deterministic_and_forceable():
    rels = make_rels(10)
    a = plan_splits(rels, "folder", 0.2, 0.2)[0]
    b = plan_splits(rels, "folder", 0.2, 0.2)[0]
    assert a == b and plan_splits(rels, "folder", 0.2, 0.2)[1] == "folder"
