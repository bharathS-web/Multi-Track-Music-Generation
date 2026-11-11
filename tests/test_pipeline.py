import json

import numpy as np
import pytest
import torch
import yaml

from mtmg import compare, cost, evaluate, generate, ppo, train
from mtmg.data import build_dataset
from mtmg.models.hybrid import MODES
from helpers import make_corpus


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pipeline")
    midi = make_corpus(tmp / "midi", n_files=24, bars=24)
    out = tmp / "processed"
    build_dataset.main(["--midi-dir", str(midi), "--out", str(out), "--bars", "4", "--stride-bars", "4",
                        "--vocab-size", "64", "--min-notes", "50", "--val-frac", "0.2", "--test-frac", "0.2"])
    return out


def tiny_config(tmp_path, data_dir, **train_overrides):
    c = yaml.safe_load(open("configs/tiny.yaml"))
    c["train"]["run_dir"] = str(tmp_path / "runs")
    c["train"].update(train_overrides)
    c["data"]["dir"] = str(data_dir)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(c))
    return path


def test_end_to_end(dataset, tmp_path):
    out = dataset
    stats = json.loads((out / "stats.json").read_text())
    assert stats["file_rejections"].get("ok", 0) == 24
    tr, va, te = (np.load(out / f"{s}.npy") for s in ("train", "val", "test"))
    assert tr.shape[1:] == (3, 64) and len(tr) > 0 and len(va) > 0 and len(te) > 0

    cfg = tiny_config(tmp_path, out)
    runs = tmp_path / "runs"
    for mode in MODES:
        train.main(["--config", str(cfg), "--mode", mode, "--run-name", mode, "--max-steps", "6"])
        assert (runs / mode / "best.pt").exists() and (runs / mode / "last.pt").exists()

    ckpt = str(runs / "hybrid" / "best.pt")
    gen_dir = tmp_path / "gen"
    generate.main(["--ckpt", ckpt, "--data-dir", str(out), "--out-dir", str(gen_dir),
                   "--num-samples", "3", "--batch-size", "2", "--prompt-bars", "1", "--device", "cpu"])
    assert len(list(gen_dir.glob("*.mid"))) == 3

    results = tmp_path / "results"
    for mode in ("hybrid", "transformer_only"):
        evaluate.main(["--ckpt", str(runs / mode / "best.pt"), "--data-dir", str(out),
                       "--out", str(results / f"{mode}.json"), "--num-gen", "3", "--batch-size", "2",
                       "--prompt-bars", "1", "--num-real", "8", "--device", "cpu"])
    res = json.loads((results / "hybrid.json").read_text())
    assert "test" in res and "generated_metrics" in res and "generated_rewards" in res
    assert set(res["gate_mamba_share"]["overall"]) == {"piano", "guitar", "bass"}
    assert len(res["interaction_matrix"]["per_layer"]) == 2               # tiny config: 2 layers, graph on
    assert "gate_mamba_share" not in json.loads((results / "transformer_only.json").read_text())

    compare.main(["--results", str(results), "--out", str(results / "summary.md")])
    table = (results / "summary.md").read_text()
    assert "| hybrid |" in table and "| transformer_only |" in table and "| Ground truth |" in table
    for key in ("chord_matching", "nctn", "qr", "sequence_similarity_bass"):
        assert key in res["generated_metrics"]


def test_resume_continues_exactly(dataset, tmp_path, monkeypatch):
    """Training 10 steps in one go == training that crashes after step 5 and is resumed."""
    cfg = tiny_config(tmp_path, dataset, save_every=3)
    train.main(["--config", str(cfg), "--run-name", "straight", "--max-steps", "10"])

    real_lr_at = train.lr_at

    def crash_at_step_5(step, *a, **k):
        if step == 5:
            raise KeyboardInterrupt
        return real_lr_at(step, *a, **k)

    monkeypatch.setattr(train, "lr_at", crash_at_step_5)
    with pytest.raises(KeyboardInterrupt):
        train.main(["--config", str(cfg), "--run-name", "crashed", "--max-steps", "10"])
    monkeypatch.setattr(train, "lr_at", real_lr_at)
    assert torch.load(tmp_path / "runs" / "crashed" / "last.pt", weights_only=False)["step"] == 3
    train.main(["--config", str(cfg), "--run-name", "crashed", "--max-steps", "10", "--resume"])

    a = torch.load(tmp_path / "runs" / "straight" / "last.pt", weights_only=False)
    b = torch.load(tmp_path / "runs" / "crashed" / "last.pt", weights_only=False)
    assert a["step"] == b["step"] == 10
    for k in a["model"]:
        assert torch.allclose(a["model"][k], b["model"][k], atol=1e-6), k


def test_ppo_smoke(dataset, tmp_path):
    cfg = tiny_config(tmp_path, dataset)
    train.main(["--config", str(cfg), "--run-name", "base", "--max-steps", "6"])
    ckpt = tmp_path / "runs" / "base" / "best.pt"
    ppo.main(["--ckpt", str(ckpt), "--data-dir", str(dataset), "--run-name", "rl",
              "--run-dir", str(tmp_path / "runs"), "--iters", "2", "--batch-size", "3", "--prompt-bars", "1",
              "--gen-bars", "2", "--eval-every", "1", "--val-batches", "2", "--device", "cpu"])
    final = tmp_path / "runs" / "rl" / "final.pt"
    assert final.exists()
    log = (tmp_path / "runs" / "rl" / "ppo_log.csv").read_text().strip().splitlines()
    assert len(log) == 3                                                    # header + 2 iterations
    before = torch.load(ckpt)["model"]
    after = torch.load(final)["model"]
    assert any(not torch.equal(before[k], after[k]) for k in before)        # the policy was updated
    evaluate.main(["--ckpt", str(final), "--data-dir", str(dataset), "--out", str(tmp_path / "rl.json"),
                   "--num-gen", "2", "--prompt-bars", "1", "--num-real", "4", "--device", "cpu"])


def test_cost_runs(capsys):
    cost.main(["--config", "configs/tiny.yaml", "--bars", "1", "2"])
    out = capsys.readouterr().out
    assert "| 1 bars |" in out and "| 2 bars |" in out
