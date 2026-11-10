"""Evaluate a checkpoint on the test split.

    python -m mtmg.evaluate --ckpt runs/hybrid_v1/best.pt --data-dir data/processed \
        --out results/hybrid_v1.json

Reports:
  * test loss / perplexity / token accuracy (teacher forcing)
  * teacher-forced chord match (argmax prediction vs. ground truth, per bar)
  * arrangement metrics for (a) real test windows and (b) generated continuations of prompts
  * the PPO reward terms (rewards.py) for the same real and generated music
  * for the hybrid models: the mean gate value per track and layer (paper Eq. 4: share of the
    within-track Mamba branch) and the learned interaction matrix A per layer (paper Eq. 11)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import metrics as M
from .constants import STEPS_PER_BAR, TRACKS
from .data.dataset import WindowDataset
from .generate import load_model, resolve_device
from .rewards import mean_rewards
from .tokenization import Vocab
from .train import evaluate


@torch.no_grad()
def gate_statistics(model, windows, device, batch_size: int = 16):
    """Mean alpha (Mamba share) per layer and track over real windows. None if the model has no gate."""
    layers = [l for l in getattr(model, "layers", []) if getattr(l, "cross", False)]
    if not layers or model.cfg.mode != "hybrid":
        return None
    for l in layers:
        l.store_gate = True
    sums = np.zeros((len(layers), len(TRACKS)))
    n = 0
    for i in range(0, len(windows), batch_size):
        x = torch.from_numpy(np.stack(windows[i:i + batch_size])).to(device)
        model.teacher_forced_logits(x)
        for li, l in enumerate(layers):
            sums[li] += l.last_gate.float().mean(dim=(0, 2)).cpu().numpy()   # (B, K, T) -> (K,)
        n += 1
    for l in layers:
        l.store_gate, l.last_gate = False, None
    mean = sums / max(1, n)
    return {"per_layer": [{t: float(v) for t, v in zip(TRACKS, row)} for row in mean],
            "overall": {t: float(v) for t, v in zip(TRACKS, mean.mean(0))}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--out", default="results/eval.json")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-gen", type=int, default=32, help="generated samples for metrics")
    ap.add_argument("--prompt-bars", type=int, default=4)
    ap.add_argument("--num-real", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    device = resolve_device(args.device)
    model = load_model(args.ckpt, device)
    vocab = Vocab.load(Path(args.data_dir) / "vocab.json")
    ds = WindowDataset(Path(args.data_dir) / f"{args.split}.npy")
    dl = DataLoader(ds, batch_size=args.batch_size)
    res = {"ckpt": args.ckpt, "mode": model.cfg.mode, "split": args.split, "windows": len(ds),
           "params": model.num_parameters(),
           "interaction_graph": bool(getattr(model.cfg, "interaction_graph", False))}
    res["test"] = evaluate(model, dl, device, None)

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(ds))
    real = [np.asarray(ds.x[i], dtype=np.int64) for i in idx[:args.num_real]]
    res["real_data_metrics"] = M.summarize(real, vocab)
    res["real_data_rewards"] = mean_rewards(real, vocab)

    # teacher-forced chord agreement: argmax prediction vs. ground truth, per bar
    cm = []
    with torch.no_grad():
        for x in real[:min(len(real), 64)]:
            xt = torch.from_numpy(x).unsqueeze(0).to(device)
            pred = model.teacher_forced_logits(xt).argmax(-1)[0].cpu().numpy()
            cm.append(M.chord_match(pred, x, vocab))
    res["teacher_forced_chord_match"] = float(np.nanmean(cm)) if cm else float("nan")

    gates = gate_statistics(model, real[:64], device)
    if gates is not None:
        res["gate_mamba_share"] = gates
    if hasattr(model, "interaction_matrices") and model.interaction_matrices():
        res["interaction_matrix"] = {"rows_listen_to_columns": list(TRACKS),
                                     "per_layer": model.interaction_matrices()}

    if args.num_gen > 0:
        P = args.prompt_bars * STEPS_PER_BAR
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        chosen = idx[:min(args.num_gen, len(ds))]
        samples = []
        for i in range(0, len(chosen), args.batch_size):
            prefix = torch.from_numpy(np.stack([np.asarray(ds.x[j], dtype=np.int64)[:, :P]
                                                for j in chosen[i:i + args.batch_size]]))
            toks = model.generate_batch(model.cfg.max_len - P, prefix, len(prefix), args.temperature,
                                        args.top_k, args.top_p, gen)
            samples += [t.numpy() for t in toks]
        res["generated_metrics"] = M.summarize(samples, vocab)
        res["generated_rewards"] = mean_rewards([s[:, P:] for s in samples], vocab)   # continuation only
        res["generation"] = {"prompt_bars": args.prompt_bars, "temperature": args.temperature,
                             "top_k": args.top_k, "top_p": args.top_p}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
