"""Generate MIDI samples from a trained checkpoint.

    python -m mtmg.generate --ckpt runs/hybrid_v1/best.pt --data-dir data/processed \
        --num-samples 4 --prompt-bars 2 --out-dir outputs/hybrid_v1

--prompt-bars N continues the first N bars of random TEST windows (the prompt is
included in the output MIDI). With --prompt-bars 0 the model starts from BOS.
Samples are generated --batch-size at a time with the cached (fast) generator.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .constants import STEPS_PER_BAR
from .models.factory import build_model
from .models.hybrid import ModelConfig
from .tokenization import Vocab, tokens_to_midi


def load_model(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(ModelConfig(**ckpt["model_cfg"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8, help="samples generated in parallel")
    ap.add_argument("--prompt-bars", type=int, default=2)
    ap.add_argument("--bars", type=int, default=0, help="total bars (0 = full model length)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--bpm", type=float, default=110.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    device = resolve_device(args.device)
    model = load_model(args.ckpt, device)
    vocab = Vocab.load(Path(args.data_dir) / "vocab.json")
    test = np.load(Path(args.data_dir) / "test.npy", mmap_mode="r")
    max_len = model.cfg.max_len
    total = min((args.bars * STEPS_PER_BAR) if args.bars else max_len, max_len)
    P = args.prompt_bars * STEPS_PER_BAR
    assert P < total, "the prompt is as long as the whole piece"
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    i = 0
    while i < args.num_samples:
        n = min(args.batch_size, args.num_samples - i)
        prefix = None
        if P > 0:
            rows = [np.asarray(test[int(rng.integers(len(test)))], dtype=np.int64)[:, :P] for _ in range(n)]
            prefix = torch.from_numpy(np.stack(rows))
        tokens = model.generate_batch(total - P, prefix, n, args.temperature, args.top_k, args.top_p, gen)
        for b in range(n):
            path = out / f"sample_{i:02d}.mid"
            tokens_to_midi(tokens[b].numpy(), vocab, args.bpm).write(str(path))
            print("wrote", path)
            i += 1


if __name__ == "__main__":
    main()
