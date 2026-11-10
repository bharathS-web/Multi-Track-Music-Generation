"""Compute cost of ONE forward pass over a piece, per model and length (no training data needed).

    python -m mtmg.cost --config configs/base.yaml --bars 16 64

Counts matrix multiplications, convolutions and attention with PyTorch's FlopCounterMode and reports
multiply-accumulates (MACs = FLOPs / 2; "G" = 1e9). Element-wise work is NOT in that count:
activations, norms, softmax and the Mamba selective-scan recurrence (run as element-wise ops in the
pure-PyTorch block). The `+scan` column estimates the scan separately (about 4 MAC-equivalents per
d_inner x d_state element per token per Mamba block), so you can see it is small.
"""
from __future__ import annotations

import argparse

import torch
import yaml
from torch.utils.flop_counter import FlopCounterMode

from .constants import NUM_TRACKS, STEPS_PER_BAR
from .models.factory import build_model
from .models.hybrid import MODES, ModelConfig


def forward_macs(mcfg: ModelConfig, steps: int) -> int:
    model = build_model(mcfg).eval()
    x = torch.randint(4, mcfg.vocab_size, (1, NUM_TRACKS, steps))
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model.teacher_forced_logits(x)
    return fc.get_total_flops() // 2


def scan_macs(mcfg: ModelConfig, steps: int) -> int:
    """Element-wise selective-scan work not seen by FlopCounterMode (estimate)."""
    if mcfg.mode == "transformer_only":
        return 0
    tokens = NUM_TRACKS * (steps + (0 if mcfg.mode == "mamba_only" else 1))
    return 4 * tokens * mcfg.n_layers * mcfg.expand * mcfg.d_model * mcfg.d_state


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--bars", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--modes", nargs="+", default=["transformer_only", "mamba_only", "hybrid"],
                    choices=list(MODES))
    ap.add_argument("--vocab-size", type=int, default=2048)
    args = ap.parse_args(argv)

    base = yaml.safe_load(open(args.config))["model"]
    base.update(grad_checkpoint=False, dropout=0.0, vocab_size=args.vocab_size)
    print(f"Forward-pass cost, {base['n_layers']} layers, d_model {base['d_model']} "
          f"(G = 1e9 multiply-accumulates)\n")
    header = "| Length | " + " | ".join(args.modes) + " |"
    print(header)
    print("|" + "---|" * (len(args.modes) + 1))
    params = {}
    for bars in args.bars:
        steps = bars * STEPS_PER_BAR
        cells, ref = [], None
        for mode in args.modes:
            torch.manual_seed(0)
            mcfg = ModelConfig(**{**base, "mode": mode, "max_len": steps})
            macs = forward_macs(mcfg, steps)
            ref = ref or macs
            params.setdefault(mode, sum(p.numel() for p in build_model(mcfg).parameters()))
            scan = scan_macs(mcfg, steps)
            extra = f" (+scan {scan / 1e9:.2f})" if scan else ""
            cells.append(f"{macs / 1e9:.1f} G{extra}, {macs / ref:.2f}x")
        print(f"| {bars} bars | " + " | ".join(cells) + " |")
    print("\nparameters at the first length: " + ", ".join(f"{m} {n / 1e6:.2f}M" for m, n in params.items()))
    print("ratios are relative to the first model in --modes.")


if __name__ == "__main__":
    main()
