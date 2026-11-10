"""Reward fine-tuning with clipped PPO (paper Sec. IX, Eq. 21-23), starting from a trained checkpoint.

    python -m mtmg.ppo --ckpt runs/hybrid_v1/best.pt --run-name hybrid_v1_ppo --iters 200

Each iteration:
  1. take `--batch-size` random TRAIN windows, keep their first `--prompt-bars` bars as prompts;
  2. sample `--gen-bars` bars per prompt from the current policy (temperature 1, no top-k/top-p, so the
     samples come from exactly the distribution whose log-probabilities PPO uses);
  3. score the generated part with rewards.py (Eq. 22); advantage = reward normalised within the batch
     (no value network: every sample of the batch is a baseline for the others);
  4. `--ppo-epochs` updates of

        L = ce_coef * L_CE(real batch) + rl_coef * L_PPO + kl_coef * KL(policy || start) - ent_coef * H

     L_CE on real music, L_PPO (Eq. 23, clipped ratio) and the entropy term are the paper's Eq. 21.
     The KL penalty towards the starting model is MY addition (common practice); together with L_CE it
     keeps the model from drifting into music that only scores well on the heuristics.

Writes runs/<run-name>/ppo_log.csv, last.pt and final.pt (same format as train.py's checkpoints, so
generate/evaluate work on them unchanged). Compare evaluate.py results before and after: the rewards
should go up WITHOUT the validation loss getting much worse.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .constants import STEPS_PER_BAR
from .data.dataset import WindowDataset
from .generate import load_model, resolve_device
from .models.hybrid import mask_special
from .rewards import RewardWeights, reward
from .tokenization import Vocab
from .train import evaluate, set_seed


def token_logprobs(model, x: torch.Tensor, entropy: bool = False):
    """Log-probability of every token of x (B, K, T) under the model's sampling distribution
    (UNK/BOS removed, temperature 1). Optionally also the entropy per position."""
    logits = mask_special(model.teacher_forced_logits(x))
    logp_all = F.log_softmax(logits, dim=-1)
    lp = logp_all.gather(-1, x.unsqueeze(-1)).squeeze(-1)
    if not entropy:
        return lp
    p = logp_all.exp()
    ent = -(p * logp_all.masked_fill(~torch.isfinite(logp_all), 0.0)).sum(-1)
    return lp, ent


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="cross-entropy checkpoint to start from (best.pt)")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--run-dir", default="runs")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16, help="rollouts per iteration (>= 2)")
    ap.add_argument("--prompt-bars", type=int, default=2)
    ap.add_argument("--gen-bars", type=int, default=8)
    ap.add_argument("--ppo-epochs", type=int, default=2)
    ap.add_argument("--clip", type=float, default=0.2, help="epsilon in Eq. 23")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ce-coef", type=float, default=1.0)
    ap.add_argument("--rl-coef", type=float, default=1.0, help="lambda_RL in Eq. 21")
    ap.add_argument("--ent-coef", type=float, default=0.01, help="lambda_H in Eq. 21")
    ap.add_argument("--kl-coef", type=float, default=0.05)
    ap.add_argument("--beta-pitch", type=float, default=1.0)
    ap.add_argument("--beta-repeat", type=float, default=1.0)
    ap.add_argument("--beta-harmony", type=float, default=1.0)
    ap.add_argument("--max-run", type=int, default=4, help="repeats allowed before R_repeat penalises")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--val-batches", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    assert args.batch_size >= 2, "advantages are normalised within the batch: use --batch-size >= 2"

    set_seed(args.seed)
    device = resolve_device(args.device)
    policy = load_model(args.ckpt, device)
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    policy.eval()                    # no dropout: the PPO ratio must compare the same function

    data = Path(args.data_dir)
    vocab = Vocab.load(data / "vocab.json")
    train_ds, val_ds = WindowDataset(data / "train.npy"), WindowDataset(data / "val.npy")
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)
    P, G = args.prompt_bars * STEPS_PER_BAR, args.gen_bars * STEPS_PER_BAR
    assert P + G <= policy.cfg.max_len, f"prompt + generated ({P + G}) > max_len {policy.cfg.max_len}"
    weights = RewardWeights(args.beta_pitch, args.beta_repeat, args.beta_harmony, args.max_run)

    run_dir = Path(args.run_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ppo_args.json").write_text(json.dumps(vars(args), indent=2))
    log_path = run_dir / "ppo_log.csv"
    cols = ["iter", "reward", "r_pitch", "r_repeat", "r_harmony", "pg_loss", "kl", "ce", "entropy",
            "clip_frac", "val_loss", "val_ppl", "secs"]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(cols)

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    ckpt = lambda: {"model": policy.state_dict(), "model_cfg": policy.cfg.to_dict(),
                    "ppo_args": vars(args), "start_ckpt": args.ckpt}
    t0 = time.time()

    for it in range(1, args.iters + 1):
        # 1-2. rollouts
        idx = rng.choice(len(train_ds), size=args.batch_size, replace=len(train_ds) < args.batch_size)
        prompts = torch.from_numpy(np.stack([np.asarray(train_ds.x[i], np.int64)[:, :P] for i in idx]))
        samples = policy.generate_batch(G, prompts, args.batch_size, temperature=1.0, top_k=0, top_p=1.0,
                                        generator=gen)
        # 3. rewards on the generated part only
        rs = [reward(s[:, P:].numpy(), vocab, weights) for s in samples]
        r = torch.tensor([x["total"] for x in rs], dtype=torch.float32, device=device)
        adv = ((r - r.mean()) / (r.std() + 1e-6))[:, None, None]

        x = samples.to(device)
        with torch.no_grad():
            old_lp = token_logprobs(policy, x)[..., P:]
            ref_lp = token_logprobs(ref, x)[..., P:]

        # 4. PPO updates (+ cross-entropy on real windows, Eq. 21)
        stats = []
        for _ in range(args.ppo_epochs):
            lp, ent = token_logprobs(policy, x, entropy=True)
            lp, ent = lp[..., P:], ent[..., P:]
            ratio = torch.exp(lp - old_lp)
            pg = -torch.min(ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv).mean()
            d = ref_lp - lp
            kl = (torch.exp(d) - d - 1).mean()                         # >= 0, unbiased KL estimate
            real = torch.from_numpy(np.stack([np.asarray(train_ds.x[i], np.int64) for i in
                                              rng.choice(len(train_ds), size=args.batch_size)])).to(device)
            ce = policy.loss(real)[0]
            loss = args.ce_coef * ce + args.rl_coef * pg + args.kl_coef * kl - args.ent_coef * ent.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            opt.step()
            clip_frac = ((ratio - 1).abs() > args.clip).float().mean()
            stats.append([pg.item(), kl.item(), ce.item(), ent.mean().item(), clip_frac.item()])
        pg_l, kl_v, ce_v, ent_v, cf = np.mean(stats, axis=0)

        val_loss = val_ppl = ""
        if it % args.eval_every == 0 or it == args.iters:
            v = evaluate(policy, val_dl, device, None, max_batches=args.val_batches)
            policy.eval()                                              # evaluate() switches to train()
            val_loss, val_ppl = round(v["loss"], 5), round(v["ppl"], 3)
            torch.save(ckpt(), run_dir / "last.pt")
        row = [it, r.mean().item()] + [float(np.mean([x[k] for x in rs])) for k in ("pitch", "repeat", "harmony")]
        row += [pg_l, kl_v, ce_v, ent_v, cf, val_loss, val_ppl, int(time.time() - t0)]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([round(v, 5) if isinstance(v, float) else v for v in row])
        print(f"iter {it}: reward {row[1]:.3f} (pitch {row[2]:.3f} repeat {row[3]:.3f} harmony {row[4]:.3f}) "
              f"kl {kl_v:.4f} ce {ce_v:.3f}" + (f" | val loss {val_loss} ppl {val_ppl}" if val_loss != "" else ""))

    torch.save(ckpt(), run_dir / "final.pt")
    print(f"done. checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
