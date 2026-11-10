"""Train one model (see MODES in models/hybrid.py).

    python -m mtmg.train --config configs/base.yaml --mode hybrid --run-name hybrid_v1
    python -m mtmg.train --config configs/tiny.yaml --run-name smoke --max-steps 20   # CPU smoke test

Resume after an interruption (Colab / Kaggle sessions end without warning):

    python -m mtmg.train --config configs/base.yaml --mode hybrid --run-name hybrid_v1 --resume

`last.pt` is written at the end of every epoch and every `train.save_every` optimizer steps. It holds the
model, optimizer, grad scaler, step counters and RNG states, and the shuffling order of each epoch is
derived from the seed, so a resumed run continues exactly where it stopped instead of restarting the
epoch. `best.pt` (lowest validation loss) holds only the model and is what generate/evaluate use.
Everything needed to reproduce a run (config, seed, versions) is saved in runs/<run-name>/.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data.dataset import WindowDataset
from .models.factory import build_model
from .models.hybrid import MODES, ModelConfig
from .tokenization import Vocab


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_at(step: int, total: int, warmup: int, base_lr: float, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def evaluate(model, loader: DataLoader, device, amp_dtype, max_batches: int = 0):
    model.eval()
    tot_loss = tot_tok = tot_correct = 0.0
    for i, x in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = x.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss, logits = model.loss(x)
        n = x.numel()
        tot_loss += loss.item() * n
        tot_tok += n
        tot_correct += (logits.argmax(-1) == x).sum().item()
    model.train()
    mean = tot_loss / max(1, tot_tok)
    return {"loss": mean, "ppl": math.exp(min(mean, 20)), "acc": tot_correct / max(1, tot_tok)}


def epoch_order(n: int, seed: int, epoch: int) -> torch.Tensor:
    """Shuffling order of one epoch, reproducible from (seed, epoch) so a run can resume mid-epoch."""
    g = torch.Generator().manual_seed(seed * 100_003 + epoch)
    return torch.randperm(n, generator=g)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--mode", choices=list(MODES))
    ap.add_argument("--data-dir")
    ap.add_argument("--run-name")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (smoke tests)")
    ap.add_argument("--resume", nargs="?", const="last", default=None,
                    help="continue runs/<run-name>/last.pt (or the given checkpoint path)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.mode:
        cfg["model"]["mode"] = args.mode
    if args.data_dir:
        cfg["data"]["dir"] = args.data_dir
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    tc = cfg["train"]
    if args.resume and not args.run_name and args.resume == "last":
        raise SystemExit("--resume needs --run-name (or a checkpoint path)")
    run_name = args.run_name or f"{cfg['model']['mode']}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(tc.get("run_dir", "runs")) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(tc["seed"])
    device = resolve_device(args.device)
    amp_dtype = None
    if tc.get("amp", True) and device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    data_dir = Path(cfg["data"]["dir"])
    vocab = Vocab.load(data_dir / "vocab.json")
    train_ds, val_ds = WindowDataset(data_dir / "train.npy"), WindowDataset(data_dir / "val.npy")
    seq_len = train_ds.x.shape[-1]
    mcfg = ModelConfig(**{**cfg["model"], "vocab_size": vocab.size, "max_len": seq_len})

    resume = None
    if args.resume:
        path = run_dir / "last.pt" if args.resume == "last" else Path(args.resume)
        if not path.exists():
            raise SystemExit(f"nothing to resume: {path} not found")
        resume = torch.load(path, map_location="cpu", weights_only=False)   # our own file
        if "optimizer" not in resume:
            raise SystemExit(f"{path} has no optimizer state (written by an older version); "
                             "start a new run instead")
        mcfg = ModelConfig(**resume["model_cfg"])                            # the run's architecture wins

    model = build_model(mcfg).to(device)
    print(f"mode={mcfg.mode} params={model.num_parameters()/1e6:.2f}M device={device} "
          f"amp={amp_dtype} train={len(train_ds)} val={len(val_ds)} seq_len={seq_len}")

    nw = cfg["data"].get("num_workers", 0)
    bs = tc["batch_size"]
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    steps_per_epoch = len(train_ds) // bs
    if steps_per_epoch == 0:
        raise SystemExit("batch_size is larger than the number of training windows")

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tc["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=tc["lr"], betas=(0.9, 0.95))
    total_steps = tc["epochs"] * steps_per_epoch
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    save_every = int(tc.get("save_every", 500))

    step, best, start_epoch, skip = 0, float("inf"), 1, 0
    log_path = run_dir / "log.csv"
    if resume:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["optimizer"])
        if resume.get("scaler"):
            scaler.load_state_dict(resume["scaler"])
        step, best = resume["step"], resume["best"]
        start_epoch, skip = resume["epoch"], resume["batch_in_epoch"]
        if skip >= steps_per_epoch:                      # it stopped at an epoch boundary
            start_epoch, skip = start_epoch + 1, 0
        set_rng_state(resume["rng"])
        print(f"resumed from step {step} (epoch {start_epoch}, batch {skip}); best val loss {best:.4f}")
    else:
        (run_dir / "config.yaml").write_text(yaml.safe_dump({**cfg, "model": mcfg.to_dict()}))
        (run_dir / "env.json").write_text(json.dumps({
            "torch": torch.__version__, "python": platform.python_version(),
            "device": str(device), "seed": tc["seed"], "train_windows": len(train_ds),
            "params": model.num_parameters()}, indent=2))
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "step", "train_loss", "val_loss", "val_ppl", "val_acc", "lr", "secs"])

    def save_last(epoch: int, batch_in_epoch: int):
        torch.save({"model": model.state_dict(), "model_cfg": mcfg.to_dict(), "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None, "step": step,
                    "epoch": epoch, "batch_in_epoch": batch_in_epoch, "best": best, "rng": rng_state()},
                   run_dir / "last.pt")

    t0, done, lr = time.time(), step >= total_steps, tc["lr"]
    model.train()
    for epoch in range(start_epoch, tc["epochs"] + 1):
        if done:
            break
        order = epoch_order(len(train_ds), tc["seed"], epoch)[: steps_per_epoch * bs]
        first = skip if epoch == start_epoch else 0
        train_dl = DataLoader(train_ds, batch_size=bs, sampler=order[first * bs:].tolist(),
                              num_workers=nw, drop_last=True, pin_memory=device.type == "cuda")
        run_loss, run_n, b = 0.0, 0, first
        for x in train_dl:
            lr = lr_at(step, total_steps, tc["warmup_steps"], tc["lr"])
            for g in opt.param_groups:
                g["lr"] = lr
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss, _ = model.loss(x)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item()
            run_n += 1
            step += 1
            b += 1
            if step % tc.get("log_every", 50) == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {run_loss/run_n:.4f} lr {lr:.2e}")
            if save_every and step % save_every == 0:
                save_last(epoch, b)
            if step >= total_steps:
                done = True
                break
        val = evaluate(model, val_dl, device, amp_dtype)
        print(f"== epoch {epoch}: train {run_loss/max(1,run_n):.4f} | val loss {val['loss']:.4f} "
              f"ppl {val['ppl']:.2f} acc {val['acc']:.3f}")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, step, run_loss / max(1, run_n), val["loss"], val["ppl"],
                                    val["acc"], lr, int(time.time() - t0)])
        if val["loss"] < best:
            best = val["loss"]
            torch.save({"model": model.state_dict(), "model_cfg": mcfg.to_dict(), "epoch": epoch,
                        "step": step, "val_loss": val["loss"]}, run_dir / "best.pt")
        save_last(epoch, b)                              # b == steps_per_epoch unless stopped early
    print(f"finished. best val loss {best:.4f}. checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
