"""Collect every evaluated run into one Markdown table (for the README / a write-up).

    python -m mtmg.compare --results results --out results/summary.md

Reads results/*.json written by evaluate.py and, for each, the training log next to its checkpoint
(runs/<name>/log.csv) for the best validation perplexity. Lower perplexity is better; for the
arrangement metrics compare against the "real data" row rather than reading higher as better.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def best_val_ppl(ckpt: str):
    log = Path(ckpt).parent / "log.csv"
    if not log.exists():
        return None
    with open(log, newline="") as f:
        losses = [float(r["val_loss"]) for r in csv.DictReader(f) if r.get("val_loss")]
    return math.exp(min(losses)) if losses else None


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    files = sorted(Path(args.results).glob("*.json"))
    if not files:
        raise SystemExit(f"no evaluate.py results (*.json) in {args.results}")
    rows, real = [], None
    for p in files:
        r = json.loads(p.read_text())
        if "test" not in r:
            continue
        g, gr = r.get("generated_metrics", {}), r.get("generated_rewards", {})
        real = real or (r.get("real_data_metrics"), r.get("real_data_rewards"))
        rows.append([p.stem, r.get("mode"), fmt(r.get("params", 0) / 1e6, 2) if r.get("params") else "–",
                     fmt(best_val_ppl(r["ckpt"]), 2), fmt(r["test"]["ppl"], 2), fmt(r["test"]["acc"]),
                     fmt(r.get("teacher_forced_chord_match")), fmt(g.get("harmony_analysis")),
                     fmt(g.get("polyphony")), fmt(g.get("upc"), 2), fmt(gr.get("total"))])
    head = ["run", "mode", "params (M)", "best val ppl", "test ppl", "test acc", "chord match (TF)",
            "gen harmony", "gen polyphony", "gen UPC", "gen reward"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    if real and real[0]:
        m, rw = real
        lines.append("| real test data | – | – | – | – | – | – | " + " | ".join(
            [fmt(m.get("harmony_analysis")), fmt(m.get("polyphony")), fmt(m.get("upc"), 2),
             fmt((rw or {}).get("total"))]) + " |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
