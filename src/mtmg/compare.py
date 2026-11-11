"""Collect every evaluated model into the paper's result tables (Markdown).

    python -m mtmg.compare --results results --out results/summary.md

Reads results/*.json written by evaluate.py (one file per model) and prints

  Perplexity      best validation and test perplexity, parameters (lower perplexity is better)
  Table I         chord matching (CM) and harmony analysis (HA)
  Table II        NCTN (%), QR (%), UPC, polyphony PP (%)
  Table III       per-track sequence similarity and rests ratio

with a "Ground truth" row computed on real test windows. Generated rows are measured on the
continuation of the prompts, so compare them with the ground-truth row rather than reading every
metric as "higher is better".
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

TRACKS = ("piano", "guitar", "bass")


def best_val_ppl(ckpt: str):
    log = Path(ckpt).parent / "log.csv"
    if not log.exists():
        return None
    with open(log, newline="") as f:
        losses = [float(r["val_loss"]) for r in csv.DictReader(f) if r.get("val_loss")]
    return math.exp(min(losses)) if losses else None


def fmt(v, nd=2, pct=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{100 * v:.1f}" if pct else f"{v:.{nd}f}"


def table(title, head, rows):
    out = [f"**{title}**", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    runs = []
    for p in sorted(Path(args.results).glob("*.json")):
        r = json.loads(p.read_text())
        if "test" in r:
            runs.append((p.stem, r))
    if not runs:
        raise SystemExit(f"no evaluate.py results (*.json) in {args.results}")
    gt = runs[0][1].get("real_data_metrics", {})

    ppl = [[name, r.get("mode", "–"), fmt(r["params"] / 1e6) if r.get("params") else "–",
            fmt(best_val_ppl(r["ckpt"])), fmt(r["test"]["ppl"]), fmt(r["test"]["acc"], 3)] for name, r in runs]
    t1 = [[name, fmt(r.get("generated_metrics", {}).get("chord_matching")),
           fmt(r.get("generated_metrics", {}).get("harmony_analysis"))] for name, r in runs]
    t1.append(["Ground truth", fmt(gt.get("chord_matching")), fmt(gt.get("harmony_analysis"))])

    def t2_row(name, m):
        return [name, fmt(m.get("nctn"), pct=True), fmt(m.get("qr"), pct=True), fmt(m.get("upc")),
                fmt(m.get("polyphony"), pct=True)]
    t2 = [t2_row(name, r.get("generated_metrics", {})) for name, r in runs] + [t2_row("Ground truth", gt)]

    def t3_row(name, m):
        return ([name] + [fmt(m.get(f"sequence_similarity_{t}")) for t in TRACKS]
                + [fmt(m.get(f"rests_ratio_{t}")) for t in TRACKS])
    t3 = [t3_row(name, r.get("generated_metrics", {})) for name, r in runs] + [t3_row("Ground truth", gt)]

    text = "\n".join([
        table("Perplexity", ["Model", "Mode", "Params (M)", "Best val ppl", "Test ppl", "Test acc"], ppl),
        table("Table I. Chord matching and harmony analysis", ["Model", "CM", "HA"], t1),
        table("Table II. Rhythm and pitch statistics", ["Model", "NCTN (%)", "QR (%)", "UPC", "PP (%)"], t2),
        table("Table III. Sequence similarity and rests ratio",
              ["Model", "Sim. piano", "Sim. guitar", "Sim. bass", "Rests piano", "Rests guitar", "Rests bass"], t3),
    ])
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
