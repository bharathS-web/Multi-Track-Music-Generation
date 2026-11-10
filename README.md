# Multi-Track Music Generation (Mamba + Cross-Track Attention)

A from-scratch **reimplementation** of the hybrid multi-track symbolic music generator
described in the paper *"Hybrid Multi-Track Music Generation using Transformers and State Space Models"*
(Veena S T, Bharath Kumar S, Nandhagopal R).

> **Status: reimplementation.** The original project code was lost. This repository rebuilds the
> system from the paper. The paper does not specify hyperparameters, tokenization, filtering
> thresholds or reward formulas, so those are my own choices (listed below). Numbers produced by
> this code come from *this* code and are **not** the paper's numbers.

The model writes three instrument tracks (piano, guitar, bass) together. Each track keeps its own
memory with a per-track **Mamba** (selective state space) block, the tracks listen to each other through
**cross-track attention**, and a learned **gate** decides, token by token, how much of each to use.

## What is implemented

| Part | Where | Notes |
|---|---|---|
| MIDI filtering (4/4, stable tempo, piano+guitar+bass) | `src/mtmg/data/filter_midi.py` | every rejection reason is counted |
| 16th-note grid tokenizer, vocab, MIDI decode | `src/mtmg/tokenization.py` | 1 token per track per step |
| Dataset builder (artist-level splits, windows) | `src/mtmg/data/build_dataset.py` | no train/test leakage between songs |
| Per-track selective SSM (Mamba-style) | `src/mtmg/models/mamba_block.py` | pure PyTorch; optional `mamba_ssm` |
| Embedding fusion (Eq. 6-8), cross-track attention (Eq. 1-2), gate (Eq. 3-4) | `src/mtmg/models/hybrid.py` | |
| **Instrument interaction graph (Eq. 11)** | `src/mtmg/models/hybrid.py` | `interaction_graph: true` (on in `configs/base.yaml`) |
| Baselines and ablations | `hybrid.py`, `flat.py` | `transformer_only`, `mamba_only`, `mamba_independent`, `hybrid_no_gate` |
| **Fast cached generation** | `generate_batch` in both model files | recurrent Mamba state + key/value cache, batched; tested equal to full recomputation |
| Training with **resume** | `src/mtmg/train.py` | `--resume` continues mid-epoch with the same data order |
| **PPO reward fine-tuning (Sec. IX, Eq. 21-23)** | `src/mtmg/ppo.py`, `src/mtmg/rewards.py` | reward terms are my definitions (the paper gives none) |
| Evaluation: perplexity, chord match, arrangement metrics, rewards, gate share, learned graph | `src/mtmg/evaluate.py`, `metrics.py` | my metric definitions, see docstrings |
| Forward-pass cost per model and length | `src/mtmg/cost.py` | no data needed |
| Results table across runs | `src/mtmg/compare.py` | Markdown, for this README |
| Tests (62) | `tests/` | causality, cached-vs-full generation, resume, PPO, end-to-end pipeline |

**Not done:** the listening study from the paper, and trained results (see [Results](#results)).

## Design decisions (not specified in the paper)

- **Representation:** 16th-note grid, 16 bars = 256 steps, 3 tracks aligned in time. Token per step: `REST`, `HOLD`,
  `UNK`, or a **pitch-set** (note/chord with up to 4 pitches). Vocabulary = the 2044 most frequent pitch-sets in the
  train split (configurable). A note lasts until the next non-`HOLD` token (one chord at a time per track).
- **Within-step track order:** at every time step the tracks are generated in the order bass -> piano -> guitar
  (`GEN_ORDER` in `constants.py`). A track is conditioned on all earlier steps AND on the already-generated tokens of
  the tracks before it in the same step, so the instruments coordinate instead of being sampled independently.
  In the hybrid models this works through cross-track attention over other tracks only (same-track memory comes from
  Mamba); `tests/test_model.py` and `tests/test_generation.py` check exhaustively that no token can see itself or
  anything generated after it.
- **Interaction graph (Eq. 11):** each track attends to every other track *separately* (one softmax per source
  track, giving Z<sub>i←j</sub>), and the results are mixed with a learned matrix A whose row i is a distribution
  over the other tracks: Attn<sub>i</sub> = Σ<sub>j</sub> A<sub>ij</sub> Z<sub>i←j</sub>. `evaluate.py` prints the
  learned A per layer, so you can check the paper's claim that piano and bass couple strongly. With
  `interaction_graph: false` a single softmax with a learned track-pair bias is used instead.
- **Gate:** one scalar per token (paper Eq. 3). Residual connections and a feed-forward block are added around it.
  `hybrid_no_gate` fixes it at 0.5 to test whether learning it helps.
- **Baselines** (defined by me, same data / token order / loss as the hybrid):
  `transformer_only` = one causal Transformer over all tracks flattened into a single stream (MMM-style);
  `mamba_only` = one Mamba stack over the same flat stream (fair Mamba baseline);
  `mamba_independent` = per-track Mamba with no interaction between tracks (ablation, expected to be weak on harmony).
  They are **not parameter-matched** by default (hybrid ~8.5M, mamba ~7.0M, transformer ~5.9M), so adjust
  `n_layers` / `d_model` or report parameter counts when comparing.
- **PPO rewards** (paper Eq. 22 names them, but doesn't define them): R<sub>pitch</sub> = minus the share of notes
  outside a plausible range for the instrument; R<sub>repeat</sub> = minus the share of onsets repeating the same
  chord more than 4 times in a row; R<sub>harmony</sub> = share of bars where all tracks share a pitch class
  (0 for silence, so silence is never rewarded). The cross-entropy term of Eq. 21 stays on during PPO, and I add a
  small KL penalty to the starting model, so the model can't drift into music that only scores well on these heuristics.

## Cost of one forward pass

`python -m mtmg.cost --config configs/base.yaml --bars 16 64` (6 layers, width 256; G = 10⁹ multiply-accumulates):

| Length | Transformer | Mamba | Hybrid |
|---|---|---|---|
| 16 bars | 6.3 G | 5.2 G (0.83×) | 11.0 G (1.75×) |
| 64 bars | 46.9 G | 21.0 G (0.45×) | 87.6 G (1.87×) |

The hybrid is the most expensive: its cross-track attention is dense over the whole past. Mamba's cost grows
linearly with length, attention's quadratically. The count covers matrix multiplications, convolutions and attention;
the Mamba scan's element-wise work adds about 0.15 G at 16 bars (printed separately by the script).

## Setup (Windows, PowerShell)

```powershell
cd C:\Users\Bharath\Music\Multi-Track_Music_Generation
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -q          # 62 tests, about a minute on CPU
```

For GPU training install the CUDA build of PyTorch first (see pytorch.org), or train on Colab / Kaggle.
`mamba-ssm` only installs on Linux with CUDA; the default pure-PyTorch Mamba works everywhere (slower).

## Data

Download the **Lakh MIDI Dataset** (the "clean" subset is a good start; see its page at colinraffel.com/projects/lmd
and check its license/terms), extract it, then:

```powershell
python -m mtmg.data.build_dataset --midi-dir D:\datasets\clean_midi --out data\processed --workers 4
```

Try `--max-files 500` first. Look at `data/processed/stats.json` (how many files each rule rejected, UNK rate) and
**listen to a round trip** before training anything: `python -m mtmg.inspect_data --data-dir data\processed`.

**Train/val/test split:** by default (`--split-by auto`) all files in one folder share a split (clean_midi has one folder
per artist, so versions of the same song cannot leak into the test set). With fewer than 100 folders (e.g. a small
`--max-files` test, or lmd_full's hash folders) it falls back to per-file splitting and prints a warning: test scores
are then optimistic. `stats.json` records the mode used. Lakh contains many near-duplicates: see
[Choi et al., ISMIR 2025](https://arxiv.org/abs/2509.16662).

**My build of the clean subset** (default settings):

| | |
|---|---|
| MIDI files found | 17,256 |
| Kept after filtering | 4,034 (train 3,642 · val 190 · test 202 files) |
| Split | by artist folder (2,197 artists) |
| 16-bar windows | train 30,693 · val 1,556 · test 1,920 |
| Vocabulary | 2,048 tokens; UNK among note onsets: 3.4% train, 6.3% val, 4.9% test |
| Main rejection reasons | missing piano 3,812 · not 4/4 2,766 · unstable tempo 2,737 · missing guitar 2,192 |

## Train, generate, evaluate

```powershell
# quick CPU smoke test (tiny model)
python -m mtmg.train --config configs/tiny.yaml --run-name smoke --max-steps 20

# real runs (GPU recommended): same data, same seed
python -m mtmg.train --config configs/base.yaml --mode hybrid            --run-name hybrid_v1
python -m mtmg.train --config configs/base.yaml --mode mamba_only        --run-name mamba_v1
python -m mtmg.train --config configs/base.yaml --mode transformer_only  --run-name transformer_v1
# ablations
python -m mtmg.train --config configs/base.yaml --mode hybrid_no_gate    --run-name hybrid_nogate_v1
python -m mtmg.train --config configs/base.yaml --mode mamba_independent --run-name mamba_indep_v1

# the session died? continue from the last checkpoint (saved every 500 steps and every epoch)
python -m mtmg.train --config configs/base.yaml --mode hybrid --run-name hybrid_v1 --resume

python -m mtmg.generate --ckpt runs\hybrid_v1\best.pt --out-dir outputs\hybrid_v1 --num-samples 8 --prompt-bars 2
python -m mtmg.evaluate --ckpt runs\hybrid_v1\best.pt --out results\hybrid_v1.json
python -m mtmg.compare  --results results --out results\summary.md
```

Each run saves its config, seed, environment and a CSV log in `runs/<name>/`.
Generated `.mid` files can be played in any MIDI player or rendered with FluidSynth + a soundfont.

### Optional: PPO reward fine-tuning

Start from a trained cross-entropy checkpoint, then evaluate both and compare:

```powershell
python -m mtmg.ppo --ckpt runs\hybrid_v1\best.pt --run-name hybrid_v1_ppo --iters 200
python -m mtmg.evaluate --ckpt runs\hybrid_v1_ppo\final.pt --out results\hybrid_v1_ppo.json
```

`runs/<name>/ppo_log.csv` tracks the reward terms, KL to the start model, the cross-entropy and the validation loss.
A good run raises the rewards without making validation perplexity much worse. If perplexity climbs, raise
`--kl-coef` or lower `--lr`.

## Results

**No trained results yet.** Only a 20-step CPU smoke run exists; the numbers above are costs, not quality.
When the runs are done, paste the output of `python -m mtmg.compare` here, with:

- validation and test perplexity first (the most reliable metric);
- parameter counts, since the models are not parameter-matched;
- the learned interaction matrix and gate share from `evaluate.py` for the hybrid;
- an honest statement if the hybrid does not beat the baselines.

## Next steps

1. Train the baselines and the hybrid on a GPU; compare validation perplexity.
2. Run the two ablations (`hybrid_no_gate`, `mamba_independent`).
3. Optional PPO fine-tuning; report reward and perplexity before and after.
4. A demo page with audio samples.
