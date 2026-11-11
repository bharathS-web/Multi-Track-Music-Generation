# Hybrid Multi-Track Music Generation (Mamba + Cross-Track Attention)

Research implementation of *"Hybrid Multi-Track Music Generation using Transformers and State Space Models"*
(Veena S T, Bharath Kumar S, Nandhagopal R), by Bharath Kumar S.

The model writes three instrument tracks (piano, guitar, bass) together. Each track keeps its own memory with a
per-track **Mamba** (selective state space) block, the tracks listen to each other through **cross-track
attention** weighted by a learned **instrument interaction graph**, and a learned **gate** decides, token by token,
how much of each to use. Training is next-token prediction, optionally followed by **PPO** fine-tuning with
music-theory rewards.

## Paper → code

| Paper | Code |
|---|---|
| Embedding fusion, Eq. 6-8; cyclic bar position, Eq. 20 | `EmbeddingFusion`, `bar_phase_features` in `src/mtmg/models/hybrid.py` |
| Cross-track attention Z<sub>i←j</sub>, Eq. 1 | `TrackAttention` (`hybrid.py`): one softmax per source track |
| Track-pair weights ρ<sub>ij</sub> / interaction graph A, Eq. 2 and 11 | `TrackAttention.interaction()`: learned A, each row a distribution over the other tracks |
| Gate, gated fusion, residual and feed-forward, Eq. 3-5 | `HybridLayer.gate_value` and `HybridLayer.forward` |
| Selective Mamba, Eq. 12-14 | `MambaBlock` in `src/mtmg/models/mamba_block.py` (or `mamba_ssm` with `use_official_mamba`) |
| Autoregressive factorisation and cross-entropy, Eq. 9-10 | `MultiTrackLM.loss`, `generate_batch` |
| Reduced cross-track width d<sub>cross</sub>, Eq. 25 | `d_cross` in the config (64 in `configs/base.yaml`) |
| Composite objective, Eq. 21; rewards, Eq. 22; clipped PPO, Eq. 23; entropy term, Eq. 16 | `src/mtmg/ppo.py`, `src/mtmg/rewards.py` |
| Algorithm 1 (pretrain with cross-entropy, then optional PPO) | `mtmg.train`, then `mtmg.ppo` |
| Dataset filtering, Sec. XII-A | `src/mtmg/data/filter_midi.py`, `build_dataset.py` |
| Metrics of Tables I-III: CM (Eq. 26), HA (Eq. 27), NCTN, QR, UPC, PP, sequence similarity, rests ratio | `src/mtmg/metrics.py`, reported by `mtmg.evaluate`, tabulated by `mtmg.compare` |
| Transformer-only and Mamba-only baselines | `transformer_only`, `mamba_only` in `src/mtmg/models/flat.py` |
| Complexity analysis, Sec. XI | `python -m mtmg.cost` |

Two ablations go beyond the paper: `hybrid_no_gate` (gate fixed at 0.5) and `mamba_independent` (no cross-track
attention at all).

## Implementation details

These are the choices the paper leaves open.

- **Representation:** 16th-note grid, 16 bars = 256 steps, 3 tracks aligned in time. One token per track per step:
  `REST`, `HOLD`, `UNK`, or a **pitch-set** (a note or chord of up to 4 pitches). A note lasts until the next
  non-`HOLD` token, so durations are encoded by `HOLD` and bar/beat position by the cyclic feature of Eq. 20.
  Vocabulary = the 2044 most frequent pitch-sets in the training split.
- **Order of the tracks within a step:** the tracks of one time step are generated in the order bass → piano →
  guitar (`GEN_ORDER` in `constants.py`). Each track sees every earlier step plus the tracks before it in the same
  step, so the instruments coordinate instead of being sampled independently.
- **Causality:** cross-track attention only reads what has already been generated. `tests/test_model.py` and
  `tests/test_generation.py` check exhaustively that no token can see itself or anything after it.
- **Interaction graph:** Eq. 2 (ρ) and Eq. 11 (A) describe the same learned per-pair weighting, implemented once:
  each track attends to every other track separately, and the results are mixed with A, whose row i sums to 1 over
  the other tracks. `evaluate.py` prints the learned A per layer, which is how the piano–bass coupling described in
  the paper can be checked.
- **Gate and residual:** one scalar gate per token (Eq. 3), computed from the layer-normalized input, mixes the two
  branches; the mixture is added to the layer input and followed by a feed-forward block (Eq. 4-5).
- **Rewards (Eq. 22):** R<sub>pitch</sub> = minus the share of notes outside a plausible range for the
  instrument; R<sub>repeat</sub> = minus the share of onsets repeating the same chord more than 4 times in a row;
  R<sub>harmony</sub> = harmony analysis (Eq. 27) of the generated continuation. The advantage is the reward
  normalised within each batch of rollouts. During PPO the cross-entropy term of Eq. 21 stays on, plus a small KL
  penalty towards the pretrained model, so the model cannot drift into music that only scores well on the heuristics.
- **Metrics:** NCTN = share of note onsets outside the bar's chord; QR = share of notes with a standard length
  (16th, 8th, dotted 8th, quarter, dotted quarter, half, dotted half, whole bars); sequence similarity = per-track
  cosine similarity of per-bar pitch-class histograms between a generated continuation and the real continuation
  of the same prompt. Chord matching (Eq. 26, adapted from PopMAG's chord accuracy) = share of bars in which the
  chord implied by the piano or guitar alone equals the bar's chord from all tracks; it needs no reference piece, so
  real music is scored the same way (the test set gives 0.742, and harmony analysis 0.684, matching Table I).
- **Sizes** (`configs/base.yaml`): 6 layers, width 256, cross-track attention width 64, 4 heads, state size 16.
  The models are not parameter-matched (hybrid ~7.4M, Mamba-only ~7.0M, Transformer-only ~5.9M); report parameter
  counts with results.

## Cost of one forward pass

`python -m mtmg.cost --config configs/base.yaml --bars 16 64` (G = 10⁹ multiply-accumulates):

| Length | Transformer-only | Mamba-only | Hybrid, d<sub>cross</sub> = 256 | Hybrid, d<sub>cross</sub> = 64 (default) |
|---|---|---|---|---|
| 16 bars | 6.3 G | 5.2 G (0.83×) | 11.0 G (1.75×) | 6.7 G (1.07×) |
| 64 bars | 46.9 G | 21.0 G (0.45×) | 87.6 G (1.87×) | 37.6 G (0.80×) |

Mamba's cost grows linearly with length and attention's quadratically. Cross-track attention reads the whole past
of the other tracks, so at full width (`d_cross: 0`) the hybrid is the most expensive model; with the reduced width
of Eq. 25 (`d_cross: 64`, the default) it costs about the same as the Transformer at 16 bars and less for long
pieces. The count covers matrix multiplications,
convolutions and attention; the Mamba scan's element-wise work adds about 0.15 G at 16 bars (shown by the script).

## Setup (Windows, PowerShell)

```powershell
cd C:\Users\Bharath\Music\Multi-Track_Music_Generation
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -q          # 75 tests, about 2 minutes on CPU
```

For GPU training install the CUDA build of PyTorch first (see pytorch.org), or train on Colab / Kaggle.
`mamba-ssm` only installs on Linux with CUDA; the default pure-PyTorch Mamba works everywhere (slower).

## Data

Download the **Lakh MIDI Dataset** (the "clean" subset; see colinraffel.com/projects/lmd and check its terms),
extract it, then:

```powershell
python -m mtmg.data.build_dataset --midi-dir D:\datasets\clean_midi --out data\processed --workers 4
```

Try `--max-files 500` first. Check `data/processed/stats.json` (how many files each rule rejected, UNK rate) and
**listen to a round trip** before training: `python -m mtmg.inspect_data --data-dir data\processed`.

**Splits are made by artist folder** (`--split-by auto`), so different versions of the same song cannot end up in
both training and test data. Lakh contains many near-duplicates; see
[Choi et al., ISMIR 2025](https://arxiv.org/abs/2509.16662).

**Dataset built from the clean subset** (default settings):

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

# main models (GPU recommended): same data, same seed
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
python -m mtmg.compare  --results results --out results\summary.md     # Tables I-III
```

Each run saves its config, seed, environment and a CSV log in `runs/<name>/`.
Generated `.mid` files can be played in any MIDI player or rendered with FluidSynth and a soundfont.

### PPO fine-tuning (Algorithm 1, steps 4-5)

```powershell
python -m mtmg.ppo --ckpt runs\hybrid_v1\best.pt --run-name hybrid_v1_ppo --iters 200
python -m mtmg.evaluate --ckpt runs\hybrid_v1_ppo\final.pt --out results\hybrid_v1_ppo.json
```

`runs/<name>/ppo_log.csv` tracks the reward terms, the KL to the pretrained model, the cross-entropy and the
validation loss. A good run raises the rewards without making validation perplexity much worse. If perplexity
climbs, raise `--kl-coef` or lower `--lr`.

## Results

The tables in the paper were produced with an earlier version of this code. This version has not been trained
yet; its results will be added here from `python -m mtmg.compare`, together with parameter counts, the learned
interaction matrix and the gate statistics.

## Not in this repository

- The listening study (Tables IV-V): it needs human raters, not code.
- MuseGAN, MMM and MMT: use their authors' official implementations on the same filtered split.
