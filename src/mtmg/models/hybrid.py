"""Hybrid multi-track language model (per-track Mamba + cross-track attention + gate).

Model family (see `factory.build_model`):

  hybrid             this file. per-track Mamba + cross-track attention + sigmoid gate (paper Eq. 3-5)
  hybrid_no_gate     this file. same, but the gate is fixed at 0.5 (ablation: is the learned gate useful?)
  mamba_independent  this file. per-track Mamba only (ablation: tracks never interact)
  mamba_only         flat.py.   ONE Mamba over all tracks flattened into a single stream (baseline)
  transformer_only   flat.py.   ONE causal Transformer over the same flattened stream (baseline)

With `interaction_graph: true` the cross-track attention follows paper Eq. 11: every track attends to
each other track separately and the results are mixed with a learned instrument interaction matrix A
(Attn_i = sum_j A_ij Z_{i<-j}). Without it, a learned per-head track-pair bias is added to one joint
attention instead (a simpler stand-in).

Tensor layout here: (B, K, T, d) = batch, track, time step, channels.

Autoregressive order. At every time step the tracks are generated in the order GEN_ORDER
(bass, piano, guitar). When predicting the token of track `a` at step `u`, the model may use
  * all tokens of every track at steps < u, and
  * the step-u tokens of the tracks that come EARLIER in GEN_ORDER.
So the tracks coordinate within a step instead of being sampled independently.

How this is implemented. Position (k, u) holds the input x[k, u-1] (BOS at u = 0) and its output
predicts x[k, u]. The model input has T+1 positions: [BOS, x_0 .. x_{T-1}].
Cross-track attention (queries from track a at step u) may read
  * hidden states of OTHER tracks at steps <= u-1, plus hidden states of EARLIER-order tracks at step u;
  * the RAW EMBEDDINGS (layer-0 states) of EARLIER-order tracks at position u+1, i.e. their step-u token.
Only raw embeddings are used for the "look-ahead" keys. Hidden states at u+1 would already have
looked at later positions and could leak the very token being predicted.
tests/test_model.py checks this property exhaustively.

Generation. `generate_batch` computes one (track, step) column at a time and keeps a recurrent state
per Mamba block plus a key/value cache for the cross-track attention, so each sampled token costs one
column instead of a full forward pass. It gives the same logits as the full forward pass
(tests/test_generation.py). `generate_recompute` is the slow reference implementation; it is also used
automatically with `use_official_mamba`, whose blocks have no compatible `step` method.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..constants import BOS, GEN_ORDER, NUM_TRACKS, SLOT_OF_TRACK, STEPS_PER_BAR, UNK
from .mamba_block import build_mamba

MODES = ("hybrid", "hybrid_no_gate", "mamba_independent", "mamba_only", "transformer_only")
CROSS_TRACK_MODES = ("hybrid", "hybrid_no_gate")


@dataclass
class ModelConfig:
    vocab_size: int = 2048
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    ffn_mult: int = 4
    dropout: float = 0.1
    max_len: int = 256
    num_tracks: int = NUM_TRACKS
    mode: str = "hybrid"
    grad_checkpoint: bool = True
    use_official_mamba: bool = False
    tie_embeddings: bool = True
    # Paper Eq. 11 (learned instrument interaction matrix). Default False so that checkpoints trained
    # before this option existed still load; configs/base.yaml turns it on.
    interaction_graph: bool = False
    # Width of the cross-track attention (paper Eq. 25, d_cross). 0 = same as d_model.
    d_cross: int = 0

    def to_dict(self):
        return asdict(self)


class EmbeddingFusion(nn.Module):
    """Paper Eq. 6-8: gated fusion of token, position and instrument embeddings."""

    def __init__(self, d: int):
        super().__init__()
        self.gate = nn.Linear(3 * d, d)
        self.w_f = nn.Linear(3 * d, d)
        self.w_r = nn.Linear(3 * d, d)

    def forward(self, tok, pos, ins):
        u = torch.cat([tok, pos, ins], dim=-1)
        g = torch.sigmoid(self.gate(u))
        return g * self.w_f(u) + (1 - g) * self.w_r(u)


def bar_phase_features(t: torch.Tensor) -> torch.Tensor:
    """Cyclic position inside the bar (paper Eq. 20): (..., 2)."""
    phase = 2 * math.pi * (t % STEPS_PER_BAR).float() / STEPS_PER_BAR
    return torch.stack([phase.sin(), phase.cos()], dim=-1)


def mask_special(logits: torch.Tensor) -> torch.Tensor:
    """UNK and BOS are never generated. Used for sampling AND for policy log-probs (PPO)."""
    logits = logits.float().clone()
    logits[..., UNK] = float("-inf")
    logits[..., BOS] = float("-inf")
    return logits


def prepare_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return mask_special(logits) / max(temperature, 1e-5)


def sample_logits(logits: torch.Tensor, top_k: int, top_p: float, generator=None) -> torch.Tensor:
    """logits: (N, V) -> sampled ids (N,). top_k=0 and top_p=1 sample from the full distribution.
    A CPU generator also works for CUDA logits."""
    if top_k and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    if top_p < 1.0:
        sp, si = torch.sort(probs, descending=True, dim=-1)
        sp = sp.masked_fill(sp.cumsum(-1) - sp > top_p, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, si, sp)
        probs = probs / probs.sum(-1, keepdim=True)
    if generator is not None:
        return torch.multinomial(probs.cpu(), 1, generator=generator).to(logits.device).squeeze(-1)
    return torch.multinomial(probs, 1).squeeze(-1)


class TrackAttention(nn.Module):
    """Cross-track attention with the in-step ordering rule (see module docstring).

    Keys = [hidden states of all positions | raw embeddings of all positions]; the mask decides which
    of them each query may read.
      graph=False  one softmax over all allowed keys + a learned per-head track-pair bias.
      graph=True   paper Eq. 11: one softmax PER SOURCE TRACK j, giving Z_{i<-j}; the outputs are mixed
                   with the learned interaction matrix A (row i: how much track i listens to each j).
    """

    def __init__(self, d: int, heads: int, num_tracks: int, dropout: float, graph: bool = False,
                 d_cross: int = 0):
        super().__init__()
        dc = d_cross or d                                       # paper Eq. 25: reduced width d_cross
        assert dc % heads == 0
        self.h, self.dh, self.K, self.dropout = heads, dc // heads, num_tracks, dropout
        self.graph = graph
        self.qkv = nn.Linear(d, 3 * dc)
        self.proj = nn.Linear(dc, d)
        if graph:
            self.graph_logits = nn.Parameter(torch.zeros(num_tracks, num_tracks))
        else:
            self.pair_bias = nn.Parameter(torch.zeros(heads, num_tracks, num_tracks))
        self.register_buffer("slot", torch.tensor(SLOT_OF_TRACK), persistent=False)
        self._cache = {}

    def interaction(self) -> torch.Tensor:
        """A (K, K): row i is a distribution over the OTHER tracks (diagonal 0). Paper Eq. 11."""
        eye = torch.eye(self.K, dtype=torch.bool, device=self.graph_logits.device)
        return torch.softmax(self.graph_logits.masked_fill(eye, float("-inf")), dim=-1)

    # ---------------------------------------------------------------- masks ----
    def raw_mask(self, T: int, device):
        """allowed: bool (T*K, 2*T*K) WITHOUT the empty-row fallback; trk: track id per position."""
        key = ("raw", T, str(device))
        if key not in self._cache:
            idx = torch.arange(T * self.K, device=device)
            u, k = idx // self.K, idx % self.K
            sl = self.slot.to(device)[k]
            qu, ku, qs, ks = u[:, None], u[None, :], sl[:, None], sl[None, :]
            other = k[:, None] != k[None, :]
            hidden_ok = other & ((ku <= qu - 1) | ((ku == qu) & (ks < qs)))
            embed_ok = other & (ks < qs) & (ku == qu + 1)
            self._cache[key] = (torch.cat([hidden_ok, embed_ok], dim=1), k)
        return self._cache[key]

    def mask(self, T: int, device):
        """Joint-softmax mask: a query with no allowed key (first track at step 0) reads itself."""
        key = ("joint", T, str(device))
        if key not in self._cache:
            allowed, k = self.raw_mask(T, device)
            TK = T * self.K
            empty = ~allowed.any(1)
            allowed = allowed.clone()
            allowed[:, :TK] |= torch.eye(TK, dtype=torch.bool, device=device) & empty[:, None]
            self._cache[key] = (allowed, k)
        return self._cache[key]

    def group_masks(self, T: int, device):
        """Per source track j: mask over track j's keys and which queries have any key from j."""
        key = ("groups", T, str(device))
        if key not in self._cache:
            allowed, _ = self.raw_mask(T, device)
            TK = T * self.K
            groups = []
            for j in range(self.K):
                cols = torch.cat([torch.arange(T, device=device) * self.K + j,
                                  TK + torch.arange(T, device=device) * self.K + j])
                m = allowed[:, cols].clone()                       # (TK, 2T)
                has = m.any(1)
                m[~has, 0] = True                                  # harmless key; its output gets weight 0
                groups.append((m, has))
            self._cache[key] = groups
        return self._cache[key]

    # -------------------------------------------------------------- forward ----
    def _heads(self, a: torch.Tensor) -> torch.Tensor:          # (B, N, d_cross) -> (B, H, N, dh)
        B, N, _ = a.shape
        return a.view(B, N, self.h, self.dh).transpose(1, 2)

    def forward(self, z: torch.Tensor, zc: torch.Tensor) -> torch.Tensor:
        """z: normalised hidden states, zc: normalised raw embeddings; both (B, K, T, d)."""
        B, K, T, d = z.shape
        flat = lambda a: a.permute(0, 2, 1, 3).reshape(B, T * K, d)          # index = u*K + k
        q, k_h, v_h = (self._heads(a) for a in self.qkv(flat(z)).chunk(3, dim=-1))
        _, k_e, v_e = (self._heads(a) for a in self.qkv(flat(zc)).chunk(3, dim=-1))
        p = self.dropout if self.training else 0.0
        if self.graph:
            A = self.interaction()
            _, trk = self.raw_mask(T, z.device)
            by_track = lambda a: a.view(B, self.h, T, K, self.dh)
            out = torch.zeros_like(q)
            for j, (m, has) in enumerate(self.group_masks(T, z.device)):
                keys = torch.cat([by_track(k_h)[:, :, :, j], by_track(k_e)[:, :, :, j]], dim=2)
                vals = torch.cat([by_track(v_h)[:, :, :, j], by_track(v_e)[:, :, :, j]], dim=2)
                z_j = F.scaled_dot_product_attention(q, keys, vals, attn_mask=m, dropout_p=p)
                w = (A[trk, j] * has).to(z_j.dtype)                          # (TK,)
                out = out + w[None, None, :, None] * z_j
        else:
            keys, values = torch.cat([k_h, k_e], dim=2), torch.cat([v_h, v_e], dim=2)
            allowed, trk = self.mask(T, z.device)
            bias = self.pair_bias[:, trk[:, None], trk[None, :]]             # (H, TK, TK)
            bias = torch.cat([bias, bias], dim=-1)
            attn_mask = bias.masked_fill(~allowed, float("-inf")).to(q.dtype)
            out = F.scaled_dot_product_attention(q, keys, values, attn_mask=attn_mask, dropout_p=p)
        out = self.proj(out.transpose(1, 2).reshape(B, T * K, self.h * self.dh))
        return out.view(B, T, K, d).permute(0, 2, 1, 3)

    def step_attend(self, q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                    key_tracks: torch.Tensor, query_track: int) -> torch.Tensor:
        """One query against an explicit key list (cached generation).
        q: (B, H, 1, dh); keys/values: (B, H, N, dh); key_tracks: (N,) track id per key. -> (B, d)"""
        B = q.size(0)
        if self.graph:
            A = self.interaction()
            out = torch.zeros_like(q)
            for j in range(self.K):
                sel = (key_tracks == j).nonzero(as_tuple=True)[0]
                if j == query_track or len(sel) == 0:
                    continue
                z_j = F.scaled_dot_product_attention(q, keys[:, :, sel], values[:, :, sel])
                out = out + A[query_track, j].to(z_j.dtype) * z_j
        else:
            bias = self.pair_bias[:, query_track, key_tracks]                # (H, N)
            out = F.scaled_dot_product_attention(q, keys, values, attn_mask=bias[:, None, :].to(q.dtype))
        return self.proj(out.transpose(1, 2).reshape(B, self.h * self.dh))


class HybridLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, self.mode = cfg.d_model, cfg.mode
        self.norm1 = nn.LayerNorm(d)
        self.mamba = build_mamba(d, cfg.d_state, cfg.d_conv, cfg.expand, cfg.use_official_mamba)
        self.cross = self.mode in CROSS_TRACK_MODES
        if self.cross:
            self.xattn = TrackAttention(d, cfg.n_heads, cfg.num_tracks, cfg.dropout, cfg.interaction_graph,
                                        cfg.d_cross)
        if self.mode == "hybrid":
            self.gate = nn.Linear(d, 1)                            # paper Eq. 3 (scalar gate per token)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, cfg.ffn_mult * d), nn.GELU(),
                                 nn.Linear(cfg.ffn_mult * d, d))
        self.drop = nn.Dropout(cfg.dropout)
        self.store_gate = False
        self.last_gate: Optional[torch.Tensor] = None

    def gate_value(self, z: torch.Tensor) -> torch.Tensor:
        """alpha in Eq. 4: weight of the Mamba (within-track) branch. Fixed 0.5 in `hybrid_no_gate`."""
        if self.mode == "hybrid":
            return torch.sigmoid(self.gate(z))
        return torch.full_like(z[..., :1], 0.5)

    def forward(self, h: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:   # both (B, K, T, d)
        B, K, T, d = h.shape
        z = self.norm1(h)
        s = self.mamba(z.reshape(B * K, T, d)).view(B, K, T, d)
        if self.cross:
            a = self.xattn(z, self.norm1(h0))
            alpha = self.gate_value(z)                            # (B, K, T, 1)
            if self.store_gate:
                self.last_gate = alpha.detach().squeeze(-1)
            mix = alpha * s + (1 - alpha) * a                      # paper Eq. 4
        else:
            mix = s
        h = h + self.drop(mix)
        return h + self.drop(self.ffn(self.norm2(h)))


class _Cache:
    """Per-generation state: Mamba states per (layer, track) and the cross-track key/value cache."""

    def __init__(self, model: "MultiTrackLM", B: int, T: int, device, dtype):
        cfg = model.cfg
        L, K, H = cfg.n_layers, cfg.num_tracks, cfg.n_heads
        dh = (cfg.d_cross or cfg.d_model) // H
        self.mamba = [[None] * K for _ in range(L)]
        if model.within_step:
            self.kh = torch.zeros(L, B, H, T * K, dh, device=device, dtype=dtype)   # hidden keys at u*K+k
            self.vh = torch.zeros_like(self.kh)
            self.ke = torch.zeros(L, B, H, K, dh, device=device, dtype=dtype)       # embedding keys, this step
            self.ve = torch.zeros_like(self.ke)


class MultiTrackLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.mode in ("hybrid", "hybrid_no_gate", "mamba_independent"), cfg.mode
        self.cfg = cfg
        d = cfg.d_model
        self.within_step = cfg.mode in CROSS_TRACK_MODES
        self.tok = nn.Embedding(cfg.vocab_size, d)
        self.track = nn.Embedding(cfg.num_tracks, d)
        self.pos = nn.Embedding(cfg.max_len + 1, d)                # T+1 input positions
        self.cyc = nn.Linear(2, d)
        self.fusion = EmbeddingFusion(d)
        self.layers = nn.ModuleList([HybridLayer(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok.weight
        self._init_weights()

    def _init_weights(self):
        """GPT-style init everywhere EXCEPT inside Mamba blocks (they have their own init)."""
        for name, m in self.named_modules():
            if "mamba" in name:
                continue
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _pos(self, t: torch.Tensor) -> torch.Tensor:
        return self.pos(t) + self.cyc(bar_phase_features(t))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """inp: (B, K, T_in) = [BOS, tokens...] (T_in <= max_len + 1) -> logits (B, K, T_in, V).
        logits[:, k, u] predicts the token of track k at step u."""
        B, K, T = inp.shape
        assert K == self.cfg.num_tracks and T <= self.cfg.max_len + 1
        dev = inp.device
        pos = self._pos(torch.arange(T, device=dev))                           # (T, d)
        ins = self.track(torch.arange(K, device=dev))                          # (K, d)
        e = self.tok(inp)
        h0 = self.fusion(e, pos[None, None].expand_as(e), ins[None, :, None].expand_as(e))
        h = h0
        for layer in self.layers:
            if self.cfg.grad_checkpoint and self.training and h.requires_grad:
                h = checkpoint(layer, h, h0, use_reentrant=False)
            else:
                h = layer(h, h0)
        return self.head(self.norm(h))

    def teacher_forced_logits(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, K, T) real tokens -> logits (B, K, T, V), aligned with x."""
        bos = torch.full_like(x[..., :1], BOS)
        return self(torch.cat([bos, x], dim=-1))[:, :, : x.size(-1)]

    def loss(self, x: torch.Tensor):
        logits = self.teacher_forced_logits(x)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), x.reshape(-1))
        return ce, logits

    # ------------------------------------------------------------ generation ----
    def _embed_one(self, tok: torch.Tensor, track: int, u: int) -> torch.Tensor:
        """Layer-0 state of one (track, position) for a batch of tokens (B,) -> (B, d)."""
        dev = tok.device
        e = self.tok(tok)
        pos = self._pos(torch.tensor([u], device=dev)).expand_as(e)
        ins = self.track(torch.tensor([track], device=dev)).expand_as(e)
        return self.fusion(e, pos, ins)

    def _keys_for(self, track: int, u: int, device):
        """Cache indices a query of `track` at step u may read (same rule as TrackAttention.raw_mask).
        Returns (hidden flat indices, embedding-key tracks, track id of every key)."""
        K, slot = self.cfg.num_tracks, SLOT_OF_TRACK
        others = torch.tensor([k for k in range(K) if k != track], dtype=torch.long, device=device)
        earlier = torch.tensor([k for k in range(K) if k != track and slot[k] < slot[track]],
                               dtype=torch.long, device=device)
        past = (torch.arange(u, device=device)[:, None] * K + others[None, :]).reshape(-1)
        hidx = torch.cat([past, u * K + earlier])
        key_tracks = torch.cat([hidx % K, earlier])
        return hidx, earlier, key_tracks

    def _column(self, cache: _Cache, track: int, u: int, inp_tok: torch.Tensor) -> torch.Tensor:
        """Compute position u of `track` through all layers (input token = x[track, u-1] or BOS).
        Every key it may read is already in the cache. Returns logits (B, V) for x[track, u]."""
        K = self.cfg.num_tracks
        h = self._embed_one(inp_tok, track, u)
        if self.within_step:
            hidx, etr, key_tracks = self._keys_for(track, u, h.device)
            empty = key_tracks.numel() == 0
        for li, layer in enumerate(self.layers):
            z = layer.norm1(h)
            s, cache.mamba[li][track] = layer.mamba.step(z, cache.mamba[li][track])
            if layer.cross:
                xa = layer.xattn
                q, k, v = (a.view(-1, xa.h, 1, xa.dh) for a in xa.qkv(z).chunk(3, dim=-1))
                cache.kh[li][:, :, u * K + track] = k[:, :, 0]
                cache.vh[li][:, :, u * K + track] = v[:, :, 0]
                if empty and xa.graph:                                 # nothing to listen to yet
                    a_out = xa.proj(z.new_zeros(z.size(0), xa.h * xa.dh))
                elif empty:                                            # joint softmax reads itself
                    self_track = torch.tensor([track], device=h.device)
                    a_out = xa.step_attend(q, k, v, self_track, track)
                else:
                    keys = torch.cat([cache.kh[li][:, :, hidx], cache.ke[li][:, :, etr]], dim=2)
                    vals = torch.cat([cache.vh[li][:, :, hidx], cache.ve[li][:, :, etr]], dim=2)
                    a_out = xa.step_attend(q, keys, vals, key_tracks, track)
                alpha = layer.gate_value(z)
                mix = alpha * s + (1 - alpha) * a_out
            else:
                mix = s
            h = h + mix
            h = h + layer.ffn(layer.norm2(h))
        return self.head(self.norm(h))

    def _store_embedding_keys(self, cache: _Cache, track: int, u: int, tok: torch.Tensor) -> None:
        """Once x[track, u-1] is fixed: keys of its raw embedding at position u, read by the
        later-order tracks of the same step."""
        h0 = self._embed_one(tok, track, u)
        for li, layer in enumerate(self.layers):
            xa = layer.xattn
            _, k, v = (a.view(-1, xa.h, xa.dh) for a in xa.qkv(layer.norm1(h0)).chunk(3, dim=-1))
            cache.ke[li][:, :, track] = k
            cache.ve[li][:, :, track] = v

    def has_fast_generation(self) -> bool:
        return all(hasattr(l.mamba, "step") for l in self.layers)

    @torch.no_grad()
    def generate_batch(self, steps: int, prefix: Optional[torch.Tensor] = None, batch_size: int = 1,
                       temperature: float = 1.0, top_k: int = 40, top_p: float = 0.95,
                       generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """prefix: (B, K, P) or None (then `batch_size` samples start from BOS).
        Returns (B, K, P + steps) on the CPU. One cached column per token."""
        if not self.has_fast_generation():
            n = prefix.size(0) if prefix is not None else batch_size
            return torch.stack([self.generate_recompute(steps, None if prefix is None else prefix[b],
                                                        temperature, top_k, top_p, generator)
                                for b in range(n)])
        was_training = self.training
        self.eval()
        p0 = next(self.parameters())
        dev, K = p0.device, self.cfg.num_tracks
        if prefix is None:
            prefix = torch.zeros(batch_size, K, 0, dtype=torch.long)
        B, _, P = prefix.shape
        total = P + steps
        assert total <= self.cfg.max_len, f"total length {total} > max_len {self.cfg.max_len}"
        out = torch.zeros(B, K, total, dtype=torch.long, device=dev)
        out[:, :, :P] = prefix.to(dev)
        cache = _Cache(self, B, total, dev, p0.dtype)
        bos = torch.full((B,), BOS, dtype=torch.long, device=dev)
        order = GEN_ORDER if self.within_step else tuple(range(K))
        for u in range(total):
            for track in order:
                logits = self._column(cache, track, u, bos if u == 0 else out[:, track, u - 1])
                if u >= P:
                    out[:, track, u] = sample_logits(prepare_logits(logits, temperature), top_k, top_p,
                                                     generator)
                if self.within_step:
                    self._store_embedding_keys(cache, track, u + 1, out[:, track, u])
        self.train(was_training)
        return out.cpu()

    @torch.no_grad()
    def generate(self, steps: int, prefix: Optional[torch.Tensor] = None, temperature: float = 1.0,
                 top_k: int = 40, top_p: float = 0.95, generator: Optional[torch.Generator] = None):
        """Single sample: prefix (K, P) or None -> (K, P + steps)."""
        pre = None if prefix is None else prefix.unsqueeze(0)
        return self.generate_batch(steps, pre, 1, temperature, top_k, top_p, generator)[0]

    @torch.no_grad()
    def generate_recompute(self, steps: int, prefix: Optional[torch.Tensor] = None, temperature: float = 1.0,
                           top_k: int = 40, top_p: float = 0.95, generator: Optional[torch.Generator] = None):
        """Reference implementation: a full forward pass for every sampled token (K per step in the
        cross-track modes). Slow, but needs nothing beyond `forward`. Returns (K, P + steps)."""
        was_training = self.training
        self.eval()
        dev = next(self.parameters()).device
        K = self.cfg.num_tracks
        x = (prefix.to(dev) if prefix is not None else torch.zeros(K, 0, dtype=torch.long, device=dev)).unsqueeze(0)
        P = x.size(-1)
        assert P + steps <= self.cfg.max_len, f"total length {P + steps} > max_len {self.cfg.max_len}"
        for t in range(P, P + steps):
            base = torch.cat([torch.full((1, K, 1), BOS, dtype=torch.long, device=dev), x], dim=-1)   # (1,K,t+1)
            if self.within_step:
                cur = torch.zeros(1, K, 1, dtype=torch.long, device=dev)   # placeholders for unsampled tracks
                for track in GEN_ORDER:
                    lg = prepare_logits(self(torch.cat([base, cur], dim=-1))[0, track, t], temperature)
                    cur[0, track, 0] = sample_logits(lg.unsqueeze(0), top_k, top_p, generator)[0]
                nxt = cur
            else:
                lg = prepare_logits(self(base)[0, :, t], temperature)          # (K, V)
                nxt = sample_logits(lg, top_k, top_p, generator).view(1, K, 1)
            x = torch.cat([x, nxt], dim=-1)
        self.train(was_training)
        return x[0].cpu()

    # ------------------------------------------------------------ inspection ----
    @torch.no_grad()
    def interaction_matrices(self) -> List[List[List[float]]]:
        """Learned A (paper Eq. 11) per layer; empty list when the graph is off."""
        return [l.xattn.interaction().cpu().tolist() for l in self.layers
                if l.cross and l.xattn.graph]
