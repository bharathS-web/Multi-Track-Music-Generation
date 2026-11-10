"""Flattened single-stream baselines (MMM-style).

All tracks are interleaved into ONE sequence, ordered by time step and, inside a step, by
GEN_ORDER (bass, piano, guitar). A standard causal language model predicts it token by token,
so every token is conditioned on the whole past and on the earlier tracks of its own step.

  mamba_only        one Mamba stack over the flat stream
  transformer_only  one causal Transformer over the flat stream

These are fair baselines for the hybrid: same data, same token order, same targets, same loss.

Generation is cached for both: a recurrent state per Mamba block, or a key/value cache per
attention layer, so every new token costs one step instead of a full forward pass.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..constants import BOS, GEN_ORDER, SLOT_OF_TRACK
from .hybrid import EmbeddingFusion, ModelConfig, bar_phase_features, prepare_logits, sample_logits
from .mamba_block import build_mamba


class CausalSelfAttention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh, self.dropout = heads, d // heads, dropout
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # (B, L, d)
        B, L, d = x.shape
        q, k, v = (a.view(B, L, self.h, self.dh).transpose(1, 2) for a in self.qkv(x).chunk(3, dim=-1))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                             dropout_p=self.dropout if self.training else 0.0)
        return self.proj(out.transpose(1, 2).reshape(B, L, d))

    def step(self, x: torch.Tensor, cache=None):
        """One new token. x: (B, d); cache: (keys, values) of all earlier tokens, each (B, H, n, dh)."""
        B, d = x.shape
        q, k, v = (a.view(B, self.h, 1, self.dh) for a in self.qkv(x).chunk(3, dim=-1))
        if cache is not None:
            k, v = torch.cat([cache[0], k], dim=2), torch.cat([cache[1], v], dim=2)
        out = F.scaled_dot_product_attention(q, k, v)                # the new token may read everything
        return self.proj(out.reshape(B, d)), (k, v)


class FlatLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.norm1 = nn.LayerNorm(d)
        self.use_mamba = cfg.mode == "mamba_only"
        if self.use_mamba:
            self.mamba = build_mamba(d, cfg.d_state, cfg.d_conv, cfg.expand, cfg.use_official_mamba)
        else:
            self.attn = CausalSelfAttention(d, cfg.n_heads, cfg.dropout)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, cfg.ffn_mult * d), nn.GELU(),
                                 nn.Linear(cfg.ffn_mult * d, d))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:           # (B, L, d)
        z = self.norm1(h)
        h = h + self.drop(self.mamba(z) if self.use_mamba else self.attn(z))
        return h + self.drop(self.ffn(self.norm2(h)))

    def can_step(self) -> bool:
        return hasattr(self.mamba, "step") if self.use_mamba else True

    def step(self, h: torch.Tensor, state):                        # h: (B, d); eval only (no dropout)
        z = self.norm1(h)
        y, state = self.mamba.step(z, state) if self.use_mamba else self.attn.step(z, state)
        h = h + y
        return h + self.ffn(self.norm2(h)), state


class FlatLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.mode in ("mamba_only", "transformer_only"), cfg.mode
        self.cfg = cfg
        d, self.K = cfg.d_model, cfg.num_tracks
        self.tok = nn.Embedding(cfg.vocab_size, d)
        self.track = nn.Embedding(cfg.num_tracks, d)
        self.pos = nn.Embedding(cfg.max_len, d)
        self.cyc = nn.Linear(2, d)
        self.fusion = EmbeddingFusion(d)
        self.layers = nn.ModuleList([FlatLayer(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok.weight
        self.register_buffer("order", torch.tensor(GEN_ORDER), persistent=False)
        self.register_buffer("slot_of", torch.tensor(SLOT_OF_TRACK), persistent=False)
        self._init_weights()

    def _init_weights(self):
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

    def _embed(self, inp: torch.Tensor, start: int = 0) -> torch.Tensor:
        """inp (B, L): the token fed at flat position p predicts the token AT position p; the
        embedding therefore carries the time step / track of the position being predicted."""
        L = inp.size(1)
        p = torch.arange(start, start + L, device=inp.device)
        t, trk = p // self.K, self.order[p % self.K]
        pos = self.pos(t) + self.cyc(bar_phase_features(t))
        e = self.tok(inp)
        return self.fusion(e, pos[None].expand_as(e), self.track(trk)[None].expand_as(e))

    def forward_flat(self, inp: torch.Tensor) -> torch.Tensor:    # (B, L) -> (B, L, V)
        h = self._embed(inp)
        for layer in self.layers:
            if self.cfg.grad_checkpoint and self.training and h.requires_grad:
                h = checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        return self.head(self.norm(h))

    def teacher_forced_logits(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, K, T) -> logits (B, K, T, V), aligned with x (track order, not slot order)."""
        B, K, T = x.shape
        assert K == self.K and T <= self.cfg.max_len
        y = x[:, self.order, :].permute(0, 2, 1).reshape(B, T * K)          # time-major, slot order
        inp = torch.cat([torch.full_like(y[:, :1], BOS), y[:, :-1]], dim=1)
        logits = self.forward_flat(inp).view(B, T, K, -1).permute(0, 2, 1, 3)   # (B, K_slot, T, V)
        return logits[:, self.slot_of]

    def loss(self, x: torch.Tensor):
        logits = self.teacher_forced_logits(x)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), x.reshape(-1))
        return ce, logits

    def has_fast_generation(self) -> bool:
        return all(l.can_step() for l in self.layers)

    @torch.no_grad()
    def generate_batch(self, steps: int, prefix: Optional[torch.Tensor] = None, batch_size: int = 1,
                       temperature: float = 1.0, top_k: int = 40, top_p: float = 0.95,
                       generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """prefix: (B, K, P) or None (then `batch_size` samples start from BOS).
        Returns (B, K, P + steps) on the CPU."""
        was_training = self.training
        self.eval()
        dev = next(self.parameters()).device
        K = self.K
        if prefix is None:
            prefix = torch.zeros(batch_size, K, 0, dtype=torch.long)
        pre = prefix.to(dev).long()
        B, _, P = pre.shape
        assert P + steps <= self.cfg.max_len, f"total length {P + steps} > max_len {self.cfg.max_len}"
        known = pre[:, self.order].permute(0, 2, 1).reshape(B, P * K)        # slot order, time-major
        toks = torch.cat([torch.full((B, 1), BOS, dtype=torch.long, device=dev), known], dim=1)
        n_new = steps * K
        new = torch.zeros(B, n_new, dtype=torch.long, device=dev)

        def pick(logits):
            return sample_logits(prepare_logits(logits, temperature), top_k, top_p, generator)

        if self.has_fast_generation():
            states = [None] * len(self.layers)

            def feed(tok: torch.Tensor, pos: int) -> torch.Tensor:          # tok (B,) -> logits (B, V)
                h = self._embed(tok[:, None], start=pos)[:, 0]
                for i, layer in enumerate(self.layers):
                    h, states[i] = layer.step(h, states[i])
                return self.head(self.norm(h))

            logits = None
            for i in range(toks.size(1)):
                logits = feed(toks[:, i], i)
            for m in range(n_new):
                new[:, m] = pick(logits)
                if m < n_new - 1:
                    logits = feed(new[:, m], toks.size(1) + m)
        else:                                                                # e.g. official mamba_ssm
            seq = toks
            for m in range(n_new):
                new[:, m] = pick(self.forward_flat(seq)[:, -1])
                seq = torch.cat([seq, new[:, m:m + 1]], dim=1)

        out_slot = new.view(B, steps, K).permute(0, 2, 1)                    # (B, K_slot, steps)
        out = out_slot[:, self.slot_of]                                      # track order
        self.train(was_training)
        return torch.cat([pre, out], dim=2).cpu()

    @torch.no_grad()
    def generate(self, steps: int, prefix: Optional[torch.Tensor] = None, temperature: float = 1.0,
                 top_k: int = 40, top_p: float = 0.95, generator: Optional[torch.Generator] = None):
        """Single sample: prefix (K, P) or None -> (K, P + steps)."""
        pre = None if prefix is None else prefix.unsqueeze(0)
        return self.generate_batch(steps, pre, 1, temperature, top_k, top_p, generator)[0]
