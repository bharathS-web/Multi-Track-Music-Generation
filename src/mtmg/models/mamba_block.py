"""Selective state-space (Mamba-style) block.

`MambaBlock` is a compact pure-PyTorch implementation (sequential scan) that
works on CPU, Windows and any GPU. For speed on Linux+CUDA you can set
`use_official_mamba: true` in the config to use `mamba_ssm.Mamba` instead.

Equations follow the paper (Sec. IV-D):
    delta_t = softplus(W_delta x_t),  A_bar = exp(delta_t * A),
    B_t = delta_t * (W_B x_t),        C_t = W_C x_t
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = max(1, math.ceil(d_model / 16))

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner,
                                padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # S4D-real initialisation of A, small positive delta bias.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))   # inverse softplus
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B, L, d_model)
        B, L, _ = x.shape
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)
        u = self.conv1d(u.transpose(1, 2))[..., :L].transpose(1, 2)   # causal depthwise conv
        u = F.silu(u)
        y = self._ssm(u)
        y = y * F.silu(z)
        return self.out_proj(y)

    def step(self, x: torch.Tensor, state=None):
        """One recurrent step for fast generation. x: (B, d_model). state: (conv_buf, h) or None.
        Gives exactly the same result as `forward` run on the whole sequence."""
        B = x.shape[0]
        if state is None:
            conv_buf = x.new_zeros(B, self.d_inner, self.d_conv)
            h = x.new_zeros(B, self.d_inner, self.d_state)
        else:
            conv_buf, h = state
        u, z = self.in_proj(x).chunk(2, dim=-1)
        conv_buf = torch.cat([conv_buf[:, :, 1:], u.unsqueeze(-1)], dim=-1)
        u = F.silu((conv_buf * self.conv1d.weight.squeeze(1)).sum(-1) + self.conv1d.bias)
        A = -torch.exp(self.A_log.float())
        dt_r, Bm, Cm = torch.split(self.x_proj(u), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt_r))
        dA = torch.exp(delta.unsqueeze(-1) * A)
        h = dA * h + delta.unsqueeze(-1) * Bm.unsqueeze(1) * u.unsqueeze(-1)
        y = (h * Cm.unsqueeze(1)).sum(-1) + u * self.D
        return self.out_proj(y * F.silu(z)), (conv_buf, h)

    def _ssm(self, u: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=u.device.type, enabled=False):
            u = u.float()
            A = -torch.exp(self.A_log.float())                         # (d_inner, N)
            dt_r, Bm, Cm = torch.split(self.x_proj(u), [self.dt_rank, self.d_state, self.d_state], dim=-1)
            delta = F.softplus(self.dt_proj(dt_r))                      # (B, L, d_inner)
            dA = torch.exp(delta.unsqueeze(-1) * A)                     # (B, L, d_inner, N)
            dBu = delta.unsqueeze(-1) * Bm.unsqueeze(2) * u.unsqueeze(-1)
            h = torch.zeros(u.shape[0], self.d_inner, self.d_state, device=u.device, dtype=u.dtype)
            ys = []
            for t in range(u.shape[1]):
                h = dA[:, t] * h + dBu[:, t]
                ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
            y = torch.stack(ys, dim=1) + u * self.D
        return y


def build_mamba(d_model: int, d_state: int, d_conv: int, expand: int, use_official: bool = False) -> nn.Module:
    if use_official:
        try:
            from mamba_ssm import Mamba  # type: ignore
            return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        except Exception as e:  # pragma: no cover
            print(f"[warn] mamba_ssm not available ({e}); falling back to pure-PyTorch MambaBlock")
    return MambaBlock(d_model, d_state, d_conv, expand)
