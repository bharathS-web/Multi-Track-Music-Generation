"""Build any model of the family from a ModelConfig."""
from __future__ import annotations

from .flat import FlatLM
from .hybrid import MODES, ModelConfig, MultiTrackLM


def build_model(cfg: ModelConfig):
    if cfg.mode in ("hybrid", "hybrid_no_gate", "mamba_independent"):
        return MultiTrackLM(cfg)
    if cfg.mode in ("mamba_only", "transformer_only"):
        return FlatLM(cfg)
    raise ValueError(f"unknown mode {cfg.mode!r}; choose one of {MODES}")
