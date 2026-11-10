"""Cached generation, the interaction graph (paper Eq. 11) and the no-gate ablation."""
import pytest
import torch

from mtmg.constants import BOS, GEN_ORDER, NUM_TRACKS, SLOT_OF_TRACK
from mtmg.models.factory import build_model
from mtmg.models.hybrid import MODES, ModelConfig, _Cache

T, V = 10, 40


def make(mode, graph=False, seed=0, jitter=True):
    torch.manual_seed(seed)
    m = build_model(ModelConfig(vocab_size=V, d_model=32, n_layers=2, n_heads=2, d_state=4, expand=2,
                                ffn_mult=2, dropout=0.0, max_len=T, mode=mode, grad_checkpoint=False,
                                interaction_graph=graph)).eval()
    if jitter:   # gate / graph start at exact zeros; move them so the tests exercise real values
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p) * 0.05)
    return m


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("mode", ["hybrid", "hybrid_no_gate", "mamba_independent"])
def test_cached_columns_match_full_forward(mode, graph):
    m = make(mode, graph)
    x = torch.randint(4, V, (2, NUM_TRACKS, T))
    with torch.no_grad():
        full = m.teacher_forced_logits(x)
        cache = _Cache(m, 2, T, "cpu", torch.float32)
        bos = torch.full((2,), BOS)
        order = GEN_ORDER if m.within_step else range(NUM_TRACKS)
        for u in range(T):
            for k in order:
                lg = m._column(cache, k, u, bos if u == 0 else x[:, k, u - 1])
                assert torch.allclose(lg, full[:, k, u], atol=1e-5), f"{mode} graph={graph} ({k},{u})"
                if m.within_step:
                    m._store_embedding_keys(cache, k, u + 1, x[:, k, u])


@pytest.mark.parametrize("mode", ["hybrid", "hybrid_no_gate", "mamba_independent"])
def test_cached_generation_equals_recompute(mode):
    m = make(mode, graph=True)
    prefix = torch.randint(4, V, (NUM_TRACKS, 2))
    fast = m.generate(T - 2, prefix, temperature=1e-4, top_k=1, top_p=1.0)
    slow = m.generate_recompute(T - 2, prefix, temperature=1e-4, top_k=1, top_p=1.0)
    assert torch.equal(fast, slow)


@pytest.mark.parametrize("mode", ["mamba_only", "transformer_only"])
def test_flat_cached_generation_equals_recompute(mode, monkeypatch):
    m = make(mode)
    prefix = torch.randint(4, V, (2, NUM_TRACKS, 2))
    fast = m.generate_batch(T - 2, prefix, temperature=1e-4, top_k=1, top_p=1.0)
    for layer in m.layers:                                   # force the no-cache path
        monkeypatch.setattr(layer, "can_step", lambda: False)
    slow = m.generate_batch(T - 2, prefix, temperature=1e-4, top_k=1, top_p=1.0)
    assert torch.equal(fast, slow)


@pytest.mark.parametrize("mode", MODES)
def test_batch_generation_shapes_and_prefix(mode):
    m = make(mode)
    prefix = torch.randint(4, V, (3, NUM_TRACKS, 4))
    g = torch.Generator().manual_seed(0)
    out = m.generate_batch(T - 4, prefix, temperature=1.0, top_k=0, top_p=1.0, generator=g)
    assert out.shape == (3, NUM_TRACKS, T) and torch.equal(out[:, :, :4], prefix)
    assert (out >= 4).logical_or(out <= 1).all()             # never UNK (2) or BOS (3)
    free = m.generate_batch(T, None, batch_size=2, generator=g)
    assert free.shape == (2, NUM_TRACKS, T)


def test_graph_no_leak_in_generation_order():
    """Same exhaustive leak test as test_model.py, for the Eq. 11 attention."""
    m = make("hybrid", graph=True)
    x = torch.randint(4, V, (2, NUM_TRACKS, T))
    base = m.teacher_forced_logits(x)
    gi = lambda k, t: t * NUM_TRACKS + SLOT_OF_TRACK[k]
    for k0 in range(NUM_TRACKS):
        for t0 in (0, 1, 5, T - 1):
            y = x.clone()
            y[:, k0, t0] = (y[:, k0, t0] - 4 + 1) % (V - 4) + 4
            out = m.teacher_forced_logits(y)
            for k in range(NUM_TRACKS):
                for t in range(T):
                    if gi(k, t) <= gi(k0, t0):
                        assert torch.allclose(base[:, k, t], out[:, k, t], atol=1e-5)


def test_interaction_matrix_is_row_stochastic_over_other_tracks():
    m = make("hybrid", graph=True)
    mats = m.interaction_matrices()
    assert len(mats) == 2
    A = torch.tensor(mats[0])
    assert torch.allclose(A.diagonal(), torch.zeros(NUM_TRACKS))
    assert torch.allclose(A.sum(-1), torch.ones(NUM_TRACKS))
    assert make("hybrid", graph=False).interaction_matrices() == []


def test_graph_weights_change_the_output():
    m = make("hybrid", graph=True)
    x = torch.randint(4, V, (1, NUM_TRACKS, T))
    with torch.no_grad():
        before = m.teacher_forced_logits(x)
        m.layers[0].xattn.graph_logits.add_(torch.randn(NUM_TRACKS, NUM_TRACKS))
        after = m.teacher_forced_logits(x)
    assert not torch.allclose(before, after)


def test_no_gate_ablation_has_no_gate_and_fixed_mix():
    m = make("hybrid_no_gate")
    assert not any("gate" in n and "fusion" not in n for n, _ in m.named_parameters())
    z = torch.randn(2, NUM_TRACKS, T, 32)
    assert torch.all(m.layers[0].gate_value(z) == 0.5)
