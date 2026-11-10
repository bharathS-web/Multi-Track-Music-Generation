import pytest
import torch

from mtmg.constants import BOS, GEN_ORDER, NUM_TRACKS, SLOT_OF_TRACK, UNK
from mtmg.models.factory import build_model
from mtmg.models.hybrid import MODES, ModelConfig
from mtmg.models.mamba_block import MambaBlock

T = 12
V = 40
BASS, PIANO, GUITAR = GEN_ORDER          # generation order: bass, piano, guitar


def tiny(mode, max_len=T):
    return ModelConfig(vocab_size=V, d_model=32, n_layers=2, n_heads=2, d_state=4, expand=2,
                       ffn_mult=2, dropout=0.0, max_len=max_len, mode=mode, grad_checkpoint=False)


def make(mode, seed=0):
    torch.manual_seed(seed)
    return build_model(tiny(mode)).eval()


def gen_index(track, t):
    return t * NUM_TRACKS + SLOT_OF_TRACK[track]


def perturb(x, track, t):
    y = x.clone()
    y[:, track, t] = (y[:, track, t] - 4 + 1) % (V - 4) + 4
    return y


@pytest.mark.parametrize("mode", MODES)
def test_shapes(mode):
    m = make(mode)
    x = torch.randint(4, V, (2, NUM_TRACKS, T))
    assert m.teacher_forced_logits(x).shape == (2, NUM_TRACKS, T, V)


@pytest.mark.parametrize("mode", MODES)
def test_no_leak_in_generation_order(mode):
    """Changing the token of track k0 at step t0 must not change the prediction of ANY token
    that is generated at or before it (including the token itself)."""
    m = make(mode)
    x = torch.randint(4, V, (2, NUM_TRACKS, T))
    base = m.teacher_forced_logits(x)
    for k0 in range(NUM_TRACKS):
        for t0 in (0, 1, 5, T - 1):
            out = m.teacher_forced_logits(perturb(x, k0, t0))
            g0 = gen_index(k0, t0)
            for k in range(NUM_TRACKS):
                for t in range(T):
                    if gen_index(k, t) <= g0:
                        assert torch.allclose(base[:, k, t], out[:, k, t], atol=1e-5), \
                            f"{mode}: prediction ({k},{t}) depends on token ({k0},{t0})"


@pytest.mark.parametrize("mode", ["hybrid", "mamba_only", "transformer_only"])
def test_tracks_condition_on_earlier_tracks_in_same_step(mode):
    m = make(mode)
    x = torch.randint(4, V, (2, NUM_TRACKS, T))
    base = m.teacher_forced_logits(x)
    t0 = 5
    out = m.teacher_forced_logits(perturb(x, BASS, t0))
    assert not torch.allclose(base[:, PIANO, t0], out[:, PIANO, t0], atol=1e-6)    # bass -> piano
    assert not torch.allclose(base[:, GUITAR, t0], out[:, GUITAR, t0], atol=1e-6)  # bass -> guitar
    out = m.teacher_forced_logits(perturb(x, PIANO, t0))
    assert not torch.allclose(base[:, GUITAR, t0], out[:, GUITAR, t0], atol=1e-6)  # piano -> guitar
    out = m.teacher_forced_logits(perturb(x, GUITAR, t0))
    assert torch.allclose(base[:, PIANO, t0], out[:, PIANO, t0], atol=1e-5)        # guitar !-> piano


def test_independent_ablation_has_no_cross_track_flow():
    m = make("mamba_independent")
    x = torch.randint(4, V, (2, NUM_TRACKS, T))
    base = m.teacher_forced_logits(x)
    out = m.teacher_forced_logits(perturb(x, BASS, 3))
    for k in (PIANO, GUITAR):
        assert torch.allclose(base[:, k], out[:, k], atol=1e-5)


def test_mamba_step_matches_forward():
    torch.manual_seed(0)
    blk = MambaBlock(16, d_state=4).eval()
    x = torch.randn(2, 9, 16)
    full = blk(x)
    state, outs = None, []
    for i in range(x.size(1)):
        y, state = blk.step(x[:, i], state)
        outs.append(y)
    assert torch.allclose(full, torch.stack(outs, 1), atol=1e-5)


@pytest.mark.parametrize("mode", MODES)
def test_greedy_generation_is_consistent_with_teacher_forcing(mode):
    """Greedy generation must pick exactly the argmax of the teacher-forced logits for the same
    sequence. This verifies the sampling order, the placeholders and the Mamba cache."""
    m = make(mode)
    prefix = torch.randint(4, V, (NUM_TRACKS, 3))
    out = m.generate(T - 3, prefix, temperature=1e-4, top_k=1, top_p=1.0)
    assert out.shape == (NUM_TRACKS, T) and (out[:, :3] == prefix).all()
    assert (out[:, 3:] != UNK).all() and (out[:, 3:] != BOS).all()
    lg = m.teacher_forced_logits(out.unsqueeze(0))[0]
    lg[..., UNK] = float("-inf")
    lg[..., BOS] = float("-inf")
    assert (lg.argmax(-1)[:, 3:] == out[:, 3:]).all()


@pytest.mark.parametrize("mode", MODES)
def test_overfits_one_batch(mode):
    torch.manual_seed(0)
    m = build_model(tiny(mode))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.randint(4, V, (4, NUM_TRACKS, T))
    first = m.loss(x)[0].item()
    for _ in range(120):
        loss, _ = m.loss(x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.7
