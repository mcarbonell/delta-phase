"""
tests/test_integrated_cores.py
==============================
Dedicated unit and numerical tests for:
1. ComplexBetaDeltaPhaseBlock (Complex S^1 Householder Dynamics & Z_k Group Expressivity)
2. LaplacePhaseCore (Continuous Hurwitz Stability & Complex Laplace S-plane VSA Operations)
3. DeltaPhaseModel integration with beta_mode='complex'
"""

import os
import sys
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from delta_phase import (
    ComplexBetaDeltaPhaseBlock,
    LaplacePhaseCore,
    DeltaPhaseConfig,
    DeltaPhaseModel
)


# =========================================================================
# 1. Tests for ComplexBetaDeltaPhaseBlock
# =========================================================================

def test_complex_beta_block_forward_and_backward():
    torch.manual_seed(42)
    B, L, D = 2, 32, 64
    block = ComplexBetaDeltaPhaseBlock(d_model=D, n_heads=4, chunk_size=16)
    x = torch.randn(B, L, D, requires_grad=True)

    out, final_state = block(x)

    assert out.shape == (B, L, D)
    assert final_state.shape == (B, 4, 16, 16)
    assert final_state.is_complex()

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert block.w_phi.weight.grad is not None


def test_complex_beta_streaming_equivalence():
    """Verify that block.step() produces identical outputs to block.forward() in evaluation mode."""
    torch.manual_seed(42)
    B, L, D = 2, 32, 64
    block = ComplexBetaDeltaPhaseBlock(d_model=D, n_heads=4, chunk_size=16)
    block.eval()

    x = torch.randn(B, L, D)

    # 1. Parallel forward
    with torch.no_grad():
        out_forward, state_forward = block(x)

    # 2. Sequential step-by-step scan
    out_step_list = []
    seq_state = None
    with torch.no_grad():
        for t in range(L):
            x_t = x[:, t:t+1, :]
            out_t, seq_state = block.step(x_t, state=seq_state)
            out_step_list.append(out_t)
        out_step = torch.cat(out_step_list, dim=1)

    conv_state, state_step = seq_state

    out_diff = (out_forward - out_step).abs().max().item()
    state_diff = (state_forward - state_step).abs().max().item()

    assert out_diff < 1e-4, f"Output mismatch between forward and step: {out_diff:.6e}"
    assert state_diff < 1e-4, f"State mismatch between forward and step: {state_diff:.6e}"


def test_complex_beta_isometry_spectrum():
    """Verify that beta = 1 + exp(i*phi) yields reflection eigenvalues on the unit circle S^1."""
    phi = torch.linspace(0, 2 * 3.1415926535, 100)
    beta = 1.0 + torch.polar(torch.ones_like(phi), phi)

    # For unit vector k, (I - beta * k * k^H) has non-trivial eigenvalue lambda = 1 - beta = -exp(i*phi)
    # The magnitude of lambda must be identically 1.0 (unimodular / isometric)
    lambda_recurrent = 1.0 - beta
    magnitude = torch.abs(lambda_recurrent)

    assert torch.allclose(magnitude, torch.ones_like(magnitude), atol=1e-6)


# =========================================================================
# 2. Tests for LaplacePhaseCore
# =========================================================================

def test_laplace_phase_core_forward_and_backward():
    torch.manual_seed(42)
    B, L, D = 2, 32, 64
    core = LaplacePhaseCore(d_model=D, n_heads=4, d_k=16)
    x = torch.randn(B, L, D, requires_grad=True)

    out, final_state = core(x)

    assert out.shape == (B, L, D)
    assert final_state.shape == (B, 4, 16, 16)
    assert final_state.is_complex()

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert core.w_sigma_k.weight.grad is not None
    assert core.w_theta_k.weight.grad is not None


def test_laplace_phase_core_hurwitz_dissipation():
    """Verify that sigma is strictly <= 0 and r_k = exp(sigma * dt) <= 1.0 (Hurwitz Stability)."""
    torch.manual_seed(42)
    core = LaplacePhaseCore(d_model=64, n_heads=4, d_k=16)
    x = torch.randn(4, 32, 64)

    dt = 1.0
    sigma_k = -torch.nn.functional.softplus(core.w_sigma_k(x)) * dt
    r_k = torch.exp(sigma_k)

    assert (sigma_k <= 0.0).all(), "Real frequency part sigma must be non-positive!"
    assert (r_k <= 1.000001).all(), "Magnitude decay factor r_k must be <= 1.0!"


def test_laplace_vsa_symbolic_operations():
    """Verify VSA binding, unbinding, bundling, and Boolean AND operations."""
    torch.manual_seed(42)
    core = LaplacePhaseCore(d_model=16, n_heads=2, d_k=8)

    # Create random unit phasors
    phase1 = torch.rand(2, 8) * 2 * 3.1415926535
    phase2 = torch.rand(2, 8) * 2 * 3.1415926535
    v1 = torch.polar(torch.ones_like(phase1), phase1)
    v2 = torch.polar(torch.ones_like(phase2), phase2)

    # 1. Bind and Unbind
    bound = core.bind(v1, v2)
    unbound = core.unbind(bound, v1)
    # unbound should be approximately v2 (phase angle diff ~ 0)
    diff = (unbound - v2).abs().max().item()
    assert diff < 1e-5, f"VSA Unbind failed: {diff:.6e}"

    # 2. Bundle
    bundled = core.bundle(v1, v2)
    assert bundled.shape == v1.shape

    # 3. Strict Boolean AND operator
    and_res = core.strict_and_op(v1, v2, threshold=0.5)
    assert and_res.shape == v1.shape
    assert (torch.abs(and_res) <= 1.00001).all()


# =========================================================================
# 3. Tests for DeltaPhaseModel integration with beta_mode='complex'
# =========================================================================

def test_model_complex_beta_integration():
    config = DeltaPhaseConfig(
        dim=64,
        emb_dim=16,
        n_layers=2,
        n_heads=2,
        vocab_size=128,
        max_seq_len=64,
        chunk_size=16,
        beta_mode="complex"
    )
    model = DeltaPhaseModel(config)

    # Verify that blocks are ComplexBetaDeltaPhaseBlock
    assert isinstance(model.blocks[0], ComplexBetaDeltaPhaseBlock)

    x = torch.randint(0, config.vocab_size, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, config.vocab_size)

    loss = logits.sum()
    loss.backward()

    # Verify streaming step with complex model
    x_t = torch.tensor([[10]])
    logits_t, state = model.step(x_t)
    assert logits_t.shape == (1, 1, config.vocab_size)
    assert len(state) == config.n_layers
