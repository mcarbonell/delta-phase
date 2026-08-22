"""
tests/test_amp_dtypes.py
========================
P2-8 Audit remediation: dtype policy tests under mixed precision.

Guarantees:
  1. All phasor cores run correctly under torch.autocast(bfloat16): the explicit
     .float() casts before cos/sin/complex keep phase math in fp32 (torch.complex
     and torch.polar do not support bfloat16, and reduced-precision phases corrupt
     the cosine kernel).
  2. Sequential-vs-chunkwise equivalence still holds under autocast within bf16
     rounding tolerances.
  3. The float64 double-precision path is unaffected.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from delta_phase.layers import (
    DeltaPhaseHolographicBlock, LaplacePhaseCore, ComplexBetaDeltaPhaseBlock
)


def _autocast_bf16():
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16)


def test_main_block_autocast_bf16_forward_backward():
    torch.manual_seed(0)
    block = DeltaPhaseHolographicBlock(d_model=64, n_heads=4, chunk_size=32)
    x = torch.randn(2, 65, 64, requires_grad=True)  # L not multiple of C
    with _autocast_bf16():
        out, state = block(x)
        loss = out.float().sum()
    loss.backward()
    assert out.shape == (2, 65, 64)
    assert torch.isfinite(out).all(), "Non-finite activations under bf16 autocast"
    assert torch.isfinite(x.grad).all(), "Non-finite gradients under bf16 autocast"


def test_main_block_equivalence_under_autocast_bf16():
    """Parallel chunkwise and sequential step must agree within bf16 rounding."""
    torch.manual_seed(0)
    block = DeltaPhaseHolographicBlock(d_model=32, n_heads=2, chunk_size=16).eval()
    x = torch.randn(1, 48, 32)
    with torch.no_grad(), _autocast_bf16():
        out_par, state_par = block(x)
        outs, st = [], None
        for t in range(x.shape[1]):
            o_t, st = block.step(x[:, t:t + 1, :], state=st)
            outs.append(o_t)
        out_seq = torch.cat(outs, dim=1)
    diff = (out_par - out_seq).abs().max().item()
    assert diff < 5e-2, f"Chunkwise vs sequential diverged under autocast: {diff:.3e}"


def test_laplace_core_autocast_bf16():
    """Regression: torch.polar/complex paths crashed on bf16 inputs before P2-8 hardening."""
    torch.manual_seed(0)
    core = LaplacePhaseCore(d_model=32, n_heads=4, d_k=8, chunk_size=8)
    x = torch.randn(2, 19, 32, requires_grad=True)
    with _autocast_bf16():
        out, state = core(x)
        loss = out.float().sum()
    loss.backward()
    assert torch.isfinite(out).all()
    assert torch.isfinite(state.real).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_complex_beta_block_autocast_bf16():
    """Regression: torch.polar crashed on bf16 projections before P2-8 hardening."""
    torch.manual_seed(0)
    block = ComplexBetaDeltaPhaseBlock(d_model=32, n_heads=4, chunk_size=16)
    x = torch.randn(2, 33, 32, requires_grad=True)
    with _autocast_bf16():
        out, state = block(x)
        loss = out.float().sum()
    loss.backward()
    assert torch.isfinite(out).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_float64_path_unaffected():
    torch.manual_seed(0)
    block = DeltaPhaseHolographicBlock(d_model=32, n_heads=2, chunk_size=16).double()
    x = torch.randn(1, 20, 32, dtype=torch.float64, requires_grad=True)
    out, state = block(x)
    out.sum().backward()
    assert out.dtype == torch.float64
    assert state.is_complex() and state.dtype == torch.complex128
    ok = torch.autograd.gradcheck(lambda x_: block(x_)[0], (x,), eps=1e-6, atol=1e-6,
                                  rtol=1e-4, nondet_tol=1e-9)
    assert ok
