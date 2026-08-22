"""
tests/test_laplace_chunkwise.py
===============================
P2-9 Audit remediation: exact chunkwise-parallel scan for LaplacePhaseCore.

The data-dependent decay r_t = exp(sigma_t dt) acts on the output rows of M, so the
intra-chunk coupling coefficients reduce to the plain phasor Gram and the decay
factorizes via log-space cumsums -> an EXACT batched triangular solve per output
channel (no approximation). These tests pin forward() (chunkwise) against
forward_sequential() (literal token loop oracle).
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from delta_phase.layers import LaplacePhaseCore


def _amplified_core(chunk_size=8):
    torch.manual_seed(7)
    core = LaplacePhaseCore(d_model=32, n_heads=2, d_k=16, chunk_size=chunk_size)
    with torch.no_grad():
        core.w_sigma_k.weight.mul_(3.0)   # stress strong decay / log-space path
        core.w_sigma_q.weight.mul_(3.0)
    return core


@pytest.mark.parametrize("L", [1, 7, 8, 9, 17, 64])
@pytest.mark.parametrize("use_init_state", [False, True])
def test_chunkwise_matches_sequential(L, use_init_state):
    core = _amplified_core()
    x = torch.randn(2, L, 32)
    M0 = torch.randn(2, 2, 16, 16, dtype=torch.complex64) if use_init_state else None
    out_c, Mc = core(x, memory_state=M0.clone() if use_init_state else None)
    out_s, Ms = core.forward_sequential(x, memory_state=M0.clone() if use_init_state else None)
    assert (out_c - out_s).abs().max().item() < 5e-5
    assert (Mc - Ms).abs().max().item() < 5e-5


@pytest.mark.parametrize("C", [8, 32])
def test_chunkwise_independent_of_chunk_size(C):
    core_a = _amplified_core(chunk_size=C)
    core_b = _amplified_core(chunk_size=C)
    x = torch.randn(2, 37, 32)
    out_a, Ma = core_a(x)
    # Same weights (identical init under same seed): compare against sequential of core_b
    out_s, Ms = core_b.forward_sequential(x)
    assert (out_a - out_s).abs().max().item() < 5e-5
    assert (Ma - Ms).abs().max().item() < 5e-5


def test_gradients_flow_through_solve():
    core = _amplified_core(chunk_size=16)
    x = torch.randn(2, 33, 32, requires_grad=True)
    out, _ = core(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert core.w_sigma_k.weight.grad is not None
