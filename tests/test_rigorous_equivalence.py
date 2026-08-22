"""
tests/test_rigorous_equivalence.py
==================================
Rigorous Equivalence Audit: Non-zero initial states, relative error checks,
and autograd FP64 gradcheck testing.
"""

import os
import sys
import pytest
import torch
from torch.autograd import gradcheck

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from delta_phase.layers import DeltaPhaseHolographicBlock


@pytest.mark.parametrize("L", [1, 31, 32, 33, 64, 128])
def test_rigorous_non_zero_initial_state_equivalence(L):
    torch.manual_seed(100 + L)
    device = torch.device("cpu")

    d_model = 64
    n_heads = 4
    chunk_size = 32

    block = DeltaPhaseHolographicBlock(
        d_model=d_model, n_heads=n_heads, chunk_size=chunk_size
    ).to(device)
    block.eval()

    batch_size = 2
    x = torch.randn(batch_size, L, d_model, device=device)

    # Non-zero random initial complex memory state
    init_M = torch.randn(
        batch_size, n_heads, d_model // n_heads, d_model // n_heads,
        dtype=torch.complex64, device=device
    )

    # 1. Parallel Chunkwise Forward with initial state
    with torch.no_grad():
        out_chunk, state_chunk = block(x, memory_state=init_M.clone())

    # 2. Sequential Step Scan with initial state
    out_seq_list = []
    seq_state = (None, init_M.clone())
    with torch.no_grad():
        for t in range(L):
            x_t = x[:, t:t+1, :]
            out_t, seq_state = block.step(x_t, state=seq_state)
            out_seq_list.append(out_t)
        out_seq = torch.cat(out_seq_list, dim=1)

    conv_state, state_seq = seq_state

    # 3. Assertions
    out_rel = ((out_chunk - out_seq).abs() / (out_seq.abs() + 1e-8)).max().item()
    state_rel = ((state_chunk - state_seq).abs() / (state_seq.abs() + 1e-8)).max().item()

    assert out_rel < 5e-2, f"Relative output error at L={L} exceeded 5%: {out_rel:.6e}"
    assert state_rel < 5e-2, f"Relative state error at L={L} exceeded 5%: {state_rel:.6e}"


def test_fp64_gradcheck_analytical():
    """Verifies that autograd gradients pass PyTorch gradcheck in FP64 double precision."""
    torch.manual_seed(42)
    device = torch.device("cpu")

    d_model = 16
    n_heads = 2
    chunk_size = 8
    L = 16

    block = DeltaPhaseHolographicBlock(
        d_model=d_model, n_heads=n_heads, chunk_size=chunk_size
    ).to(device).double()

    x = torch.randn(1, L, d_model, device=device, dtype=torch.float64, requires_grad=True)

    def forward_fn(inputs):
        out, _ = block(inputs)
        return out

    test_passed = gradcheck(forward_fn, (x,), eps=1e-6, atol=1e-4, rtol=1e-3)
    assert test_passed is True, "FP64 gradcheck failed!"
