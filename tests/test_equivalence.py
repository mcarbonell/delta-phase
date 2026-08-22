"""
tests/test_equivalence.py
==========================
Automated Equivalence Test: Sequential Step vs Parallel Chunkwise Forward.
Tests exact output matching and state convergence across sequence lengths:
L in [1, 31, 32, 33, 64, 128, 256].
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from delta_phase.layers import DeltaPhaseHolographicBlock


@pytest.mark.parametrize("L", [1, 31, 32, 33, 64, 128, 256])
def test_sequential_step_vs_parallel_chunkwise_equivalence(L):
    torch.manual_seed(42 + L)
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

    # 1. Parallel Chunkwise Forward
    with torch.no_grad():
        out_chunk, state_chunk = block(x)

    # 2. Sequential Step-by-Step Scan
    out_seq_list = []
    seq_state = None
    with torch.no_grad():
        for t in range(L):
            x_t = x[:, t:t+1, :]
            out_t, seq_state = block.step(x_t, state=seq_state)
            out_seq_list.append(out_t)
        out_seq = torch.cat(out_seq_list, dim=1)

    conv_state, state_seq = seq_state

    # 3. Assertions with explicit tolerances
    out_diff = (out_chunk - out_seq).abs().max().item()
    state_diff = (state_chunk - state_seq).abs().max().item()

    assert out_diff < 1e-4, f"Output divergence at L={L}: {out_diff:.6e} >= 1e-4"
    assert state_diff < 1e-4, f"State divergence at L={L}: {state_diff:.6e} >= 1e-4"
