"""
tests/test_triton_dispatcher.py
===============================
Tests the differentiability and fallback mechanisms of delta_phase_chunkwise_fused dispatcher.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from delta_phase.kernels.triton_chunk_delta import (
    delta_phase_chunkwise_fused, _chunkwise_delta_reference, gram_matrix_reference
)
from delta_phase.kernels import triton_available

try:
    from delta_phase.kernels.triton_chunk_delta import (
        _triton_fused_phase_gram_kernel, gram_matrix_triton
    )
    HAS_KERNEL_SYMBOL = True
except ImportError:
    HAS_KERNEL_SYMBOL = False


def test_dispatcher_gradient_flow():
    B, H, L, D = 2, 2, 32, 16
    C = 16

    theta_k = torch.randn(B, H, L, D, requires_grad=True)
    theta_q = torch.randn(B, H, L, D, requires_grad=True)
    v = torch.randn(B, H, L, D, requires_grad=True)
    raw_beta = torch.randn(B, H, L, requires_grad=True)
    beta = torch.sigmoid(raw_beta)

    out, final_state = delta_phase_chunkwise_fused(
        theta_k, theta_q, v, beta, chunk_size=C
    )

    assert out.shape == (B, H, L, D)
    assert final_state.shape == (B, H, D, D)

    loss = out.sum()
    loss.backward()

    assert theta_k.grad is not None
    assert theta_q.grad is not None
    assert v.grad is not None
    assert raw_beta.grad is not None
    assert not torch.isnan(theta_k.grad).any()
    assert not torch.isnan(raw_beta.grad).any()


def test_dispatcher_inference_matches_reference():
    B, H, L, D = 2, 2, 32, 16
    C = 16

    theta_k = torch.randn(B, H, L, D)
    theta_q = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    beta = torch.sigmoid(torch.randn(B, H, L))

    with torch.no_grad():
        out_disp, state_disp = delta_phase_chunkwise_fused(
            theta_k, theta_q, v, beta, chunk_size=C
        )
        out_ref, state_ref = _chunkwise_delta_reference(
            theta_k, theta_q, v, beta, chunk_size=C
        )

    out_diff = (out_disp - out_ref).abs().max().item()
    state_diff = (state_disp - state_ref).abs().max().item()

    assert out_diff < 1e-6
    assert state_diff < 1e-6


# ---------------------------------------------------------------------
# Tiled Gram kernel: reference correctness (CPU) + GPU parity (skipped w/o CUDA)
# ---------------------------------------------------------------------

def test_gram_reference_matches_naive_loop():
    """gram_matrix_reference must equal the literal cos(theta_i - theta_j) definition."""
    torch.manual_seed(0)
    N, C, D = 3, 8, 16
    theta = torch.randn(N, C, D)
    beta = torch.rand(N, C)

    ref = gram_matrix_reference(theta, beta)

    naive = torch.zeros(N, C, C)
    for n in range(N):
        for i in range(C):
            for j in range(C):
                if i > j:
                    val = torch.cos(theta[n, i] - theta[n, j]).sum() / D * beta[n, i]
                    naive[n, i, j] = val

    assert (ref - naive).abs().max().item() < 1e-5
    # Upper triangle, diagonal and only-lower are exactly as specified.
    assert torch.triu(ref, diagonal=0).abs().max().item() == 0.0


@pytest.mark.skipif(not (triton_available() and HAS_KERNEL_SYMBOL),
                    reason="Triton + CUDA not available")
def test_triton_gram_kernel_matches_reference():
    """GPU parity incl. the masked-lane regression (dk not multiple of BLOCK_D)
    and the C=16/dk=16 config that failed on T4 before the tl.where fix."""
    torch.manual_seed(0)
    for (C, dk) in [(16, 16), (32, 32), (32, 48), (64, 64), (64, 96)]:
        N = 4
        theta = torch.randn(N, C, dk, device="cuda")
        beta = torch.rand(N, C, device="cuda")
        beta[0] = 0.0

        out = gram_matrix_triton(theta, beta)
        ref = gram_matrix_reference(theta, beta)
        assert (out - ref).abs().max().item() < 1e-3, f"parity failed at C={C}, dk={dk}"
        assert torch.triu(out, diagonal=0).abs().max().item() == 0.0
