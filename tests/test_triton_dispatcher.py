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
    delta_phase_chunkwise_fused, _chunkwise_delta_reference
)


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
