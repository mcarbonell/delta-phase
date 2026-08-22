"""
tests/test_core.py
==================
Unit and functional tests for DeltaPhase core configuration and model.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from delta_phase import DeltaPhaseConfig, DeltaPhaseModel


@pytest.fixture
def default_config():
    return DeltaPhaseConfig(
        dim=128,
        emb_dim=32,
        n_layers=2,
        n_heads=4,
        vocab_size=256,
        max_seq_len=256,
        chunk_size=32,
        conv_kernel_size=4,
        num_banks=2
    )


@pytest.fixture
def model(default_config):
    torch.manual_seed(42)
    return DeltaPhaseModel(default_config)


def test_config_initialization(default_config):
    assert default_config.dim == 128
    assert default_config.n_heads == 4
    assert default_config.chunk_size == 32
    assert default_config.vocab_size == 256


def test_model_forward_shape(model, default_config):
    batch_size = 2
    seq_len = 64
    x = torch.randint(0, default_config.vocab_size, (batch_size, seq_len))
    logits = model(x)
    assert logits.shape == (batch_size, seq_len, default_config.vocab_size), (
        f"Expected shape {(batch_size, seq_len, default_config.vocab_size)}, got {logits.shape}"
    )


def test_model_backward_gradient_flow(model, default_config):
    x = torch.randint(0, default_config.vocab_size, (2, 32))
    logits = model(x)
    loss = logits.sum()
    loss.backward()

    # Check that all trainable parameters received gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_params) > 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has no gradient!"
            assert not torch.isnan(p.grad).any(), f"Parameter {name} has NaN in gradient!"
            assert not torch.isinf(p.grad).any(), f"Parameter {name} has Inf in gradient!"


def test_streaming_step_consistency(model, default_config):
    model.eval()
    x_t = torch.tensor([[42]])
    logits_t, state = model.step(x_t)
    assert logits_t.shape == (1, 1, default_config.vocab_size)
    assert state is not None
    assert len(state) == default_config.n_layers


def test_substrate_report_executes(model):
    # Verify that printing the report does not raise exceptions
    model.print_substrate_report()

