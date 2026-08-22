"""
tests/test_smoke_mqar.py
========================
Fast smoke test for Multi-Query Associative Recall (MQAR) training and evaluation.
"""

import os
import sys
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from delta_phase import DeltaPhaseConfig, DeltaPhaseModel
from tests.benchmark_rigorous_mqar import generate_zoology_mqar_batch, DeltaPhaseMQAR


def test_mqar_batch_generator_shapes_and_markers():
    batch_size = 4
    seq_len = 64
    num_pairs = 4
    vocab_size = 128

    x, y = generate_zoology_mqar_batch(
        batch_size=batch_size, seq_len=seq_len, num_pairs=num_pairs,
        vocab_size=vocab_size, device="cpu"
    )

    assert x.shape == (batch_size, seq_len)
    assert y.shape == (batch_size, seq_len)

    # Check query marker presence
    query_positions = (y != -100)
    assert query_positions.sum().item() == batch_size * num_pairs
    assert (x[query_positions] >= 2).all()


def test_mqar_quick_smoke_training_step():
    torch.manual_seed(42)
    device = "cpu"
    vocab_size = 128
    d_model = 32
    n_heads = 2
    chunk_size = 16

    model = DeltaPhaseMQAR(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
        chunk_size=chunk_size, num_layers=1
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    initial_loss = None
    final_loss = None

    for step in range(15):
        x, y = generate_zoology_mqar_batch(
            batch_size=8, seq_len=64, num_pairs=4,
            vocab_size=vocab_size, device=device
        )
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert initial_loss is not None
    assert final_loss is not None
    assert not torch.isnan(torch.tensor(final_loss))
