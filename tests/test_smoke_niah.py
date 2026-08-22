"""
tests/test_smoke_niah.py
========================
Fast smoke test for Needle-In-A-Haystack (NIAH) randomized evaluation and gating.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.benchmark_niah_e2e_colab import (
    generate_niah_single_eval, DeltaPhaseNIAHModel, VOCAB_SIZE, KEY_LO, KEY_HI, VAL_LO, VAL_HI
)


def test_niah_random_needle_generation():
    seq_len = 128
    depth = 0.5
    device = "cpu"

    seq, gold_v = generate_niah_single_eval(seq_len=seq_len, depth=depth, device=device)

    assert seq.shape == (1, seq_len)
    assert VAL_LO <= gold_v <= VAL_HI
    # Final token is the query needle key
    assert KEY_LO <= seq[0, -1].item() <= KEY_HI


def test_niah_model_eval_step():
    torch.manual_seed(42)
    device = "cpu"
    model = DeltaPhaseNIAHModel(
        vocab_size=VOCAB_SIZE, d_model=32, n_heads=2,
        chunk_size=16, num_layers=1, beta_mode="learned"
    ).to(device)
    model.eval()

    seq, gold_v = generate_niah_single_eval(seq_len=64, depth=0.25, device=device)

    with torch.no_grad():
        logits = model(seq)
        pred = logits[0, -1, :].argmax().item()

    assert logits.shape == (1, 64, VOCAB_SIZE)
    assert 0 <= pred < VOCAB_SIZE
