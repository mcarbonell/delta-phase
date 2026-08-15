"""
Refined Induction Circuit Test with proper Causal Next-Token Shift
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from delta_phase import DeltaPhaseConfig, DeltaPhaseModel

def test_causal_induction():
    device = 'cpu'
    vocab_size = 64
    dim = 64
    n_heads = 2
    n_layers = 2
    
    config = DeltaPhaseConfig(
        dim=dim,
        emb_dim=32,
        n_layers=n_layers,
        n_heads=n_heads,
        vocab_size=vocab_size,
        max_seq_len=256,
        chunk_size=32,
        conv_kernel_size=4,
        num_banks=2
    )
    model = DeltaPhaseModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    batch_size = 32
    seq_len = 64
    
    print("Testing Causal Next-Token Induction Convergence...")
    for step in range(1, 301):
        # Input tokens: background 10..63
        x = torch.randint(10, vocab_size, (batch_size, seq_len), device=device)
        y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
        
        for b in range(batch_size):
            k = torch.randint(1, 9, (1,)).item()
            v = torch.randint(1, 9, (1,)).item()
            
            # Place pair at pos
            pos = torch.randint(2, 20, (1,)).item()
            x[b, pos] = k
            x[b, pos + 1] = v
            
            # Query key at pos 50 -> target is v at output of pos 50 (predicting pos 51)
            x[b, 50] = k
            y[b, 50] = v
            
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 50 == 0:
            with torch.no_grad():
                preds = logits[:, 50, :].argmax(dim=-1)
                target = y[:, 50]
                acc = (preds == target).float().mean().item() * 100.0
            print(f"Step {step:3d} | Loss: {loss.item():.4f} | Target Match: {acc:6.2f}%")

if __name__ == '__main__':
    test_causal_induction()
