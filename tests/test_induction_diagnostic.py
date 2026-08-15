"""
Diagnostic test for 2-layer DeltaPhase Induction Circuit
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from delta_phase import DeltaPhaseConfig, DeltaPhaseModel

def test_induction_head_learning():
    device = 'cpu'
    vocab_size = 64
    dim = 64
    n_heads = 2
    n_layers = 2 # Minimum 2 layers for induction circuit
    
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    batch_size = 32
    seq_len = 64
    
    print("Testing 2-Layer Induction Circuit Convergence...")
    for step in range(1, 101):
        # Generate batch where token pairs [A, B] appear, then [A] appears at the end and targets [B]
        x = torch.randint(10, vocab_size, (batch_size, seq_len), device=device)
        y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
        
        for b in range(batch_size):
            k = torch.randint(1, 9, (1,)).item()
            v = torch.randint(1, 9, (1,)).item()
            pos = torch.randint(2, 20, (1,)).item()
            x[b, pos] = k
            x[b, pos + 1] = v
            
            # Query at pos 50
            x[b, 50] = k
            y[b, 50] = v
            
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0 or step == 100:
            with torch.no_grad():
                preds = logits[:, 50, :].argmax(dim=-1)
                target = y[:, 50]
                acc = (preds == target).float().mean().item() * 100.0
            print(f"Step {step:3d} | Loss: {loss.item():.4f} | Target Match: {acc:.2f}%")

if __name__ == '__main__':
    test_induction_head_learning()
