import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from delta_phase import DeltaPhaseConfig, DeltaPhaseModel

def main():
    print("[INFO] Running DeltaPhase Minimal Training Demo...")
    
    config = DeltaPhaseConfig(
        dim=256,
        emb_dim=64,
        n_layers=4,
        n_heads=4,
        vocab_size=4096,
        max_seq_len=128,
        chunk_size=64
    )
    
    model = DeltaPhaseModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    
    # Synthetic batch of tokens
    batch_size = 4
    seq_len = 64
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    y = torch.roll(x, shifts=-1, dims=-1) # Target next tokens
    
    model.train()
    for step in range(5):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, config.vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        print(f"Step {step+1}/5 - Loss: {loss.item():.4f}")
        
    print("[OK] Training demo completed cleanly!")
    model.print_substrate_report()

if __name__ == "__main__":
    main()
