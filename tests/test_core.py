import sys
import os
import torch

# Path setup for standalone testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from delta_phase import DeltaPhaseConfig, DeltaPhaseModel

def test_delta_phase_standalone():
    config = DeltaPhaseConfig(
        dim=256,
        emb_dim=64,
        n_layers=3,
        n_heads=4,
        vocab_size=8192,
        max_seq_len=256,
        chunk_size=64
    )
    
    model = DeltaPhaseModel(config)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] DeltaPhase standalone model initialized with {params:,} parameters.")
    
    # Forward Pass Test (Batch=2, Seq=64)
    x = torch.randint(0, config.vocab_size, (2, 64))
    logits = model(x)
    print(f"Forward Logits shape: {logits.shape} (Expected: [2, 64, 8192])")
    assert logits.shape == (2, 64, config.vocab_size), "Logits shape mismatch!"
    
    # Backward Pass Test
    loss = logits.sum()
    loss.backward()
    print("[OK] Backward pass completed cleanly with no errors!")
    
    # Streaming Step Test (Batch=1, Seq=1)
    x_t = torch.tensor([[101]])
    logits_t, state = model.step(x_t)
    print(f"Streaming Logits shape: {logits_t.shape} (Expected: [1, 1, 8192])")
    assert logits_t.shape == (1, 1, config.vocab_size), "Streaming logits shape mismatch!"
    
    # Print Substrate Selection Report
    model.print_substrate_report()
    print("=== ALL DELTAPHASE STANDALONE TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    test_delta_phase_standalone()
