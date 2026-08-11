"""
test_equivalence.py
===================
Automated Equivalence Test: Sequential Step vs Parallel Chunkwise Forward.
Tests exact output matching and state convergence across sequence lengths:
L in [1, 63, 64, 65, 128, 1024]
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from delta_phase.layers import DeltaPhaseHolographicBlock

def test_equivalence():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    d_model = 64
    n_heads = 4
    chunk_size = 64
    
    block = DeltaPhaseHolographicBlock(d_model=d_model, n_heads=n_heads, chunk_size=chunk_size).to(device)
    block.eval()
    
    lengths = [1, 63, 64, 65, 128, 1024]
    
    print("=" * 80)
    print("RUNNING SEQUENTIAL STEP VS PARALLEL CHUNKWISE FORWARD EQUIVALENCE TEST")
    print("=" * 80)
    
    all_passed = True
    for L in lengths:
        x = torch.randn(2, L, d_model, device=device) # Batch=2
        
        # 1. Parallel Chunkwise Forward
        with torch.no_grad():
            out_chunk, state_chunk = block(x)
            
        # 2. Sequential Step-by-Step Scan (with state buffering)
        out_seq_list = []
        seq_state = None
        with torch.no_grad():
            for t in range(L):
                x_t = x[:, t:t+1, :]
                out_t, seq_state = block.step(x_t, state=seq_state)
                out_seq_list.append(out_t)
            out_seq = torch.cat(out_seq_list, dim=1)
            
        conv_state, state_seq = seq_state
        
        # Measure Max Absolute Difference
        out_diff = (out_chunk - out_seq).abs().max().item()
        state_diff = (state_chunk - state_seq).abs().max().item()
        
        status = "[OK]" if out_diff < 1e-4 and state_diff < 1e-4 else "[FAIL]"
        if status == "[FAIL]":
            all_passed = False
            
        print(f"Length L={L:<5} | Output Max Diff: {out_diff:.6e} | State Max Diff: {state_diff:.6e} | {status}")
        
    print("=" * 80)
    if all_passed:
        print("=== ALL EQUIVALENCE TESTS PASSED SUCCESSFULLY! ===")
    else:
        raise ValueError("Equivalence test failed!")

if __name__ == "__main__":
    test_equivalence()
