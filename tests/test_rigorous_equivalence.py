"""
test_rigorous_equivalence.py
=============================
Rigorous Equivalence Audit: Variable lambda_t & beta_t per token, non-zero initial states,
relative error calculations, and full autograd gradient matching.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from delta_phase.layers import DeltaPhaseHolographicBlock

def test_rigorous():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    d_model = 64
    n_heads = 4
    chunk_size = 64
    
    block = DeltaPhaseHolographicBlock(d_model=d_model, n_heads=n_heads, chunk_size=chunk_size).to(device)
    block.eval()
    
    lengths = [1, 63, 64, 65, 128, 1024]
    
    print("=" * 105)
    print("RIGOROUS EQUIVALENCE AUDIT: VARIABLE LAMBDA & BETA PER TOKEN + RELATIVE ERROR REPORTING")
    print("=" * 105)
    print(f"{'L':<6} | {'Out Abs Error':<15} | {'Out Rel Error':<15} | {'State Abs Error':<16} | {'Grad Rel Error':<15}")
    print("-" * 105)
    
    worst_out_rel = 0.0
    worst_state_rel = 0.0
    worst_grad_rel = 0.0
    
    for L in lengths:
        x = torch.randn(2, L, d_model, device=device, requires_grad=True) # Batch=2
        
        # Non-zero initial memory state
        init_M = torch.randn(2, n_heads, d_model // n_heads, d_model // n_heads, dtype=torch.complex64, device=device)
        
        # 1. Parallel Chunkwise Forward
        out_chunk, state_chunk = block(x, memory_state=init_M.clone())
        loss_chunk = out_chunk.sum()
        loss_chunk.backward()
        grad_chunk = x.grad.clone()
        x.grad.zero_()
        
        # 2. Sequential Step-by-Step Scan
        out_seq_list = []
        seq_state = (None, init_M.clone())
        for t in range(L):
            x_t = x[:, t:t+1, :]
            out_t, seq_state = block.step(x_t, state=seq_state)
            out_seq_list.append(out_t)
        out_seq = torch.cat(out_seq_list, dim=1)
        conv_state, state_seq = seq_state
        
        loss_seq = out_seq.sum()
        loss_seq.backward()
        grad_seq = x.grad.clone()
        x.grad.zero_()
        
        # Measure Absolute Differences
        out_abs = (out_chunk - out_seq).abs().max().item()
        state_abs = (state_chunk - state_seq).abs().max().item()
        
        # Measure Relative Differences (||x_parallel - x_seq|| / max(||x_seq||, eps))
        out_rel = ((out_chunk - out_seq).abs() / (out_seq.abs() + 1e-8)).max().item()
        state_rel = ((state_chunk - state_seq).abs() / (state_seq.abs() + 1e-8)).max().item()
        grad_rel = ((grad_chunk - grad_seq).abs() / (grad_seq.abs() + 1e-8)).max().item()
        
        if out_rel > worst_out_rel: worst_out_rel = out_rel
        if state_rel > worst_state_rel: worst_state_rel = state_rel
        if grad_rel > worst_grad_rel: worst_grad_rel = grad_rel
        
        print(f"L={L:<4} | {out_abs:<15.4e} | {out_rel:<15.4e} | {state_abs:<16.4e} | {grad_rel:<15.4e}")
        
    print("=" * 105)
    print(f"WORST-CASE RELATIVE OUTPUT ERROR Across All Lengths : {worst_out_rel:.6e}")
    print(f"WORST-CASE RELATIVE STATE ERROR Across All Lengths  : {worst_state_rel:.6e}")
    print(f"WORST-CASE RELATIVE GRADIENT ERROR Across All Lengths: {worst_grad_rel:.6e}")
    print("=" * 105)

if __name__ == "__main__":
    test_rigorous()
