"""
High-Efficiency Dense MQAR & Zero-Shot NIAH Benchmark for DeltaPhase
"""

import os
import sys
import io
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from delta_phase import DeltaPhaseConfig, DeltaPhaseModel

def generate_dense_mqar_batch(batch_size: int, seq_len: int = 128, num_pairs: int = 8, vocab_size: int = 256, device='cpu'):
    """
    Generates dense Multi-Query Associative Recall (MQAR) batches.
    First half: stores key-value pairs [K1][V1][K2][V2]...
    Second half: queries [K3] -> [V3], [K1] -> [V1]... with dense supervision on all query positions.
    """
    tokens = torch.randint(100, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
    
    half_len = seq_len // 2
    pair_spacing = half_len // num_pairs
    query_spacing = (seq_len - half_len) // num_pairs
    
    for b in range(batch_size):
        chosen_keys = torch.randperm(45)[:num_pairs] + 1     # Keys: 1..45
        chosen_vals = torch.randperm(45)[:num_pairs] + 50    # Values: 50..94
        
        # 1. Store pairs in first half
        for p in range(num_pairs):
            pos = p * pair_spacing
            tokens[b, pos] = chosen_keys[p]
            tokens[b, pos + 1] = chosen_vals[p]
            
        # 2. Query pairs in second half with randomized order
        perm = torch.randperm(num_pairs)
        for q in range(num_pairs):
            q_pos = half_len + q * query_spacing
            k_idx = perm[q]
            tokens[b, q_pos] = chosen_keys[k_idx]
            targets[b, q_pos] = chosen_vals[k_idx]
            
    return tokens, targets

def train_and_eval_niah():
    device = 'cpu'
    vocab_size = 128
    dim = 64
    n_heads = 2 # d_k = 32
    n_layers = 1
    chunk_size = 32
    
    print("="*85)
    print("🚀 ENTRENAMIENTO DEL NÚCLEO DELTAPHASE EN MQAR DENSO (L=128)")
    print("="*85)
    
    config = DeltaPhaseConfig(
        dim=dim,
        emb_dim=32,
        n_layers=n_layers,
        n_heads=n_heads,
        vocab_size=vocab_size,
        max_seq_len=65536,
        chunk_size=chunk_size,
        conv_kernel_size=4,
        num_banks=2
    )
    model = DeltaPhaseModel(config).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    model.train()
    t0 = time.perf_counter()
    for step in range(1, 151):
        x, y = generate_dense_mqar_batch(batch_size=32, seq_len=128, num_pairs=8, vocab_size=vocab_size, device=device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if step % 25 == 0 or step == 150:
            with torch.no_grad():
                mask = (y != -100)
                preds = logits.argmax(dim=-1)
                acc = (preds[mask] == y[mask]).float().mean().item() * 100.0
            print(f"  Paso {step:3d}/150 | Loss: {loss.item():.4f} | Precisión MQAR: {acc:6.2f}%")
            
    train_time = time.perf_counter() - t0
    print(f"\n[OK] Entrenamiento completado en {train_time:.2f}s.")
    
    print("\n" + "="*85)
    print("🧭 2. EVALUACIÓN ZERO-SHOT NEEDLE IN A HAYSTACK (NIAH) A LONGITUD EXTREMA")
    print("="*85)
    
    model.eval()
    lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    depths = [0.10, 0.25, 0.50, 0.75, 0.90]
    
    results = {}
    timings = {}
    
    for L in lengths:
        results[L] = {}
        timings[L] = {}
        for d in depths:
            trials = 10
            hits = 0
            lat_list = []
            
            for t in range(trials):
                # Unique Needle Key & Value
                k_needle = 15
                v_needle = 85
                
                # Haystack of noise tokens (range 100..vocab_size-1)
                seq = torch.randint(100, vocab_size, (1, L), device=device)
                pos = max(1, min(int(L * d), L - 4))
                seq[0, pos] = k_needle
                seq[0, pos + 1] = v_needle
                seq[0, -1] = k_needle # Query at end
                
                t_start = time.perf_counter()
                with torch.no_grad():
                    logits = model(seq)
                    pred = logits[0, -1, :].argmax().item()
                lat_list.append(time.perf_counter() - t_start)
                
                if pred == v_needle:
                    hits += 1
                    
            acc = (hits / trials) * 100.0
            avg_lat = sum(lat_list) / len(lat_list)
            results[L][d] = acc
            timings[L][d] = avg_lat
            print(f"  [Contexto: {L:>6,} | Profundidad: {int(d*100):>2}%] -> Match: {acc:>5.1f}% | Latencia: {avg_lat*1000:>6.1f} ms")
            
    print("\n" + "="*95)
    print("📊 MATRIZ DE CALOR NIAH DELTAPHASE (EXTRAPOLACIÓN ZERO-SHOT DESDE L=128)")
    print("="*95)
    depth_headers = " | ".join([f"{int(d*100):>4}% Depth" for d in depths])
    print(f"{'Context Length':<16} | {depth_headers} | {'Mean Latency':<12}")
    print("-" * 95)
    
    for L in lengths:
        row = []
        lats = []
        for d in depths:
            acc = results[L][d]
            lat = timings[L][d]
            lats.append(lat)
            if acc == 100.0:
                badge = f"🟩 {acc:>5.1f}%"
            elif acc >= 70.0:
                badge = f"🟨 {acc:>5.1f}%"
            else:
                badge = f"🟥 {acc:>5.1f}%"
            row.append(f"{badge:>12}")
        mean_l = sum(lats) / len(lats)
        print(f"{L:<16,} | {' | '.join(row)} | {mean_l*1000:>9.2f} ms")
    print("="*95)

if __name__ == '__main__':
    train_and_eval_niah()
