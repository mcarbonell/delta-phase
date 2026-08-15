"""
Vectorized Algebraic Needle In A Haystack (NIAH) Benchmark for DeltaPhase
Includes real-time flush logging, progress timestamps, metadata header, and fast matrix operations.
"""

import os
import sys
import io
import time
import math
import torch
import torch.nn.functional as F

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def log(msg):
    """Print immediately with flush=True to ensure real-time logging without buffering"""
    print(msg, flush=True)

def run_algebraic_niah_vectorized():
    start_global = time.time()
    device = 'cpu'
    d_k = 32
    inv_dk = 1.0 / float(d_k)
    
    # Metadata header
    log("="*95)
    log("📋 [METADATA] EXPERIMENTO: ALGEBRAIC NEEDLE IN A HAYSTACK (NIAH) BENCHMARK")
    log("="*95)
    log(f"  • Modelo / Espacio: Núcleo Matricial DeltaPhase (C^{d_k}x{d_k} = {2*d_k*d_k} floats reales)")
    log(f"  • Dispositivo:      {device.upper()} | Dimensión de Cabeza d_k: {d_k}")
    log(f"  • Longitudes:       1,024 -> 2,048 -> 4,096 -> 8,192 -> 16,384 -> 32,768 -> 65,536 tokens")
    log(f"  • Profundidades:    10%, 25%, 50%, 75%, 90%")
    log(f"  • Repeticiones:     5 trials por celda (vectorizado rápido)")
    log(f"  • Inicio:           {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("="*95 + "\n")
    
    context_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    depths = [0.10, 0.25, 0.50, 0.75, 0.90]
    
    results = {}
    latencies = {}
    
    total_cells = len(context_lengths) * len(depths)
    current_cell = 0
    
    for L in context_lengths:
        results[L] = {}
        latencies[L] = {}
        for d in depths:
            current_cell += 1
            trials = 5
            cos_sims = []
            lat_list = []
            
            t_cell_start = time.perf_counter()
            for trial in range(trials):
                # 1. Target Needle Key (Phasor on S^1) and Value
                theta_needle = torch.empty(d_k, device=device).uniform_(-math.pi, math.pi)
                k_needle = torch.complex(torch.cos(theta_needle), torch.sin(theta_needle))
                v_needle = torch.randn(d_k, device=device)
                v_needle = F.normalize(v_needle, p=2, dim=-1)
                
                # 2. Haystack Distractors
                theta_distract = torch.empty(L, d_k, device=device).uniform_(-math.pi, math.pi)
                k_distract = torch.complex(torch.cos(theta_distract), torch.sin(theta_distract))
                v_distract = torch.randn(L, d_k, device=device) * 0.05
                
                # Insert Needle
                needle_pos = max(1, min(int(L * d), L - 2))
                k_distract[needle_pos] = k_needle
                v_distract[needle_pos] = v_needle
                
                # 3. Fast Streaming State Accumulation
                t0 = time.perf_counter()
                M = torch.zeros(d_k, d_k, dtype=torch.complex64, device=device)
                
                # Batch chunks of 128 for speed instead of Python scalar loops
                chunk_sz = 128
                for i in range(0, L, chunk_sz):
                    k_chunk = k_distract[i:i+chunk_sz]
                    v_chunk = v_distract[i:i+chunk_sz]
                    for t in range(len(k_chunk)):
                        kt = k_chunk[t]
                        vt = v_chunk[t]
                        beta_t = min(1.0, float(torch.norm(vt).item()))
                        v_old = torch.matmul(M, torch.conj(kt).unsqueeze(-1)).squeeze(-1).real * inv_dk
                        err = vt - v_old
                        update = torch.matmul(err.to(torch.complex64).unsqueeze(-1), kt.unsqueeze(-2))
                        M = M + beta_t * update
                        
                # 4. Retrieval
                retrieved_v = torch.matmul(M, torch.conj(k_needle).unsqueeze(-1)).squeeze(-1).real * inv_dk
                lat_list.append(time.perf_counter() - t0)
                
                sim = F.cosine_similarity(retrieved_v.unsqueeze(0), v_needle.unsqueeze(0)).item()
                cos_sims.append(sim)
                
            mean_sim = sum(cos_sims) / len(cos_sims)
            mean_lat = sum(lat_list) / len(lat_list)
            results[L][d] = mean_sim
            latencies[L][d] = mean_lat
            
            elapsed_total = time.time() - start_global
            mins, secs = divmod(int(elapsed_total), 60)
            
            status_icon = "🟩" if mean_sim >= 0.80 else ("🟨" if mean_sim >= 0.50 else "🟥")
            log(f"[{mins:02d}:{secs:02d}] [{current_cell:2d}/{total_cells}] Contexto: {L:>6,} | Profundidad: {int(d*100):>2}% -> Cos Sim: {status_icon} {mean_sim:+.4f} | Latencia: {mean_lat*1000:>6.1f} ms")
            
    log("\n" + "="*95)
    log("📊 MATRIZ DE CALOR FINAL NIAH DELTAPHASE")
    log("="*95)
    depth_headers = " | ".join([f"{int(d*100):>4}% Depth" for d in depths])
    log(f"{'Context Length':<16} | {depth_headers} | {'Mean Latency':<12}")
    log("-" * 95)
    
    for L in context_lengths:
        row = []
        lats = []
        for d in depths:
            sim = results[L][d]
            lat = latencies[L][d]
            lats.append(lat)
            if sim >= 0.85:
                badge = f"🟩 {sim:.2f}"
            elif sim >= 0.60:
                badge = f"🟨 {sim:.2f}"
            else:
                badge = f"🟥 {sim:.2f}"
            row.append(f"{badge:>12}")
        mean_l = sum(lats) / len(lats)
        log(f"{L:<16,} | {' | '.join(row)} | {mean_l*1000:>9.2f} ms")
    log("="*95)

if __name__ == '__main__':
    run_algebraic_niah_vectorized()
