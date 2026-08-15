"""
Head Dimension Sweep (d_k = 32 vs 64 vs 128) on Algebraic NIAH Benchmark
Evaluates memory retention across context lengths (512 to 8,192 tokens)
with real-time progress logging and comparative heatmaps.
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
    print(msg, flush=True)

def run_dk_sweep_benchmark():
    start_global = time.time()
    device = 'cpu'
    
    context_lengths = [512, 1024, 2048, 4096, 8192]
    depths = [0.10, 0.25, 0.50, 0.75, 0.90]
    dk_values = [32, 64, 128]
    
    # Metadata Header
    log("="*95)
    log("📋 [METADATA] EXPERIMENTO: BARRIDO DE DIMENSIÓN DE CABEZA (d_k = 32 -> 64 -> 128)")
    log("="*95)
    log(f"  • Modelo / Espacio: Núcleo Matricial DeltaPhase (C^{{d_k x d_k}})")
    log(f"  • Valores de d_k:   {dk_values} (Capacidad relativa: 1x -> 4x -> 16x)")
    log(f"  • Longitudes:       512 -> 1,024 -> 2,048 -> 4,096 -> 8,192 tokens")
    log(f"  • Profundidades:    10%, 25%, 50%, 75%, 90%")
    log(f"  • Repeticiones:     5 trials por celda")
    log(f"  • Dispositivo:      {device.upper()} | Inicio: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("="*95 + "\n")
    
    all_dk_results = {}
    all_dk_timings = {}
    
    total_cells = len(dk_values) * len(context_lengths) * len(depths)
    current_cell = 0
    
    for d_k in dk_values:
        inv_dk = 1.0 / float(d_k)
        log(f"\n🚀 INICIANDO BARRIDO CON d_k = {d_k} (Matriz C^{d_k}x{d_k} = {2*d_k*d_k} floats)")
        log("-" * 80)
        
        results = {}
        latencies = {}
        
        for L in context_lengths:
            results[L] = {}
            latencies[L] = {}
            for d in depths:
                current_cell += 1
                trials = 5
                cos_sims = []
                lat_list = []
                
                for trial in range(trials):
                    # 1. Target Needle
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
                    
                    # 3. State Accumulation
                    t0 = time.perf_counter()
                    M = torch.zeros(d_k, d_k, dtype=torch.complex64, device=device)
                    
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
                
                badge = "🟩" if mean_sim >= 0.85 else ("🟨" if mean_sim >= 0.50 else "🟥")
                log(f"[{mins:02d}:{secs:02d}] [{current_cell:2d}/{total_cells}] (d_k={d_k:3d}) L={L:>5,} | Depth={int(d*100):>2}% -> Sim: {badge} {mean_sim:+.4f} | Lat: {mean_lat*1000:>5.1f} ms")
                
        all_dk_results[d_k] = results
        all_dk_timings[d_k] = latencies

    # Final Comparative Heatmaps
    log("\n" + "="*95)
    log("📊 MATRICES DE CALOR COMPARATIVAS (d_k = 32 vs 64 vs 128)")
    log("="*95)
    
    depth_headers = " | ".join([f"{int(d*100):>4}% Depth" for d in depths])
    
    for d_k in dk_values:
        log(f"\n--- MATRIZ DE CALOR PARA d_k = {d_k} ({2*d_k*d_k:,} parámetros reales) ---")
        log(f"{'Context Length':<16} | {depth_headers} | {'Mean Latency':<12}")
        log("-" * 95)
        for L in context_lengths:
            row = []
            lats = []
            for d in depths:
                sim = all_dk_results[d_k][L][d]
                lat = all_dk_timings[d_k][L][d]
                lats.append(lat)
                if sim >= 0.85:
                    badge = f"🟩 {sim:.2f}"
                elif sim >= 0.50:
                    badge = f"🟨 {sim:.2f}"
                else:
                    badge = f"🟥 {sim:.2f}"
                row.append(f"{badge:>12}")
            mean_l = sum(lats) / len(lats)
            log(f"{L:<16,} | {' | '.join(row)} | {mean_l*1000:>9.2f} ms")
            
    log("\n" + "="*95)
    log("📈 RESUMEN DEL IMPACTO DE ESCALAR d_k EN LA RETENCIÓN A LARGO PLAZO (10% Depth)")
    log("="*95)
    log(f"{'Context Length':<16} | {'d_k = 32':<15} | {'d_k = 64':<15} | {'d_k = 128':<15} | {'Ganancia (128 vs 32)':<20}")
    log("-" * 95)
    for L in context_lengths:
        s32 = all_dk_results[32][L][0.10]
        s64 = all_dk_results[64][L][0.10]
        s128 = all_dk_results[128][L][0.10]
        gain = s128 - s32
        log(f"{L:<16,} | {s32:+.4f}{' ':>8} | {s64:+.4f}{' ':>8} | {s128:+.4f}{' ':>8} | {gain:+.4f}")
    log("="*95)

if __name__ == '__main__':
    run_dk_sweep_benchmark()
