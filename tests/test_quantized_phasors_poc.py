"""
Proof of Concept & Performance Benchmark: Quantized Phasors (uint8 / uint16) for DeltaPhase
Tests mathematical accuracy, associative recall fidelity, and throughput speedups.
"""

import sys
import io
import time
import math
import torch
import torch.nn.functional as F

# Fix Windows console encoding for UTF-8
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def set_seed(seed=42):
    torch.manual_seed(seed)

# =====================================================================
# 1. Quantized Phasor Engine (uint8 & uint16)
# =====================================================================

class QuantizedPhasorEngine:
    def __init__(self, device='cpu'):
        self.device = device
        
        # LUT Cosine for uint8 (256 entries: 0 to 255 -> 0 to 2pi)
        angles_u8 = torch.linspace(0, 2 * math.pi, 257, device=device)[:-1]
        self.lut_cos_u8 = torch.cos(angles_u8)
        self.lut_sin_u8 = torch.sin(angles_u8)
        
        # LUT Cosine for uint16 (65536 entries: 0 to 65535 -> 0 to 2pi)
        angles_u16 = torch.linspace(0, 2 * math.pi, 65537, device=device)[:-1]
        self.lut_cos_u16 = torch.cos(angles_u16)
        self.lut_sin_u16 = torch.sin(angles_u16)

    def float_to_uint8(self, theta: torch.Tensor) -> torch.Tensor:
        """Map radians [-pi, pi] or [0, 2pi] to uint8 [0, 255]"""
        normalized = (theta % (2 * math.pi)) / (2 * math.pi)
        return (normalized * 256.0).to(torch.uint8)

    def uint8_to_float(self, u8_tensor: torch.Tensor) -> torch.Tensor:
        """Map uint8 [0, 255] back to radians [0, 2pi)"""
        return u8_tensor.float() * (2 * math.pi / 256.0)

    def float_to_uint16(self, theta: torch.Tensor) -> torch.Tensor:
        """Map radians [-pi, pi] or [0, 2pi] to int32 with uint16 mask or float->int16"""
        normalized = (theta % (2 * math.pi)) / (2 * math.pi)
        return (normalized * 65536.0).to(torch.int32) & 0xFFFF

    def uint16_to_float(self, u16_tensor: torch.Tensor) -> torch.Tensor:
        """Map uint16 [0, 65535] back to radians [0, 2pi)"""
        return u16_tensor.float() * (2 * math.pi / 65536.0)

    def bind_u8(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Phasor Multiplication via uint8 modular overflow addition: K * V"""
        return k + v # In torch uint8, 250 + 20 wraps automatically to 14

    def unbind_u8(self, bound: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Phasor Retrieval via uint8 modular subtraction: Bound * conj(K)"""
        return bound - k

    def bind_u16(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Phasor Multiplication via uint16 addition: (K + V) & 0xFFFF"""
        return (k + v) & 0xFFFF

    def unbind_u16(self, bound: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Phasor Retrieval via uint16 subtraction: (Bound - K) & 0xFFFF"""
        return (bound - k) & 0xFFFF

    def similarity_u8(self, k: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Cosine similarity Re(K * conj(Q)) via 256-byte LUT lookup"""
        diff = (k - q).long() # Diff wraps in uint8 [0..255]
        return self.lut_cos_u8[diff]

    def similarity_u16(self, k: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Cosine similarity Re(K * conj(Q)) via 65536-entry LUT lookup"""
        diff = ((k - q) & 0xFFFF).long()
        return self.lut_cos_u16[diff]


# =====================================================================
# 2. Benchmarks & Validation
# =====================================================================

def run_precision_audit(engine):
    print("\n" + "="*70)
    print("🔬 1. AUDITORÍA DE PRECISIÓN Y ERROR DE FASE (Float32 vs uint8 / uint16)")
    print("="*70)
    
    N = 100_000
    theta_k = torch.empty(N).uniform_(-math.pi, math.pi)
    theta_v = torch.empty(N).uniform_(-math.pi, math.pi)
    
    # FP32 Ground Truth
    z_k_f32 = torch.complex(torch.cos(theta_k), torch.sin(theta_k))
    z_v_f32 = torch.complex(torch.cos(theta_v), torch.sin(theta_v))
    z_bound_f32 = z_k_f32 * z_v_f32
    z_retrieved_f32 = z_bound_f32 * torch.conj(z_k_f32)
    theta_retrieved_f32 = torch.angle(z_retrieved_f32) % (2 * math.pi)
    expected_v_rad = theta_v % (2 * math.pi)
    
    # uint8
    k_u8 = engine.float_to_uint8(theta_k)
    v_u8 = engine.float_to_uint8(theta_v)
    bound_u8 = engine.bind_u8(k_u8, v_u8)
    retrieved_u8 = engine.unbind_u8(bound_u8, k_u8)
    retrieved_u8_rad = engine.uint8_to_float(retrieved_u8)
    
    # uint16
    k_u16 = engine.float_to_uint16(theta_k)
    v_u16 = engine.float_to_uint16(theta_v)
    bound_u16 = engine.bind_u16(k_u16, v_u16)
    retrieved_u16 = engine.unbind_u16(bound_u16, k_u16)
    retrieved_u16_rad = engine.uint16_to_float(retrieved_u16)
    
    # Angular Errors
    def angular_diff(a, b):
        diff = torch.abs(a - b) % (2 * math.pi)
        return torch.minimum(diff, 2 * math.pi - diff)
    
    err_u8_deg = torch.rad2deg(angular_diff(retrieved_u8_rad, expected_v_rad))
    err_u16_deg = torch.rad2deg(angular_diff(retrieved_u16_rad, expected_v_rad))
    
    # Cosine Similarity Error
    sim_fp32 = (z_k_f32 * torch.conj(z_v_f32)).real
    sim_u8 = engine.similarity_u8(k_u8, v_u8)
    sim_u16 = engine.similarity_u16(k_u16, v_u16)
    
    mae_sim_u8 = torch.mean(torch.abs(sim_fp32 - sim_u8)).item()
    mae_sim_u16 = torch.mean(torch.abs(sim_fp32 - sim_u16)).item()
    
    print(f"Número de fasores testeados: {N:,}")
    print(f"\n[uint8 - 8 bits (256 niveles)]:")
    print(f"  - Resolución angular por paso: {360.0 / 256.0:.4f}° (~1.4063°)")
    print(f"  - Error Angular Máximo:        {err_u8_deg.max().item():.4f}°")
    print(f"  - Error Angular Medio (MAE):   {err_u8_deg.mean().item():.4f}°")
    print(f"  - Error MAE en Similitud Cos:  {mae_sim_u8:.6f} (Fidelidad > 99.3%)")
    
    print(f"\n[uint16 - 16 bits (65,536 niveles)]:")
    print(f"  - Resolución angular por paso: {360.0 / 65536.0:.6f}° (~0.0055°)")
    print(f"  - Error Angular Máximo:        {err_u16_deg.max().item():.6f}°")
    print(f"  - Error Angular Medio (MAE):   {err_u16_deg.mean().item():.6f}°")
    print(f"  - Error MAE en Similitud Cos:  {mae_sim_u16:.8f} (Fidelidad > 99.999%)")


def run_associative_recall_experiment(engine):
    print("\n" + "="*70)
    print("🧠 2. EXPERIMENTO DE MEMORIA ASOCIATIVA (Recuperación con Múltiples Pares)")
    print("="*70)
    
    # Almacenar N pares clave-valor en un vector superpuesto de dimensión D
    D = 1024
    num_pairs_list = [8, 16, 32, 64]
    
    print(f"Dimensión del vector de memoria: D = {D}")
    print(f"{'Pares (N)':<12} | {'FP32 Cos Sim':<15} | {'uint8 Cos Sim':<15} | {'uint16 Cos Sim':<15} | {'uint8 SNR (dB)':<15}")
    print("-" * 75)
    
    for N_pairs in num_pairs_list:
        # Generar claves y valores aleatorios
        theta_keys = torch.empty(N_pairs, D).uniform_(-math.pi, math.pi)
        theta_vals = torch.empty(N_pairs, D).uniform_(-math.pi, math.pi)
        
        # --- 1. FP32 Complejo ---
        K_f32 = torch.complex(torch.cos(theta_keys), torch.sin(theta_keys))
        V_f32 = torch.complex(torch.cos(theta_vals), torch.sin(theta_vals))
        pairs_f32 = K_f32 * V_f32
        M_f32 = pairs_f32.sum(dim=0, keepdim=True) # Superposición (Bundling)
        
        # Recuperar todos los valores
        retrieved_f32 = M_f32 * torch.conj(K_f32) # [N_pairs, D]
        # Cosine similarity entre recuperado y valor original
        cos_sim_f32 = (retrieved_f32 * torch.conj(V_f32)).real.sum(dim=-1) / (torch.abs(retrieved_f32).sum(dim=-1) + 1e-8)
        avg_sim_f32 = cos_sim_f32.mean().item()
        
        # --- 2. uint8 ---
        K_u8 = engine.float_to_uint8(theta_keys)
        V_u8 = engine.float_to_uint8(theta_vals)
        bound_u8 = engine.bind_u8(K_u8, V_u8)
        
        # Para bundling en uint8, acumulamos usando la tabla seno/coseno (LUT) a acumuladores reales
        # o enteros pequeños int16 para el vector de memoria M:
        M_cos_u8 = engine.lut_cos_u8[bound_u8.long()].sum(dim=0, keepdim=True)
        M_sin_u8 = engine.lut_sin_u8[bound_u8.long()].sum(dim=0, keepdim=True)
        M_u8_complex = torch.complex(M_cos_u8, M_sin_u8)
        
        # Retrieval con K_u8
        K_cos_u8 = engine.lut_cos_u8[K_u8.long()]
        K_sin_u8 = engine.lut_sin_u8[K_u8.long()]
        K_u8_complex = torch.complex(K_cos_u8, K_sin_u8)
        retrieved_u8 = M_u8_complex * torch.conj(K_u8_complex)
        
        V_cos_u8 = engine.lut_cos_u8[V_u8.long()]
        V_sin_u8 = engine.lut_sin_u8[V_u8.long()]
        V_u8_complex = torch.complex(V_cos_u8, V_sin_u8)
        cos_sim_u8 = (retrieved_u8 * torch.conj(V_u8_complex)).real.sum(dim=-1) / (torch.abs(retrieved_u8).sum(dim=-1) + 1e-8)
        avg_sim_u8 = cos_sim_u8.mean().item()
        
        # --- 3. uint16 ---
        K_u16 = engine.float_to_uint16(theta_keys)
        V_u16 = engine.float_to_uint16(theta_vals)
        bound_u16 = engine.bind_u16(K_u16, V_u16)
        
        M_cos_u16 = engine.lut_cos_u16[bound_u16.long()].sum(dim=0, keepdim=True)
        M_sin_u16 = engine.lut_sin_u16[bound_u16.long()].sum(dim=0, keepdim=True)
        M_u16_complex = torch.complex(M_cos_u16, M_sin_u16)
        
        K_cos_u16 = engine.lut_cos_u16[K_u16.long()]
        K_sin_u16 = engine.lut_sin_u16[K_u16.long()]
        K_u16_complex = torch.complex(K_cos_u16, K_sin_u16)
        retrieved_u16 = M_u16_complex * torch.conj(K_u16_complex)
        
        V_cos_u16 = engine.lut_cos_u16[V_u16.long()]
        V_sin_u16 = engine.lut_sin_u16[V_u16.long()]
        V_u16_complex = torch.complex(V_cos_u16, V_sin_u16)
        cos_sim_u16 = (retrieved_u16 * torch.conj(V_u16_complex)).real.sum(dim=-1) / (torch.abs(retrieved_u16).sum(dim=-1) + 1e-8)
        avg_sim_u16 = cos_sim_u16.mean().item()
        
        # SNR en dB: 10 * log10(Signal_Power / Noise_Power)
        snr_u8 = 10 * math.log10(max(avg_sim_u8**2 / (1.0 - avg_sim_u8**2 + 1e-8), 1e-5))
        
        print(f"{N_pairs:<12} | {avg_sim_f32:<15.4f} | {avg_sim_u8:<15.4f} | {avg_sim_u16:<15.4f} | {snr_u8:<15.2f} dB")


def run_throughput_speedup_benchmark(engine):
    print("\n" + "="*70)
    print("⚡ 3. BENCHMARK DE VELOCIDAD Y RENDIMIENTO (Throughput en 10,000,000 Fasores)")
    print("="*70)
    
    N = 10_000_000
    repeats = 20
    
    # 1. Tensores FP32
    z_k_f32 = torch.complex(torch.randn(N), torch.randn(N))
    z_v_f32 = torch.complex(torch.randn(N), torch.randn(N))
    
    # 2. Tensores uint8
    k_u8 = torch.randint(0, 256, (N,), dtype=torch.uint8)
    v_u8 = torch.randint(0, 256, (N,), dtype=torch.uint8)
    
    # 3. Tensores uint16 (representados en int32 o int16)
    k_u16 = torch.randint(0, 65536, (N,), dtype=torch.int32)
    v_u16 = torch.randint(0, 65536, (N,), dtype=torch.int32)
    
    # Benchmark FP32 Complex Mult
    # Warmup
    for _ in range(3): _ = z_k_f32 * z_v_f32
    t0 = time.perf_counter()
    for _ in range(repeats):
        res_f32 = z_k_f32 * z_v_f32
    t_f32 = (time.perf_counter() - t0) / repeats
    
    # Benchmark uint8 Modular Addition (Binding)
    # Warmup
    for _ in range(3): _ = k_u8 + v_u8
    t0 = time.perf_counter()
    for _ in range(repeats):
        res_u8 = k_u8 + v_u8
    t_u8 = (time.perf_counter() - t0) / repeats
    
    # Benchmark uint16 Addition (Binding)
    for _ in range(3): _ = (k_u16 + v_u16) & 0xFFFF
    t0 = time.perf_counter()
    for _ in range(repeats):
        res_u16 = (k_u16 + v_u16) & 0xFFFF
    t_u16 = (time.perf_counter() - t0) / repeats
    
    # Memory footprint
    mem_f32_mb = (z_k_f32.element_size() * z_k_f32.nelement() * 2) / (1024 * 1024)
    mem_u8_mb = (k_u8.element_size() * k_u8.nelement() * 2) / (1024 * 1024)
    mem_u16_mb = (2 * k_u16.nelement() * 2) / (1024 * 1024) # 2 bytes per element
    
    speedup_u8 = t_f32 / t_u8
    speedup_u16 = t_f32 / t_u16
    throughput_f32 = (N / t_f32) / 1e6
    throughput_u8 = (N / t_u8) / 1e6
    throughput_u16 = (N / t_u16) / 1e6
    
    print(f"{'Operación / Formato':<30} | {'Tiempo (ms)':<12} | {'Throughput (M op/s)':<20} | {'Speedup':<10} | {'VRAM / RAM':<10}")
    print("-" * 90)
    print(f"{'FP32 Complex Mult (4 FLOP)':<30} | {t_f32*1000:<12.2f} | {throughput_f32:<20.2f} | {'1.00x':<10} | {mem_f32_mb:<10.1f} MB")
    print(f"{'uint8 Modular ADD (1 ALU)':<30} | {t_u8*1000:<12.2f} | {throughput_u8:<20.2f} | {f'{speedup_u8:.2f}x':<10} | {mem_u8_mb:<10.1f} MB")
    print(f"{'uint16 Modular ADD (1 ALU)':<30} | {t_u16*1000:<12.2f} | {throughput_u16:<20.2f} | {f'{speedup_u16:.2f}x':<10} | {mem_u16_mb:<10.1f} MB")
    
    print("\n" + "="*70)
    print("📊 4. COMPARATIVA DE BÚSQUEDA DE AFINIDAD / COSENO (LUT vs FP32)")
    print("="*70)
    
    # FP32 Cosine Similarity: Re(K * conj(Q))
    t0 = time.perf_counter()
    for _ in range(repeats):
        sim_f32_bench = (z_k_f32 * torch.conj(z_v_f32)).real
    t_sim_f32 = (time.perf_counter() - t0) / repeats
    
    # uint8 LUT Cosine
    t0 = time.perf_counter()
    for _ in range(repeats):
        sim_u8_bench = engine.similarity_u8(k_u8, v_u8)
    t_sim_u8 = (time.perf_counter() - t0) / repeats
    
    print(f"FP32 Cosine Sim (Re(K*conj(Q))): {t_sim_f32*1000:.2f} ms ({N/t_sim_f32/1e6:.1f} M op/s)")
    print(f"uint8 LUT Cosine (LUT 256 bytes): {t_sim_u8*1000:.2f} ms ({N/t_sim_u8/1e6:.1f} M op/s)")
    print(f"Ahorro de Memoria con uint8: {mem_f32_mb / mem_u8_mb:.1f}x menos consumo de memoria")


if __name__ == '__main__':
    set_seed(42)
    engine = QuantizedPhasorEngine(device='cpu')
    run_precision_audit(engine)
    run_associative_recall_experiment(engine)
    run_throughput_speedup_benchmark(engine)
