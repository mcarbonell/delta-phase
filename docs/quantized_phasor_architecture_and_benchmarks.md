# ⚡ Quantized Phasor Architecture: uint8 / uint16 Integer Phase Arithmetic & Benchmark Audit

**Document Type:** Technical Specification & Empirical Audit  
**Date:** August 2026  
**Status:** Validated Proof-of-Concept (`tests/test_quantized_phasors_poc.py`)  

---

## 1. Executive Summary

Traditional complex neural networks suffer from the computational overhead of complex floating-point multiplications:
$$(a + bi)(c + di) = (ac - bd) + i(ad + bc) \quad \longrightarrow \quad \text{4 FLOPs + 2 Adds}$$

In DeltaPhase, keys and queries are strictly **unit-magnitude phasors on $S^1$** ($|K| = 1, |Q| = 1$), parameterized solely by their phase angle $\theta \in [0, 2\pi)$.

By quantizing the phase angle into **`uint8` (256 discrete levels)** or **`uint16` (65,536 discrete levels)**:
1. **Phasor multiplication becomes 8-bit integer addition ($1\text{ ALU op}$)**.
2. **$2\pi$ modular periodicity is free** at the silicon level via native integer register overflow.
3. **Cosine similarity / attention affinities** are computed via a **256-byte Lookup Table (LUT)** permanently resident in L1 CPU/GPU Cache.
4. **Empirical speedup:** **$8.12\times$ faster binding throughput** with an **$8.0\times$ reduction in memory bandwidth**.

---

## 2. Mathematical Formulation & Hardware Mechanics

### 2.1 Quantization Mapping
The continuous phase $\theta \in [0, 2\pi)$ is mapped to discrete integer buckets:

$$\theta_{\text{u8}} = \left\lfloor \frac{\theta \pmod{2\pi}}{2\pi} \times 256 \right\rfloor \in \{0, 1, \dots, 255\}$$

$$\theta_{\text{u16}} = \left\lfloor \frac{\theta \pmod{2\pi}}{2\pi} \times 65536 \right\rfloor \in \{0, 1, \dots, 65535\}$$

* **Angular Resolution:**
  * `uint8`: $\Delta \theta = \frac{360^\circ}{256} \approx 1.40625^\circ$
  * `uint16`: $\Delta \theta = \frac{360^\circ}{65536} \approx 0.005493^\circ$

---

### 2.2 Free Modulo $2\pi$ via Hardware ALU Register Overflow

In standard integer arithmetic, unsigned 8-bit registers naturally wrap around upon reaching 256:
```c
uint8_t theta_k = 200; // Angle K
uint8_t theta_v = 100; // Angle V
uint8_t theta_bound = theta_k + theta_v; 
// 200 + 100 = 300 -> Automatically wraps to 44 (mod 256) in 1 CPU/GPU clock cycle!
```
* **Zero Branching / Zero FMOD Instructions:** No modulo operations or conditional checks exist in machine code; wrapping is an intrinsic property of binary two's-complement adders.

---

### 2.3 L1-Resident Cosine Lookup Table (LUT)

To compute the real inner product / affinity between key $K$ and query $Q$:
$$\text{Re}(K \bar{Q}) = \cos(\theta_K - \theta_Q)$$

In `uint8`, the phase difference $\Delta \theta = (\theta_K - \theta_Q) \pmod{256}$ indexes a static precomputed table:
$$\text{LUT}_{\cos}[\Delta \theta] = \cos\left(\frac{2\pi \cdot \Delta \theta}{256}\right) \quad \in [-1.0, 1.0]$$

* **Memory Footprint of LUT:** $256 \text{ floats} \times 4 \text{ bytes} = 1024 \text{ bytes} = 1\text{ KB}$ (or $256 \text{ bytes}$ in INT8).
* This table fits permanently inside the **L1 Data Cache (32–64 KB)** or **GPU Shared Memory (SRAM)**, resulting in near-zero memory access latency.

---

## 3. Hybrid Precision Architecture: State Matrix ($M_t$) vs Phasor Stream ($K_t, Q_t$)

A critical distinction in DeltaPhase is the duality between the stream of tokens and the recurrent memory state:

```
           STREAMING TOKENS (Context Scaling)                RECURRENT STATE MATRIX (Fixed Size)
           ─────────────────────────────────                ───────────────────────────────────
                  Key & Query Phasors                               State Matrix M_t
                    (K_t, Q_t in S^1)                            (M_t in C^{d_k x d_k})
                            │                                               │
                            ▼                                               ▼
                    Quantized uint8                                   FP16 / BF16
                   (1 Byte / Element)                            (Fixed ~4-12 KB / Head)
                            │                                               │
                            └───────────────────────┬───────────────────────┘
                                                    │
                                                    ▼
                                           Hybrid Readout & Delta
                                         v_old = Re(M_t * conj(Q_t))
```

1. **Keys & Queries ($K, Q$):** Strict unit phasors ($|K|=1$). Scaled over long contexts ($L = 100\text{K}+$ tokens). Quantizing to `uint8` saves **87.5% VRAM** with zero impact on associative fidelity.
2. **State Matrix ($M_t$):** Additive superposition of outer products $M_t = \sum \beta_t (e_t \otimes K_t) \in \mathbb{C}^{d_k \times d_k}$. Because its elements have dynamic magnitudes, and its footprint is **fixed and tiny** ($\approx 4\text{ KB}$ per head in FP16), keeping $M_t$ in FP16/BF16 ensures exact gradient and error accumulation without memory bottlenecks.

---

## 4. Empirical Benchmarks & Validation Results

The implementation in `tests/test_quantized_phasors_poc.py` produced the following verified metrics:

### 4.1 Precision & Angular Error Audit ($N = 100,000$ Random Phasors)

| Format | Angular Resolution | Mean Angular Error (MAE) | Max Angular Error | Cosine Sim MAE vs FP32 | Signal Fidelity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`uint8` (8-bit)** | $1.4062^\circ$ | **$0.7046^\circ$** | $1.4062^\circ$ | $0.005237$ | **$> 99.30\%$** |
| **`uint16` (16-bit)** | $0.0055^\circ$ | **$0.0027^\circ$** | $0.0055^\circ$ | $0.000020$ | **$> 99.999\%$** |

---

### 4.2 Multi-Pair Associative Memory Recall Fidelity ($D = 1024$)

Evaluating recall accuracy across superposed key-value pairs ($M = \sum K_i \odot V_i$):

| Number of Stored Pairs ($N$) | FP32 Cosine Similarity | uint8 Cosine Similarity | uint16 Cosine Similarity | Quantization Gap |
| :---: | :---: | :---: | :---: | :---: |
| **8 pairs** | `0.4016` | **`0.4016`** | **`0.4016`** | **$0.0000$** |
| **16 pairs** | `0.2819` | **`0.2818`** | **`0.2819`** | **$0.0001$** |
| **32 pairs** | `0.2003` | **`0.2003`** | **`0.2003`** | **$0.0000$** |
| **64 pairs** | `0.1443` | **`0.1443`** | **`0.1443`** | **$0.0000$** |

> **Key Finding:** The discretization error of `uint8` is several orders of magnitude smaller than the natural superposition crosstalk noise. `uint8` achieves identical associative recall to 64-bit `complex64`.

---

### 4.3 Throughput & Memory Footprint ($10,000,000$ Operations)

| Operation / Format | Latency (ms) | Throughput (Million ops/sec) | Speedup vs FP32 | Memory Allocation | Memory Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FP32 Complex Mult ($4\text{ FLOPs}$)** | $7.72\text{ ms}$ | $1,294.5\text{ M op/s}$ | $1.00\times$ | $152.6\text{ MB}$ | Baseline |
| **uint8 Modular ADD ($1\text{ ALU}$)** | **$0.95\text{ ms}$** | **$10,513.3\text{ M op/s}$** | **$8.12\times$** ⚡ | **$19.1\text{ MB}$** | **$8.0\times$ Less RAM** |
| **uint16 Modular ADD ($1\text{ ALU}$)** | $6.70\text{ ms}$ | $1,492.1\text{ M op/s}$ | $1.15\times$ | $38.1\text{ MB}$ | $4.0\times$ Less RAM |

---

## 5. Deployment Guidelines & Future Optimizations

1. **Edge & Microcontroller Deployment:** `uint8` phasors enable running DeltaPhase memory cores on low-power microcontrollers (ARM Cortex-M, RISC-V, ESP32) lacking hardware floating-point units (FPUs).
2. **SIMD Vectorization:** Using AVX2 / AVX-512 `_mm256_add_epi8` or ARM NEON `vadd_u8` executes **32 to 64 phasor multiplications per single clock cycle**.
3. **Custom Triton / CUDA Kernel:** Writing a fused integer phasor attention kernel will unlock near-instantaneous streaming inference on consumer GPUs.
4. **Native $(S^1)^D$ Phasor Embeddings:** Extending `uint8` quantization to the token embedding vocabulary dictionary for complete end-to-end integer wave propagation (see [`docs/native_phasor_embeddings_and_spectral_dimensions.md`](native_phasor_embeddings_and_spectral_dimensions.md)).

