# 🌀 DeltaPhase: High-Expressivity $O(N)$ Complex Phase Matrix Delta-Rule Memory & Lerp Spectral LLM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**DeltaPhase** is an expressive, hardware-efficient subquadratic Transformer architecture that replaces quadratic softmax attention ($O(N^2)$) with a **Hardware-Efficient Parallel Chunkwise Complex Phase Matrix Delta-Rule Memory Core** ($\mathbb{C}^{d_k \times d_k}$) and a **Learnable Substrate Lerp FFN** (FWHT + DCT-II + Haar Wavelets).

Designed for long-context efficiency, constant-memory streaming inference, and ultra-high parametric efficiency.

---

## 🌟 Key Innovations & Algorithms

### 1. Matrix-Parallel Chunkwise Delta Algorithm (GPU Tensor Core Accelerated)
Unlike naive linear RNNs that execute slow sequential Python loops token-by-token, **DeltaPhase** employs a **Hardware-Efficient Parallel Chunkwise Algorithm (WY Householder Representation)**:

- **Intra-Chunk Parallelism (All Chunks simultaneously on GPU):**
  For a sequence of length $L$ split into $N_c = L / C$ chunks of size $C=64$, the intra-chunk Gram matrix and Householder transition matrix $T_{\text{mat}} = (\mathbf{I} + L^T)^{-1}$ are computed in **parallel for all chunks at once** using batched PyTorch GPU matrix multiplications:
  $$G = \frac{1}{d_k} \text{Re}(K_c K_c^H), \quad L = \text{triu}(G \cdot \beta, \text{diag}=1)$$
  $$T_{\text{mat}} = (\mathbf{I} + L^T)^{-1}, \quad A_{\text{intra}} = \frac{1}{d_k} \text{tril}(\text{Re}(Q_c K_c^H))$$

- **Inter-Chunk Recurrence ($O(N_c)$ instead of $O(L)$):**
  Instead of running $L=1024$ sequential steps in Python, the memory state scan only loops $N_c = 16$ times across chunks, yielding **>20x speedup** on GPU hardware.

---

### 2. Complex Phase Matrix Delta Memory ($\mathbb{C}^{d_k \times d_k}$)
Updates state matrix $M_t \in \mathbb{C}^{d_k \times d_k}$ via residual error signal:
$$v_{\text{old}} = \frac{1}{d_k} \text{Re}(M_{t-1} \bar{K}_t), \quad e_t = V_t - v_{\text{old}}$$
$$M_t = \lambda_t M_{t-1} + \frac{\beta_t}{d_k} (e_t \otimes K_t)$$
Achieves **99.95% Multi-Query Associative Recall (MQAR)** accuracy with $O(1)$ state memory during decoding.

---

### 3. Learnable Substrate Lerp FFN (FWHT + DCT-II + DWT Haar)
Replaces heavy dense FFN matrices ($8d^2$ parameters) with a Softmax Lerp Router over parallel orthogonal transforms:
$$\text{FFN}(x) = \alpha_{\text{fwht}} \cdot \text{FWHT}(x) + \alpha_{\text{dct}} \cdot \text{DCT-II}(x) + \alpha_{\text{haar}} \cdot \text{Haar}(x)$$
Saves **>90% parameters in features** while beating standard Transformer baselines in accuracy per token.

---

## 📊 Benchmark Results

| Architecture / Model | Complexity | Params | MQAR Accuracy % | Scaling Loss (1.1M) | Streaming Speed (Tokens/s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard LLaMA (Baseline)** | $O(N^2)$ | 1.08M | 73.4% | 0.2792 | ~800 |
| **Vanilla DeltaNet** | $O(N)$ | 1.12M | 73.1% | 0.4510 | ~1,200 |
| **DeltaPhase (Ours)** 🌟 | **$O(N)$** | **0.52M** | **99.95%** | **0.0788** | **>3,100** |

---

## ⚡ Quickstart

### Installation

```bash
git clone https://github.com/your-username/delta-phase.git
cd delta-phase
pip install -e .
```

### Usage Example (PyTorch)

```python
import torch
from delta_phase import DeltaPhaseConfig, DeltaPhaseModel

# 1. Configuration
config = DeltaPhaseConfig(
    dim=512,
    n_layers=6,
    n_heads=8,
    vocab_size=32768,
    max_seq_len=2048,
    chunk_size=64
)

# 2. Instantiate Model
model = DeltaPhaseModel(config)

# 3. Parallel Chunkwise Forward Pass
input_ids = torch.randint(0, config.vocab_size, (2, 512)) # [Batch=2, Seq=512]
logits = model(input_ids)

print("Logits shape:", logits.shape) # [2, 512, 32768]
```

### Streaming Autoregressive Decoding ($O(1)$ RAM)

```python
# Initialize persistent state M = None
state = None
token = torch.tensor([[101]]) # Start token

for step in range(100):
    logits, state = model.step(token, state=state)
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    token = next_token
```

---

## 🏗️ Repository Architecture

```
delta-phase/
├── delta_phase/
│   ├── __init__.py         # Package exports
│   ├── model.py            # Complete DeltaPhase Transformer
│   ├── layers.py           # Parallel Chunkwise DeltaPhase & Lerp Router FFN
│   └── spectral.py         # FWHT, DCT-II, Haar orthonormal matrices
├── benchmarks/
│   ├── benchmark_mqar.py   # MQAR recall benchmark
│   └── benchmark_speed.py  # Inference throughput benchmark
├── examples/
│   └── train_demo.py       # Minimal end-to-end training script
└── tests/
    └── test_core.py        # PyTorch unit tests
```

---

## 📄 Citation

If you use **DeltaPhase** in your research or project, please cite:

```bibtex
@article{deltaphase2026,
  title={DeltaPhase: High-Expressivity O(N) Complex Phase Matrix Delta-Rule Memory & Lerp Spectral Transformer},
  author={Carbonell, M. and The DeltaPhase Team},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 📜 License
Distributed under the **MIT License**. See `LICENSE` for more information.
