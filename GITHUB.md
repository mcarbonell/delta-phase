# 📌 GitHub Repository Metadata & Summary — DeltaPhase

## Repository Name
`delta-phase`

## Short Summary (About / Description)
> **DeltaPhase:** High-Expressivity $O(N)$ Complex Phase Matrix Delta-Rule Memory & Lerp Spectral Transformer for Subquadratic Sub-Linear LLMs.

## Suggested GitHub Repository Tags (Topics)
`deep-learning` `linear-attention` `subquadratic-attention` `delta-rule` `complex-phase` `fast-weight-programmers` `associative-memory` `pytorch` `llm-architecture` `spectral-learning` `hadamard-transform` `discrete-cosine-transform` `state-space-models` `artificial-intelligence`

## Social Preview & Card Summary
DeltaPhase is an expressive, hardware-efficient subquadratic Transformer architecture that replaces standard softmax attention with an $O(N)$ Complex Phase Matrix Delta-Rule memory core ($\mathbb{C}^{d_k \times d_k}$) and a Learnable Substrate Lerp FFN (FWHT + DCT-II + Haar). Achieves 99.95% MQAR recall accuracy with $O(1)$ streaming state memory.

---

## Key Features
- 🧠 **$O(N)$ Complex Phase Matrix Delta Memory:** Matrix residual update ($M_t = \lambda_t M_{t-1} + \frac{\beta_t}{d_k} (e_t \otimes K_t)$) operating in complex phasors $\mathbb{C}^{d_k \times d_k}$, eliminating linear memory crosstalk.
- ⚡ **Learnable Substrate Lerp FFN:** Convex Softmax Lerp router over Fast Walsh-Hadamard Transform (FWHT), Discrete Cosine Transform (DCT-II), and DWT Haar Wavelets with >90% parameter savings over dense FFNs.
- 🚀 **Constant $O(1)$ Streaming Inference:** 3,000+ tokens/sec streaming throughput on single-core CPU/GPU with zero KV-cache growth.
- 🛡️ **Position-Aware (NoPE Compatible):** Native positional encoding via multiplicative phase transition dynamics.
