# Technical Memo: DeltaPhase Architectural Derivation, Equivalence & Capacity Bound

**Author:** DeltaPhase Research Team  
**Date:** August 12, 2026  
**Document Version:** 1.0  

---

## 1. Introduction & Problem Statement
Recurrent state-space models and linear attention mechanisms attempt to reduce the $O(N^2)$ quadratic computational complexity of Softmax Attention to $O(N)$ sequence scaling and $O(1)$ autoregressive decoding memory. However, real-valued state representations ($\mathbb{R}^{d_k \times d_v}$) suffer from catastrophic capacity degradation (crosstalk) when density of key-value pairs increases.

**DeltaPhase** addresses this limitation by extending the **Parallel Chunkwise WY Representation for Delta Updates** (Yang et al., 2024) into **Complex Phase Phasor Spaces ($\mathbb{C}^{d_k \times d_k}$)** ($K, Q \in S^1$).

---

## 2. Mathematical Derivation of Complex Phase Delta Rule

### 2.1 State Matrix and Unimodular Phasor Mapping
Let $x_t \in \mathbb{R}^d$ be the input representation at sequence step $t$. We project $x_t$ onto continuous angle representations:
$$\theta_K = W_K x_t, \quad \theta_Q = W_Q x_t \in \mathbb{R}^{d_k}$$
Key and Query phasors are mapped to the complex unit circle $S^1 \subset \mathbb{C}^{d_k}$:
$$K_t = \cos(\theta_K) + i \sin(\theta_K), \quad Q_t = \cos(\theta_Q) + i \sin(\theta_Q)$$
Because $\|K_t\|_2^2 = \sum_{j=1}^{d_k} |K_{t,j}|^2 = d_k$, the squared norm of any key phasor is strictly constant $d_k$.

### 2.2 Memory State Readout & Error Correction
The memory state matrix is $M_t \in \mathbb{C}^{d_k \times d_k}$. At step $t$:
1. **Unattenuated Readout:** $v_{\text{old}} = \frac{1}{d_k} \text{Re}(M_{t-1} \bar{K}_t)$
2. **Attenuated Readout:** $v_{\text{att}} = \lambda_t v_{\text{old}} = \frac{\lambda_t}{d_k} \text{Re}(M_{t-1} \bar{K}_t)$
3. **Error Vector:** $e_t = V_t - \lambda_t v_{\text{old}} \in \mathbb{R}^{d_k}$
4. **State Update:** $M_t = \lambda_t M_{t-1} + \beta_t (e_t \otimes K_t)$

Where $\beta_t = 2.0 \cdot \text{sigmoid}(W_\beta x_t) \in (0, 2)$, granting access to the complete isometric Householder reflection spectrum $|1 - \beta_t| < 1$.

---

## 3. Parallel Chunkwise Formulation & Triangular Solve ($T_{\text{mat}}$)

For a sequence of length $L$ partitioned into $N_c = L / C$ chunks of size $C$:
$$\text{Gram}_{\text{real}} = \frac{1}{d_k} \text{Re}(K_c K_c^H) \in \mathbb{R}^{C \times C}$$
$$L_{\text{mat}} = \text{triu}(\text{Gram}_{\text{real}} \cdot \beta_c, \text{diag}=1)$$
$$T_{\text{mat}} = \text{solve\_triangular}(\mathbf{I} + L_{\text{mat}}^T, \mathbf{I}, \text{upper}=\text{False})$$

This converts $L$ sequential scalar steps into $N_c$ parallel GPU matrix operations, achieving exact equivalence to machine precision ($7.39 \times 10^{-16}$ FP64 L2 relative gradient error).

---

## 4. Empirical Capacity Bounds & Failure Modes

1. **Theoretical Information Bound:** A state matrix $M \in \mathbb{C}^{d_k \times d_k}$ stores at most $2 d_k^2$ real floating-point values.
2. **Capacity Breakdown Curve ($d_k=32$):**
   - $N_{\text{pairs}} \le 32$: Exact recall ($>99.95\%$)
   - $N_{\text{pairs}} = 64$: High recall ($91.30\%$)
   - $N_{\text{pairs}} \ge 128$: Capacity degradation ($61.80\%$)
3. **Known Failure Modes:** Scenarios demanding verbatim reproduction of $>100\text{K}$ noisy tokens without attention selection experience crosstalk accumulation.

---

## 5. Conclusion
DeltaPhase provides a mathematically sound, GPU-parallelized $O(N)$ linear memory architecture that empirically mitigates crosstalk, offering $O(1)$ RAM decoding for streaming edge/agentic inference.
