import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .spectral import create_hadamard_matrix, create_dct2_matrix, create_haar_matrix

# ---------------------------------------------------------------------------
# Dtype policy (AMP / autocast):
# Phasor trigonometry (cos/sin -> torch.complex / torch.polar) is ALWAYS computed
# in float32 (or float64 on request) regardless of autocast context:
#  1) reduced-precision phases (bf16/fp16) lose phase fidelity and corrupt the
#     cosine-difference kernel Re(K conj(Q)),
#  2) torch.complex / torch.polar do not support bfloat16 inputs.
# Everything else (conv, norms, projections, FFN) participates in autocast
# normally. Do not remove the explicit `.float()` casts below.
# ---------------------------------------------------------------------------

class LogicPhaseCore(nn.Module):
    """
    LogicPhase Symbolic Phasor Operators on S^1 (BIND, UNBIND, NOT, AND)
    """
    def __init__(self, d_k: int = 32):
        super().__init__()
        self.d_k = d_k

    def bind(self, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """BIND(K, V) -> Phasor Hadamard product K * V"""
        V_complex = V.to(torch.complex64) if not V.is_complex() else V
        return K * V_complex

    def unbind(self, K: torch.Tensor, M_bind: torch.Tensor) -> torch.Tensor:
        """UNBIND(K, M) -> Conjugate readout conj(K) * M"""
        return (torch.conj(K) * M_bind).real

    def not_op(self, Q: torch.Tensor) -> torch.Tensor:
        """NOT(Q) -> Phase Shift by pi radians (-Q) for destructive wave cancellation"""
        return Q * torch.complex(torch.tensor(-1.0, device=Q.device), torch.tensor(0.0, device=Q.device))

    def bundle(self, Q1: torch.Tensor, Q2: torch.Tensor) -> torch.Tensor:
        """
        BUNDLE(Q1, Q2) -> Vector Superposition (Plate 1995 VSA Bundling / Set Union)
        Sum of phasors normalized to unit magnitude. Represents memory set superposition.
        """
        superpos = Q1 + Q2
        mags = torch.abs(superpos) + 1e-8
        return superpos / mags

    def strict_and_op(self, r1: torch.Tensor, r2: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        STRICT_AND(r1, r2) -> Strict Logical Conjunction (Intersection Gate)
        Returns minimum activation when both inputs exceed threshold, strictly 0.0000 otherwise.
        """
        mask = (r1 > threshold) & (r2 > threshold)
        return torch.minimum(r1, r2) * mask.float()

    def and_op(self, Q1: torch.Tensor, Q2: torch.Tensor) -> torch.Tensor:
        """Coherent superposition bundling for query phasors"""
        return self.bundle(Q1, Q2)

class ShortCausalConv1D(nn.Module):
    """Depthwise 1D Causal Convolution (kernel_size=4) for local token binding"""
    def __init__(self, d_model: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=d_model
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x_t = x.transpose(1, 2)
        conv_out = self.conv(x_t)[:, :, :L].transpose(1, 2)
        return x + self.act(conv_out)

    def step(self, x_t: torch.Tensor, conv_state=None):
        B, L, D = x_t.shape
        if conv_state is None:
            conv_state = torch.zeros(B, D, self.kernel_size - 1, device=x_t.device, dtype=x_t.dtype)
        inputs = torch.cat([conv_state, x_t.transpose(1, 2)], dim=2)
        new_conv_state = inputs[:, :, 1:]
        conv_out = F.conv1d(inputs, self.conv.weight, self.conv.bias, groups=D)[:, :, -1:]
        out = x_t + self.act(conv_out.transpose(1, 2))
        return out, new_conv_state

class LearnableSubstrateLerpFFN(nn.Module):
    """FFN con router Softmax Lerp aprendible entre FWHT, DCT-II y DWT Haar (Vectorizado)"""
    def __init__(self, d_model: int, num_banks: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_banks = num_banks
        self.substrate_logits = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
        
        self.register_buffer('mat_fwht', create_hadamard_matrix(d_model))
        self.register_buffer('mat_dct', create_dct2_matrix(d_model))
        self.register_buffer('mat_haar', create_haar_matrix(d_model))
        
        self.phi1_fwht = nn.Parameter(torch.zeros(num_banks, d_model))
        self.phi2_fwht = nn.Parameter(torch.zeros(num_banks, d_model))
        self.w1_fwht = nn.Parameter(torch.ones(num_banks, d_model))
        self.w2_fwht = nn.Parameter(torch.ones(num_banks, d_model))
        
        self.phi1_dct = nn.Parameter(torch.zeros(num_banks, d_model))
        self.phi2_dct = nn.Parameter(torch.zeros(num_banks, d_model))
        self.w1_dct = nn.Parameter(torch.ones(num_banks, d_model))
        self.w2_dct = nn.Parameter(torch.ones(num_banks, d_model))
        
        self.phi1_haar = nn.Parameter(torch.zeros(num_banks, d_model))
        self.phi2_haar = nn.Parameter(torch.zeros(num_banks, d_model))
        self.w1_haar = nn.Parameter(torch.ones(num_banks, d_model))
        self.w2_haar = nn.Parameter(torch.ones(num_banks, d_model))
        
        self.combine = nn.Linear(num_banks * d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.substrate_logits, dim=0)
        
        # 1. Rama FWHT (Vectorizada con broadcasting)
        h_fwht = F.linear(x, self.mat_fwht).unsqueeze(-2)
        outs_fwht = (torch.cos(h_fwht + self.phi1_fwht) * self.w1_fwht + 
                     torch.sin(h_fwht + self.phi2_fwht) * self.w2_fwht).flatten(-2)
        out_fwht = F.linear(self.combine(outs_fwht), self.mat_fwht.t())
        
        # 2. Rama DCT-II (Vectorizada con broadcasting)
        h_dct = F.linear(x, self.mat_dct).unsqueeze(-2)
        outs_dct = (torch.cos(h_dct + self.phi1_dct) * self.w1_dct + 
                    torch.sin(h_dct + self.phi2_dct) * self.w2_dct).flatten(-2)
        out_dct = F.linear(self.combine(outs_dct), self.mat_dct.t())
        
        # 3. Rama DWT Haar (Vectorizada con broadcasting)
        h_haar = F.linear(x, self.mat_haar).unsqueeze(-2)
        outs_haar = (torch.cos(h_haar + self.phi1_haar) * self.w1_haar + 
                     torch.sin(h_haar + self.phi2_haar) * self.w2_haar).flatten(-2)
        out_haar = F.linear(self.combine(outs_haar), self.mat_haar.t())
        
        # Combinación Lerp Convexa
        return weights[0] * out_fwht + weights[1] * out_dct + weights[2] * out_haar

    def get_substrate_probabilities(self):
        probs = F.softmax(self.substrate_logits, dim=0)
        return probs[0].item(), probs[1].item(), probs[2].item()

class DeltaPhaseHolographicBlock(nn.Module):
    """
    Parallel Chunkwise Complex Delta-Phase Memory Layer (v300/v305).
    Computes intra-chunk transitions and outputs via parallel GPU matmuls.
    """
    def __init__(self, d_model: int, n_heads: int = 8, conv_kernel_size: int = 4, chunk_size: int = 64, num_banks: int = 4, beta_mode: str = "learned"):
        super().__init__()
        if beta_mode not in ("learned", "fixed"):
            raise ValueError(f"beta_mode must be 'learned' or 'fixed', got {beta_mode!r}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.inv_dk = 1.0 / float(self.d_k)
        self.chunk_size = chunk_size
        self.beta_mode = beta_mode
        self.last_beta = None
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_retrieved = nn.LayerNorm(d_model)
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=conv_kernel_size)
        
        self.w_k = nn.Linear(d_model, d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_beta = nn.Linear(d_model, n_heads)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = LearnableSubstrateLerpFFN(d_model, num_banks=num_banks)

    def forward(self, x: torch.Tensor, memory_state=None):
        res = x
        normed = self.norm1(x)
        conv_x = self.causal_conv(normed)
        B, L, D = conv_x.shape
        C = self.chunk_size
        inv_dk = self.inv_dk

        pad_len = (C - (L % C)) % C
        if pad_len > 0:
            conv_x = F.pad(conv_x, (0, 0, 0, pad_len))
            L_padded = L + pad_len
        else:
            L_padded = L

        theta_k = self.w_k(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        theta_q = self.w_q(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        beta = 2.0 * torch.sigmoid(self.w_beta(conv_x)).transpose(1, 2)
        if self.beta_mode == "fixed":
            beta = torch.ones_like(beta)
        if pad_len > 0:
            mask = torch.ones(B, self.n_heads, L_padded, device=x.device)
            mask[:, :, L:] = 0.0
            beta = beta * mask
        
        theta_k_f = theta_k if theta_k.dtype == torch.float64 else theta_k.float()
        theta_q_f = theta_q if theta_q.dtype == torch.float64 else theta_q.float()
        K = torch.complex(torch.cos(theta_k_f), torch.sin(theta_k_f))
        Q = torch.complex(torch.cos(theta_q_f), torch.sin(theta_q_f))

        num_chunks = L_padded // C
        Q_c = Q.view(B, self.n_heads, num_chunks, C, self.d_k)
        K_c = K.view(B, self.n_heads, num_chunks, C, self.d_k)
        V_c = v.view(B, self.n_heads, num_chunks, C, self.d_k)
        beta_c = beta.view(B, self.n_heads, num_chunks, C)

        # 1. Parallel Matmuls for Gram & Transition Matrix T_mat across all chunks at once
        Gram_real = torch.matmul(K_c, torch.conj(K_c).transpose(-1, -2)).real * inv_dk
        L_mat = torch.triu(Gram_real * beta_c.unsqueeze(-1), diagonal=1)
        I_mat = torch.eye(C, device=x.device).view(1, 1, 1, C, C)
        T_mat = torch.linalg.solve_triangular(I_mat + L_mat.transpose(-1, -2), I_mat, upper=False)

        # 2. Inter-chunk scan (num_chunks iterations instead of token-by-token loop)
        complex_dtype = torch.complex128 if x.dtype == torch.float64 else torch.complex64
        if memory_state is None:
            M_state = torch.zeros(B, self.n_heads, self.d_k, self.d_k, dtype=complex_dtype, device=x.device)
        else:
            M_state = memory_state

        out_chunks = []
        for c in range(num_chunks):
            qc, kc, vc, bc, tc = Q_c[:, :, c], K_c[:, :, c], V_c[:, :, c], beta_c[:, :, c], T_mat[:, :, c]
            v_old = torch.matmul(M_state, torch.conj(kc).transpose(-1, -2)).real.transpose(-1, -2) * inv_dk
            E_c = torch.matmul(tc, vc - v_old)
            U_c = bc.unsqueeze(-1) * E_c
            o_inter = torch.matmul(M_state, torch.conj(qc).transpose(-1, -2)).real.transpose(-1, -2) * inv_dk
            A_intra = torch.tril(torch.matmul(qc, torch.conj(kc).transpose(-1, -2)).real) * inv_dk
            out_chunks.append(torch.matmul(A_intra, U_c) + o_inter)
            M_state = M_state + torch.matmul(U_c.to(complex_dtype).transpose(-1, -2), kc)

        retrieved = torch.cat(out_chunks, dim=2)[:, :, :L].transpose(1, 2).reshape(B, L, D)
        retrieved_norm = self.norm_retrieved(retrieved)
        self.last_beta = beta[:, :, :L].detach()

        x = res + self.out_proj(retrieved_norm)
        x = x + self.ffn(self.norm2(x))
        return x, M_state

    def step(self, x_t: torch.Tensor, state=None):
        """Single-token O(1) streaming step during autoregressive decoding"""
        res = x_t
        normed = self.norm1(x_t)
        
        if state is None:
            conv_state, memory_state = None, None
        elif isinstance(state, tuple):
            conv_state, memory_state = state
        else:
            conv_state, memory_state = None, state
            
        conv_x, new_conv_state = self.causal_conv.step(normed, conv_state=conv_state)
        B, L, D = conv_x.shape # L=1
        inv_dk = self.inv_dk
        
        theta_k = self.w_k(conv_x).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
        theta_q = self.w_q(conv_x).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(conv_x).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
        beta = 2.0 * torch.sigmoid(self.w_beta(conv_x)).transpose(1, 2)
        if self.beta_mode == "fixed":
            beta = torch.ones_like(beta)
        self.last_beta = beta.detach()

        theta_k_f = theta_k if theta_k.dtype == torch.float64 else theta_k.float()
        theta_q_f = theta_q if theta_q.dtype == torch.float64 else theta_q.float()
        K = torch.complex(torch.cos(theta_k_f), torch.sin(theta_k_f))
        Q = torch.complex(torch.cos(theta_q_f), torch.sin(theta_q_f))

        complex_dtype = torch.complex128 if x_t.dtype == torch.float64 else torch.complex64
        if memory_state is None:
            memory_state = torch.zeros(B, self.n_heads, self.d_k, self.d_k, dtype=complex_dtype, device=x_t.device)
            
        kt, qt, vt, bt = K[:, :, 0], Q[:, :, 0], v[:, :, 0], beta[:, :, 0]
        v_old = torch.matmul(memory_state, torch.conj(kt).unsqueeze(-1)).squeeze(-1).real * inv_dk
        err = vt - v_old
        update = torch.matmul(err.to(complex_dtype).unsqueeze(-1), kt.unsqueeze(-2))
        memory_state = memory_state + bt.unsqueeze(-1).unsqueeze(-1) * update
        
        retrieved_t = torch.matmul(memory_state, torch.conj(qt).unsqueeze(-1)).squeeze(-1).real * inv_dk
        retrieved = retrieved_t.reshape(B, 1, D)
        retrieved_norm = self.norm_retrieved(retrieved)
        
        x = res + self.out_proj(retrieved_norm)
        x = x + self.ffn(self.norm2(x))
        return x, (new_conv_state, memory_state)


class ComplexBetaDeltaPhaseBlock(nn.Module):
    """
    Complex Beta Householder DeltaPhase Block:
    Uses complex Householder parameterization: beta_t = 1 + exp(i * phi_t).
    The eigenvalues of the recurrent update matrix I - beta_t K_t K_t* are lambda = -exp(i * phi_t) in S^1.
    Natively computes cyclic group operations (Z_k modular addition) in a single token step
    while maintaining isometric non-expansion.
    """
    def __init__(self, d_model: int, n_heads: int = 4, conv_kernel_size: int = 4, chunk_size: int = 32, num_banks: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.inv_dk = 1.0 / float(self.d_k)
        self.chunk_size = chunk_size
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_retrieved = nn.LayerNorm(d_model)
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=conv_kernel_size)
        
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_phi = nn.Linear(d_model, n_heads, bias=False)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = LearnableSubstrateLerpFFN(d_model, num_banks=num_banks)
        self.last_beta = None

    def forward(self, x: torch.Tensor, memory_state=None):
        res = x
        normed = self.norm1(x)
        conv_x = self.causal_conv(normed)
        B, L, D = conv_x.shape
        inv_dk = self.inv_dk
        complex_dtype = torch.complex128 if x.dtype == torch.float64 else torch.complex64
        
        theta_k = self.w_k(conv_x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        theta_q = self.w_q(conv_x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(conv_x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        phi_beta = self.w_phi(conv_x).transpose(1, 2)

        compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        theta_k = theta_k.to(compute_dtype)
        theta_q = theta_q.to(compute_dtype)
        v = v.to(compute_dtype)
        phi_beta = phi_beta.to(compute_dtype)

        K = torch.polar(torch.ones_like(theta_k), theta_k)
        Q = torch.polar(torch.ones_like(theta_q), theta_q)
        beta_complex = 1.0 + torch.polar(torch.ones_like(phi_beta), phi_beta)
        self.last_beta = beta_complex.detach()

        if memory_state is None:
            M_state = torch.zeros(B, self.n_heads, self.d_k, self.d_k, dtype=complex_dtype, device=x.device)
        else:
            M_state = memory_state

        out_list = []
        for t in range(L):
            kt, qt, vt, bt = K[:, :, t], Q[:, :, t], v[:, :, t], beta_complex[:, :, t]
            v_old = torch.matmul(M_state, torch.conj(kt).unsqueeze(-1)).squeeze(-1).real * inv_dk
            err = vt - v_old
            err_c = err.to(M_state.dtype)
            update = bt.unsqueeze(-1).unsqueeze(-1) * torch.matmul(err_c.unsqueeze(-1), kt.unsqueeze(-2))
            M_state = M_state + update
            out_t = torch.matmul(M_state, torch.conj(qt).unsqueeze(-1)).squeeze(-1).real * inv_dk
            out_list.append(out_t)
            
        retrieved = torch.stack(out_list, dim=2).transpose(1, 2).reshape(B, L, D)
        retrieved_norm = self.norm_retrieved(retrieved)
        
        x = res + self.out_proj(retrieved_norm)
        x = x + self.ffn(self.norm2(x))
        return x, M_state

    def step(self, x_t: torch.Tensor, state=None):
        res = x_t
        normed = self.norm1(x_t)
        if state is None:
            conv_state, memory_state = None, None
        elif isinstance(state, tuple):
            conv_state, memory_state = state
        else:
            conv_state, memory_state = None, state
            
        conv_x, new_conv_state = self.causal_conv.step(normed, conv_state=conv_state)
        B, L, D = conv_x.shape
        inv_dk = self.inv_dk
        complex_dtype = torch.complex128 if x_t.dtype == torch.float64 else torch.complex64
        
        theta_k = self.w_k(conv_x).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
        theta_q = self.w_q(conv_x).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(conv_x).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
        phi_beta = self.w_phi(conv_x).transpose(1, 2)

        compute_dtype = torch.float64 if x_t.dtype == torch.float64 else torch.float32
        theta_k = theta_k.to(compute_dtype)
        theta_q = theta_q.to(compute_dtype)
        v = v.to(compute_dtype)
        phi_beta = phi_beta.to(compute_dtype)

        K = torch.polar(torch.ones_like(theta_k), theta_k)
        Q = torch.polar(torch.ones_like(theta_q), theta_q)
        beta_complex = 1.0 + torch.polar(torch.ones_like(phi_beta), phi_beta)
        self.last_beta = beta_complex.detach()

        if memory_state is None:
            memory_state = torch.zeros(B, self.n_heads, self.d_k, self.d_k, dtype=complex_dtype, device=x_t.device)

        kt, qt, vt, bt = K[:, :, 0], Q[:, :, 0], v[:, :, 0], beta_complex[:, :, 0]
        v_old = torch.matmul(memory_state, torch.conj(kt).unsqueeze(-1)).squeeze(-1).real * inv_dk
        err = vt - v_old
        err_c = err.to(memory_state.dtype)
        update = bt.unsqueeze(-1).unsqueeze(-1) * torch.matmul(err_c.unsqueeze(-1), kt.unsqueeze(-2))
        memory_state = memory_state + update
        
        retrieved_t = torch.matmul(memory_state, torch.conj(qt).unsqueeze(-1)).squeeze(-1).real * inv_dk
        retrieved = retrieved_t.reshape(B, 1, D)
        retrieved_norm = self.norm_retrieved(retrieved)
        
        x = res + self.out_proj(retrieved_norm)
        x = x + self.ffn(self.norm2(x))
        return x, (new_conv_state, memory_state)



class LaplacePhaseCore(nn.Module):
    """
    Delta-Laplace Phase Memory Core:
    Operates over complex frequency s = sigma + i*theta in the Laplace s-plane.
    Guarantees Hurwitz Stability via strictly non-positive real dissipation sigma <= 0 (Re(s) <= 0),
    mapped to the Z-plane unit disk |z| = exp(sigma * dt) <= 1 via Continuous Zero-Order Hold (ZOH).

    Recurrence (per token t):
        v_old = Re(M k_t^H)/d_k
        M_t   = diag(r_t) M_{t-1} + beta_t (v_t - v_old) k_t^T ,   r_t = exp(sigma_t dt) in (0,1)
        out_t = Re(M_t q_t^H)/d_k

    Because the decay diag(r_t) acts on the OUTPUT rows (left multiplication), the
    intra-chunk coupling coefficients are the plain phasor Gram <k_s, k_m^H>/d_k and the
    data-dependent decay factorizes in log-space. This enables an EXACT chunkwise-parallel
    form (no approximation): per output channel o, a C x C lower-triangular system
        e_m + sum_{s<m} P_{s,m-1}[o] * beta_s * <k_s, k_m^H>/d_k * e_s = v_m[o] - D_{m-1}[o] * (M0 k_m^H)[o]/d_k
    is solved batched over (B, H, d_k), with P_{s,t} = prod_{j=s+1..t} r_j and
    D_t = prod_{j<=t} r_j accumulated via log-space cumsums.

    forward() runs the vectorized chunkwise path; forward_sequential() keeps the
    literal token-loop reference used by equivalence tests.
    """
    LOG_FLOOR = -80.0  # clamp for log-space decay cumsums (fp32-safe)

    def __init__(self, d_model=64, n_heads=4, d_k=16, chunk_size=32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.inv_dk = 1.0 / float(d_k)
        self.chunk_size = int(chunk_size)

        self.w_theta_k = nn.Linear(d_model, d_model, bias=False)
        self.w_sigma_k = nn.Linear(d_model, d_model, bias=False)
        self.w_theta_q = nn.Linear(d_model, d_model, bias=False)
        self.w_sigma_q = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_beta = nn.Linear(d_model, n_heads, bias=False)

    def _project(self, x: torch.Tensor, time_scale: float):
        """Projections with explicit fp32/fp64 phasor computation (AMP-safe, see module docstring)."""
        B, L, D = x.shape
        dt = 1.0 / float(time_scale)
        compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32

        theta_k = (self.w_theta_k(x).to(compute_dtype) * dt).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        sigma_k = (-F.softplus(self.w_sigma_k(x).to(compute_dtype)) * dt).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        theta_q = (self.w_theta_q(x).to(compute_dtype) * dt).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        sigma_q = (-F.softplus(self.w_sigma_q(x).to(compute_dtype)) * dt).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).to(compute_dtype).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        beta = (2.0 * torch.sigmoid(self.w_beta(x).to(compute_dtype)) * dt).transpose(1, 2)

        r_k = torch.exp(sigma_k)
        K = torch.complex(r_k * torch.cos(theta_k), r_k * torch.sin(theta_k))
        r_q = torch.exp(sigma_q)
        Q = torch.complex(r_q * torch.cos(theta_q), r_q * torch.sin(theta_q))
        return K, Q, v, beta, r_k

    def forward(self, x: torch.Tensor, memory_state=None, time_scale: float = 1.0):
        K, Q, v, beta, r_k = self._project(x, time_scale)
        B, H, L, dk = K.shape
        complex_dtype = torch.complex128 if x.dtype == torch.float64 else torch.complex64
        if memory_state is None:
            M = torch.zeros(B, H, dk, dk, dtype=complex_dtype, device=x.device)
        else:
            M = memory_state

        C = min(self.chunk_size, max(L, 1))
        pad_len = (C - L % C) % C
        if pad_len:
            K = F.pad(K, (0, 0, 0, pad_len))
            Q = F.pad(Q, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            r_k = F.pad(r_k, (0, 0, 0, pad_len), value=1.0)
            beta = F.pad(beta, (0, pad_len))
        NC = (L + pad_len) // C

        Kv = K.view(B, H, NC, C, dk)                         # [.., m, c]
        Qv = Q.view(B, H, NC, C, dk)
        vv = v.view(B, H, NC, C, dk)                         # [m, o]
        rv = r_k.view(B, H, NC, C, dk)                       # [m, o]
        bv = beta.view(B, H, NC, C)                          # [m]

        out_chunks = []
        for ci in range(NC):
            kc, qc = Kv[:, :, ci], Qv[:, :, ci]              # (B,H,C,dk) complex
            vo = vv[:, :, ci]                                # (B,H,m,o)
            bc = bv[:, :, ci]                                # (B,H,m)
            rc = rv[:, :, ci]                                # (B,H,m,o)
            Sr = torch.log(rc).clamp_min(self.LOG_FLOOR).cumsum(dim=2)          # (B,H,m,o) = log D_m
            G = torch.exp(Sr)                                                   # D_m[o]
            Gprev = torch.cat([torch.ones_like(G[:, :, :1, :]), G[:, :, :-1, :]], dim=2)  # D_{m-1}[o]

            g_km = torch.matmul(kc, torch.conj(kc).transpose(-1, -2)).real * self.inv_dk   # [s,m]
            h_qm = torch.matmul(kc, torch.conj(qc).transpose(-1, -2)).real * self.inv_dk   # [s,m]

            B0k = torch.matmul(M, torch.conj(kc).transpose(-1, -2)).real * self.inv_dk     # (B,H,o,m)
            B0q = torch.matmul(M, torch.conj(qc).transpose(-1, -2)).real * self.inv_dk     # (B,H,o,m)

            # Channel-batched lower-triangular system (one C x C system per output channel o):
            #   e_m[o] + sum_{s<m} exp(S_{m-1}[o]-S_s[o]) * beta_s * g_km[s,m] * e_s[o]
            #       = v[m,o] - D_{m-1}[o] * (M0 k_m^H)[o,m]/d_k
            Sro = Sr.permute(0, 1, 3, 2)                                        # (B,H,o,m)
            Sro_prev = torch.cat([torch.zeros_like(Sro[..., :1]), Sro[..., :-1]], dim=-1)
            P_oms = torch.exp(Sro_prev.unsqueeze(-1) - Sro.unsqueeze(-2))       # (B,H,o,m,s) = D_{m-1}/D_s
            beta_s = bc.view(B, H, 1, 1, C)
            g_smT = g_km.transpose(-1, -2).unsqueeze(2)                         # (B,H,1,m,s)
            Acoef = torch.eye(C, dtype=Sr.dtype, device=Sr.device) \
                + torch.tril(P_oms * beta_s * g_smT, diagonal=-1)               # (B,H,o,m,s) lower in (m,s)

            RHS_om = (vo - Gprev * B0k.transpose(-1, -2)).permute(0, 1, 3, 2)        # (B,H,o,m)
            E_om = torch.linalg.solve_triangular(Acoef, RHS_om.unsqueeze(-1), upper=False).squeeze(-1)  # (B,H,o,m)
            Et = E_om.transpose(-1, -2)                                         # (B,H,m,o)
            BEt = bc.unsqueeze(-1) * Et                                         # beta_s * e_s[s,o]

            # Outputs: o_m[o] = D_m[o]*(M0 q_m^H)[o]/dk + sum_{s<=m} (D_m/D_s)[o] * beta_s e_s[o] * h_qm[s,m]
            Ointer = G * B0q.transpose(-1, -2)                                  # (B,H,m,o)
            FE = torch.exp(-Sr) * BEt                                           # (B,H,s,o)
            # h_qm is [s,m]; we need the s<=m triangle: tril on the transposed [m,s] layout.
            Hl = torch.tril(h_qm.transpose(-1, -2), diagonal=0).transpose(-1, -2)  # (B,H,s,m), s<=m
            Z = torch.matmul(Hl.transpose(-1, -2), FE)                          # (B,H,m,o)
            out_chunks.append(Ointer + G * Z)

            # State carry: M_C[o,c] = D_C[o] M[o,c] + sum_s (D_C/D_s)[o] * beta_s e_s[o] * k_s[c]
            Wdec = torch.exp(Sr[:, :, -1:, :] - Sr)                             # (B,H,s,o) = D_C/D_s
            WE = Wdec * BEt                                                     # (B,H,s,o)
            M = torch.exp(Sr[:, :, -1, :]).unsqueeze(-1) * M \
                + torch.matmul(WE.to(M.dtype).transpose(-1, -2), kc)

        retrieved = torch.cat(out_chunks, dim=2)[..., :L, :]                    # (B,H,L,dk)
        retrieved = retrieved.transpose(1, 2).reshape(B, L, self.n_heads * self.d_k)

        if pad_len:
            # Undo the fictitious decay applied during padded steps (beta=0 there => no writes).
            pad_G = rv[:, :, -1, -pad_len:, :].prod(dim=2)                      # (B,H,o): decay over pad steps of last chunk
            safe = torch.where(pad_G > 1e-30, pad_G, torch.ones_like(pad_G))
            M = M / safe.to(M.dtype).unsqueeze(-1)

        return retrieved, M

    def forward_sequential(self, x: torch.Tensor, memory_state=None, time_scale: float = 1.0):
        """Literal token-by-token reference implementation (equivalence oracle for forward())."""
        K, Q, v, beta, r_k = self._project(x, time_scale)
        B, H, L, dk = K.shape
        complex_dtype = torch.complex128 if x.dtype == torch.float64 else torch.complex64
        if memory_state is None:
            M = torch.zeros(B, H, dk, dk, dtype=complex_dtype, device=x.device)
        else:
            M = memory_state

        out_list = []
        for t in range(L):
            kt, qt, vt, bt = K[:, :, t], Q[:, :, t], v[:, :, t], beta[:, :, t]
            v_old = torch.matmul(M, torch.conj(kt).unsqueeze(-1)).squeeze(-1).real * self.inv_dk
            err = vt - v_old
            M = M * r_k[:, :, t].unsqueeze(-1) + bt.unsqueeze(-1).unsqueeze(-1) * torch.matmul(err.to(M.dtype).unsqueeze(-1), kt.unsqueeze(-2))
            out_t = torch.matmul(M, torch.conj(qt).unsqueeze(-1)).squeeze(-1).real * self.inv_dk
            out_list.append(out_t)

        retrieved = torch.cat(out_list, dim=-1).view(B, self.n_heads, L, self.d_k).transpose(1, 2).reshape(B, L, self.d_model)
        return retrieved, M

    def bind(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """VSA Phasor Binding: k * v"""
        return k * v

    def unbind(self, bound: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """VSA Phasor Unbinding via Complex Conjugate: bound * conj(k)"""
        return bound * torch.conj(k)

    def bundle(self, r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
        """VSA Vector Bundling (Superposition / Set Union): r1 + r2"""
        return r1 + r2

    def strict_and_op(self, r1: torch.Tensor, r2: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Strict Boolean Intersection Gate (AND): Zero if missing, thresholded min amplitude"""
        mag1, mag2 = torch.abs(r1), torch.abs(r2)
        gate = (mag1 > threshold) & (mag2 > threshold)
        min_mag = torch.minimum(mag1, mag2)
        phase = torch.angle(r1 + r2)
        result_mag = torch.where(gate, min_mag, torch.zeros_like(min_mag))
        return torch.complex(result_mag * torch.cos(phase), result_mag * torch.sin(phase))
