import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .spectral import create_hadamard_matrix, create_dct2_matrix, create_haar_matrix

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

class LearnableSubstrateLerpFFN(nn.Module):
    """FFN con router Softmax Lerp aprendible entre FWHT, DCT-II y DWT Haar"""
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
        
        # 1. FWHT Branch
        h_fwht = F.linear(x, self.mat_fwht)
        outs_fwht = [torch.cos(h_fwht + self.phi1_fwht[b]) * self.w1_fwht[b] + torch.sin(h_fwht + self.phi2_fwht[b]) * self.w2_fwht[b] for b in range(self.num_banks)]
        out_fwht = F.linear(self.combine(torch.cat(outs_fwht, dim=-1)), self.mat_fwht.t())
        
        # 2. DCT-II Branch
        h_dct = F.linear(x, self.mat_dct)
        outs_dct = [torch.cos(h_dct + self.phi1_dct[b]) * self.w1_dct[b] + torch.sin(h_dct + self.phi2_dct[b]) * self.w2_dct[b] for b in range(self.num_banks)]
        out_dct = F.linear(self.combine(torch.cat(outs_dct, dim=-1)), self.mat_dct.t())
        
        # 3. DWT Haar Branch
        h_haar = F.linear(x, self.mat_haar)
        outs_haar = [torch.cos(h_haar + self.phi1_haar[b]) * self.w1_haar[b] + torch.sin(h_haar + self.phi2_haar[b]) * self.w2_haar[b] for b in range(self.num_banks)]
        out_haar = F.linear(self.combine(torch.cat(outs_haar, dim=-1)), self.mat_haar.t())
        
        return weights[0] * out_fwht + weights[1] * out_dct + weights[2] * out_haar

    def get_substrate_probabilities(self):
        probs = F.softmax(self.substrate_logits, dim=0)
        return probs[0].item(), probs[1].item(), probs[2].item()

def _scan_chunk(
    K_c: torch.Tensor,
    Q_c: torch.Tensor,
    v_c: torch.Tensor,
    beta_c: torch.Tensor,
    lam_c: torch.Tensor,
    M_init: torch.Tensor,
    inv_dk: float
):
    """Chunked scan helper for Delta-Phase complex memory"""
    B, C, H, dk = K_c.shape
    M = M_init
    out_list = []
    
    for t in range(C):
        k_t = K_c[:, t]
        q_t = Q_c[:, t]
        v_t = v_c[:, t]
        beta_t = beta_c[:, t]
        lam_t = lam_c[:, t]
        
        k_conj = torch.conj(k_t)
        q_conj = torch.conj(q_t)
        
        # 1. Readout prediction
        v_old = torch.matmul(M, k_conj.unsqueeze(-1)).squeeze(-1).real
        err = v_t - v_old
        
        # 2. Residual complex update
        update = torch.matmul(err.to(torch.complex64).unsqueeze(-1), k_t.unsqueeze(-2))
        M = lam_t * M + (beta_t * inv_dk) * update
        
        # 3. Query readout
        ret = torch.matmul(M, q_conj.unsqueeze(-1)).squeeze(-1).real
        out_list.append(ret)
        
    retrieved_chunk = torch.stack(out_list, dim=1)
    return retrieved_chunk, M

class DeltaPhaseHolographicBlock(nn.Module):
    """Core O(N) Complex Phase Delta Memory Block"""
    def __init__(self, d_model: int, n_heads: int = 8, conv_kernel_size: int = 4, chunk_size: int = 128, num_banks: int = 4):
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
        
        self.w_k = nn.Linear(d_model, d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_beta = nn.Linear(d_model, n_heads)
        self.w_lambda = nn.Linear(d_model, n_heads)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = LearnableSubstrateLerpFFN(d_model, num_banks=num_banks)

    def forward(self, x: torch.Tensor, memory_state=None):
        res = x
        normed = self.norm1(x)
        conv_x = self.causal_conv(normed)
        B, L, D = conv_x.shape
        
        theta_k = self.w_k(conv_x).view(B, L, self.n_heads, self.d_k)
        theta_q = self.w_q(conv_x).view(B, L, self.n_heads, self.d_k)
        v = self.w_v(conv_x).view(B, L, self.n_heads, self.d_k)
        beta = torch.sigmoid(self.w_beta(conv_x)).view(B, L, self.n_heads, 1, 1)
        lam = (0.85 + 0.149 * torch.sigmoid(self.w_lambda(conv_x))).view(B, L, self.n_heads, 1, 1)
        
        K = torch.polar(torch.ones_like(theta_k), theta_k)
        Q = torch.polar(torch.ones_like(theta_q), theta_q)
        
        if memory_state is None:
            M = torch.zeros(B, self.n_heads, self.d_k, self.d_k, dtype=torch.complex64, device=x.device)
        else:
            M = memory_state
            
        chunk_size = self.chunk_size
        num_chunks = max(1, L // chunk_size)
        retrieved_chunks = []
        
        for c in range(num_chunks):
            start = c * chunk_size
            end = min((c + 1) * chunk_size, L)
            
            K_c = K[:, start:end]
            Q_c = Q[:, start:end]
            v_c = v[:, start:end]
            beta_c = beta[:, start:end]
            lam_c = lam[:, start:end]
            
            if self.training:
                ret_c, M = checkpoint(
                    _scan_chunk,
                    K_c, Q_c, v_c, beta_c, lam_c, M, self.inv_dk,
                    use_reentrant=False
                )
            else:
                ret_c, M = _scan_chunk(K_c, Q_c, v_c, beta_c, lam_c, M, self.inv_dk)
                
            retrieved_chunks.append(ret_c)
            
        retrieved = torch.cat(retrieved_chunks, dim=1).view(B, L, D)
        retrieved_norm = self.norm_retrieved(retrieved)
        
        x = res + self.out_proj(retrieved_norm)
        x = x + self.ffn(self.norm2(x))
        return x, M

    def step(self, x_t: torch.Tensor, memory_state=None):
        """Single-token O(1) streaming step during autoregressive decoding"""
        res = x_t
        normed = self.norm1(x_t)
        conv_x = self.causal_conv(normed)
        B, L, D = conv_x.shape # L=1
        
        theta_k = self.w_k(conv_x).view(B, 1, self.n_heads, self.d_k)
        theta_q = self.w_q(conv_x).view(B, 1, self.n_heads, self.d_k)
        v = self.w_v(conv_x).view(B, 1, self.n_heads, self.d_k)
        beta = torch.sigmoid(self.w_beta(conv_x)).view(B, 1, self.n_heads, 1, 1)
        lam = (0.85 + 0.149 * torch.sigmoid(self.w_lambda(conv_x))).view(B, 1, self.n_heads, 1, 1)
        
        K = torch.polar(torch.ones_like(theta_k), theta_k)
        Q = torch.polar(torch.ones_like(theta_q), theta_q)
        
        if memory_state is None:
            M = torch.zeros(B, self.n_heads, self.d_k, self.d_k, dtype=torch.complex64, device=x_t.device)
        else:
            M = memory_state
            
        ret_c, M_next = _scan_chunk(K, Q, v, beta, lam, M, self.inv_dk)
        retrieved = ret_c.view(B, 1, D)
        retrieved_norm = self.norm_retrieved(retrieved)
        
        x = res + self.out_proj(retrieved_norm)
        x = x + self.ffn(self.norm2(x))
        return x, M_next
