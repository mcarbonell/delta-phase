"""
===========================================================================
P0-1: Capacity-Matched MQAR Control — Colab/Kaggle Self-Contained Notebook
===========================================================================
Upload this .py to Colab/Kaggle and run:
    !python benchmark_capacity_matched_colab.py --device cuda

Or in a notebook cell:
    %run benchmark_capacity_matched_colab.py --device cuda --steps 1500

Everything is self-contained — no external imports from delta_phase/.
GPU recommended (T4/P100 on free tier, A100 on Pro+).

Experiment:  Isolate phasor geometry vs 2x capacity in MQAR.
Arms:
  1. DeltaPhase Complex d_k=32  (state: 2*32^2 = 2048 floats/head)
  2. Gated DeltaNet Real d_k=32 (state: 32^2 = 1024 floats/head)  [reference]
  3. Gated DeltaNet Real d_k=45 (state: 45^2 = 2025 floats/head)  [iso-floats control]
  4. Causal Transformer Softmax  (control+)

Per-arm LR sweep: {1e-3, 3e-3, 5e-3}, best selected via 300-step pilot.
5 seeds, pair sweep {8, 16, 32}, 1500 steps, early stop @ 99.5%.
===========================================================================
"""

import os
import sys
import time
import math
import json
import random
import datetime
import platform
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# SELF-CONTAINED SPECTRAL UTILITIES (from delta_phase/spectral.py)
# ==============================================================================

def create_hadamard_matrix(n: int) -> torch.Tensor:
    H = torch.tensor([[1.0]], dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)
    return H / math.sqrt(n)

def create_dct2_matrix(n: int) -> torch.Tensor:
    C = torch.zeros((n, n), dtype=torch.float32)
    for k in range(n):
        for i in range(n):
            if k == 0:
                C[k, i] = 1.0 / math.sqrt(n)
            else:
                C[k, i] = math.sqrt(2.0 / n) * math.cos(
                    math.pi * k * (2 * i + 1) / (2.0 * n)
                )
    return C

def create_haar_matrix(n: int) -> torch.Tensor:
    if n == 1:
        return torch.tensor([[1.0]], dtype=torch.float32)
    H_sub = create_haar_matrix(n // 2)
    low = torch.cat([H_sub, H_sub], dim=1) / math.sqrt(2)
    high = torch.zeros((n // 2, n), dtype=torch.float32)
    for i in range(n // 2):
        high[i, 2 * i] = 1.0 / math.sqrt(2)
        high[i, 2 * i + 1] = -1.0 / math.sqrt(2)
    return torch.cat([low, high], dim=0)


# ==============================================================================
# SELF-CONTAINED LAYERS (from delta_phase/layers.py — only what's needed)
# ==============================================================================

class ShortCausalConv1D(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=d_model, out_channels=d_model,
            kernel_size=kernel_size, padding=kernel_size - 1,
            groups=d_model
        )
        self.act = nn.SiLU()

    def forward(self, x):
        B, L, D = x.shape
        conv_out = self.conv(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
        return x + self.act(conv_out)


class LearnableSubstrateLerpFFN(nn.Module):
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

    def forward(self, x):
        weights = F.softmax(self.substrate_logits, dim=0)

        h_fwht = F.linear(x, self.mat_fwht).unsqueeze(-2)
        outs_fwht = (torch.cos(h_fwht + self.phi1_fwht) * self.w1_fwht +
                     torch.sin(h_fwht + self.phi2_fwht) * self.w2_fwht).flatten(-2)
        out_fwht = F.linear(self.combine(outs_fwht), self.mat_fwht.t())

        h_dct = F.linear(x, self.mat_dct).unsqueeze(-2)
        outs_dct = (torch.cos(h_dct + self.phi1_dct) * self.w1_dct +
                    torch.sin(h_dct + self.phi2_dct) * self.w2_dct).flatten(-2)
        out_dct = F.linear(self.combine(outs_dct), self.mat_dct.t())

        h_haar = F.linear(x, self.mat_haar).unsqueeze(-2)
        outs_haar = (torch.cos(h_haar + self.phi1_haar) * self.w1_haar +
                     torch.sin(h_haar + self.phi2_haar) * self.w2_haar).flatten(-2)
        out_haar = F.linear(self.combine(outs_haar), self.mat_haar.t())

        return weights[0] * out_fwht + weights[1] * out_dct + weights[2] * out_haar


class DeltaPhaseHolographicBlock(nn.Module):
    def __init__(self, d_model, n_heads=8, conv_kernel_size=4,
                 chunk_size=64, num_banks=4):
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

        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = LearnableSubstrateLerpFFN(d_model, num_banks=num_banks)

    def forward(self, x, memory_state=None):
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
        if pad_len > 0:
            mask = torch.ones(B, self.n_heads, L_padded, device=x.device)
            mask[:, :, L:] = 0.0
            beta = beta * mask

        theta_k_f = theta_k.float()
        theta_q_f = theta_q.float()
        K = torch.complex(torch.cos(theta_k_f), torch.sin(theta_k_f))
        Q = torch.complex(torch.cos(theta_q_f), torch.sin(theta_q_f))

        num_chunks = L_padded // C
        Q_c = Q.view(B, self.n_heads, num_chunks, C, self.d_k)
        K_c = K.view(B, self.n_heads, num_chunks, C, self.d_k)
        V_c = v.view(B, self.n_heads, num_chunks, C, self.d_k)
        beta_c = beta.view(B, self.n_heads, num_chunks, C)

        Gram_real = torch.matmul(K_c, torch.conj(K_c).transpose(-1, -2)).real * inv_dk
        L_mat = torch.triu(Gram_real * beta_c.unsqueeze(-1), diagonal=1)
        I_mat = torch.eye(C, device=x.device).view(1, 1, 1, C, C)
        T_mat = torch.linalg.solve_triangular(
            I_mat + L_mat.transpose(-1, -2), I_mat, upper=False
        )

        complex_dtype = torch.complex64
        if memory_state is None:
            M_state = torch.zeros(B, self.n_heads, self.d_k, self.d_k,
                                  dtype=complex_dtype, device=x.device)
        else:
            M_state = memory_state

        out_chunks = []
        for c in range(num_chunks):
            qc, kc, vc, bc, tc = (Q_c[:, :, c], K_c[:, :, c], V_c[:, :, c],
                                   beta_c[:, :, c], T_mat[:, :, c])
            v_old = torch.matmul(
                M_state, torch.conj(kc).transpose(-1, -2)
            ).real.transpose(-1, -2) * inv_dk
            E_c = torch.matmul(tc, vc - v_old)
            U_c = bc.unsqueeze(-1) * E_c
            o_inter = torch.matmul(
                M_state, torch.conj(qc).transpose(-1, -2)
            ).real.transpose(-1, -2) * inv_dk
            A_intra = torch.tril(
                torch.matmul(qc, torch.conj(kc).transpose(-1, -2)).real
            ) * inv_dk
            out_chunks.append(torch.matmul(A_intra, U_c) + o_inter)
            M_state = M_state + torch.matmul(
                U_c.to(complex_dtype).transpose(-1, -2), kc
            )

        retrieved = torch.cat(out_chunks, dim=2)[:, :, :L].transpose(1, 2).reshape(B, L, D)
        retrieved_norm = self.norm_retrieved(retrieved)
        x = res + self.out_proj(retrieved_norm)
        x = x + self.ffn(self.norm2(x))
        return x, M_state


# ==============================================================================
# DATA GENERATOR (Zoology MQAR standard)
# ==============================================================================

PAD_ID = 0
QUERY_MARKER = 1
TOKEN_OFFSET = 2
NUM_CONTENT_TOKENS = 512
VOCAB_SIZE = TOKEN_OFFSET + NUM_CONTENT_TOKENS


def generate_zoology_mqar_batch(batch_size=32, seq_len=128, num_pairs=8,
                                 vocab_size=VOCAB_SIZE, device='cpu'):
    num_tokens = vocab_size - TOKEN_OFFSET
    gap = 2
    assert 2 * num_pairs + gap + 2 * num_pairs <= seq_len

    rand_t = torch.rand(batch_size, num_tokens, device=device)
    sampled = torch.argsort(rand_t, dim=-1)[:, :2 * num_pairs] + TOKEN_OFFSET
    keys = sampled[:, :num_pairs]
    vals = sampled[:, num_pairs:]

    x = torch.full((batch_size, seq_len), PAD_ID, dtype=torch.long, device=device)
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)

    kv = torch.stack([keys, vals], dim=2).view(batch_size, 2 * num_pairs)
    x[:, :2 * num_pairs] = kv

    q_perm = torch.argsort(torch.rand(batch_size, num_pairs, device=device), dim=-1)
    query_keys = torch.gather(keys, 1, q_perm)
    query_vals = torch.gather(vals, 1, q_perm)

    pos_q = (2 * num_pairs + gap + 2 * torch.arange(num_pairs, device=device)
             ).unsqueeze(0).expand(batch_size, -1)

    x.scatter_(1, pos_q, QUERY_MARKER)
    x.scatter_(1, pos_q + 1, query_keys)
    y.scatter_(1, pos_q + 1, query_vals)
    return x, y


# ==============================================================================
# MODELS
# ==============================================================================

class DeltaPhaseMQAR(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=128, n_heads=4,
                 chunk_size=32, num_layers=2, max_len=4096):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            DeltaPhaseHolographicBlock(d_model=d_model, n_heads=n_heads,
                                       chunk_size=chunk_size)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_emb(pos)
        for block in self.blocks:
            h, _ = block(h)
        return self.head(h)


class RealGatedDeltaNetBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_k=None,
                 chunk_size=32, conv_kernel_size=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else d_model // n_heads
        self.internal_dim = self.n_heads * self.d_k
        self.chunk_size = chunk_size

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=conv_kernel_size)

        self.w_k = nn.Linear(d_model, self.internal_dim, bias=False)
        self.w_q = nn.Linear(d_model, self.internal_dim, bias=False)
        self.w_v = nn.Linear(d_model, self.internal_dim, bias=False)
        self.w_beta = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(self.internal_dim, d_model, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        res = x
        normed = self.norm1(x)
        conv_x = self.causal_conv(normed)
        B, L, D = conv_x.shape
        C = self.chunk_size

        pad_len = (C - (L % C)) % C
        if pad_len > 0:
            conv_x = F.pad(conv_x, (0, 0, 0, pad_len))
            L_padded = L + pad_len
        else:
            L_padded = L

        k = self.w_k(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        q = self.w_q(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        beta = 2.0 * torch.sigmoid(self.w_beta(conv_x)).transpose(1, 2)

        k = F.normalize(k, p=2, dim=-1)
        q = F.normalize(q, p=2, dim=-1)

        if pad_len > 0:
            mask = torch.ones(B, self.n_heads, L_padded, device=x.device)
            mask[:, :, L:] = 0.0
            beta = beta * mask

        num_chunks = L_padded // C
        Q_c = q.view(B, self.n_heads, num_chunks, C, self.d_k)
        K_c = k.view(B, self.n_heads, num_chunks, C, self.d_k)
        V_c = v.view(B, self.n_heads, num_chunks, C, self.d_k)
        beta_c = beta.view(B, self.n_heads, num_chunks, C)

        Gram = torch.matmul(K_c, K_c.transpose(-1, -2))
        L_mat = torch.triu(Gram * beta_c.unsqueeze(-1), diagonal=1)
        I_mat = torch.eye(C, device=x.device).view(1, 1, 1, C, C)
        T_mat = torch.linalg.solve_triangular(
            I_mat + L_mat.transpose(-1, -2), I_mat, upper=False
        )

        M_state = torch.zeros(B, self.n_heads, self.d_k, self.d_k,
                              device=x.device, dtype=x.dtype)
        out_chunks = []
        for c in range(num_chunks):
            qc, kc, vc, bc, tc = (Q_c[:, :, c], K_c[:, :, c], V_c[:, :, c],
                                   beta_c[:, :, c], T_mat[:, :, c])
            v_old = torch.matmul(M_state, kc.transpose(-1, -2)).transpose(-1, -2)
            E_c = torch.matmul(tc, vc - v_old)
            U_c = bc.unsqueeze(-1) * E_c
            o_inter = torch.matmul(M_state, qc.transpose(-1, -2)).transpose(-1, -2)
            A_intra = torch.tril(torch.matmul(qc, kc.transpose(-1, -2)))
            out_chunks.append(torch.matmul(A_intra, U_c) + o_inter)
            M_state = M_state + torch.matmul(U_c.transpose(-1, -2), kc)

        retrieved = torch.cat(out_chunks, dim=2)[:, :, :L].transpose(1, 2)
        retrieved = retrieved.reshape(B, L, self.internal_dim)
        x = res + self.out_proj(retrieved)
        x = x + self.mlp(self.norm2(x))
        return x


class RealGatedDeltaNetMQAR(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=128, n_heads=4,
                 d_k=None, chunk_size=32, num_layers=2, max_len=4096):
        super().__init__()
        self.d_k = d_k if d_k is not None else d_model // n_heads
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            RealGatedDeltaNetBlock(d_model=d_model, n_heads=n_heads,
                                   d_k=self.d_k, chunk_size=chunk_size)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


class CausalMHABlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, conv_kernel_size=4):
        super().__init__()
        self.conv = ShortCausalConv1D(d_model, kernel_size=conv_kernel_size)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        x = self.conv(x)
        res = x
        norm_x = self.norm1(x)
        L = x.shape[1]
        causal_mask = torch.triu(
            torch.full((L, L), float('-inf'), device=x.device), diagonal=1
        )
        attn_out, _ = self.mha(norm_x, norm_x, norm_x,
                                attn_mask=causal_mask, is_causal=False)
        x = res + attn_out
        return x + self.ffn(self.norm2(x))


class CausalTransformerMQAR(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=128, n_heads=4,
                 num_layers=2, max_len=4096):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            CausalMHABlock(d_model=d_model, n_heads=n_heads)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# ==============================================================================
# EVALUATION & TRAINING
# ==============================================================================

def evaluate_mqar_accuracy(model, num_eval_batches=15, batch_size=32,
                            seq_len=128, num_pairs=8, vocab_size=VOCAB_SIZE,
                            device='cpu'):
    model.eval()
    total_correct = 0
    total_queries = 0
    with torch.no_grad():
        for _ in range(num_eval_batches):
            tokens, targets = generate_zoology_mqar_batch(
                batch_size=batch_size, seq_len=seq_len,
                num_pairs=num_pairs, vocab_size=vocab_size, device=device
            )
            logits = model(tokens)
            preds = torch.argmax(logits, dim=-1)
            mask = (targets != -100)
            total_correct += (preds[mask] == targets[mask]).sum().item()
            total_queries += mask.sum().item()
    return (total_correct / max(total_queries, 1)) * 100.0


def fmt(seconds):
    m, s = int(seconds // 60), int(seconds % 60)
    if m >= 60:
        return f"{m//60:02d}h {m%60:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def train_model(name, model, total_steps=1500, batch_size=32, seq_len=128,
                num_pairs=8, vocab_size=VOCAB_SIZE, lr=3e-3, device='cpu',
                log_interval=50, early_stop_acc=99.5,
                idx=1, total=1, t0_global=0.0):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    crit = nn.CrossEntropyLoss(ignore_index=-100)
    model.train()
    t0 = time.time()

    print(f"\n[{time.strftime('%H:%M:%S')}] [{idx}/{total}] {name} (lr={lr:.1e})",
          flush=True)
    print(f"   L={seq_len}, N_pairs={num_pairs}, max {total_steps} steps | "
          f"Elapsed: {fmt(time.time()-t0_global)}", flush=True)

    rloss = 0.0
    stimes = []
    s50 = s95 = None
    final_step = total_steps
    stopped = False
    best_acc = 0.0

    for step in range(1, total_steps + 1):
        st = time.time()
        tok, tgt = generate_zoology_mqar_batch(
            batch_size=batch_size, seq_len=seq_len, num_pairs=num_pairs,
            vocab_size=vocab_size, device=device
        )
        opt.zero_grad()
        logits = model(tok)
        loss = crit(logits.view(-1, vocab_size), tgt.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        dt = time.time() - st
        stimes.append(dt)
        if len(stimes) > 50: stimes.pop(0)
        rloss += loss.item()

        if step % log_interval == 0 or step == total_steps:
            al = rloss / min(step, log_interval)
            va = evaluate_mqar_accuracy(model, 10, batch_size, seq_len,
                                        num_pairs, vocab_size, device)
            best_acc = max(best_acc, va)
            if s50 is None and va >= 50: s50 = step
            if s95 is None and va >= 95: s95 = step
            avg_t = sum(stimes) / len(stimes)
            eta = (total_steps - step) * avg_t
            print(f"   [{time.strftime('%H:%M:%S')}] {step:4d}/{total_steps} "
                  f"({step/total_steps*100:5.1f}%) | L:{al:6.4f} | "
                  f"Acc:{va:6.2f}% | ETA:{fmt(eta)}", flush=True)
            rloss = 0.0
            if early_stop_acc and va >= early_stop_acc:
                final_step = step
                stopped = True
                print(f"   [EARLY STOP] {va:.2f}% @ step {step}", flush=True)
                break
            model.train()

    wc = time.time() - t0
    fa = evaluate_mqar_accuracy(model, 20, batch_size, seq_len,
                                 num_pairs, vocab_size, device)
    best_acc = max(best_acc, fa)
    tag = "ES" if stopped else "Full"
    print(f"   [{name}] {tag} step {final_step} | {fmt(wc)} | "
          f"Acc:{fa:.2f}%", flush=True)
    return model, {
        "wallclock": wc, "final_step": final_step,
        "step_to_50": s50 or total_steps, "step_to_95": s95 or total_steps,
        "early_stopped": stopped, "final_acc": fa, "best_acc": best_acc
    }


def select_best_lr(arm_key, factory, lrs, pilot_seed=42, pilot_steps=300,
                    bs=32, seq_len=128, npairs=8, vs=VOCAB_SIZE, dev='cpu'):
    print(f"\n{'~'*70}", flush=True)
    print(f"  LR PILOT [{arm_key}]: {lrs}, seed={pilot_seed}, "
          f"{pilot_steps} steps", flush=True)
    best_lr, best_a = lrs[0], -1.0
    for lr in lrs:
        torch.manual_seed(pilot_seed)
        np.random.seed(pilot_seed)
        random.seed(pilot_seed)
        m = factory()
        _, met = train_model(f"{arm_key} pilot lr={lr:.1e}", m,
                              pilot_steps, bs, seq_len, npairs, vs, lr, dev,
                              pilot_steps, None, 0, 0, 0.0)
        a = met["best_acc"]
        print(f"     lr={lr:.1e} => {a:.2f}%", flush=True)
        if a > best_a: best_a, best_lr = a, lr
    print(f"  => Best LR [{arm_key}]: {best_lr:.1e} ({best_a:.2f}%)", flush=True)
    return best_lr


# ==============================================================================
# MAIN SUITE
# ==============================================================================

def run_suite(seeds=(42, 137, 2024, 7, 999), pair_sweep=(8, 16, 32),
              steps=1500, es_acc=99.5, lr_cands=(1e-3, 3e-3, 5e-3),
              pilot_steps=300, device='cpu'):

    t0g = time.time()
    d_model, n_heads, nl = 128, 4, 2
    dk_c, dk_iso = d_model // n_heads, 45  # 32 vs 45

    arms = {
        "DeltaPhase_Complex_dk32": {
            "dk": dk_c, "sfh": 2*dk_c**2,
            "f": lambda: DeltaPhaseMQAR(VOCAB_SIZE, d_model, n_heads, 32, nl)
        },
        "GatedDeltaNet_Real_dk32": {
            "dk": dk_c, "sfh": dk_c**2,
            "f": lambda: RealGatedDeltaNetMQAR(VOCAB_SIZE, d_model, n_heads, dk_c, 32, nl)
        },
        "GatedDeltaNet_Real_dk45_ISO": {
            "dk": dk_iso, "sfh": dk_iso**2,
            "f": lambda: RealGatedDeltaNetMQAR(VOCAB_SIZE, d_model, n_heads, dk_iso, 32, nl)
        },
        "Transformer_Causal": {
            "dk": dk_c, "sfh": "KV-cache",
            "f": lambda: CausalTransformerMQAR(VOCAB_SIZE, d_model, n_heads, nl)
        },
    }
    akeys = list(arms.keys())

    # Header
    print("=" * 110, flush=True)
    print("P0-1: CAPACITY-MATCHED MQAR CONTROL EXPERIMENT", flush=True)
    print("Goal: Isolate phasor geometry benefit vs 2x capacity", flush=True)
    print("=" * 110, flush=True)
    dev_name = "CUDA " + torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else device.upper()
    print(f"  Device:    {dev_name}", flush=True)
    print(f"  Date UTC:  {datetime.datetime.now(datetime.timezone.utc).isoformat()}", flush=True)
    print(f"  PyTorch:   {torch.__version__}", flush=True)
    print(f"  Seeds:     {list(seeds)}", flush=True)
    print(f"  Pairs:     {list(pair_sweep)}", flush=True)
    print(f"  Steps:     {steps} (ES@{es_acc}%)", flush=True)
    print(f"  LR cands:  {list(lr_cands)}", flush=True)

    print(f"\n  {'Arm':<38s} | {'dk':>3} | {'Floats/head':>11s} | {'Params':>10s}",
          flush=True)
    print(f"  {'─'*38} | {'─'*3} | {'─'*11} | {'─'*10}", flush=True)
    for ak, sp in arms.items():
        m = sp["f"]()
        tp = sum(p.numel() for p in m.parameters())
        sfh = f"{sp['sfh']:,}" if isinstance(sp['sfh'], int) else sp['sfh']
        print(f"  {ak:<38s} | {sp['dk']:>3} | {sfh:>11s} | {tp:>10,}", flush=True)
        del m

    print(f"\n  Complex dk=32: {2*dk_c**2} fl/h | Real dk=45: {dk_iso**2} fl/h "
          f"(ratio: {dk_iso**2/(2*dk_c**2):.3f})", flush=True)
    print("=" * 110, flush=True)

    all_res = {}
    for npairs in pair_sweep:
        L_tr = 128 if npairs <= 16 else 256
        evals = [L_tr, 2*L_tr, 4*L_tr]
        bk = f"pairs_{npairs}"
        all_res[bk] = {"L_train": L_tr, "eval_lengths": evals}

        print(f"\n{'#'*110}", flush=True)
        print(f"  BLOCK N_PAIRS={npairs} | L_train={L_tr} | Eval={evals}", flush=True)
        print(f"{'#'*110}", flush=True)

        # LR pilot
        blrs = {}
        for ak in akeys:
            if pilot_steps > 0:
                blrs[ak] = select_best_lr(ak, arms[ak]["f"], list(lr_cands),
                                           42, pilot_steps, 32, L_tr, npairs,
                                           VOCAB_SIZE, device)
            else:
                blrs[ak] = lr_cands[0] if len(lr_cands) == 1 else 3e-3

        print(f"\n  Selected LRs for N_pairs={npairs}:", flush=True)
        for ak, lr in blrs.items():
            print(f"    {ak:<38s}: {lr:.1e}", flush=True)

        total_runs = len(akeys) * len(seeds)
        rc = 0
        for ak in akeys:
            sp = arms[ak]
            lr = blrs[ak]
            all_res[bk][ak] = {
                "lr": lr, "dk": sp["dk"], "sfh": sp["sfh"],
                "wc": [], "s50": [], "s95": [], "fa": [], "ba": [],
                "la": {L: [] for L in evals}
            }
            for si, seed in enumerate(seeds, 1):
                rc += 1
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                model = sp["f"]()
                tm, met = train_model(
                    f"{ak} [S{seed}]", model, steps, 32, L_tr, npairs,
                    VOCAB_SIZE, lr, device, 50, es_acc, rc, total_runs, t0g
                )
                d = all_res[bk][ak]
                d["wc"].append(met["wallclock"])
                d["s50"].append(met["step_to_50"])
                d["s95"].append(met["step_to_95"])
                d["fa"].append(met["final_acc"])
                d["ba"].append(met["best_acc"])

                print(f"   Eval lengths {evals}...", flush=True)
                for Le in evals:
                    a = evaluate_mqar_accuracy(tm, 20, 32, Le, npairs,
                                               VOCAB_SIZE, device)
                    d["la"][Le].append(a)
                    print(f"      L={Le:4d}: {a:6.2f}%", flush=True)
                del tm, model
                if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n\n{'='*130}", flush=True)
    print(f"  SUMMARY TABLE P0-1 (mean +/- SE, {len(seeds)} seeds)", flush=True)
    print(f"{'='*130}", flush=True)
    print(f"  {'NP':<4} | {'Arm':<38s} | {'dk':>3} | {'Fl/h':>6} | "
          f"{'LR':>7} | {'L_train':>14s} | {'OOD 2x':>14s} | "
          f"{'OOD 4x':>14s} | {'S>95%':>6}", flush=True)
    print(f"  {'─'*128}", flush=True)

    summary = {}
    for npairs in pair_sweep:
        bk = f"pairs_{npairs}"
        bl = all_res[bk]
        evl = bl["eval_lengths"]
        summary[bk] = {}
        for ak in akeys:
            if ak not in bl: continue
            d = bl[ak]
            ss = []
            for L in evl:
                v = d["la"][L]
                m = np.mean(v)
                se = np.std(v)/math.sqrt(len(v)) if len(v)>1 else 0.0
                ss.append(f"{m:5.2f}+/-{se:4.2f}")
            s95m = np.mean(d["s95"])
            sfh = f"{d['sfh']:,}" if isinstance(d['sfh'], int) else str(d['sfh'])
            print(f"  {npairs:<4} | {ak:<38s} | {d['dk']:>3} | {sfh:>6} | "
                  f"{d['lr']:.1e} | {ss[0]:>14s} | {ss[1]:>14s} | "
                  f"{ss[2]:>14s} | {s95m:>6.0f}", flush=True)
            summary[bk][ak] = {
                "dk": d["dk"], "sfh": d["sfh"], "lr": d["lr"],
                "accs": {str(L): d["la"][L] for L in evl},
                "acc_summary": {str(L): ss[i] for i, L in enumerate(evl)},
                "s95_mean": s95m
            }
        print(f"  {'─'*128}", flush=True)

    # Diagnostic
    print(f"\n  CONFOUND DIAGNOSIS:", flush=True)
    for npairs in pair_sweep:
        bk = f"pairs_{npairs}"
        bl = all_res[bk]
        Lt = bl["eval_lengths"][0]
        ca = bl.get("DeltaPhase_Complex_dk32", {}).get("la", {}).get(Lt, [])
        r32 = bl.get("GatedDeltaNet_Real_dk32", {}).get("la", {}).get(Lt, [])
        r45 = bl.get("GatedDeltaNet_Real_dk45_ISO", {}).get("la", {}).get(Lt, [])
        if ca and r32 and r45:
            cm, rm, im = np.mean(ca), np.mean(r32), np.mean(r45)
            g_orig = cm - rm
            g_iso = cm - im
            print(f"\n    N_pairs={npairs}:", flush=True)
            print(f"      Complex dk=32: {cm:.2f}%", flush=True)
            print(f"      Real dk=32:    {rm:.2f}% (gap: {g_orig:+.2f}pp)", flush=True)
            print(f"      Real dk=45:    {im:.2f}% (gap: {g_iso:+.2f}pp)", flush=True)
            if g_iso > 2.0:
                print(f"      => H1 SUPPORTED: phasor benefit persists "
                      f"({g_iso:+.2f}pp)", flush=True)
            elif g_iso > 0.0:
                print(f"      => WEAK SIGNAL: residual {g_iso:+.2f}pp", flush=True)
            else:
                print(f"      => H0 SUPPORTED: iso-floats real closes gap "
                      f"({g_iso:+.2f}pp)", flush=True)
            cap_pct = max(0, min(100, (1 - g_iso/max(g_orig, 1e-6)) * 100))
            print(f"      Capacity explains ~{cap_pct:.0f}% of original gap",
                  flush=True)

    # Save
    twc = time.time() - t0g
    out_path = "capacity_matched_mqar_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "P0-1 Capacity-Matched MQAR",
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "seeds": list(seeds), "pairs": list(pair_sweep),
            "lr_candidates": list(lr_cands), "steps": steps,
            "device": device, "total_sec": twc,
            "summary": summary
        }, f, indent=2, default=str)
    print(f"\n  Results saved: {os.path.abspath(out_path)}", flush=True)
    print(f"  Total time: {fmt(twc)} ({twc:.0f}s)", flush=True)
    print("=" * 110, flush=True)


# ==============================================================================
# CONFIGURACIÓN PARA COLAB / JUPYTER (Edita aquí si quieres antes de ejecutar)
# ==============================================================================
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "steps": 3000,                         # 1500 pasos recomendados para convergencia
    "seeds": (42, 137, 2024, 7, 999),       # 5 semillas para Nivel 2
    "pairs": (8, 16, 32),                   # Barrido de 8, 16 y 32 pares K-V
    "early_stop_acc": 99.5,                 # Parar temprano si llega a 99.5%
    "lr_candidates": (1e-3, 3e-3, 5e-3),    # Mini-sweep de Learning Rate por brazo
    "pilot_steps": 300,                     # Pasos para elegir el mejor LR
    "quick_smoke_test": False               # Cambiar a True para prueba rápida de 2 min
}

if __name__ == '__main__':
    # Detectar si se pasan argumentos de terminal o si corre directo en notebook
    in_notebook = 'ipykernel' in sys.modules or 'google.colab' in sys.modules
    
    if not in_notebook and len(sys.argv) > 1:
        p = argparse.ArgumentParser(description="P0-1 Capacity-Matched MQAR")
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--steps", type=int, default=1500)
        p.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 2024, 7, 999])
        p.add_argument("--pairs", type=int, nargs="+", default=[8, 16, 32])
        p.add_argument("--es", type=float, default=99.5)
        p.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 3e-3, 5e-3])
        p.add_argument("--pilot-steps", type=int, default=300)
        p.add_argument("--quick", action="store_true")
        a, _ = p.parse_known_args()

        if a.quick:
            run_suite((42,), (32,), 300, a.es, (3e-3,), 0, a.device)
        else:
            run_suite(tuple(a.seeds), tuple(a.pairs), a.steps, a.es,
                      tuple(a.lrs), a.pilot_steps, a.device)
    else:
        # Modo celda directa de Colab/Kaggle
        print(f"🚀 Ejecutando benchmark directamente en Colab/Notebook...")
        if CONFIG["quick_smoke_test"]:
            print("⚠️ Modo Quick Smoke Test activado.")
            run_suite((42,), (32,), 300, CONFIG["early_stop_acc"], (3e-3,), 0, CONFIG["device"])
        else:
            run_suite(
                CONFIG["seeds"],
                CONFIG["pairs"],
                CONFIG["steps"],
                CONFIG["early_stop_acc"],
                CONFIG["lr_candidates"],
                CONFIG["pilot_steps"],
                CONFIG["device"]
            )

