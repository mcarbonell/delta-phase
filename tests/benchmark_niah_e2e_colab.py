"""
===========================================================================
P0-2: End-to-End NIAH with Randomized Needles & Learned Gating — Colab/Kaggle
===========================================================================
Upload/paste this script into a single cell in Google Colab (with GPU T4/A100)
and run directly (Shift + Enter).

Zero external imports — 100% self-contained in pure PyTorch.

Audit Remediation (R2 from docs/project_audit_2026-08.md):
  1. Re-randomizes needle identity (key, value) on EVERY trial (keys 1..32, vals 33..96).
  2. Data-dependent gating beta_t is LEARNED end-to-end (NOT an oracle simulation).
  3. Evaluates selective gating (learned) vs fixed-beta (beta=1.0) ablation control.
  4. Tests zero-shot length extrapolation up to 16,384+ tokens across 5 depths (10%, 25%, 50%, 75%, 90%).
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
# CONFIGURACIÓN PARA COLAB / JUPYTER (Edita aquí antes de ejecutar)
# ==============================================================================
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "steps": 800,                          # 800 pasos para convergencia de lectura/escritura
    "train_len": 128,                       # Longitud de entrenamiento base (MQAR dinámico)
    "train_pairs": 8,                       # Pares K-V por secuencia durante entrenamiento
    "batch_size": 32,                       # Tamaño de batch
    "lr": 4e-3,                             # Tasa de aprendizaje
    "early_stop_acc": 99.0,                 # Parar temprano si llega a 99.0% en train
    "eval_lengths": [256, 512, 1024, 2048, 4096, 8192, 16384], # Longitudes NIAH OOD
    "depths": [0.10, 0.25, 0.50, 0.75, 0.90], # Profundidades de aguja (%)
    "trials_per_cell": 20,                  # Ensayos aleatorios por celda
    "seeds": (42, 137, 2024),               # 3 semillas independientes
    "arms": ["learned", "fixed"],           # Gating aprendido vs Gating fijo (beta=1)
    "quick_smoke_test": False               # True para prueba rápida de 2 min
}

VOCAB_SIZE = 129
QUERY_MARKER = 128
KEY_LO, KEY_HI = 1, 33      # Keys: 1..32
VAL_LO, VAL_HI = 33, 97     # Values: 33..96
NOISE_LO, NOISE_HI = 97, 128 # Noise filler tokens

# ==============================================================================
# SELF-CONTAINED SPECTRAL UTILITIES
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
                C[k, i] = math.sqrt(2.0 / n) * math.cos(math.pi * k * (2 * i + 1) / (2.0 * n))
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
# SELF-CONTAINED DELTAPHASE LAYERS & MODEL
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
    def __init__(self, d_model: int, num_banks: int = 2):
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
    def __init__(self, d_model, n_heads=4, conv_kernel_size=4,
                 chunk_size=32, num_banks=2, beta_mode="learned"):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.inv_dk = 1.0 / float(self.d_k)
        self.chunk_size = chunk_size
        self.beta_mode = beta_mode  # "learned" or "fixed"

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
        self.last_beta = None

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

        if self.beta_mode == "fixed":
            beta = torch.ones(B, self.n_heads, L_padded, device=x.device, dtype=x.dtype)
        else:
            beta = 2.0 * torch.sigmoid(self.w_beta(conv_x)).transpose(1, 2)

        self.last_beta = beta[:, :, :L].detach()

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


class DeltaPhaseNIAHModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=128, n_heads=4,
                 chunk_size=32, num_layers=2, beta_mode="learned"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            DeltaPhaseHolographicBlock(d_model=d_model, n_heads=n_heads,
                                       chunk_size=chunk_size, num_banks=2,
                                       beta_mode=beta_mode)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.embedding(x)
        for block in self.blocks:
            h, _ = block(h)
        return self.head(h)


# ==============================================================================
# DATA GENERATORS
# ==============================================================================

def generate_training_batch(batch_size, seq_len=128, num_pairs=8, device='cpu'):
    """
    Entrenamiento dinámico en MQAR (Zoology standard):
    - Primera mitad: Pares K-V contiguos [K, V]
    - Segunda mitad: Consultas aleatorias con marcador [QUERY_MARKER, K] -> Target V
    - Supervisión estricta en posiciones de consulta.
    """
    tokens = torch.randint(NOISE_LO, NOISE_HI, (batch_size, seq_len), device=device)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)

    half_len = seq_len // 2
    pair_spacing = max(2, half_len // num_pairs)
    query_spacing = max(2, (seq_len - half_len) // num_pairs)

    for b in range(batch_size):
        chosen_keys = torch.randperm(KEY_HI - KEY_LO)[:num_pairs] + KEY_LO
        chosen_vals = torch.randperm(VAL_HI - VAL_LO)[:num_pairs] + VAL_LO

        for p in range(num_pairs):
            pos = p * pair_spacing
            tokens[b, pos] = chosen_keys[p]
            tokens[b, pos + 1] = chosen_vals[p]

        perm = torch.randperm(num_pairs)
        for q in range(num_pairs):
            q_pos = half_len + q * query_spacing
            k_idx = perm[q]
            tokens[b, q_pos] = QUERY_MARKER
            tokens[b, q_pos + 1] = chosen_keys[k_idx]
            targets[b, q_pos + 1] = chosen_vals[k_idx]

    return tokens, targets


def generate_niah_single_eval(seq_len, depth, device='cpu'):
    """
    Genera 1 secuencia NIAH con aguja 100% ALEATORIA:
    - Aguja (k_needle, v_needle) aleatoria e inédita en cada invocación.
    - Insertada a la profundidad 'depth' (10%..90%).
    - Relleno de ruido aleatorio (97..128).
    - Consulta final: [..., QUERY_MARKER, k_needle] -> Objetivo v_needle.
    """
    seq = torch.randint(NOISE_LO, NOISE_HI, (1, seq_len), device=device)
    k_needle = int(torch.randint(KEY_LO, KEY_HI, (1,)).item())
    v_needle = int(torch.randint(VAL_LO, VAL_HI, (1,)).item())

    max_pos = seq_len - 4
    pos = max(2, min(int(round(depth * max_pos)), max_pos))
    pos -= pos % 2

    seq[0, pos] = k_needle
    seq[0, pos + 1] = v_needle
    seq[0, -2] = QUERY_MARKER
    seq[0, -1] = k_needle

    return seq, v_needle


# ==============================================================================
# EVALUATION & TRAINING FUNCTIONS
# ==============================================================================

def train_arm(beta_mode, seed, steps, train_len, train_pairs, batch_size, lr, early_stop_acc, device, log_interval=50):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = DeltaPhaseNIAHModel(vocab_size=VOCAB_SIZE, d_model=128, n_heads=4,
                                 chunk_size=32, num_layers=2, beta_mode=beta_mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    t0 = time.time()
    final_step = steps
    for step in range(1, steps + 1):
        x, y = generate_training_batch(batch_size, train_len, num_pairs=train_pairs, device=device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % log_interval == 0 or step == steps:
            mask = (y != -100)
            preds = logits.argmax(dim=-1)
            acc = (preds[mask] == y[mask]).float().mean().item() * 100.0
            print(f"   [{beta_mode:>7} | seed {seed}] paso {step:4d}/{steps} | loss {loss.item():6.4f} | train acc {acc:6.2f}% | {time.time()-t0:7.1f}s", flush=True)
            if acc >= early_stop_acc and step >= 200:
                final_step = step
                print(f"   [Early Stop] ({acc:.2f}% >= {early_stop_acc}%) at step {step}", flush=True)
                break

    wallclock = time.time() - t0
    # In-distribution validation
    with torch.no_grad():
        x_val, y_val = generate_training_batch(64, train_len, num_pairs=train_pairs, device=device)
        preds = model(x_val).argmax(dim=-1)
        mask = (y_val != -100)
        in_dist_acc = (preds[mask] == y_val[mask]).float().mean().item() * 100.0

    print(f"   [Trained] [{beta_mode} | seed {seed}] in {wallclock:.1f}s | in-dist acc {in_dist_acc:.2f}%", flush=True)
    return model, {"wallclock": wallclock, "final_step": final_step, "in_dist_acc": in_dist_acc}


def evaluate_niah_matrix(model, lengths, depths, trials, device):
    model.eval()
    results = {}
    for L in lengths:
        results[L] = {}
        for d in depths:
            hits = 0
            for _ in range(trials):
                seq, gold_v = generate_niah_single_eval(L, d, device=device)
                with torch.no_grad():
                    pred = model(seq)[0, -1, :].argmax().item()
                if pred == gold_v:
                    hits += 1
            results[L][d] = 100.0 * hits / trials
        row = " | ".join(f"{results[L][d]:5.1f}%" for d in depths)
        print(f"      L={L:>6,}: {row}", flush=True)
    return results


# ==============================================================================
# MAIN SUITE
# ==============================================================================

def run_suite(arms=("learned", "fixed"), seeds=(42, 137, 2024), steps=800,
              train_len=128, train_pairs=8, batch_size=32, lr=4e-3, early_stop_acc=99.0,
              eval_lengths=(256, 512, 1024, 2048, 4096, 8192, 16384),
              depths=(0.10, 0.25, 0.50, 0.75, 0.90), trials=20,
              device='cpu'):

    t0_global = time.time()
    dev_name = "CUDA " + torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else device.upper()

    print("=" * 112, flush=True)
    print("P0-2 END-TO-END NIAH: RANDOMIZED NEEDLES + LEARNED GATING (Audit Remediation)", flush=True)
    print("=" * 112, flush=True)
    print(f"  * Protocol: Randomized needle key/value per trial (keys 1..32, vals 33..96)", flush=True)
    print(f"  * Training: Dynamic MQAR (L={train_len}, {train_pairs} pairs, batch={batch_size}, lr={lr:.1e}, max {steps} steps)", flush=True)
    print(f"  * Evaluation: Context lengths {list(eval_lengths)} x depths {[int(d*100) for d in depths]}% x {trials} trials", flush=True)
    print(f"  * Device: {dev_name} | PyTorch {torch.__version__}", flush=True)
    print(f"  * Arms: {list(arms)} | Seeds: {list(seeds)}", flush=True)
    print("=" * 112 + "\n", flush=True)

    payload = {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "P0-2 end-to-end NIAH: randomized needles, learned gating vs fixed-beta control",
        "protocol": {"train_len": train_len, "train_pairs": train_pairs, "steps": steps,
                      "batch_size": batch_size, "lr": lr, "early_stop_acc": early_stop_acc,
                      "eval_lengths": list(eval_lengths), "depths": list(depths), "trials": trials,
                      "seeds": list(seeds), "device": device},
        "arms": {},
    }

    for arm in arms:
        payload["arms"][arm] = {"seeds": {}}
        acc_matrices, in_dist_accs = [], []

        for seed in seeds:
            print(f"\n[Arm: {arm} | Seed: {seed}] Starting training...", flush=True)
            model, tmetrics = train_arm(arm, seed, steps, train_len, train_pairs, batch_size, lr, early_stop_acc, device)
            in_dist_accs.append(tmetrics["in_dist_acc"])

            print(f"   [Eval] Zero-Shot NIAH Retrieval Matrix (Random Needle per Trial):", flush=True)
            depth_header = " | ".join([f"{int(d*100):>4}%" for d in depths])
            print(f"      Depth: {depth_header}", flush=True)
            matrix = evaluate_niah_matrix(model, eval_lengths, depths, trials, device)
            acc_matrices.append(matrix)

            payload["arms"][arm]["seeds"][str(seed)] = {
                **tmetrics,
                "matrix": {str(L): {str(d): matrix[L][d] for d in depths} for L in eval_lengths},
            }

        # Agregado sobre semillas
        agg = {}
        for L in eval_lengths:
            agg[str(L)] = {str(d): {
                "mean": float(np.mean([m[L][d] for m in acc_matrices])),
                "se": float(np.std([m[L][d] for m in acc_matrices]) / math.sqrt(len(acc_matrices))) if len(acc_matrices) > 1 else 0.0,
            } for d in depths}
        payload["arms"][arm]["summary"] = {
            "mean_in_dist_acc": float(np.mean(in_dist_accs)),
            "aggregate_matrix": agg,
        }

    print("\n" + "=" * 112, flush=True)
    print("AGGREGATE NIAH MATRIX (Mean +/- SE across seeds) - Exact Match at Final Position:", flush=True)
    print("=" * 112, flush=True)
    for arm in arms:
        s = payload["arms"][arm]["summary"]
        print(f"\n  Arm: {arm.upper()} | In-dist acc: {s['mean_in_dist_acc']:.2f}%")
        header = "     " + "".join([f"{int(float(d)*100):>9}%" for d in depths])
        print(header)
        for L in eval_lengths:
            cells = "".join([f"{s['aggregate_matrix'][str(L)][str(d)]['mean']:>7.1f}%+/-{s['aggregate_matrix'][str(L)][str(d)]['se']:<3.1f}" for d in depths])
            print(f"     L={L:>7,}: {cells}")
    print("=" * 112, flush=True)

    out_file = "niah_e2e_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved to: {os.path.abspath(out_file)}", flush=True)
    print(f"Total suite time: {time.time()-t0_global:.1f}s", flush=True)
    return payload


if __name__ == '__main__':
    in_notebook = 'ipykernel' in sys.modules or 'google.colab' in sys.modules
    
    if not in_notebook and len(sys.argv) > 1:
        p = argparse.ArgumentParser(description="P0-2 End-to-End NIAH")
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--steps", type=int, default=800)
        p.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 2024])
        p.add_argument("--lengths", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192, 16384])
        p.add_argument("--quick", action="store_true")
        a, _ = p.parse_known_args()

        if a.quick:
            run_suite(arms=["learned"], seeds=[42], steps=50, train_len=128, batch_size=16,
                      eval_lengths=[256], depths=[0.5], trials=2, device=a.device)
        else:
            run_suite(arms=["learned", "fixed"], seeds=tuple(a.seeds), steps=a.steps,
                      eval_lengths=tuple(a.lengths), device=a.device)
    else:
        print("Running P0-2 NIAH in Colab/Notebook environment...")
        if CONFIG["quick_smoke_test"]:
            print("Quick Smoke Test active.")
            run_suite(arms=["learned"], seeds=[42], steps=50, train_len=128, batch_size=16,
                      eval_lengths=[256], depths=[0.5], trials=2, device=CONFIG["device"])
        else:
            run_suite(
                arms=CONFIG["arms"],
                seeds=CONFIG["seeds"],
                steps=CONFIG["steps"],
                train_len=CONFIG["train_len"],
                train_pairs=CONFIG["train_pairs"],
                batch_size=CONFIG["batch_size"],
                lr=CONFIG["lr"],
                early_stop_acc=CONFIG["early_stop_acc"],
                eval_lengths=CONFIG["eval_lengths"],
                depths=CONFIG["depths"],
                trials=CONFIG["trials_per_cell"],
                device=CONFIG["device"]
            )
