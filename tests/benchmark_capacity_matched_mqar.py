"""
Benchmark P0-1: Control de Capacidad Igualada para MQAR
=======================================================
Experimento diseñado para resolver el confound de capacidad identificado en
la auditoría project_audit_2026-08.md (§3.1, §5 R1):

  El estado complejo C^{32x32} almacena 2*32^2 = 2048 flotantes reales por
  cabeza, mientras que el estado real R^{32x32} almacena solo 32^2 = 1024.
  La ventaja "+22.82%" puede deberse a geometría fasorial O a simplemente
  tener el doble de memoria.

Controles añadidos:
  1. Real d_k=45 (iso-floats): 45^2 = 2025 floats ≈ 2048 del complejo.
     Proyecciones w_k/w_q/w_v amplian de d_model -> n_heads*45.
  2. Per-arm LR sweep: {1e-3, 3e-3, 5e-3}. Se reporta el MEJOR lr por brazo,
     eliminando el confound de tuning asimétrico.

Protocolo:
  - Datos generados on-the-fly (idéntico a benchmark_rigorous_mqar.py).
  - 5 semillas independientes por configuración.
  - Barrido de pares: {8, 16, 32}.
  - Early stopping al 99.5%.
  - JSON crudo archivado para auditoría.

Hipótesis a dirimir:
  H0: La ventaja compleja se explica enteramente por 2x capacidad de estado.
      => Real d_k=45 iguala o supera a Complex d_k=32.
  H1: Existe un beneficio geométrico/fasorial genuino.
      => Complex d_k=32 supera a Real d_k=45 a pesar de presupuesto similar.

Uso:
  python benchmark_capacity_matched_mqar.py --device cpu
  python benchmark_capacity_matched_mqar.py --device cpu --quick    # smoke test
  python benchmark_capacity_matched_mqar.py --device cpu --pairs 32  # solo el bloque crítico
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

# Fix Windows console utf-8 encoding if needed
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from delta_phase.layers import DeltaPhaseHolographicBlock, ShortCausalConv1D

# ==============================================================================
# 1. GENERADOR DE DATOS (idéntico a benchmark_rigorous_mqar.py)
# ==============================================================================

PAD_ID = 0
QUERY_MARKER = 1
TOKEN_OFFSET = 2
NUM_CONTENT_TOKENS = 512
VOCAB_SIZE = TOKEN_OFFSET + NUM_CONTENT_TOKENS  # 514


def generate_zoology_mqar_batch(
    batch_size: int = 32,
    seq_len: int = 128,
    num_pairs: int = 8,
    vocab_size: int = VOCAB_SIZE,
    device: str = 'cpu'
):
    num_tokens = vocab_size - TOKEN_OFFSET
    gap = 2
    tokens_needed = 2 * num_pairs
    assert 2 * num_pairs + gap + 2 * num_pairs <= seq_len, (
        f"Demasiados pares ({num_pairs}) para longitud L={seq_len}"
    )

    rand_t = torch.rand(batch_size, num_tokens, device=device)
    sampled = torch.argsort(rand_t, dim=-1)[:, :tokens_needed] + TOKEN_OFFSET
    keys = sampled[:, :num_pairs]
    vals = sampled[:, num_pairs:]

    x = torch.full((batch_size, seq_len), PAD_ID, dtype=torch.long, device=device)
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)

    kv = torch.stack([keys, vals], dim=2).view(batch_size, 2 * num_pairs)
    x[:, :2 * num_pairs] = kv

    q_perm = torch.argsort(torch.rand(batch_size, num_pairs, device=device), dim=-1)
    query_keys = torch.gather(keys, 1, q_perm)
    query_vals = torch.gather(vals, 1, q_perm)

    gap = 2
    pos_q = (2 * num_pairs + gap + 2 * torch.arange(num_pairs, device=device)).unsqueeze(0).expand(batch_size, -1)

    x.scatter_(1, pos_q, QUERY_MARKER)
    x.scatter_(1, pos_q + 1, query_keys)
    y.scatter_(1, pos_q + 1, query_vals)
    return x, y


# ==============================================================================
# 2. MODELOS
# ==============================================================================

# --- A. DeltaPhase Complex (original, d_k = d_model // n_heads) ---
class DeltaPhaseMQAR(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=128, n_heads=4,
                 chunk_size=32, num_layers=2, max_len=4096):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
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


# --- B. Real Gated DeltaNet con d_k configurable ---
class RealGatedDeltaNetBlock(nn.Module):
    """
    Gated DeltaNet real R^(d_k x d_k) con d_k configurable.
    Cuando d_k != d_model // n_heads, las proyecciones k/q/v mapean
    d_model -> n_heads * d_k (dimensión interna diferente de d_model).
    """
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

        # Projections: d_model -> internal_dim (may differ from d_model)
        self.w_k = nn.Linear(d_model, self.internal_dim, bias=False)
        self.w_q = nn.Linear(d_model, self.internal_dim, bias=False)
        self.w_v = nn.Linear(d_model, self.internal_dim, bias=False)
        self.w_beta = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(self.internal_dim, d_model, bias=False)

        # Standard MLP (same budget as original)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
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

        # Real L2 Normalization (standard DeltaNet)
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

        # Gram matrix & Chunkwise solve
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
            qc = Q_c[:, :, c]
            kc = K_c[:, :, c]
            vc = V_c[:, :, c]
            bc = beta_c[:, :, c]
            tc = T_mat[:, :, c]

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


# --- C. Causal MHA Transformer (control positivo, sin cambios) ---
class CausalMHABlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, conv_kernel_size=4):
        super().__init__()
        self.conv = ShortCausalConv1D(d_model, kernel_size=conv_kernel_size)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
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
# 3. EVALUACIÓN Y ENTRENAMIENTO
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
            correct = (preds[mask] == targets[mask]).sum().item()
            total_correct += correct
            total_queries += mask.sum().item()
    return (total_correct / max(total_queries, 1)) * 100.0


def format_duration(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    if mins >= 60:
        hrs = int(mins // 60)
        mins = int(mins % 60)
        return f"{hrs:02d}h {mins:02d}m {secs:02d}s"
    return f"{mins:02d}m {secs:02d}s"


def train_mqar_model(
    name, model, total_steps=1500, batch_size=32, seq_len=128,
    num_pairs=8, vocab_size=VOCAB_SIZE, lr=3e-3, device='cpu',
    log_interval=50, early_stop_acc=99.5,
    global_model_idx=1, total_models=1, global_start_time=0.0
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    t_start = time.time()

    elapsed_global = time.time() - global_start_time if global_start_time > 0 else 0
    print(f"\n[{time.strftime('%H:%M:%S')}] [{global_model_idx}/{total_models}] "
          f"{name} (lr={lr:.1e})", flush=True)
    print(f"   Config: L={seq_len}, N_pairs={num_pairs}, max {total_steps} pasos | "
          f"Global: {format_duration(elapsed_global)}", flush=True)

    running_loss = 0.0
    step_times = []
    step_to_50 = None
    step_to_95 = None
    final_step = total_steps
    early_stopped = False
    best_acc = 0.0

    for step in range(1, total_steps + 1):
        t0 = time.time()
        tokens, targets = generate_zoology_mqar_batch(
            batch_size=batch_size, seq_len=seq_len, num_pairs=num_pairs,
            vocab_size=vocab_size, device=device
        )

        optimizer.zero_grad()
        logits = model(tokens)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        dt = time.time() - t0
        step_times.append(dt)
        if len(step_times) > 50:
            step_times.pop(0)

        running_loss += loss.item()

        if step % log_interval == 0 or step == total_steps:
            avg_loss = running_loss / min(step, log_interval)
            val_acc = evaluate_mqar_accuracy(
                model, num_eval_batches=10, batch_size=batch_size,
                seq_len=seq_len, num_pairs=num_pairs, vocab_size=vocab_size,
                device=device
            )
            best_acc = max(best_acc, val_acc)

            if step_to_50 is None and val_acc >= 50.0:
                step_to_50 = step
            if step_to_95 is None and val_acc >= 95.0:
                step_to_95 = step

            avg_step_time = sum(step_times) / len(step_times)
            steps_left = total_steps - step
            model_eta = steps_left * avg_step_time
            pct = (step / total_steps) * 100.0

            print(
                f"   [{time.strftime('%H:%M:%S')}] Paso {step:4d}/{total_steps} "
                f"({pct:5.1f}%) | Loss: {avg_loss:6.4f} | "
                f"Acc: {val_acc:6.2f}% | ETA: {format_duration(model_eta)}",
                flush=True
            )
            running_loss = 0.0

            if early_stop_acc is not None and val_acc >= early_stop_acc:
                final_step = step
                early_stopped = True
                print(f"   [EARLY STOP] {val_acc:.2f}% >= {early_stop_acc:.1f}% "
                      f"en paso {step}", flush=True)
                break

            model.train()

    wallclock = time.time() - t_start
    # Final evaluation for the returned accuracy
    final_acc = evaluate_mqar_accuracy(
        model, num_eval_batches=20, batch_size=batch_size,
        seq_len=seq_len, num_pairs=num_pairs, vocab_size=vocab_size,
        device=device
    )
    best_acc = max(best_acc, final_acc)

    status = "Early Stop" if early_stopped else "Full"
    print(f"   [{name}] {status} en paso {final_step} | "
          f"{format_duration(wallclock)} | Final Acc: {final_acc:.2f}%", flush=True)

    return model, {
        "wallclock": wallclock,
        "final_step": final_step,
        "step_to_50": step_to_50 if step_to_50 is not None else total_steps,
        "step_to_95": step_to_95 if step_to_95 is not None else total_steps,
        "early_stopped": early_stopped,
        "final_acc": final_acc,
        "best_acc": best_acc
    }


# ==============================================================================
# 4. LR SELECTION (mini-sweep per arm)
# ==============================================================================

def select_best_lr(
    arm_key, model_factory, lr_candidates, pilot_seed=42,
    pilot_steps=300, batch_size=32, seq_len=128, num_pairs=8,
    vocab_size=VOCAB_SIZE, device='cpu'
):
    """
    Ejecuta un piloto corto con cada LR candidato y devuelve el mejor.
    Usa una sola semilla y pocos pasos para minimizar el coste.
    """
    print(f"\n{'─'*80}", flush=True)
    print(f"  LR PILOT para [{arm_key}]: candidatos {lr_candidates}, "
          f"seed={pilot_seed}, {pilot_steps} pasos", flush=True)
    print(f"{'─'*80}", flush=True)

    best_lr = lr_candidates[0]
    best_acc = -1.0

    for lr in lr_candidates:
        torch.manual_seed(pilot_seed)
        np.random.seed(pilot_seed)
        random.seed(pilot_seed)

        model = model_factory()
        _, metrics = train_mqar_model(
            name=f"{arm_key} pilot lr={lr:.1e}",
            model=model,
            total_steps=pilot_steps,
            batch_size=batch_size,
            seq_len=seq_len,
            num_pairs=num_pairs,
            vocab_size=vocab_size,
            lr=lr,
            device=device,
            log_interval=pilot_steps,  # solo log al final
            early_stop_acc=None,       # sin early stop en piloto
            global_model_idx=0,
            total_models=0,
            global_start_time=0.0
        )
        acc = metrics["best_acc"]
        print(f"     lr={lr:.1e} => Acc={acc:.2f}%", flush=True)
        if acc > best_acc:
            best_acc = acc
            best_lr = lr

    print(f"  => Mejor LR para [{arm_key}]: {best_lr:.1e} (Acc={best_acc:.2f}%)",
          flush=True)
    return best_lr


# ==============================================================================
# 5. SUITE PRINCIPAL
# ==============================================================================

def print_architecture_inventory(name, model, d_model, n_heads, d_k, vocab_size):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    state_floats = 2 * d_k * d_k if 'Complex' in name else d_k * d_k
    print(f"\n  [ARCH] {name}", flush=True)
    print(f"     d_model={d_model}, n_heads={n_heads}, d_k={d_k}, "
          f"Vocab={vocab_size}", flush=True)
    print(f"     Params: Total={total:,} | Trainable={trainable:,}", flush=True)
    print(f"     Estado recurrente/cabeza: {state_floats:,} floats "
          f"({state_floats * 4 / 1024:.1f} KB FP32)", flush=True)


def run_capacity_matched_suite(
    seeds=(42, 137, 2024, 7, 999),
    pair_sweep=(8, 16, 32),
    steps_per_train=1500,
    early_stop_acc=99.5,
    lr_candidates=(1e-3, 3e-3, 5e-3),
    pilot_steps=300,
    device='cpu'
):
    global_start_time = time.time()

    # Config
    vocab_size = VOCAB_SIZE
    d_model = 128
    n_heads = 4
    d_k_complex = d_model // n_heads  # = 32
    d_k_iso = 45                       # 45^2 = 2025 ≈ 2*32^2 = 2048
    num_layers = 2

    arm_specs = {
        "DeltaPhase_Complex_dk32": {
            "d_k": d_k_complex,
            "state_floats_per_head": 2 * d_k_complex**2,
            "factory": lambda: DeltaPhaseMQAR(
                vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
                num_layers=num_layers
            ),
        },
        "GatedDeltaNet_Real_dk32": {
            "d_k": d_k_complex,
            "state_floats_per_head": d_k_complex**2,
            "factory": lambda: RealGatedDeltaNetMQAR(
                vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
                d_k=d_k_complex, num_layers=num_layers
            ),
        },
        "GatedDeltaNet_Real_dk45_ISOFLTS": {
            "d_k": d_k_iso,
            "state_floats_per_head": d_k_iso**2,
            "factory": lambda: RealGatedDeltaNetMQAR(
                vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
                d_k=d_k_iso, num_layers=num_layers
            ),
        },
        "Transformer_Causal": {
            "d_k": d_k_complex,
            "state_floats_per_head": "N/A (KV-cache)",
            "factory": lambda: CausalTransformerMQAR(
                vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
                num_layers=num_layers
            ),
        },
    }

    arm_keys = list(arm_specs.keys())

    # ── Header ──────────────────────────────────────────────────────────────
    print("=" * 110, flush=True)
    print("PROTOCOLO P0-1: CONTROL DE CAPACIDAD IGUALADA MQAR", flush=True)
    print("Objetivo: Aislar beneficio geometrico fasorial vs. capacidad 2x", flush=True)
    print("=" * 110, flush=True)
    print(f"  Dispositivo:         {device.upper()} ({platform.processor() or 'N/A'})",
          flush=True)
    print(f"  Fecha UTC:           "
          f"{datetime.datetime.now(datetime.timezone.utc).isoformat()}", flush=True)
    print(f"  Python/PyTorch:      Python {platform.python_version()} | "
          f"PyTorch {torch.__version__}", flush=True)
    print(f"  Semillas ({len(seeds)}):      {list(seeds)}", flush=True)
    print(f"  Barrido de pares:    {list(pair_sweep)}", flush=True)
    print(f"  Max pasos/modelo:    {steps_per_train} (Early Stop @ {early_stop_acc}%)",
          flush=True)
    print(f"  LR candidatos:       {list(lr_candidates)}", flush=True)
    print(f"  Pasos piloto LR:     {pilot_steps}", flush=True)

    print(f"\n{'─'*110}", flush=True)
    print("  INVENTARIO DE BRAZOS EXPERIMENTALES:", flush=True)
    print(f"{'─'*110}", flush=True)

    print(f"\n  {'Brazo':<40s} | {'d_k':>4s} | {'Estado/cabeza (floats)':>22s} | "
          f"{'Params totales':>15s}", flush=True)
    print(f"  {'─'*40} | {'─'*4} | {'─'*22} | {'─'*15}", flush=True)

    for arm_key, spec in arm_specs.items():
        sample_model = spec["factory"]()
        total_p = sum(p.numel() for p in sample_model.parameters())
        sfh = spec["state_floats_per_head"]
        sfh_str = f"{sfh:,}" if isinstance(sfh, int) else str(sfh)
        print(f"  {arm_key:<40s} | {spec['d_k']:>4} | {sfh_str:>22s} | "
              f"{total_p:>15,}", flush=True)
        del sample_model

    print(f"\n  Nota: Complex dk=32 tiene 2*32^2 = {2*d_k_complex**2} floats/cabeza; "
          f"Real dk=45 tiene 45^2 = {d_k_iso**2} floats/cabeza "
          f"(ratio: {d_k_iso**2 / (2*d_k_complex**2):.3f})", flush=True)
    print("=" * 110, flush=True)

    # ── Per-pair LR selection ───────────────────────────────────────────────
    # Hacemos el piloto para cada (arm, n_pairs) por separado, ya que el
    # punto de saturación cambia con n_pairs.

    all_results = {}

    for n_pairs in pair_sweep:
        L_train = 128 if n_pairs <= 16 else 256
        eval_lengths = [L_train, 2 * L_train, 4 * L_train]

        print(f"\n\n{'#'*110}", flush=True)
        print(f"  BLOQUE N_PAIRS = {n_pairs} | L_train = {L_train} | "
              f"Eval = {eval_lengths}", flush=True)
        print(f"{'#'*110}", flush=True)

        block_key = f"pairs_{n_pairs}"
        all_results[block_key] = {
            "L_train": L_train,
            "eval_lengths": eval_lengths,
        }

        # ── LR Pilot per arm ──
        best_lrs = {}
        for arm_key in arm_keys:
            spec = arm_specs[arm_key]
            best_lrs[arm_key] = select_best_lr(
                arm_key=arm_key,
                model_factory=spec["factory"],
                lr_candidates=list(lr_candidates),
                pilot_seed=42,
                pilot_steps=pilot_steps,
                batch_size=32,
                seq_len=L_train,
                num_pairs=n_pairs,
                vocab_size=vocab_size,
                device=device
            )

        print(f"\n  LRs seleccionados para N_pairs={n_pairs}:", flush=True)
        for arm_key, lr in best_lrs.items():
            print(f"    {arm_key:<40s}: {lr:.1e}", flush=True)

        # ── Main multi-seed runs ──
        total_runs_block = len(arm_keys) * len(seeds)
        run_counter = 0

        for arm_key in arm_keys:
            spec = arm_specs[arm_key]
            lr = best_lrs[arm_key]

            all_results[block_key][arm_key] = {
                "lr_selected": lr,
                "d_k": spec["d_k"],
                "state_floats_per_head": spec["state_floats_per_head"],
                "wallclocks": [],
                "steps_to_50": [],
                "steps_to_95": [],
                "final_accs": [],
                "best_accs": [],
                "length_accs": {L: [] for L in eval_lengths}
            }

            for seed_idx, seed in enumerate(seeds, 1):
                run_counter += 1
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                model = spec["factory"]()
                model_name = f"{arm_key} [Seed {seed}]"

                trained_model, metrics = train_mqar_model(
                    name=model_name,
                    model=model,
                    total_steps=steps_per_train,
                    batch_size=32,
                    seq_len=L_train,
                    num_pairs=n_pairs,
                    vocab_size=vocab_size,
                    lr=lr,
                    device=device,
                    log_interval=50,
                    early_stop_acc=early_stop_acc,
                    global_model_idx=run_counter,
                    total_models=total_runs_block,
                    global_start_time=global_start_time
                )

                arm_data = all_results[block_key][arm_key]
                arm_data["wallclocks"].append(metrics["wallclock"])
                arm_data["steps_to_50"].append(metrics["step_to_50"])
                arm_data["steps_to_95"].append(metrics["step_to_95"])
                arm_data["final_accs"].append(metrics["final_acc"])
                arm_data["best_accs"].append(metrics["best_acc"])

                # Length generalization eval
                print(f"   Eval en longitudes {eval_lengths}...", flush=True)
                for L_eval in eval_lengths:
                    acc = evaluate_mqar_accuracy(
                        trained_model, num_eval_batches=20, batch_size=32,
                        seq_len=L_eval, num_pairs=n_pairs,
                        vocab_size=vocab_size, device=device
                    )
                    arm_data["length_accs"][L_eval].append(acc)
                    print(f"      L={L_eval:4d}: {acc:6.2f}%", flush=True)

                del trained_model
                del model

    # ══════════════════════════════════════════════════════════════════════
    # RESUMEN ESTADÍSTICO FINAL
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*130}", flush=True)
    print(f"  TABLA RESUMEN P0-1: CAPACIDAD IGUALADA (media +/- SE, "
          f"{len(seeds)} semillas)", flush=True)
    print(f"{'='*130}", flush=True)

    header = (f"  {'N_pairs':<8s} | {'Brazo':<40s} | {'d_k':>4s} | "
              f"{'Floats/h':>9s} | {'LR':>7s} | {'L_train Acc':>14s} | "
              f"{'OOD 2x':>14s} | {'OOD 4x':>14s} | {'Steps>95%':>10s}")
    print(header, flush=True)
    print(f"  {'─'*128}", flush=True)

    summary_json = {}

    for n_pairs in pair_sweep:
        block_key = f"pairs_{n_pairs}"
        block = all_results[block_key]
        eval_lengths = block["eval_lengths"]
        summary_json[block_key] = {}

        for arm_key in arm_keys:
            if arm_key not in block:
                continue
            arm = block[arm_key]

            # Compute stats for each eval length
            acc_strs = []
            acc_means = []
            for L in eval_lengths:
                vals = arm["length_accs"][L]
                mean_v = float(np.mean(vals))
                se_v = float(np.std(vals) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
                acc_strs.append(f"{mean_v:5.2f}+/-{se_v:4.2f}")
                acc_means.append(mean_v)

            s95 = float(np.mean(arm["steps_to_95"]))
            sfh = arm["state_floats_per_head"]
            sfh_str = f"{sfh:,}" if isinstance(sfh, int) else str(sfh)

            print(f"  {n_pairs:<8d} | {arm_key:<40s} | {arm['d_k']:>4d} | "
                  f"{sfh_str:>9s} | {arm['lr_selected']:.1e} | "
                  f"{acc_strs[0]:>14s} | {acc_strs[1]:>14s} | "
                  f"{acc_strs[2]:>14s} | {s95:>10.0f}", flush=True)

            summary_json[block_key][arm_key] = {
                "d_k": arm["d_k"],
                "state_floats_per_head": sfh,
                "lr_selected": arm["lr_selected"],
                "L_train_acc": acc_strs[0],
                "OOD_2x_acc": acc_strs[1],
                "OOD_4x_acc": acc_strs[2],
                "mean_steps_to_95": s95,
                "raw_accs": {str(L): arm["length_accs"][L] for L in eval_lengths}
            }

        print(f"  {'─'*128}", flush=True)

    print(f"{'='*130}", flush=True)

    # ── Diagnóstico del confound ──
    print(f"\n  DIAGNÓSTICO DE CONFOUND:", flush=True)
    for n_pairs in pair_sweep:
        block_key = f"pairs_{n_pairs}"
        block = all_results[block_key]
        L_train = block["eval_lengths"][0]

        complex_accs = block.get("DeltaPhase_Complex_dk32", {}).get("length_accs", {}).get(L_train, [])
        real32_accs = block.get("GatedDeltaNet_Real_dk32", {}).get("length_accs", {}).get(L_train, [])
        real45_accs = block.get("GatedDeltaNet_Real_dk45_ISOFLTS", {}).get("length_accs", {}).get(L_train, [])

        if complex_accs and real32_accs and real45_accs:
            c_mean = np.mean(complex_accs)
            r32_mean = np.mean(real32_accs)
            r45_mean = np.mean(real45_accs)

            gap_original = c_mean - r32_mean
            gap_isoflts = c_mean - r45_mean

            print(f"\n    N_pairs={n_pairs} (L_train={L_train}):", flush=True)
            print(f"      Complex dk=32:     {c_mean:.2f}%", flush=True)
            print(f"      Real dk=32:        {r32_mean:.2f}% "
                  f"(gap vs complex: {gap_original:+.2f}pp)", flush=True)
            print(f"      Real dk=45 (iso):  {r45_mean:.2f}% "
                  f"(gap vs complex: {gap_isoflts:+.2f}pp)", flush=True)

            if gap_isoflts > 2.0:
                print(f"      => H1 SOPORTADA: ventaja fasorial "
                      f"({gap_isoflts:+.2f}pp) persiste con presupuesto igualado",
                      flush=True)
            elif gap_isoflts > 0.0:
                print(f"      => SEÑAL DEBIL: ventaja residual "
                      f"({gap_isoflts:+.2f}pp), posiblemente ruido",
                      flush=True)
            else:
                print(f"      => H0 SOPORTADA: el iso-floats real "
                      f"cierra/supera el gap ({gap_isoflts:+.2f}pp)",
                      flush=True)

            capacity_explains_pct = max(0, min(100,
                (1 - gap_isoflts / max(gap_original, 1e-6)) * 100
            ))
            print(f"      Fraccion del gap original explicada por capacidad: "
                  f"~{capacity_explains_pct:.0f}%", flush=True)

    # ── Save JSON ──
    total_wallclock = time.time() - global_start_time
    results_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "docs",
        "capacity_matched_mqar_results.json"
    ))
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "P0-1 Capacity-Matched MQAR Control",
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "hypothesis": "H0: Complex advantage = 2x capacity. "
                          "H1: Genuine phasor geometry benefit.",
            "seeds": list(seeds),
            "pair_sweep": list(pair_sweep),
            "lr_candidates": list(lr_candidates),
            "pilot_steps": pilot_steps,
            "steps_per_train": steps_per_train,
            "device": device,
            "total_wallclock_sec": total_wallclock,
            "summary": summary_json,
            "raw_data": {
                k: {
                    kk: vv for kk, vv in v.items()
                    if kk not in ("factory",)
                }
                for k, v in all_results.items()
            }
        }, f, indent=2, default=str)

    print(f"\n  Resultados guardados: {results_path}", flush=True)
    print(f"  Tiempo total suite: {format_duration(total_wallclock)} "
          f"({total_wallclock:.0f}s)", flush=True)
    print(f"{'='*110}", flush=True)

    return all_results, summary_json


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="P0-1: Capacity-Matched MQAR Control Experiment"
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 137, 2024, 7, 999])
    parser.add_argument("--pairs", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--early-stop-acc", type=float, default=99.5)
    parser.add_argument("--lr-candidates", type=float, nargs="+",
                        default=[1e-3, 3e-3, 5e-3])
    parser.add_argument("--pilot-steps", type=int, default=300)
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 1 seed, 1 pair count, 200 steps")

    args = parser.parse_args()

    if args.quick:
        run_capacity_matched_suite(
            seeds=(42,),
            pair_sweep=(32,),       # el bloque critico
            steps_per_train=200,
            early_stop_acc=args.early_stop_acc,
            lr_candidates=(3e-3,),  # sin sweep en quick
            pilot_steps=0,
            device=args.device
        )
    else:
        run_capacity_matched_suite(
            seeds=tuple(args.seeds),
            pair_sweep=tuple(args.pairs),
            steps_per_train=args.steps,
            early_stop_acc=args.early_stop_acc,
            lr_candidates=tuple(args.lr_candidates),
            pilot_steps=args.pilot_steps,
            device=args.device
        )
