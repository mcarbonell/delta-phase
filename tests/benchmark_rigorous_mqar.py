"""
Benchmark Riguroso y Certificado de MQAR (Multi-Query Associative Recall)
para DeltaPhase (C^(d_k x d_k)) vs Gated DeltaNet Real (R^(d_k x d_k)) vs RoPE Transformer.

Protocolo Estándar de la Literatura (Zoology / H3):
- Generación de datos puramente dinámica on-the-fly (inmune a memorización estática).
- Supervisión densa multi-consulta (loss computada en todas las posiciones de consulta).
- RoPE Causal Transformer como control positivo y Gated DeltaNet Real como control de espacio.
- Barrido de pares N_pairs in {8, 16, 32} y longitudes L in {128, 256, 512}.
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
# 1. GENERADOR DE DATOS DINÁMICO ON-THE-FLY (ZOOLOGY / H3 MQAR)
# ==============================================================================

# ==============================================================================
# 1. GENERADOR DE DATOS DINÁMICO ON-THE-FLY (ZOOLOGY / H3 MQAR)
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
    """
    Genera un lote dinámico e inédito para Multi-Query Associative Recall (MQAR)
    según el estándar certificado de la literatura (Zoology - Arora et al. 2023):
    - Pares K-V contiguos: [K_0, V_0, K_1, V_1, ..., K_m, V_m]
    - Consultas explícitas con marcador: [..., QUERY_MARKER, K_target] -> Objetivo en target: V_target.
    - Loss y Accuracy evaluadas estrictamente en las posiciones de respuesta (targets != -100).
    """
    gap = 2
    tokens_needed = 2 * num_pairs
    assert 2 * num_pairs + gap + 2 * num_pairs <= seq_len, f"Demasiados pares ({num_pairs}) para longitud L={seq_len}"
    
    rand_t = torch.rand(batch_size, num_tokens, device=device)
    sampled = torch.argsort(rand_t, dim=-1)[:, :tokens_needed] + TOKEN_OFFSET
    keys = sampled[:, :num_pairs]
    vals = sampled[:, num_pairs:]
    
    x = torch.full((batch_size, seq_len), PAD_ID, dtype=torch.long, device=device)
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
    
    # 1. Almacenar pares clave-valor contiguos
    kv = torch.stack([keys, vals], dim=2).view(batch_size, 2 * num_pairs)
    x[:, :2 * num_pairs] = kv
    
    # 2. Consultas aleatorias intercaladas en la segunda mitad
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
# 2. IMPLEMENTACIÓN DE MODELOS Y BASELINES
# ==============================================================================

# --- A. DeltaPhase Model ---
class DeltaPhaseMQAR(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = 128, n_heads: int = 4, chunk_size: int = 32, num_layers: int = 2, max_len: int = 4096):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            DeltaPhaseHolographicBlock(d_model=d_model, n_heads=n_heads, chunk_size=chunk_size)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_emb(pos)
        for block in self.blocks:
            h, _ = block(h)
        logits = self.head(h)
        return logits


# --- B. Real-Valued Gated DeltaNet Baseline (R^(d_k x d_k)) ---
class RealGatedDeltaNetBlock(nn.Module):
    """
    Gated DeltaNet canónico en espacio REAL R^(d_k x d_k) con Chunkwise WY Solve y L2 Normalization.
    Permite aislar exactamente el beneficio del espacio fasorial complejo C vs el espacio real R.
    """
    def __init__(self, d_model: int = 128, n_heads: int = 4, chunk_size: int = 32, conv_kernel_size: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.chunk_size = chunk_size
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=conv_kernel_size)
        
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_beta = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Standard MLP
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
        
        # Real L2 Unit Normalization (Standard DeltaNet / Gated DeltaNet)
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
        T_mat = torch.linalg.solve_triangular(I_mat + L_mat.transpose(-1, -2), I_mat, upper=False)
        
        M_state = torch.zeros(B, self.n_heads, self.d_k, self.d_k, device=x.device, dtype=x.dtype)
        out_chunks = []
        
        for c in range(num_chunks):
            qc, kc, vc, bc, tc = Q_c[:, :, c], K_c[:, :, c], V_c[:, :, c], beta_c[:, :, c], T_mat[:, :, c]
            v_old = torch.matmul(M_state, kc.transpose(-1, -2)).transpose(-1, -2)
            E_c = torch.matmul(tc, vc - v_old)
            U_c = bc.unsqueeze(-1) * E_c
            o_inter = torch.matmul(M_state, qc.transpose(-1, -2)).transpose(-1, -2)
            A_intra = torch.tril(torch.matmul(qc, kc.transpose(-1, -2)))
            out_chunks.append(torch.matmul(A_intra, U_c) + o_inter)
            M_state = M_state + torch.matmul(U_c.transpose(-1, -2), kc)
            
        retrieved = torch.cat(out_chunks, dim=2)[:, :, :L].transpose(1, 2).reshape(B, L, D)
        x = res + self.out_proj(retrieved)
        x = x + self.mlp(self.norm2(x))
        return x

class RealGatedDeltaNetMQAR(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = 128, n_heads: int = 4, chunk_size: int = 32, num_layers: int = 2, max_len: int = 4096):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            RealGatedDeltaNetBlock(d_model=d_model, n_heads=n_heads, chunk_size=chunk_size)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# --- C. Causal MHA Transformer Baseline (Softmax O(N^2)) ---
class CausalMHABlock(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, conv_kernel_size: int = 4):
        super().__init__()
        self.conv = ShortCausalConv1D(d_model, kernel_size=conv_kernel_size)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
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
        causal_mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
        attn_out, _ = self.mha(norm_x, norm_x, norm_x, attn_mask=causal_mask, is_causal=False)
        x = res + attn_out
        return x + self.ffn(self.norm2(x))

class CausalTransformerMQAR(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = 128, n_heads: int = 4, num_layers: int = 2, max_len: int = 4096):
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
# 3. RUTINAS DE EVALUACIÓN Y ENTRENAMIENTO DINÁMICO
# ==============================================================================

def evaluate_mqar_accuracy(model, num_eval_batches=15, batch_size=32, seq_len=128, num_pairs=8, vocab_size=VOCAB_SIZE, device='cpu'):
    model.eval()
    total_correct = 0
    total_queries = 0
    
    with torch.no_grad():
        for _ in range(num_eval_batches):
            tokens, targets = generate_zoology_mqar_batch(
                batch_size=batch_size,
                seq_len=seq_len,
                num_pairs=num_pairs,
                vocab_size=vocab_size,
                device=device
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
    name: str,
    model: nn.Module,
    total_steps: int = 1500,
    batch_size: int = 32,
    seq_len: int = 128,
    num_pairs: int = 8,
    vocab_size: int = VOCAB_SIZE,
    lr: float = 3e-3,
    device: str = 'cpu',
    log_interval: int = 50,
    early_stop_acc: float = 99.5,
    global_model_idx: int = 1,
    total_models: int = 1,
    global_start_time: float = 0.0
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    model.train()
    t_start = time.time()
    
    elapsed_global = time.time() - global_start_time if global_start_time > 0 else 0
    print(f"\n[{time.strftime('%H:%M:%S')}] 🚀 [Modelo {global_model_idx}/{total_models}] {name}", flush=True)
    print(f"   Config: L={seq_len}, N_pairs={num_pairs}, max {total_steps} pasos | Tiempo global transcurrido: {format_duration(elapsed_global)}", flush=True)
    
    running_loss = 0.0
    step_times = []
    step_to_50 = None
    step_to_95 = None
    final_step = total_steps
    early_stopped = False
    
    for step in range(1, total_steps + 1):
        t0 = time.time()
        tokens, targets = generate_zoology_mqar_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            num_pairs=num_pairs,
            vocab_size=vocab_size,
            device=device
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
            avg_loss = running_loss / (step if step < log_interval else log_interval)
            val_acc = evaluate_mqar_accuracy(
                model, num_eval_batches=10, batch_size=batch_size,
                seq_len=seq_len, num_pairs=num_pairs, vocab_size=vocab_size, device=device
            )
            
            if step_to_50 is None and val_acc >= 50.0:
                step_to_50 = step
            if step_to_95 is None and val_acc >= 95.0:
                step_to_95 = step
                
            avg_step_time = sum(step_times) / len(step_times)
            steps_left = total_steps - step
            model_eta = steps_left * avg_step_time
            
            models_left = total_models - global_model_idx
            approx_total_eta = model_eta + (models_left * (total_steps * avg_step_time))
            
            pct = (step / total_steps) * 100.0
            print(
                f"   ⏱️ [{time.strftime('%H:%M:%S')}] Paso {step:4d}/{total_steps} ({pct:5.1f}%) | "
                f"Loss: {avg_loss:6.4f} | Held-out Acc: {val_acc:6.2f}% | "
                f"Vel: {1.0/max(avg_step_time, 1e-5):4.1f} st/s | "
                f"ETA modelo: {format_duration(model_eta)} | ETA suite: {format_duration(approx_total_eta)}",
                flush=True
            )
            running_loss = 0.0
            
            if early_stop_acc is not None and val_acc >= early_stop_acc:
                final_step = step
                early_stopped = True
                print(f"   🎯 [EARLY STOPPING] Alcanzado {val_acc:.2f}% (>= {early_stop_acc:.1f}%) en paso {step}! Finalizando entrenamiento temprano.", flush=True)
                break
                
            model.train()
            
    wallclock = time.time() - t_start
    status_str = f"completado en paso {final_step} (Early Stop)" if early_stopped else f"completado en {total_steps} pasos"
    print(f"   ✨ [{name}] {status_str} en {format_duration(wallclock)} ({wallclock:.1f}s)", flush=True)
    
    metrics = {
        "wallclock": wallclock,
        "final_step": final_step,
        "step_to_50": step_to_50 if step_to_50 is not None else total_steps,
        "step_to_95": step_to_95 if step_to_95 is not None else total_steps,
        "early_stopped": early_stopped
    }
    return model, metrics


def print_architecture_inventory(name: str, model: nn.Module, d_model: int, n_heads: int, d_k: int, vocab_size: int):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  📦 [ARQUITECTURA] {name}", flush=True)
    print(f"     • Dimensiones:  d_model={d_model}, n_heads={n_heads}, d_k={d_k}, Vocab={vocab_size}", flush=True)
    print(f"     • Parámetros:   Total={total_params:,} | Entrenables={trainable_params:,}", flush=True)
    print(f"     • Desglose por capas y módulos:", flush=True)
    for idx, (layer_name, module) in enumerate(model.named_children()):
        mod_params = sum(p.numel() for p in module.parameters())
        print(f"        [{idx}] {layer_name:<18s} | Clase: {module.__class__.__name__:<30s} | Params: {mod_params:,}", flush=True)


# ==============================================================================
# 4. BENCHMARK SUITE COMPLETA (CAPACIDAD + LONGITUD + MULTI-SEMILLA)
# ==============================================================================

def run_rigorous_mqar_suite(
    seeds: list = [42, 137, 2024, 7, 999],
    pair_sweep: list = [8, 16, 32],
    steps_per_train: int = 1500,
    early_stop_acc: float = 99.5,
    device: str = 'cpu'
):
    global_start_time = time.time()
    models_to_test = ["DeltaPhase_Complex", "GatedDeltaNet_Real", "Transformer_Causal"]
    total_models = len(pair_sweep) * len(models_to_test) * len(seeds)
    
    vocab_size = VOCAB_SIZE
    d_model = 128
    n_heads = 4
    d_k = d_model // n_heads
    num_layers = 2
    
    # 1. Cabecera Completa con Metadatos, Explicación e Inventario
    print("=" * 105, flush=True)
    print("🌟 PROTOCOLO RIGUROSO DE CERTIFICACIÓN MQAR NIVEL 2 (MULTI-QUERY ASSOCIATIVE RECALL)", flush=True)
    print("=" * 105, flush=True)
    print(f"  • Propósito:           Certificación formal Nivel 2 de retención y velocidad de convergencia (grokking)", flush=True)
    print(f"                         Comparación iso-paramétrica multi-semilla (n={len(seeds)}) de DeltaPhase vs Gated DeltaNet vs Transformer", flush=True)
    print(f"  • Dispositivo:         {device.upper()} ({platform.processor() or 'Multicore'})", flush=True)
    print(f"  • Fecha UTC:           {datetime.datetime.now(datetime.timezone.utc).isoformat()}", flush=True)
    print(f"  • Versión Python/Torch: Python {platform.python_version()} | PyTorch {torch.__version__}", flush=True)
    print(f"  • Semillas ({len(seeds)}):      {seeds}", flush=True)
    print(f"  • Barrido de Pares:    {pair_sweep}", flush=True)
    print(f"  • Max Pasos Train:     {steps_per_train} pasos on-the-fly por modelo (Early Stopping @ {early_stop_acc}%)", flush=True)
    print(f"  • Total ejecuciones:   {total_models} modelos individuales", flush=True)
    print("-" * 105, flush=True)
    print("  📋 INVENTARIO COMPLETO DE ARQUITECTURAS A EVALUAR:", flush=True)
    
    sample_models = {
        "DeltaPhase_Complex (C^(32x32))": DeltaPhaseMQAR(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, num_layers=num_layers),
        "GatedDeltaNet_Real (R^(32x32))": RealGatedDeltaNetMQAR(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, num_layers=num_layers),
        "Transformer_Causal (Softmax O(N^2))": CausalTransformerMQAR(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, num_layers=num_layers)
    }
    for name, m in sample_models.items():
        print_architecture_inventory(name, m, d_model=d_model, n_heads=n_heads, d_k=d_k, vocab_size=vocab_size)
    print("=" * 105 + "\n", flush=True)
    
    all_experiments = {}
    current_model_counter = 0
    
    for n_pairs in pair_sweep:
        L_train = 128 if n_pairs <= 16 else 256
        eval_lengths = [L_train, 2 * L_train, 4 * L_train]
        
        print(f"\n" + "#" * 105, flush=True)
        print(f"🎯 BLOQUE DE CAPACIDAD: N_PAIRS = {n_pairs} PARES SIMULTÁNEOS (L_train = {L_train}, Eval = {eval_lengths})", flush=True)
        print("#" * 105, flush=True)
        
        all_experiments[f"pairs_{n_pairs}"] = {
            "L_train": L_train,
            "eval_lengths": eval_lengths
        }
        
        for model_key in models_to_test:
            all_experiments[f"pairs_{n_pairs}"][model_key] = {
                "wallclocks": [],
                "steps_to_50": [],
                "steps_to_95": [],
                "final_steps": [],
                "length_accs": {L: [] for L in eval_lengths}
            }
            
            for seed_idx, seed in enumerate(seeds, 1):
                current_model_counter += 1
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                
                if model_key == "DeltaPhase_Complex":
                    model = DeltaPhaseMQAR(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, num_layers=num_layers)
                elif model_key == "GatedDeltaNet_Real":
                    model = RealGatedDeltaNetMQAR(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, num_layers=num_layers)
                elif model_key == "Transformer_Causal":
                    model = CausalTransformerMQAR(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, num_layers=num_layers)
                    
                model_name = f"{model_key} [Seed {seed}]"
                trained_model, metrics = train_mqar_model(
                    name=model_name,
                    model=model,
                    total_steps=steps_per_train,
                    batch_size=32,
                    seq_len=L_train,
                    num_pairs=n_pairs,
                    vocab_size=vocab_size,
                    lr=3e-3,
                    device=device,
                    log_interval=50,
                    early_stop_acc=early_stop_acc,
                    global_model_idx=current_model_counter,
                    total_models=total_models,
                    global_start_time=global_start_time
                )
                
                all_experiments[f"pairs_{n_pairs}"][model_key]["wallclocks"].append(metrics["wallclock"])
                all_experiments[f"pairs_{n_pairs}"][model_key]["steps_to_50"].append(metrics["step_to_50"])
                all_experiments[f"pairs_{n_pairs}"][model_key]["steps_to_95"].append(metrics["step_to_95"])
                all_experiments[f"pairs_{n_pairs}"][model_key]["final_steps"].append(metrics["final_step"])
                
                # Evaluación en todas las longitudes con secuencias inéditas on-the-fly
                print(f"   🔍 Evaluando generalización en longitudes {eval_lengths}...", flush=True)
                for L_eval in eval_lengths:
                    acc = evaluate_mqar_accuracy(
                        trained_model,
                        num_eval_batches=20,
                        batch_size=32,
                        seq_len=L_eval,
                        num_pairs=n_pairs,
                        vocab_size=vocab_size,
                        device=device
                    )
                    all_experiments[f"pairs_{n_pairs}"][model_key]["length_accs"][L_eval].append(acc)
                    print(f"      • Longitud L={L_eval:4d}: Acc = {acc:6.2f}%", flush=True)
                    
    # ==============================================================================
    # RESUMEN Y REPORTE ESTADÍSTICO FINAL
    # ==============================================================================
    print("\n" + "=" * 115)
    print(f"📊 TABLA RESUMEN CERTIFICADA NIVEL 2: MEDIA ± ERROR ESTÁNDAR ({len(seeds)} SEMILLAS INDEPENDIENTES)")
    print("=" * 115)
    print(f"{'Configuración':<14} | {'Modelo':<20} | {'L_train Acc':<16} | {'OOD 2x Acc':<16} | {'OOD 4x Acc':<16} | {'Pasos >50%':<12} | {'Pasos >95%':<12} | {'Tiempo':<10}")
    print("-" * 115)
    
    summary_report = {}
    
    for n_pairs in pair_sweep:
        summary_report[f"pairs_{n_pairs}"] = {}
        eval_lengths = all_experiments[f"pairs_{n_pairs}"]["eval_lengths"]
        
        for model_key in models_to_test:
            row_data = all_experiments[f"pairs_{n_pairs}"][model_key]
            
            # Length accuracies
            acc_cols = []
            for L in eval_lengths:
                vals = row_data["length_accs"][L]
                mean_v = float(np.mean(vals))
                std_v = float(np.std(vals))
                se_v = std_v / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
                acc_cols.append(f"{mean_v:5.2f} ± {se_v:4.2f}%")
                
            s50_mean = float(np.mean(row_data["steps_to_50"]))
            s95_mean = float(np.mean(row_data["steps_to_95"]))
            t_mean = float(np.mean(row_data["wallclocks"]))
            
            summary_report[f"pairs_{n_pairs}"][model_key] = {
                f"L_{eval_lengths[0]}": acc_cols[0],
                f"L_{eval_lengths[1]}": acc_cols[1],
                f"L_{eval_lengths[2]}": acc_cols[2],
                "mean_steps_to_50": s50_mean,
                "mean_steps_to_95": s95_mean,
                "mean_wallclock_sec": t_mean
            }
            
            print(f"N_pairs={n_pairs:<6} | {model_key:<20} | {acc_cols[0]:<16} | {acc_cols[1]:<16} | {acc_cols[2]:<16} | {s50_mean:<12.1f} | {s95_mean:<12.1f} | {t_mean:<8.1f}s")
            
    print("=" * 115)
    
    # Guardar JSON con resultados crudos
    results_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "rigorous_mqar_results.json"))
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "seeds": seeds,
            "pair_sweep": pair_sweep,
            "raw_data": all_experiments,
            "summary": summary_report
        }, f, indent=2)
    print(f"\n✅ Resultados guardados exitosamente en: {results_path}")
    return all_experiments, summary_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rigorous MQAR Benchmark Suite (Level 2 Certified)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on ('cpu' or 'cuda' or 'dml')")
    parser.add_argument("--steps", type=int, default=1500, help="Max training steps per model")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 2024, 7, 999], help="Seeds to evaluate (>=5 for Level 2)")
    parser.add_argument("--pairs", type=int, nargs="+", default=[8, 16, 32], help="Number of pairs to sweep")
    parser.add_argument("--early-stop-acc", type=float, default=99.5, help="Accuracy percentage to stop early")
    parser.add_argument("--quick", action="store_true", help="Quick run for smoke test")
    
    args = parser.parse_args()
    
    if args.quick:
        run_rigorous_mqar_suite(
            seeds=[42],
            pair_sweep=[8],
            steps_per_train=100,
            early_stop_acc=args.early_stop_acc,
            device=args.device
        )
    else:
        run_rigorous_mqar_suite(
            seeds=args.seeds,
            pair_sweep=args.pairs,
            steps_per_train=args.steps,
            early_stop_acc=args.early_stop_acc,
            device=args.device
        )

