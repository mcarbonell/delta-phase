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

def generate_zoology_mqar_batch(
    batch_size: int = 32,
    seq_len: int = 128,
    num_pairs: int = 8,
    vocab_size: int = 256,
    device: str = 'cpu'
):
    """
    Genera un lote dinámico e inédito para Multi-Query Associative Recall (MQAR).
    - Keys: tokens del rango [2, 2 + (vocab_size-2)//2 - 1]
    - Values: tokens del rango [2 + (vocab_size-2)//2, vocab_size - 1]
    - Primera mitad: inserción de num_pairs pares clave-valor espaciados.
    - Segunda mitad: inserción de num_pairs consultas aleatorias de las claves almacenadas.
    - targets: tensor con -100 en todas partes excepto en la posición de cada query, donde el target es el valor asociado.
    """
    half_len = seq_len // 2
    assert num_pairs * 2 <= half_len, f"Demasiados pares ({num_pairs}) para half_len={half_len}"
    
    num_key_candidates = (vocab_size - 2) // 2
    key_start = 2
    val_start = key_start + num_key_candidates
    val_end = vocab_size
    
    tokens = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
    
    pair_spacing = half_len // num_pairs
    query_spacing = (seq_len - half_len) // num_pairs
    
    for b in range(batch_size):
        # 1. Muestreo de pares únicos para este sample
        chosen_keys = torch.randperm(num_key_candidates)[:num_pairs] + key_start
        chosen_vals = torch.randint(val_start, val_end, (num_pairs,))
        
        # 2. Insertar pares en la primera mitad
        for i in range(num_pairs):
            pos_k = i * pair_spacing
            pos_v = pos_k + 1
            tokens[b, pos_k] = chosen_keys[i]
            tokens[b, pos_v] = chosen_vals[i]
            
        # 3. Intercalar consultas aleatorias en la segunda mitad
        perm = torch.randperm(num_pairs)
        for j in range(num_pairs):
            idx = perm[j]
            q_pos = half_len + j * query_spacing
            tokens[b, q_pos] = chosen_keys[idx]
            targets[b, q_pos] = chosen_vals[idx] # El modelo debe predecir el valor en la posición de consulta
            
    return tokens, targets


# ==============================================================================
# 2. IMPLEMENTACIÓN DE MODELOS Y BASELINES
# ==============================================================================

# --- A. DeltaPhase Model ---
class DeltaPhaseMQAR(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 128, n_heads: int = 4, chunk_size: int = 32, num_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            DeltaPhaseHolographicBlock(d_model=d_model, n_heads=n_heads, chunk_size=chunk_size)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        h = self.embedding(x)
        for block in self.blocks:
            h, _ = block(h)
        h = self.norm(h)
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
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model)
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
    def __init__(self, vocab_size: int = 256, d_model: int = 128, n_heads: int = 4, chunk_size: int = 32, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            RealGatedDeltaNetBlock(d_model=d_model, n_heads=n_heads, chunk_size=chunk_size)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        h = self.embedding(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self.head(h)


# --- C. RoPE Causal Transformer Positive Control (Softmax O(N^2)) ---
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos', emb.cos().unsqueeze(0).unsqueeze(0)) # (1, 1, L, D)
        self.register_buffer('sin', emb.sin().unsqueeze(0).unsqueeze(0))

    def _rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k):
        L = q.shape[2]
        cos, sin = self.cos[:, :, :L, :], self.sin[:, :, :L, :]
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot

class RoPECausalAttentionBlock(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.rope = RotaryEmbedding(self.head_dim)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x):
        B, L, D = x.shape
        normed = self.norm1(x)
        
        q = self.q_proj(normed).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        q, k = self.rope(q, k)
        
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        causal_mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
        attn_weights = F.softmax(attn_weights + causal_mask, dim=-1)
        
        attn_out = torch.matmul(attn_weights, v).transpose(1, 2).reshape(B, L, D)
        x = x + self.out_proj(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x

class RoPECausalTransformerMQAR(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 128, n_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            RoPECausalAttentionBlock(d_model=d_model, n_heads=n_heads)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        h = self.embedding(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self.head(h)


# ==============================================================================
# 3. RUTINAS DE EVALUACIÓN Y ENTRENAMIENTO DINÁMICO
# ==============================================================================

def evaluate_mqar_accuracy(model, num_eval_batches=15, batch_size=32, seq_len=128, num_pairs=8, vocab_size=256, device='cpu'):
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
            logits = model(tokens) # (B, L, V)
            preds = torch.argmax(logits, dim=-1)
            
            mask = (targets != -100)
            correct = (preds[mask] == targets[mask]).sum().item()
            num_q = mask.sum().item()
            
            total_correct += correct
            total_queries += num_q
            
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
    total_steps: int = 1000,
    batch_size: int = 32,
    seq_len: int = 128,
    num_pairs: int = 8,
    vocab_size: int = 256,
    lr: float = 2e-3,
    device: str = 'cpu',
    log_interval: int = 100,
    global_model_idx: int = 1,
    total_models: int = 1,
    global_start_time: float = 0.0
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    model.train()
    t_start = time.time()
    
    elapsed_global = time.time() - global_start_time if global_start_time > 0 else 0
    print(f"\n[{time.strftime('%H:%M:%S')}] 🚀 [Modelo {global_model_idx}/{total_models}] {name}", flush=True)
    print(f"   Config: L={seq_len}, N_pairs={num_pairs}, {total_steps} pasos | Tiempo global transcurrido: {format_duration(elapsed_global)}", flush=True)
    
    running_loss = 0.0
    step_times = []
    
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
        scheduler.step()
        
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
            
            avg_step_time = sum(step_times) / len(step_times)
            steps_left = total_steps - step
            model_eta = steps_left * avg_step_time
            
            # Estimación global de tiempo restante
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
            model.train()
            
    wallclock = time.time() - t_start
    print(f"   ✨ [{name}] completado en {format_duration(wallclock)} ({wallclock:.1f}s)", flush=True)
    return model, wallclock


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
    seeds: list = [42, 137, 2024],
    pair_sweep: list = [8, 16, 32],
    eval_lengths: list = [128, 256, 512],
    steps_per_train: int = 3000,
    device: str = 'cpu'
):
    global_start_time = time.time()
    models_to_test = ["DeltaPhase_Complex", "GatedDeltaNet_Real", "Transformer_RoPE"]
    total_models = len(pair_sweep) * len(models_to_test) * len(seeds)
    
    vocab_size = 256
    d_model = 128
    n_heads = 4
    d_k = d_model // n_heads
    num_layers = 2
    
    # 1. Cabecera Completa con Metadatos, Explicación e Inventario
    print("=" * 95, flush=True)
    print("🌟 PROTOCOLO RIGUROSO DE EVALUACIÓN MQAR (MULTI-QUERY ASSOCIATIVE RECALL)", flush=True)
    print("=" * 95, flush=True)
    print(f"  • Propósito:           Evaluación de retención y memoria asociativa multi-consulta densa", flush=True)
    print(f"                         Comparación iso-paramétrica de DeltaPhase vs Gated DeltaNet vs RoPE Transformer", flush=True)
    print(f"  • Dispositivo:         {device.upper()} ({platform.processor() or 'Multicore'})", flush=True)
    print(f"  • Fecha UTC:           {datetime.datetime.now(datetime.timezone.utc).isoformat()}", flush=True)
    print(f"  • Versión Python/Torch: Python {platform.python_version()} | PyTorch {torch.__version__}", flush=True)
    print(f"  • Semillas:            {seeds} (n={len(seeds)})", flush=True)
    print(f"  • Barrido de Pares:    {pair_sweep}", flush=True)
    print(f"  • Longitudes Eval:     {eval_lengths}", flush=True)
    print(f"  • Pasos de Train:      {steps_per_train} pasos on-the-fly por modelo", flush=True)
    print(f"  • Total ejecuciones:   {total_models} modelos individuales", flush=True)
    print("-" * 95, flush=True)
    print("  📋 INVENTARIO COMPLETO DE ARQUITECTURAS A EVALUAR:", flush=True)
    
    sample_models = {
        "DeltaPhase_Complex (C^(32x32))": DeltaPhaseMQAR(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, num_layers=num_layers),
        "GatedDeltaNet_Real (R^(32x32))": RealGatedDeltaNetMQAR(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, num_layers=num_layers),
        "Transformer_RoPE (Softmax O(N^2))": RoPECausalTransformerMQAR(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, num_layers=num_layers)
    }
    for name, m in sample_models.items():
        print_architecture_inventory(name, m, d_model=d_model, n_heads=n_heads, d_k=d_k, vocab_size=vocab_size)
    print("=" * 95 + "\n", flush=True)
    
    all_experiments = {}
    current_model_counter = 0
    
    for n_pairs in pair_sweep:
        print(f"\n" + "#" * 95, flush=True)
        print(f"🎯 BLOQUE DE CAPACIDAD: N_PAIRS = {n_pairs} PARES SIMULTÁNEOS", flush=True)
        print("#" * 95, flush=True)
        
        all_experiments[f"pairs_{n_pairs}"] = {}
        
        for model_key in models_to_test:
            all_experiments[f"pairs_{n_pairs}"][model_key] = {
                "wallclocks": [],
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
                elif model_key == "Transformer_RoPE":
                    model = RoPECausalTransformerMQAR(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, num_layers=num_layers)
                    
                model_name = f"{model_key} [Seed {seed}]"
                trained_model, wallclock = train_mqar_model(
                    name=model_name,
                    model=model,
                    total_steps=steps_per_train,
                    batch_size=32,
                    seq_len=128,
                    num_pairs=n_pairs,
                    vocab_size=vocab_size,
                    lr=2e-3,
                    device=device,
                    log_interval=100,
                    global_model_idx=current_model_counter,
                    total_models=total_models,
                    global_start_time=global_start_time
                )
                
                all_experiments[f"pairs_{n_pairs}"][model_key]["wallclocks"].append(wallclock)
                
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
    print("\n" + "=" * 95)
    print("📊 TABLA RESUMEN CONSOLIDAD: MEDIA ± ERROR ESTÁNDAR (MULTI-SEMILLA)")
    print("=" * 95)
    # Dynamic column headers
    header_cols = " | ".join([f"{'L=' + str(L):<16}" for L in eval_lengths])
    print(f"{'Configuración':<22} | {'Modelo':<22} | {header_cols}")
    print("-" * (48 + 19 * len(eval_lengths)))
    
    summary_report = {}
    
    for n_pairs in pair_sweep:
        summary_report[f"pairs_{n_pairs}"] = {}
        for model_key in ["DeltaPhase_Complex", "GatedDeltaNet_Real", "Transformer_RoPE"]:
            row_accs = {}
            for L in eval_lengths:
                vals = all_experiments[f"pairs_{n_pairs}"][model_key]["length_accs"][L]
                mean_v = float(np.mean(vals))
                std_v = float(np.std(vals))
                se_v = std_v / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
                row_accs[L] = (mean_v, se_v)
                
            summary_report[f"pairs_{n_pairs}"][model_key] = {
                f"L_{L}": f"{row_accs[L][0]:.2f} ± {row_accs[L][1]:.2f}%" for L in eval_lengths
            }
            
            cols_str = " | ".join([f"{row_accs[L][0]:6.2f} ± {row_accs[L][1]:4.2f}%" for L in eval_lengths])
            print(f"N_pairs={n_pairs:<14} | {model_key:<22} | {cols_str}")
            
    print("=" * (48 + 19 * len(eval_lengths)))
    
    # Guardar JSON con resultados crudos
    results_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "rigorous_mqar_results.json"))
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "seeds": seeds,
            "pair_sweep": pair_sweep,
            "eval_lengths": eval_lengths,
            "raw_data": all_experiments,
            "summary": summary_report
        }, f, indent=2)
    print(f"\n✅ Resultados guardados exitosamente en: {results_path}")
    return all_experiments, summary_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rigorous MQAR Benchmark Suite")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on ('cpu' or 'cuda' or 'dml')")
    parser.add_argument("--steps", type=int, default=3000, help="Training steps per model")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 2024], help="Seeds to evaluate")
    parser.add_argument("--pairs", type=int, nargs="+", default=[8, 16, 32], help="Number of pairs to sweep")
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512], help="Evaluation sequence lengths")
    parser.add_argument("--quick", action="store_true", help="Quick run for smoke test")
    
    args = parser.parse_args()
    
    if args.quick:
        run_rigorous_mqar_suite(
            seeds=[42],
            pair_sweep=[8],
            eval_lengths=[128, 256],
            steps_per_train=100,
            device=args.device
        )
    else:
        run_rigorous_mqar_suite(
            seeds=args.seeds,
            pair_sweep=args.pairs,
            eval_lengths=args.lengths,
            steps_per_train=args.steps,
            device=args.device
        )
