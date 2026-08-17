"""
tests/test_zk_group_expressivity.py
===================================
Rigorous benchmark comparing Complex Householder Beta DeltaPhase (beta_t = 1 + e^(i*phi_t), eigenvalues in S^1),
Real Beta Gated DeltaNet (beta_t in (0, 2), eigenvalues in (-1, 1)), Fixed Real Isometric Beta (beta=2.0),
and Softmax Causal Transformer (MHA) on Cumulative Modular Addition over Cyclic Groups Z_k (v353 Protocol).

Target Modulos:
- Z_7  (Odd Prime): Pure non-trivial cyclic group without non-trivial subgroups.
- Z_9  (Odd Composite 3^2): Tests compositionality vs single-step rotation.
- Z_12 (Even Composite 2^2 * 3): Parity shortcut vs full 12-state rotation.
"""

import os
import sys
import json
import time
import math
import argparse
import platform
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ==============================================================================
# 1. GENERACIÓN DE LOTES DE ARITMÉTICA MODULAR Z_k ON-THE-FLY
# ==============================================================================

def generate_zk_batch(batch_size: int = 64, seq_len: int = 64, modulo: int = 7, device: str = 'cpu'):
    """
    Genera secuencias acumulativas on-the-fly:
    - Input x: tokens x_t in {0, 1, ..., modulo-1}
    - Target y: (sum_{i=0}^t x_i) mod modulo
    """
    elements = torch.randint(0, modulo, (batch_size, seq_len), device=device)
    targets = torch.cumsum(elements, dim=1) % modulo
    return elements, targets


def format_duration(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}m {s:02d}s"

# ==============================================================================
# 2. BLOQUES DE ARQUITECTURA ISO-PARAMÉTRICOS
# ==============================================================================

class ShortCausalConv1D(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size - 1, groups=d_model)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        conv_out = self.conv(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
        return x + self.act(conv_out)


# --- A. DeltaPhase (Complex Beta in S^1) ---
class ComplexBetaDeltaPhaseBlock(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4, d_k: int = 16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.inv_dk = 1.0 / float(d_k)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.conv = ShortCausalConv1D(d_model, kernel_size=4)
        self.w_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_phi = nn.Linear(d_model, n_heads, bias=False)
        
        self.out_proj = nn.Linear(n_heads * d_k, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        nx = self.conv(self.norm1(x))
        B, L, D = nx.shape
        
        theta_k = self.w_k(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        theta_q = self.w_q(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        phi_beta = self.w_phi(nx).transpose(1, 2)
        
        K = torch.polar(torch.ones_like(theta_k), theta_k)
        Q = torch.polar(torch.ones_like(theta_q), theta_q)
        beta_complex = 1.0 + torch.polar(torch.ones_like(phi_beta), phi_beta)
        
        M_state = torch.zeros(B, self.n_heads, self.d_k, self.d_k, dtype=torch.complex64, device=x.device)
        out_list = []
        
        for t in range(L):
            kt, qt, vt, bt = K[:, :, t], Q[:, :, t], v[:, :, t], beta_complex[:, :, t]
            v_old = torch.matmul(M_state, torch.conj(kt).unsqueeze(-1)).squeeze(-1).real * self.inv_dk
            err = vt - v_old
            err_c = err.to(torch.complex64)
            update = bt.unsqueeze(-1).unsqueeze(-1) * torch.matmul(err_c.unsqueeze(-1), kt.unsqueeze(-2))
            M_state = M_state + update
            out_t = torch.matmul(M_state, torch.conj(qt).unsqueeze(-1)).squeeze(-1).real * self.inv_dk
            out_list.append(out_t)
            
        out_concat = torch.stack(out_list, dim=2).transpose(1, 2).reshape(B, L, self.n_heads * self.d_k)
        out = res + self.out_proj(out_concat)
        return out + self.ffn(self.norm2(out))


# --- B. Gated DeltaNet (Real Beta in (0, 2)) ---
class RealBetaDeltaNetBlock(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4, d_k: int = 16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.inv_dk = 1.0 / float(d_k)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.conv = ShortCausalConv1D(d_model, kernel_size=4)
        self.w_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_beta = nn.Linear(d_model, n_heads, bias=False)
        
        self.out_proj = nn.Linear(n_heads * d_k, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        nx = self.conv(self.norm1(x))
        B, L, D = nx.shape
        
        k = F.normalize(self.w_k(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2), p=2, dim=-1)
        q = F.normalize(self.w_q(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2), p=2, dim=-1)
        v = self.w_v(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        beta_real = 2.0 * torch.sigmoid(self.w_beta(nx)).transpose(1, 2)
        
        M_state = torch.zeros(B, self.n_heads, self.d_k, self.d_k, device=x.device)
        out_list = []
        
        for t in range(L):
            kt, qt, vt, bt = k[:, :, t], q[:, :, t], v[:, :, t], beta_real[:, :, t]
            v_old = torch.matmul(M_state, kt.unsqueeze(-1)).squeeze(-1) * self.inv_dk
            err = vt - v_old
            update = bt.unsqueeze(-1).unsqueeze(-1) * torch.matmul(err.unsqueeze(-1), kt.unsqueeze(-2))
            M_state = M_state + update
            out_t = torch.matmul(M_state, qt.unsqueeze(-1)).squeeze(-1) * self.inv_dk
            out_list.append(out_t)
            
        out_concat = torch.stack(out_list, dim=2).transpose(1, 2).reshape(B, L, self.n_heads * self.d_k)
        out = res + self.out_proj(out_concat)
        return out + self.ffn(self.norm2(out))


# --- C. Fixed Real Isometric Beta Block (beta = 2.0) ---
class FixedIsometricRealBetaBlock(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4, d_k: int = 16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.inv_dk = 1.0 / float(d_k)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.conv = ShortCausalConv1D(d_model, kernel_size=4)
        self.w_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * d_k, bias=False)
        
        self.out_proj = nn.Linear(n_heads * d_k, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        nx = self.conv(self.norm1(x))
        B, L, D = nx.shape
        
        k = F.normalize(self.w_k(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2), p=2, dim=-1)
        q = F.normalize(self.w_q(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2), p=2, dim=-1)
        v = self.w_v(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        
        M_state = torch.zeros(B, self.n_heads, self.d_k, self.d_k, device=x.device)
        out_list = []
        
        for t in range(L):
            kt, qt, vt = k[:, :, t], q[:, :, t], v[:, :, t]
            v_old = torch.matmul(M_state, kt.unsqueeze(-1)).squeeze(-1) * self.inv_dk
            err = vt - v_old
            update = 2.0 * torch.matmul(err.unsqueeze(-1), kt.unsqueeze(-2))
            M_state = M_state + update
            out_t = torch.matmul(M_state, qt.unsqueeze(-1)).squeeze(-1) * self.inv_dk
            out_list.append(out_t)
            
        out_concat = torch.stack(out_list, dim=2).transpose(1, 2).reshape(B, L, self.n_heads * self.d_k)
        out = res + self.out_proj(out_concat)
        return out + self.ffn(self.norm2(out))


# --- D. Transformer Causal (Multi-Head Softmax Attention Baseline) ---
class CausalTransformerBlock(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4, d_k: int = 16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.scale = 1.0 / math.sqrt(d_k)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.conv = ShortCausalConv1D(d_model, kernel_size=4)
        self.w_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.out_proj = nn.Linear(n_heads * d_k, d_model)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        nx = self.conv(self.norm1(x))
        B, L, D = nx.shape
        
        q = self.w_q(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2) # (B, H, L, d_k)
        k = self.w_k(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        causal_mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
        attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)
        attn_probs = F.softmax(attn_weights, dim=-1)
        
        out_heads = torch.matmul(attn_probs, v).transpose(1, 2).reshape(B, L, self.n_heads * self.d_k)
        out = res + self.out_proj(out_heads)
        return out + self.ffn(self.norm2(out))


# --- Envoltorio Completo del Modelo de Lenguaje ---
class ZkModelLM(nn.Module):
    def __init__(self, block_cls, modulo: int = 7, d_model: int = 64, n_layers: int = 2, n_heads: int = 4, d_k: int = 16):
        super().__init__()
        self.modulo = modulo
        self.d_model = d_model
        self.embed = nn.Embedding(modulo, d_model)
        self.pos_embed = nn.Embedding(1024, d_model)
        self.layers = nn.ModuleList([block_cls(d_model=d_model, n_heads=n_heads, d_k=d_k) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, modulo)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


# ==============================================================================
# 3. EVALUACIÓN Y ENTRENAMIENTO CON MONITORIZACIÓN DE CURVAS Y EARLY STOPPING
# ==============================================================================

def evaluate_zk_accuracy(model: nn.Module, modulo: int = 7, seq_len: int = 64, num_batches: int = 20, batch_size: int = 64, device: str = 'cpu') -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = generate_zk_batch(batch_size=batch_size, seq_len=seq_len, modulo=modulo, device=device)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += x.numel()
    return (correct / total) * 100.0


def train_zk_model_with_curves(
    name: str,
    model: nn.Module,
    modulo: int = 7,
    seq_len: int = 64,
    total_steps: int = 1500,
    batch_size: int = 64,
    lr: float = 2e-3,
    weight_decay: float = 0.0,
    log_interval: int = 50,
    early_stop_acc: float = 99.5,
    device: str = 'cpu',
    global_model_idx: int = 1,
    total_models: int = 1,
    global_start_time: float = 0.0
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    t_start = time.time()
    
    elapsed_global = time.time() - global_start_time if global_start_time > 0 else 0
    chance_acc = 100.0 / modulo
    print(f"\n[{time.strftime('%H:%M:%S')}] 🚀 [Modelo {global_model_idx}/{total_models}] {name} (Z_{modulo} | Azar: {chance_acc:.2f}%)", flush=True)
    print(f"   Config: L={seq_len}, max {total_steps} pasos | Opt: AdamW (lr={lr:.1e}, wd={weight_decay:.1e}) | Tiempo transcurrido: {format_duration(elapsed_global)}", flush=True)
    
    running_loss = 0.0
    step_times = []
    step_to_50 = None
    step_to_80 = None
    step_to_95 = None
    final_step = total_steps
    early_stopped = False
    
    for step in range(1, total_steps + 1):
        t0 = time.time()
        x, y = generate_zk_batch(batch_size=batch_size, seq_len=seq_len, modulo=modulo, device=device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, modulo), y.view(-1))
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
            val_acc = evaluate_zk_accuracy(model, modulo=modulo, seq_len=seq_len, num_batches=10, batch_size=batch_size, device=device)
            
            if step_to_50 is None and val_acc >= 50.0:
                step_to_50 = step
            if step_to_80 is None and val_acc >= 80.0:
                step_to_80 = step
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
                f"Loss: {avg_loss:6.4f} | Acc: {val_acc:6.2f}% | "
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
    final_val_acc = evaluate_zk_accuracy(model, modulo=modulo, seq_len=seq_len, num_batches=25, batch_size=batch_size, device=device)
    status_str = f"completado en paso {final_step} (Early Stop)" if early_stopped else f"completado en {total_steps} pasos"
    print(f"   ✨ [{name}] {status_str} en {format_duration(wallclock)} ({wallclock:.1f}s) | Acc Final: {final_val_acc:.2f}%", flush=True)
    
    metrics = {
        "final_acc": final_val_acc,
        "wallclock": wallclock,
        "final_step": final_step,
        "step_to_50": step_to_50 if step_to_50 is not None else total_steps,
        "step_to_80": step_to_80 if step_to_80 is not None else total_steps,
        "step_to_95": step_to_95 if step_to_95 is not None else total_steps,
        "early_stopped": early_stopped
    }
    return model, metrics


def print_architecture_inventory(name: str, model: nn.Module, d_model: int, n_heads: int, d_k: int, modulo: int):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  📦 [ARQUITECTURA] {name}", flush=True)
    print(f"     • Dimensiones:  d_model={d_model}, n_heads={n_heads}, d_k={d_k}, Modulo={modulo}", flush=True)
    print(f"     • Parámetros:   Total={total_params:,} | Entrenables={trainable_params:,}", flush=True)
    print(f"     • Desglose por capas y módulos:", flush=True)
    for idx, (layer_name, module) in enumerate(model.named_children()):
        mod_params = sum(p.numel() for p in module.parameters())
        print(f"        [{idx}] {layer_name:<18s} | Clase: {module.__class__.__name__:<30s} | Params: {mod_params:,}", flush=True)


# ==============================================================================
# 4. SUITE DE BENCHMARK RIGUROSA MULTI-SEMILLA Y MULTI-GRUPO
# ==============================================================================

def run_rigorous_zk_suite(
    modulos: list = [7, 9, 12],
    seeds: list = [42, 137, 2024],
    steps_per_train: int = 1500,
    seq_len: int = 64,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    lr: float = 2e-3,
    weight_decay: float = 0.0,
    early_stop_acc: float = 99.5,
    device: str = 'cpu'
):
    global_start_time = time.time()
    d_k = d_model // n_heads
    
    arms = {
        "Transformer_Causal": CausalTransformerBlock,
        "DeltaPhase_Complex": ComplexBetaDeltaPhaseBlock,
        "GatedDeltaNet_Real": RealBetaDeltaNetBlock,
        "DeltaNet_FixedIso":  FixedIsometricRealBetaBlock,
    }
    
    total_models = len(modulos) * len(arms) * len(seeds)
    
    # Cabecera reglamentaria
    print("=" * 110, flush=True)
    print("🌟 PROTOCOLO RIGUROSO DE EXPRESIVIDAD EN GRUPOS CÍCLICOS Z_k & TRANSICIONES DE FASE (GROKKING)", flush=True)
    print("=" * 110, flush=True)
    print(f"  • Propósito:           Evaluación de álgebra fasorial unitaria U(d) vs reflexiones ortogonales O(d)", flush=True)
    print(f"                         y atención cuadrática en grupos primos (Z_7), compuestos impares (Z_9) y pares (Z_12)", flush=True)
    print(f"  • Dispositivo:         {device.upper()} ({platform.processor() or 'Multicore'})", flush=True)
    print(f"  • Fecha UTC:           {datetime.datetime.now(datetime.timezone.utc).isoformat()}", flush=True)
    print(f"  • Versión Python/Torch: Python {platform.python_version()} | PyTorch {torch.__version__}", flush=True)
    print(f"  • Optimizador:         AdamW (lr={lr:.1e}, weight_decay={weight_decay:.1e}, beta1=0.9, beta2=0.999, grad_clip=1.0)", flush=True)
    print(f"  • Semillas ({len(seeds)}):      {seeds}", flush=True)
    print(f"  • Módulos evaluados:   Z_{modulos} a longitud L={seq_len}", flush=True)
    print(f"  • Max Pasos Train:     {steps_per_train} pasos on-the-fly (Early Stop @ {early_stop_acc}%)", flush=True)
    print(f"  • Total ejecuciones:   {total_models} modelos individuales", flush=True)
    print("-" * 110, flush=True)
    print("  📋 INVENTARIO COMPLETO DE ARQUITECTURAS A EVALUAR (para Z_7):", flush=True)
    
    for arm_name, arm_cls in arms.items():
        sample_model = ZkModelLM(arm_cls, modulo=7, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_k=d_k)
        print_architecture_inventory(arm_name, sample_model, d_model=d_model, n_heads=n_heads, d_k=d_k, modulo=7)
    print("=" * 110 + "\n", flush=True)
    
    all_experiments = {}
    current_model_counter = 0
    
    for k in modulos:
        mod_type = "Primo Impar" if k == 7 else ("Compuesto Impar (3^2)" if k == 9 else "Compuesto Par (2^2*3)")
        chance = 100.0 / k
        print("\n" + "#" * 110, flush=True)
        print(f"🎯 BLOQUE DE GRUPO CÍCLICO: Z_{k} [{mod_type}] (Nivel de azar uniforme = {chance:.2f}%)", flush=True)
        print("#" * 110, flush=True)
        
        all_experiments[f"Z_{k}"] = {
            "type": mod_type,
            "chance": chance,
            "arms": {}
        }
        
        for arm_name, arm_cls in arms.items():
            all_experiments[f"Z_{k}"]["arms"][arm_name] = {
                "final_accs": [],
                "wallclocks": [],
                "steps_to_50": [],
                "steps_to_80": [],
                "steps_to_95": [],
                "final_steps": []
            }
            
            for seed in seeds:
                current_model_counter += 1
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                model = ZkModelLM(arm_cls, modulo=k, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_k=d_k)
                model_name = f"{arm_name} [Seed {seed}]"
                
                trained_model, metrics = train_zk_model_with_curves(
                    name=model_name,
                    model=model,
                    modulo=k,
                    seq_len=seq_len,
                    total_steps=steps_per_train,
                    batch_size=64,
                    lr=lr,
                    weight_decay=weight_decay,
                    log_interval=50,
                    early_stop_acc=early_stop_acc,
                    device=device,
                    global_model_idx=current_model_counter,
                    total_models=total_models,
                    global_start_time=global_start_time
                )
                
                all_experiments[f"Z_{k}"]["arms"][arm_name]["final_accs"].append(metrics["final_acc"])
                all_experiments[f"Z_{k}"]["arms"][arm_name]["wallclocks"].append(metrics["wallclock"])
                all_experiments[f"Z_{k}"]["arms"][arm_name]["steps_to_50"].append(metrics["step_to_50"])
                all_experiments[f"Z_{k}"]["arms"][arm_name]["steps_to_80"].append(metrics["step_to_80"])
                all_experiments[f"Z_{k}"]["arms"][arm_name]["steps_to_95"].append(metrics["step_to_95"])
                all_experiments[f"Z_{k}"]["arms"][arm_name]["final_steps"].append(metrics["final_step"])
                
    # ==============================================================================
    # 5. TABLA RESUMEN CONSOLIDADA
    # ==============================================================================
    print("\n" + "=" * 120)
    print(f"📊 TABLA RESUMEN CERTIFICADA Z_k: MEDIA ± ERROR ESTÁNDAR ({len(seeds)} SEMILLAS INDEPENDIENTES)")
    print("=" * 120)
    print(f"{'Grupo':<8} | {'Tipo':<20} | {'Azar':<8} | {'Modelo':<22} | {'Precisión Final':<16} | {'Pasos >50%':<12} | {'Pasos >80%':<12} | {'Tiempo (s)':<10}")
    print("-" * 120)
    
    for k in modulos:
        chance = all_experiments[f"Z_{k}"]["chance"]
        mod_type = all_experiments[f"Z_{k}"]["type"]
        for arm_name in arms.keys():
            data = all_experiments[f"Z_{k}"]["arms"][arm_name]
            acc_vals = data["final_accs"]
            mean_acc = float(np.mean(acc_vals))
            se_acc = float(np.std(acc_vals) / math.sqrt(len(acc_vals))) if len(acc_vals) > 1 else 0.0
            
            s50 = float(np.mean(data["steps_to_50"]))
            s80 = float(np.mean(data["steps_to_80"]))
            t_mean = float(np.mean(data["wallclocks"]))
            
            print(f"Z_{k:<6} | {mod_type:<20} | {chance:5.2f}%  | {arm_name:<22} | {mean_acc:5.2f} ± {se_acc:4.2f}% | {s50:<12.1f} | {s80:<12.1f} | {t_mean:<8.1f}s")
        print("-" * 120)
    # Guardar JSON con resultados crudos
    results_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "rigorous_zk_results.json"))
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "seeds": seeds,
            "modulos": modulos,
            "lr": lr,
            "weight_decay": weight_decay,
            "raw_data": all_experiments
        }, f, indent=2)
    print(f"\n✅ Resultados Z_k guardados exitosamente en: {results_path}", flush=True)
    return all_experiments


def run_lr_sweep(modulos=[7], seeds=[42], lrs=[1e-3, 2e-3, 3e-3, 5e-3], steps=1000, device='cpu'):
    print("=" * 110, flush=True)
    print("🔬 BARRIDO SISTEMÁTICO DE TASAS DE APRENDIZAJE (LR SWEEP & ESTABILIDAD GROKKING)", flush=True)
    print("=" * 110, flush=True)
    print(f"  • LRs a evaluar:       {lrs}", flush=True)
    print(f"  • Pasos por run:       {steps}", flush=True)
    print(f"  • Módulos / Semillas:  Z_{modulos} | Semillas: {seeds}", flush=True)
    print("=" * 110 + "\n", flush=True)
    
    sweep_results = {}
    for lr_val in lrs:
        print(f"\n🎯 [BARRIDO LR = {lr_val:.1e}] Ejecutando suite...", flush=True)
        res = run_rigorous_zk_suite(
            modulos=modulos,
            seeds=seeds,
            steps_per_train=steps,
            lr=lr_val,
            weight_decay=0.0,
            device=device
        )
        sweep_results[f"lr_{lr_val}"] = res
    return sweep_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rigorous Z_k Group Expressivity Benchmark")
    parser.add_argument("--device", type=str, default="cpu", help="Device ('cpu', 'cuda', 'dml')")
    parser.add_argument("--steps", type=int, default=1500, help="Training steps per model")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 2024], help="Seeds to evaluate")
    parser.add_argument("--modulos", type=int, nargs="+", default=[7, 9, 12], help="Modulos Z_k to test")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--early-stop-acc", type=float, default=99.5, help="Accuracy threshold for early stopping")
    parser.add_argument("--lr-sweep", action="store_true", help="Run a systematic LR sweep")
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 2e-3, 3e-3, 5e-3], help="LRs for sweep")
    parser.add_argument("--quick", action="store_true", help="Quick run for smoke test")
    
    args = parser.parse_args()
    
    if args.lr_sweep:
        run_lr_sweep(
            modulos=args.modulos,
            seeds=args.seeds,
            lrs=args.lrs,
            steps=args.steps,
            device=args.device
        )
    elif args.quick:
        run_rigorous_zk_suite(
            modulos=[7],
            seeds=[42],
            steps_per_train=100,
            seq_len=args.seq_len,
            lr=args.lr,
            weight_decay=args.weight_decay,
            early_stop_acc=args.early_stop_acc,
            device=args.device
        )
    else:
        run_rigorous_zk_suite(
            modulos=args.modulos,
            seeds=args.seeds,
            steps_per_train=args.steps,
            seq_len=args.seq_len,
            lr=args.lr,
            weight_decay=args.weight_decay,
            early_stop_acc=args.early_stop_acc,
            device=args.device
        )
