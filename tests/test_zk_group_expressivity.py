"""
tests/test_zk_group_expressivity.py
===================================
Rigorous benchmark comparing Complex Householder Beta (beta_t = 1 + e^(i*phi_t), eigenvalues in S^1)
against Real Householder Beta (beta_t in R, eigenvalues in {-1, 1}) on Modular Addition over Cyclic Groups Z_k.
"""

import os
import sys
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

def generate_zk_batch(batch_size=64, seq_len=64, k_mod=7, device='cpu'):
    """
    Generates cumulative modular addition sequences over Z_k:
    Input tokens x_t in {0, 1, ..., k_mod-1}
    Target at position t is (sum_{i=0}^t x_i) mod k_mod
    """
    tokens = torch.randint(0, k_mod, (batch_size, seq_len), dtype=torch.long, device=device)
    targets = torch.cumsum(tokens, dim=1) % k_mod
    return tokens, targets

class ComplexBetaRecurrentCell(nn.Module):
    """Complex Householder Beta parameterized as beta_t = 1 + exp(i * phi_t) with eigenvalues on S^1"""
    def __init__(self, k_mod=7, d_model=32):
        super().__init__()
        self.k_mod = k_mod
        self.d_model = d_model
        self.emb = nn.Embedding(k_mod, d_model)
        self.w_phi = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.head = nn.Linear(d_model * 2, k_mod)

    def forward(self, x):
        B, L = x.shape
        h = self.emb(x) # (B, L, D)
        phi = self.w_phi(h) # Phase angle
        beta_complex = 1.0 + torch.complex(torch.cos(phi), torch.sin(phi))
        
        # Unit phasor key
        theta_k = self.w_k(h)
        k = torch.complex(torch.cos(theta_k), torch.sin(theta_k))
        
        # Recurrent state in C^d
        state = torch.zeros(B, self.d_model, dtype=torch.complex64, device=x.device)
        outputs = []
        
        for t in range(L):
            kt = k[:, t]
            bt = beta_complex[:, t]
            # State rotation via unit phasor and complex beta
            state = (1.0 - bt) * state + bt * kt
            # Readout real and imag components
            features = torch.cat([state.real, state.imag], dim=-1)
            outputs.append(self.head(features))
            
        return torch.stack(outputs, dim=1)

class RealBetaRecurrentCell(nn.Module):
    """Real Householder Beta parameterized as beta_t in (0, 2) with real eigenvalues in (-1, 1)"""
    def __init__(self, k_mod=7, d_model=32):
        super().__init__()
        self.k_mod = k_mod
        self.d_model = d_model
        self.emb = nn.Embedding(k_mod, d_model)
        self.w_beta = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.head = nn.Linear(d_model, k_mod)

    def forward(self, x):
        B, L = x.shape
        h = self.emb(x)
        beta_real = 2.0 * torch.sigmoid(self.w_beta(h))
        k_real = F.normalize(self.w_k(h), p=2, dim=-1)
        
        state = torch.zeros(B, self.d_model, device=x.device)
        outputs = []
        
        for t in range(L):
            kt = k_real[:, t]
            bt = beta_real[:, t]
            state = (1.0 - bt) * state + bt * kt
            outputs.append(self.head(state))
            
        return torch.stack(outputs, dim=1)

def evaluate_zk(model, k_mod=7, seq_len=64, num_batches=20, batch_size=64, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = generate_zk_batch(batch_size, seq_len, k_mod, device)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            # Evaluate accuracy on final sequence step
            correct += (preds[:, -1] == y[:, -1]).sum().item()
            total += batch_size
    return (correct / total) * 100.0

def train_zk_model(model, k_mod=7, seq_len=64, steps=1000, lr=3e-3, device='cpu'):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    model.train()
    
    for s in range(1, steps + 1):
        x, y = generate_zk_batch(64, seq_len, k_mod, device)
        opt.zero_grad()
        logits = model(x)
        loss = crit(logits.view(-1, k_mod), y.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
    return model

def run_zk_benchmark(k_mods=[7, 12], seq_len=64, seeds=[42, 137, 2024], steps=1000, device='cpu'):
    print("=" * 95, flush=True)
    print("🌟 BENCHMARK RIGUROSO DE EXPRESIVIDAD EN GRUPOS CÍCLICOS Z_k (v350 AUDIT)", flush=True)
    print("=" * 95, flush=True)
    print(f"  • Propósito:       Evaluación de representación exacta de aritmética modular Z_k en S^1 vs R", flush=True)
    print(f"  • Dispositivo:     {device.upper()} ({platform.processor() or 'Multicore'})", flush=True)
    print(f"  • Fecha UTC:       {datetime.datetime.now(datetime.timezone.utc).isoformat()}", flush=True)
    print(f"  • Semillas ({len(seeds)}):  {seeds}", flush=True)
    print(f"  • Grupos Cíclicos: Z_{k_mods} a longitud L={seq_len}", flush=True)
    print("-" * 95, flush=True)
    
    results = {}
    
    for k in k_mods:
        results[f"Z_{k}"] = {"complex": [], "real": []}
        print(f"\n🎯 Evaluando Grupo Cíclico Z_{k} (Azar uniforme = {100.0/k:.2f}%):", flush=True)
        
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # 1. Complex Model
            m_comp = ComplexBetaRecurrentCell(k_mod=k, d_model=32).to(device)
            m_comp = train_zk_model(m_comp, k_mod=k, seq_len=seq_len, steps=steps, lr=3e-3, device=device)
            acc_c = evaluate_zk(m_comp, k_mod=k, seq_len=seq_len, device=device)
            results[f"Z_{k}"]["complex"].append(acc_c)
            
            # 2. Real Model
            m_real = RealBetaRecurrentCell(k_mod=k, d_model=32).to(device)
            m_real = train_zk_model(m_real, k_mod=k, seq_len=seq_len, steps=steps, lr=3e-3, device=device)
            acc_r = evaluate_zk(m_real, k_mod=k, seq_len=seq_len, device=device)
            results[f"Z_{k}"]["real"].append(acc_r)
            
            print(f"   [Semilla {seed:4d}] Z_{k:2d} -> Complex: {acc_c:6.2f}% | Real: {acc_r:6.2f}% | Ventaja: {acc_c - acc_r:+6.2f}%", flush=True)
            
    print("\n" + "=" * 95)
    print(f"📊 RESUMEN FINAL CERTIFICADO: EXPRESIVIDAD EN GRUPOS CÍCLICOS Z_k ({len(seeds)} SEMILLAS)")
    print("=" * 95)
    print(f"{'Grupo':<10} | {'Azar':<10} | {'Complex Beta (S^1)':<24} | {'Real Beta (R)':<24} | {'Ventaja Fasorial':<16}")
    print("-" * 95)
    for k in k_mods:
        vals_c = results[f"Z_{k}"]["complex"]
        vals_r = results[f"Z_{k}"]["real"]
        mc, sec = float(np.mean(vals_c)), float(np.std(vals_c)/math.sqrt(len(vals_c)))
        mr, ser = float(np.mean(vals_r)), float(np.std(vals_r)/math.sqrt(len(vals_r)))
        chance = 100.0 / k
        gap = mc - mr
        print(f"Z_{k:<8} | {chance:5.2f}%    | {mc:5.2f} ± {sec:4.2f}%              | {mr:5.2f} ± {ser:4.2f}%              | {gap:+6.2f}% Gap")
    print("=" * 95)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 2024])
    args = parser.parse_args()
    
    run_zk_benchmark(k_mods=[7, 12], seq_len=64, seeds=args.seeds, steps=args.steps, device=args.device)
