"""
test_quaternion_binding_poc.py
==============================
Frontera E (docs/vision_speculative_visions_and_long_term_frontiers.md):
Binding Cuaterniónico S³ (no conmutativo) vs Fasorial U(1) (conmutativo).

Hipótesis bajo test: la NO-COMUTATIVIDAD del binding cuaterniónico codifica orden
estructural (trazas g1g2 != g2g1) y por tanto desambigua pares ordenados mejor que
el binding U(1), especialmente bajo interferencia — sin apoyarse en work-arounds
posicionales aprendidos.

Diseño experimental:
  Brazos (estado de memoria IGUALADO a ~8192 floats reales):
    - complex_u1 : n_heads=4, dk=32  -> 4 x C^{32x32}  = 4 x 2048 = 8192 floats
    - quat_s3    : n_heads=2, dkq=32 -> 2 x H^{32x32}  = 2 x 4096 = 8192 floats
    Ambos: mismo d_model=128, misma conv causal, mismo esquema pre-norm, mismo MLP,
    mismo optimizador/presupuesto. Difieren SOLO en el álgebra del binding.

  Tareas:
    - standard : MQAR estilo certificado (pares dirigidos K->V). Control de sanidad;
                 prior declarado: PARIDAD entre brazos (aquí la no-conmutatividad
                 no aporta nada).
    - ordered  : MQAR ORDENADO — bigrams ordenados (a,b) -> v con (b,a) -> v' != v
                 presentes SIMULTÁNEAMENTE en el mismo ejemplo (conflicto garantizado),
                 consultas [MARKER, ka, kb] -> v(orden). La única señal que distingue
                 (a,b) de (b,a) es el orden; el binding conmutativo debe compensarlo
                 con trucos posicionales aprendidos, el cuaterniónico lo tiene en el
                 álgebra.

  Métrica: precisión exacta en posiciones supervisadas de consulta, media ± SE sobre
  3 semillas, datos on-the-fly. JSON crudo: docs/quaternion_binding_poc_results.json

Prior honesto registrado de antemano (lección ablation-router / confound-capacidad):
se ESPERA neutralidad en 'standard'; cualquier victoria en 'ordered' solo cuenta si
supera ±1 SE y se sostiene al revisar presupuesto iso-floats.
"""

import os
import sys
import json
import math
import time
import argparse
import datetime
import platform

import numpy as np
import torch
import torch.nn as nn

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

RESULTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "quaternion_binding_poc_results.json"))

# Vocabulario compartido por ambas tareas
PAD = 0
MARKER = 1
KEY_LO, KEY_HI = 2, 18       # 16 entidades
VAL_LO, VAL_HI = 34, 98      # 64 valores
NOISE_LO, NOISE_HI = 98, 130
VOCAB_SIZE = 130


# =====================================================================
# Álgebra cuaternional (representación real, último dim=4 como (w,x,y,z))
# =====================================================================

def qmul(a, b):
    """Producto de Hamilton con broadcasting en todas las dims salvo la última."""
    a_w, a_v = a[..., 0:1], a[..., 1:5]
    b_w, b_v = b[..., 0:1], b[..., 1:5]
    w = a_w * b_w - (a_v * b_v).sum(-1, keepdim=True)
    ax, ay, az = a_v[..., 0:1], a_v[..., 1:2], a_v[..., 2:3]
    bx, by, bz = b_v[..., 0:1], b_v[..., 1:2], b_v[..., 2:3]
    v = (a_w * b_v + b_w * a_v
         + torch.cat([ay * bz - az * by,
                      az * bx - ax * bz,
                      ax * by - ay * bx], dim=-1))
    return torch.cat([w, v], dim=-1)


def qconj(a):
    return torch.cat([a[..., 0:1], -a[..., 1:5]], dim=-1)


def qnormalize(a):
    return a / (a.pow(2).sum(-1, keepdim=True).sqrt() + 1e-8)


# =====================================================================
# Bloques (secuenciales, estilo PoC autocontenido)
# =====================================================================

class ShortCausalConv(nn.Module):
    def __init__(self, d, k=4):
        super().__init__()
        self.conv = nn.Conv1d(d, d, k, padding=k - 1, groups=d)
        self.act = nn.SiLU()

    def forward(self, x):
        B, L, D = x.shape
        return x + self.act(self.conv(x.transpose(1, 2))[:, :, :L].transpose(1, 2))


class ComplexU1Block(nn.Module):
    """Bloque delta-rule fasorial U(1): estado C^{dk x dk} por cabeza."""

    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.n_heads, self.dk = n_heads, d_model // n_heads
        self.inv_dk = 1.0 / self.dk
        self.norm1 = nn.LayerNorm(d_model)
        self.conv = ShortCausalConv(d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_beta = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))

    def forward(self, x):
        res = x
        B, L, _ = x.shape
        h = self.conv(self.norm1(x))
        H, dk, inv = self.n_heads, self.dk, self.inv_dk
        tk = self.w_k(h).view(B, L, H, dk).transpose(1, 2)
        tq = self.w_q(h).view(B, L, H, dk).transpose(1, 2)
        vo = self.w_v(h).view(B, L, H, dk).transpose(1, 2)
        beta = 2.0 * torch.sigmoid(self.w_beta(h)).transpose(1, 2)
        K = torch.polar(torch.ones_like(tk), tk)
        Q = torch.polar(torch.ones_like(tq), tq)

        M = torch.zeros(B, H, dk, dk, dtype=torch.complex64, device=x.device)
        outs = []
        for t in range(L):
            kt, qt, vt, bt = K[:, :, t], Q[:, :, t], vo[:, :, t], beta[:, :, t]
            v_old = torch.matmul(M, torch.conj(kt).unsqueeze(-1)).squeeze(-1).real * inv
            e = vt - v_old
            M = M + bt.unsqueeze(-1).unsqueeze(-1) * e.to(M.dtype).unsqueeze(-1) @ kt.unsqueeze(-2)
            outs.append(torch.matmul(M, torch.conj(qt).unsqueeze(-1)).squeeze(-1).real * inv)
        r = torch.stack(outs, dim=2).transpose(1, 2).reshape(B, L, H * dk)
        x = res + self.out_proj(r)
        return x + self.mlp(self.norm2(x))


class QuaternionS3Block(nn.Module):
    """Bloque delta-rule cuaterniónico S³: estado H^{dkq x dkq} por cabeza.

    Claves/consultas: cuaterniones UNITARIOS por canal (viven en S³).
    Valores/errores: reales (filas del estado). Escritura: M[r,c] += b*e[r]*k[c].
    Lectura: Re(sum_c M[r,c]*conj(q)[c]) — la parte real del producto interno
    hermitiano cuaterniónico es simétrica => Gram real (nota técnica Frontera E).
    """

    def __init__(self, d_model=128, n_heads=2, dkq=32):
        super().__init__()
        self.n_heads, self.dkq = n_heads, dkq
        self.inv_dk = 1.0 / dkq
        self.norm1 = nn.LayerNorm(d_model)
        self.conv = ShortCausalConv(d_model)
        self.w_k = nn.Linear(d_model, n_heads * dkq * 4)
        self.w_q = nn.Linear(d_model, n_heads * dkq * 4)
        self.w_v = nn.Linear(d_model, n_heads * dkq)
        self.w_beta = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(n_heads * dkq, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))

    def _readout(self, M, qbar):
        """Re(sum_c M[r,c]*qbar[c]) -> (B,H,dkq) real."""
        prod = qmul(M, qbar.unsqueeze(3))          # (B,H,r,c,4)
        return prod[..., 0].sum(-1) * self.inv_dk  # componente w

    def forward(self, x):
        res = x
        B, L, _ = x.shape
        h = self.conv(self.norm1(x))
        H, dkq = self.n_heads, self.dkq
        tk = qnormalize(self.w_k(h).view(B, L, H, dkq, 4).transpose(1, 2))
        tq = qnormalize(self.w_q(h).view(B, L, H, dkq, 4).transpose(1, 2))
        vo = self.w_v(h).view(B, L, H, dkq).transpose(1, 2)
        beta = 2.0 * torch.sigmoid(self.w_beta(h)).transpose(1, 2)

        M = torch.zeros(B, H, dkq, dkq, 4, device=x.device)
        outs = []
        for t in range(L):
            kt, qt, vt, bt = tk[:, :, t], tq[:, :, t], vo[:, :, t], beta[:, :, t]
            v_old = self._readout(M, qconj(qt))                                  # (B,H,dkq)
            e = vt - v_old
            # OJO broadcasting: bt debe ser 5-D (B,H,1,1,1) para alinear con (B,H,r,c,4)
            M = M + bt.view(B, H, 1, 1, 1) \
                * e.unsqueeze(-1).unsqueeze(-1) * kt.unsqueeze(2)                # M[r,c] += b*e[r]*k[c]
            outs.append(self._readout(M, qconj(qt)))
        r = torch.stack(outs, dim=2).transpose(1, 2).reshape(B, L, H * dkq)
        x = res + self.out_proj(r)
        return x + self.mlp(self.norm2(x))


def build_arm(name, n_layers=2):
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(VOCAB_SIZE, 128)
            BlockCls, kwargs = (ComplexU1Block, dict(n_heads=4)) if name == "complex_u1" \
                else (QuaternionS3Block, dict(n_heads=2, dkq=32))
            self.blocks = nn.ModuleList([BlockCls(**kwargs) for _ in range(n_layers)])
            self.head = nn.Linear(128, VOCAB_SIZE)

        def forward(self, x):
            h = self.embed(x)
            for blk in self.blocks:
                h = blk(h)
            return self.head(h)

    return Wrapper()


def state_floats(name):
    return {"complex_u1": 4 * 2 * 32 * 32, "quat_s3": 2 * 4 * 32 * 32}[name]


# =====================================================================
# Generadores de datos on-the-fly
# =====================================================================

def gen_standard(batch, seq_len=128, n_pairs=8, device="cpu"):
    """MQAR certificado simplificado: [K,V]*P al inicio; consultas [MARKER,K]->V después."""
    x = torch.full((batch, seq_len), PAD, dtype=torch.long, device=device)
    y = torch.full((batch, seq_len), -100, dtype=torch.long, device=device)
    keys = torch.randint(KEY_LO, KEY_HI, (batch, n_pairs), device=device)
    vals = torch.randint(VAL_LO, VAL_HI, (batch, n_pairs), device=device)
    for b in range(batch):
        for i in range(n_pairs):
            x[b, 2 * i], x[b, 2 * i + 1] = keys[b, i], vals[b, i]
        perm = torch.randperm(n_pairs, device=device)
        base = 2 * n_pairs + 2
        for j in range(n_pairs):
            i = int(perm[j])
            qpos = base + 2 * j
            if qpos + 1 >= seq_len:
                break
            x[b, qpos], x[b, qpos + 1] = MARKER, keys[b, i]
            y[b, qpos + 1] = vals[b, i]
    return x, y


def gen_ordered(batch, seq_len=128, n_triples=12, n_queries=6, device="cpu", guarantee_conflicts=True):
    """MQAR ORDENADO: triples (a,b)->v almacenados; (b,a)->v' != v también presente.
    Consultas [MARKER, ka, kb] -> v(orden). El orden es la ÚNICA señal desambiguadora."""
    x = torch.full((batch, seq_len), PAD, dtype=torch.long, device=device)
    y = torch.full((batch, seq_len), -100, dtype=torch.long, device=device)
    for b in range(batch):
        facts = []
        i = 0
        while i < n_triples:
            a, bb = torch.randint(KEY_LO, KEY_HI, (2,), device=device).tolist()
            if a == bb:
                continue
            v = int(torch.randint(VAL_LO, VAL_HI, (1,), device=device))
            facts.append(((a, bb), v))
            if guarantee_conflicts and i % 2 == 0 and i + 1 < n_triples:
                v2 = VAL_LO + (v - VAL_LO + 1 + int(torch.randint(1, VAL_HI - VAL_LO - 1, (1,), device=device))) % (VAL_HI - VAL_LO)
                facts.append(((bb, a), v2))
                i += 1
            i += 1
        facts = facts[:n_triples]
        for j, ((a, bb), v) in enumerate(facts):
            x[b, 3 * j], x[b, 3 * j + 1], x[b, 3 * j + 2] = a, bb, v
        lookup = dict(facts)
        qperm = torch.randperm(len(facts), device=device)[:n_queries]
        base = seq_len // 2
        for j, fi in enumerate(qperm.tolist()):
            (a, bb), v = facts[fi]
            qpos = base + 3 * j
            if qpos + 2 >= seq_len:
                break
            x[b, qpos], x[b, qpos + 1], x[b, qpos + 2] = MARKER, a, bb
            y[b, qpos + 2] = lookup[(a, bb)]
    return x, y


TASKS = {
    "standard": lambda b, L, device="cpu": gen_standard(b, L, n_pairs=8, device=device),
    "ordered": lambda b, L, device="cpu": gen_ordered(b, L, n_triples=12, n_queries=6, device=device),
}


# =====================================================================
# Entrenamiento / evaluación
# =====================================================================

def evaluate(model, task, seq_len, batches=8, batch=32, device="cpu"):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(batches):
            x, y = TASKS[task](batch, seq_len, device=device)
            preds = model(x).argmax(-1)
            mask = y != -100
            correct += (preds[mask] == y[mask]).sum().item()
            total += mask.sum().item()
    model.train()
    return 100.0 * correct / max(total, 1)


def train_one(arm, task, seed, steps, batch, lr, early_stop, device, log_every=50):
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_arm(arm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss(ignore_index=-100)
    t0 = time.time(); final_step = steps
    for step in range(1, steps + 1):
        x, y = TASKS[task](batch, 128, device=device)
        opt.zero_grad()
        loss = crit(model(x).view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps:
            acc = evaluate(model, task, 128, batches=4, batch=batch, device=device)
            print(f"   [{arm:>10}|{task:>8}|s{seed}] paso {step:4d}/{steps} | loss {loss.item():6.4f} | acc {acc:6.2f}% | {time.time()-t0:7.1f}s", flush=True)
            if acc >= early_stop:
                final_step = step
                print(f"   🎯 Early stop ({acc:.2f}% >= {early_stop}%)", flush=True)
                break
    wall = time.time() - t0
    acc = evaluate(model, task, 128, batches=15, batch=batch, device=device)
    print(f"   ✨ [{arm}|{task}|s{seed}] final acc {acc:.2f}% | paso {final_step} | {wall:.1f}s", flush=True)
    return {"final_acc": acc, "step_stop": final_step, "wallclock": wall}


def save(p):
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(p, f, indent=2, ensure_ascii=False)


def main():
    p = argparse.ArgumentParser(description="Frontera E PoC: binding cuaterniónico S³ vs fasorial U(1)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--early-stop", type=float, default=99.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 2024])
    p.add_argument("--arms", nargs="+", default=["complex_u1", "quat_s3"])
    p.add_argument("--tasks", nargs="+", default=["standard", "ordered"])
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.steps, args.seeds, args.arms, args.tasks = 30, [42], ["complex_u1", "quat_s3"], ["ordered"]
        args.early_stop = 200.0

    arms, tasks, seeds = args.arms, args.tasks, args.seeds
    total_runs = len(arms) * len(tasks) * len(seeds)
    print("=" * 108, flush=True)
    print("🧪 FRONTERA E — BINDING CUATERNIÓNICO S³ vs FASORIAL U(1) (PoC, iso-presupuesto de estado)", flush=True)
    print("=" * 108, flush=True)
    print(f"  • Dispositivo: {args.device.upper()} | Python {platform.python_version()} | PyTorch {torch.__version__}", flush=True)
    print(f"  • Fecha UTC:   {datetime.datetime.now(datetime.timezone.utc).isoformat()}", flush=True)
    print(f"  • Presupuesto de estado (flotantes reales): " +
          " | ".join(f"{a}: {state_floats(a):,}" for a in arms), flush=True)
    print(f"  • Tareas: {tasks} | Pasos: {args.steps} | Semillas: {seeds} | Early stop ≥ {args.early_stop}%", flush=True)
    print("  • Prior registrado: paridad esperada en 'standard'; victoria solo creíble en 'ordered' si supera ±1 SE.", flush=True)
    print("-" * 108, flush=True)

    inv = {a: build_arm(a) for a in arms}
    for a in arms:
        n = sum(pp.numel() for pp in inv[a].parameters())
        print(f"     • {a:<11} params modelo: {n:,}", flush=True)
    print("=" * 108 + "\n", flush=True)

    payload = {"date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "purpose": "Frontera E: quaternionic S3 binding vs U(1) phasor binding",
               "protocol": vars(args), "results": {}}

    for task in tasks:
        payload["results"][task] = {}
        for arm in arms:
            accs, walls = [], []
            for seed in seeds:
                print(f"\n🚀 [{task}] {arm} seed={seed}", flush=True)
                m = train_one(arm, task, seed, args.steps, args.batch, args.lr, args.early_stop, args.device)
                accs.append(m["final_acc"]); walls.append(m["wallclock"])
                payload["results"][task].setdefault(arm, {"seeds": {}})
                payload["results"][task][arm]["seeds"][str(seed)] = m
                save(payload)
            mean, se = float(np.mean(accs)), float(np.std(accs) / math.sqrt(len(accs)))
            payload["results"][task][arm]["summary"] = {
                "mean_acc": mean, "se_acc": se, "mean_wallclock": float(np.mean(walls))}
            save(payload)
            print(f"   📊 [{task}] {arm}: {mean:.2f} ± {se:.2f}%", flush=True)

    print("\n" + "=" * 108, flush=True)
    print("📊 RESUMEN (media ± SE sobre semillas):", flush=True)
    verdict = {}
    for task in tasks:
        line = []
        for arm in arms:
            s = payload["results"][task].get(arm, {}).get("summary")
            line.append(f"{arm}: {s['mean_acc']:.2f} ± {s['se_acc']:.2f}%" if s else f"{arm}: n/a")
        if all(a in payload["results"][task] and "summary" in payload["results"][task][a] for a in arms):
            s3 = payload["results"][task]["quat_s3"]["summary"]
            u1 = payload["results"][task]["complex_u1"]["summary"]
            d = s3["mean_acc"] - u1["mean_acc"]
            se_pool = math.sqrt(s3["se_acc"] ** 2 + u1["se_acc"] ** 2)
            if se_pool < 1e-9:
                verdict[task] = {"delta_s3_minus_u1": round(d, 3), "sigma": None,
                                 "note": "SE=0 (semillas insuficientes) — sigma no evaluable"}
            else:
                verdict[task] = {"delta_s3_minus_u1": round(d, 3), "sigma": round(abs(d) / se_pool, 2)}
        print(f"  {task:>9}: " + " | ".join(line), flush=True)
    payload["verdict"] = verdict
    save(payload)
    if verdict.get("ordered"):
        d = verdict["ordered"]["delta_s3_minus_u1"]
        sig = verdict["ordered"].get("sigma")
        print(f"\n🔎 DELTA en tarea ordenada (S³ − U(1)): {d:+.2f} puntos", flush=True)
        if sig is None:
            print("   ⚪ Sigma no evaluable (SE=0 con semillas insuficientes)", flush=True)
        elif d > 0 and sig > 1:
            print(f"   🟢 Señal POSITIVA para S³: {sig:.2f} sigma — supera ±1 SE, merece repetición", flush=True)
        elif sig is not None and abs(sig) > 1:
            print(f"   🔴 Resultado NEGATIVO para S³ ({sig:.2f} sigma): el brazo cuaterniónico rinde PEOR — "
                  f"registrar como evidencia contra la hipótesis tal como fue implementada", flush=True)
        else:
            print(f"   ⚪ Consistente con neutralidad ({sig} sigma)", flush=True)
    print(f"\n💾 Resultados: {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
