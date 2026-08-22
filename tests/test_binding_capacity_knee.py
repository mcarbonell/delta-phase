"""
test_binding_capacity_knee.py
=============================
Frontera E — Falsable #1: rodilla de capacidad del binding ordenado S³.

Teoría bajo test (asociador lineal con claves casi-ortogonales, escritura delta β=1):
  ret = e_true + ruido ,   ruido = sum_{i!=j} e_i * G_ij ,  G_ij ~ (0, 1/d_eff)
  => ||ruido|| = sqrt((N-1)/d_eff)        [paseo aleatorio]
  => cos(ret, e_true) ~= 1/sqrt(1 + (N-1)/d_eff)

con d_eff = dimensión real efectiva del canal de claves:
  - quat_s3     : dkq=45 x 4 componentes = d_eff 180
  - u1 (ambos)  : dk=90 x 2 (Re,Im)       = d_eff 180
Presupuesto de estado igualado por construcción (mismo que v2).

Predicciones falsables:
  P1 (ley raíz): ||ruido||(N) empírico sigue sqrt((N-1)/180) hasta ~5% en TODOS los brazos.
  P2 (rodilla): existe N crítico donde top-1 cae de >99% a azar; ubicación consistente con
     d_eff=180 y separación típica entre valores aleatorios (dv=64).
  P3 (dos firmas de fallo): en condición CONFLICTO, hadamard tiene un SUELO algebráico
     (~75%: primer orden siempre falla) independiente de N; en condición LIMPIA su curva es
     puramente ruidosa e indistinguible de los otros brazos (el fallo era estructural, no de capacidad).

Modos: LIMPIO (sin revertidos — ley de ruido pura) y CONFLICTO (50% revertidos — suelo algebráico).
Barrido: N en {16, 32, 64, 128, 256, 512, 1024}. Trials adaptativos. Todo zero-shot (sin entrenar).

Salida: docs/binding_capacity_knee_results.json
"""

import os
import sys
import json
import math
import argparse
import datetime

import numpy as np
import torch

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RESULTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "binding_capacity_knee_results.json"))

DK_COMPLEX, DK_QUAT = 90, 45
DV = 64
N_ENTITIES = 600          # >= N_max dirigidos sin colisiones forzadas
ARMS = ["hadamard_u1", "quat_s3", "roletag_u1"]
STATE_FLOATS = {"hadamard_u1": 2 * DK_COMPLEX * DV, "roletag_u1": 2 * DK_COMPLEX * DV,
                "quat_s3": 4 * DK_QUAT * DV}


def qmul(a, b):
    aw, av = a[..., 0:1], a[..., 1:5]
    bw, bv = b[..., 0:1], b[..., 1:5]
    w = aw * bw - (av * bv).sum(-1, keepdim=True)
    ax, ay, az = av[..., 0:1], av[..., 1:2], av[..., 2:3]
    bx, by, bz = bv[..., 0:1], bv[..., 1:2], bv[..., 2:3]
    v = (aw * bv + bw * av
         + torch.cat([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], dim=-1))
    return torch.cat([w, v], dim=-1)


def qconj(a):
    return torch.cat([a[..., 0:1], -a[..., 1:5]], dim=-1)


def unit_phasor(dk, seed):
    g = torch.Generator().manual_seed(seed)
    th = torch.rand(dk, generator=g) * 2 * math.pi
    return torch.complex(torch.cos(th), torch.sin(th))


def unit_quat(dk, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(dk, 4, generator=g)
    return q / q.norm(dim=-1, keepdim=True)


def unit_vecs(n, d, seed):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, d, generator=g)
    return v / v.norm(dim=-1, keepdim=True)


class AlgebraMemory:
    def __init__(self, arm, n_entities=N_ENTITIES, device="cpu"):
        self.arm, self.device = arm, device
        if arm == "quat_s3":
            self.dk, self.d_eff = DK_QUAT, 4 * DK_QUAT
            self.E = torch.stack([unit_quat(self.dk, 5000 + i) for i in range(n_entities)]).to(device)
        else:
            self.dk, self.d_eff = DK_COMPLEX, 2 * DK_COMPLEX
            self.E = torch.stack([unit_phasor(self.dk, 5000 + i) for i in range(n_entities)]).to(device)
            g = torch.Generator().manual_seed(777)
            th = torch.rand(self.dk, generator=g) * 2 * math.pi
            self.rho = torch.complex(torch.cos(th), torch.sin(th)).to(device)

    def key(self, a, b):
        ka, kb = self.E[a], self.E[b]
        if self.arm == "hadamard_u1":
            return ka * kb
        if self.arm == "roletag_u1":
            return ka * (self.rho * kb)
        return qmul(ka, kb)

    def new_state(self, dv):
        if self.arm == "quat_s3":
            return torch.zeros(self.dk, dv, 4, device=self.device)
        return torch.zeros(self.dk, dv, dtype=torch.complex64, device=self.device)

    def write(self, M, K, e_vec, beta=1.0):
        if self.arm == "quat_s3":
            M += beta * e_vec.unsqueeze(0).unsqueeze(-1) * K.unsqueeze(1)
        else:
            M += beta * e_vec.to(M.dtype).unsqueeze(0) * K.unsqueeze(1)

    def read(self, M, Q):
        if self.arm == "quat_s3":
            return qmul(M, qconj(Q).unsqueeze(1))[..., 0].sum(0) / self.dk
        return (M * torch.conj(Q).unsqueeze(1)).real.sum(0) / self.dk


def sample_facts(n_facts, seed, conflict_frac, n_entities):
    """Pares ordenados únicos ((a,b), vid). Con conflict_frac>0 añade revertidos en conflicto
    DENTRO del presupuesto (cada revertido consume un slot). Sin duplicados ni inversos no deseados."""
    g = torch.Generator().manual_seed(seed)
    items, used = [], set()
    n_conflicts = int(round(conflict_frac * n_facts))
    added_conflicts = 0
    while len(items) < n_facts:
        a = int(torch.randint(0, n_entities, (1,), generator=g))
        b = int(torch.randint(0, n_entities, (1,), generator=g))
        if a == b or (a, b) in used:
            continue
        items.append(((a, b), len(items)))
        used.add((a, b))
        if added_conflicts < n_conflicts and len(items) < n_facts and (b, a) not in used:
            items.append(((b, a), len(items)))
            used.add((b, a))
            added_conflicts += 1
    return items


@torch.no_grad()
def eval_point(arm, n_facts, conflict, trials, device="cpu"):
    mem = AlgebraMemory(arm, n_entities=max(N_ENTITIES, 4 * n_facts), device=device)
    d_eff = mem.d_eff
    accs, noise_norms, sims_true = [], [], []
    for t in range(trials):
        facts = sample_facts(n_facts, seed=(90_000 + t * 977 + n_facts) % (2**31), 
                             conflict_frac=0.5 if conflict else 0.0,
                             n_entities=max(N_ENTITIES, 4 * n_facts))
        embs = unit_vecs(n_facts, DV, 60_000 + t).to(device)
        vid2row = {vid: i for i, (_, vid) in enumerate(facts)}
        E_mat = torch.stack([embs[vid] for _, vid in facts])          # (N, dv) filas alineadas

        M = mem.new_state(DV)
        keys = [mem.key(*pair) for pair, _ in facts]
        for (pair, vid), K in zip(facts, keys):
            mem.write(M, K, embs[vid])

        hits = 0
        for idx, (pair, vid) in enumerate(facts):
            ret = mem.read(M, keys[idx])
            true_vec = E_mat[idx]
            noise = ret - true_vec                                     # G_jj == 1 exacto
            noise_norms.append(noise.norm().item())
            denom = ret.norm() + 1e-12
            sims_true.append(float(true_vec @ ret) / denom)
            pred_row = int(torch.argmax(E_mat @ ret))
            hits += int(pred_row == idx)
        accs.append(100.0 * hits / len(facts))
    return {"mean_acc": float(np.mean(accs)), "se_acc": float(np.std(accs) / math.sqrt(len(accs))),
            "mean_noise": float(np.mean(noise_norms)),
            "theory_noise": math.sqrt(max(n_facts - 1, 0) / d_eff),
            "mean_cos_true": float(np.mean(sims_true)),
            "theory_cos": 1.0 / math.sqrt(1.0 + max(n_facts - 1, 0) / d_eff)}


def main():
    ap = argparse.ArgumentParser(description="Frontera E: rodilla de capacidad del binding ordenado")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nfacts", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512, 1024])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.nfacts = [16, 64, 256]

    modes = [("LIMPIO", False), ("CONFLICTO", True)]
    payload = {"date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "purpose": "Frontera E: capacity knee + sqrt(N) noise-law verification",
               "protocol": {"nfacts": args.nfacts, "modes": [m for m, _ in modes],
                             "state_floats": STATE_FLOATS, "d_eff_target": 180},
               "results": {}}

    print("=" * 108, flush=True)
    print("🧪 FRONTERA E — RODILLA DE CAPACIDAD Y LEY DE RUIDO √N (zero-shot)", flush=True)
    print("=" * 108, flush=True)
    print(f"  • Teoría: ||ruido|| = sqrt((N-1)/{180}) ; cos_true ≈ 1/sqrt(1+(N-1)/{180})", flush=True)
    print("  • Modos: LIMPIO (ley de ruido pura) y CONFLICTO (suelo algebráico de hadamard)", flush=True)
    print(f"  • Fecha UTC: {datetime.datetime.now(datetime.timezone.utc).isoformat()} | device={args.device}", flush=True)
    print("=" * 108 + "\n", flush=True)

    for mode_name, conflict in modes:
        payload["results"][mode_name] = {}
        print(f"━━━ MODO {mode_name} {'(50% revertidos)' if conflict else '(sin revertidos)'} ━━━", flush=True)
        print(f"{'N':>6} | {'brazo':<12} | {'top-1':>14} | {'‖ruido‖ emp':>11} | {'teoría':>8} | {'cos_true':>8} | {'teoría':>7}", flush=True)
        print("-" * 88, flush=True)
        for n in args.nfacts:
            trials = max(8, min(30, int(24_000 // n)))
            for arm in ARMS:
                r = eval_point(arm, n, conflict, trials, device=args.device)
                payload["results"][mode_name].setdefault(str(n), {})[arm] = {
                    **r, "trials": trials}
                print(f"{n:>6} | {arm:<12} | {r['mean_acc']:7.2f}±{r['se_acc']:4.1f}% | "
                      f"{r['mean_noise']:11.3f} | {r['theory_noise']:8.3f} | "
                      f"{r['mean_cos_true']:8.3f} | {r['theory_cos']:7.3f}", flush=True)
            print("-" * 88, flush=True)

    # Rodillas
    print("\n🔎 RODILLAS (último N con top-1 ≥ umbral):", flush=True)
    knees = {}
    for mode_name, _ in modes:
        for thr in (95, 50):
            row = {}
            for arm in ARMS:
                best = None
                for n in args.nfacts:
                    r = payload["results"][mode_name].get(str(n), {}).get(arm)
                    if r and r["mean_acc"] >= thr:
                        best = n
                row[f"{arm}_N@{thr}%"] = best
            knees.setdefault(mode_name, {})[f"umbral_{thr}%"] = row
            print(f"   [{mode_name}] ≥{thr}%: {row}", flush=True)
    payload["knees"] = knees

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Resultados: {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
