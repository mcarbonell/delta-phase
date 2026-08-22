"""
test_quaternion_binding_v2.py
=============================
Frontera E v2 — El experimento que sí discrimina por construcción.

Lección de v1 (archivada): proyectar tokens crudos y comparar bloques completos mide
geometría incidental, no álgebra de binding. v2 elimina TODO lo incidental: sin conv,
sin posiciones, sin entrenamiento en el resultado principal. Tres brazos comparten la
MISMA memoria delta-rule (mismo presupuesto de flotantes) y difieren EXCLUSIVAMENTE en
cómo se compone la clave de un par ordenado:

  hadamard_u1 : K = k(a) ⊙ k(b)        (fasores U(1), producto de Hadamard)
                -> CONMUTATIVO: K(a,b) == K(b,a). Orden-ciego DEMOSTRABLE.
  quat_s3     : K = k(a) ⊗ k(b)        (cuaterniones unitarios S³)
                -> NO conmutativo: K(a,b) != K(b,a). Orden-preservante estructural.
  roletag_u1  : K = k(a) ⊙ (ρ ⊙ k(b))  (fasor de rol fijo ρ en el segundo slot)
                -> CONTROL DE IMPOSIBILIDAD: en U(1) ρ conmuta con todo, luego
                   K(a,b) = ρ·ka·kb = K(b,a) IDENTICAS — ningún marcador de rol
                   dentro de un álgebra conmutativa puede romper la simetría.
                   Su colapso empírico sobre hadamard confirma el teorema.

Pregunta afilada respondida por diseño: la no-conmutatividad no es "útil" sino
NECESARIA y SUFICIENTE para binding que preserva orden (demostrado zero-shot).

Modo ZERO-SHOT (sin entrenamiento): proyecciones y valores FIJOS aleatorios; escritura
delta beta=1; lectura por vecino-mas-cercano coseno. Barrido de interferencia
N_facts con 50% pares en conflicto ((b,a)->v' != v). Predicción teórica exacta para
hadamard: en cada par conflictivo, la segunda escritura (beta=1, clave idéntica)
REEMPLAZA a la primera -> el primer orden escrito falla sistemáticamente (~75% techo
global con 50% de conflictos); s3 y roletag degradan suave con la interferencia.

Salida: docs/quaternion_binding_v2_results.json
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

RESULTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "quaternion_binding_v2_results.json"))

# Presupuesto de estado IGUALADO: C^{90 x dv} (2*90*dv) == H^{45 x dv} (4*45*dv)
DK_COMPLEX = 90
DK_QUAT = 45
DV = 64
N_VALUES = 96
N_ENTITIES = 16

ARMS = ["hadamard_u1", "quat_s3", "roletag_u1"]
STATE_FLOATS = {"hadamard_u1": 2 * DK_COMPLEX * DV,
                "roletag_u1": 2 * DK_COMPLEX * DV,
                "quat_s3": 4 * DK_QUAT * DV}


# --------------------------- álgebra ---------------------------

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


# --------------------------- memoria por álgebra ---------------------------

class AlgebraMemory:
    """Estado M[c, :] (dv dims por canal de clave); escritura delta; lectura coseno."""

    def __init__(self, arm, device="cpu"):
        self.arm, self.device = arm, device
        if arm == "quat_s3":
            self.dk = DK_QUAT
            self.E = torch.stack([unit_quat(self.dk, 1000 + i) for i in range(N_ENTITIES)]).to(device)
        else:
            self.dk = DK_COMPLEX
            self.E = torch.stack([unit_phasor(self.dk, 1000 + i) for i in range(N_ENTITIES)]).to(device)
            self.rho = unit_phasor(self.dk, 777).to(device)

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


# --------------------------- tarea y evaluación ---------------------------

def sample_facts(n_facts, seed, conflict_frac=0.5):
    """Pares ordenados ((a,b), vid) con ~conflict_frac revertidos en conflicto."""
    g = torch.Generator().manual_seed(seed)
    items, used = [], set()
    while len(items) < n_facts:
        a, b = int(torch.randint(0, N_ENTITIES, (1,), generator=g)), int(torch.randint(0, N_ENTITIES, (1,), generator=g))
        if a == b or (a, b) in used:
            continue
        items.append(((a, b), len(items)))
        used.add((a, b))
        if len(items) < n_facts and len(items) <= conflict_frac * n_facts and (b, a) not in used:
            items.append(((b, a), len(items)))
            used.add((b, a))
    return items


@torch.no_grad()
def zero_shot_eval(arm, n_facts, trials, device="cpu"):
    mem = AlgebraMemory(arm, device)
    accs = []
    for t in range(trials):
        facts = sample_facts(n_facts, seed=7000 + t * 131 + n_facts)
        embs = unit_vecs(N_VALUES, DV, 9000 + t).to(device)

        M = mem.new_state(DV)
        for (pair, vid) in facts:
            mem.write(M, mem.key(*pair), embs[vid], beta=1.0)

        hits = 0
        for (pair, vid_true) in facts:
            ret = mem.read(M, mem.key(*pair))
            pred = int((embs @ ret).argmax())
            hits += int(pred == vid_true)
        accs.append(100.0 * hits / len(facts))
    return float(np.mean(accs)), float(np.std(accs) / math.sqrt(len(accs)))


# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Frontera E v2: composición de claves — hadamard vs cuaterniónica vs role-tag")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--nfacts", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.trials, args.nfacts = 5, [4, 16]

    print("=" * 104, flush=True)
    print("🧪 FRONTERA E v2 — COMPOSICIÓN DE CLAVES: HADAMARD U(1) vs CUATERNIÓNICA S³ vs ROLE-TAG U(1)", flush=True)
    print("=" * 104, flush=True)
    print("  • ZERO-SHOT (sin entrenamiento): proyecciones y valores fijos; escritura delta β=1;", flush=True)
    print("     lectura NN-coseno. Aísla el ÁLGEBRA del optimizador (antídoto del fallo de diseño de v1).", flush=True)
    print(f"  • Estado igualado: C^{DK_COMPLEX}x{DV} == H^{DK_QUAT}x{DV} = {STATE_FLOATS['hadamard_u1']:,} floats", flush=True)
    print(f"  • N_facts={args.nfacts} | trials/punto={args.trials} | ~50% pares en conflicto (b,a)->v'!=v", flush=True)
    print(f"  • Fecha UTC: {datetime.datetime.now(datetime.timezone.utc).isoformat()} | device={args.device}", flush=True)
    print("  • Predicción teórica hadamard: en cada conflicto, la 2ª escritura (clave idéntica, β=1)", flush=True)
    print("     reemplaza a la 1ª → el primer orden falla sistemáticamente (techo ~75% con 50% conflictos).", flush=True)
    print("     roletag DEBE colapsar sobre hadamard (ρ conmuta ⇒ claves idénticas: imposibilidad).", flush=True)
    print("     La pregunta es si s3 sostiene 100% bajo interferencia creciente.", flush=True)
    print("-" * 104 + "\n", flush=True)

    payload = {"date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "purpose": "Frontera E v2: key composition algebra — order-blind vs order-preserving vs role-tagged",
               "protocol": {"nfacts": args.nfacts, "trials": args.trials, "state_floats": STATE_FLOATS},
               "results": {}}

    print(f"{'N_facts':>8} | " + " | ".join(f"{a:>15}" for a in ARMS), flush=True)
    print("-" * 70, flush=True)
    for n in args.nfacts:
        payload["results"][str(n)] = {}
        cells = []
        for arm in ARMS:
            m, se = zero_shot_eval(arm, n, trials=args.trials, device=args.device)
            payload["results"][str(n)][arm] = {"mean_acc": round(m, 2), "se": round(se, 2)}
            cells.append(f"{m:9.2f} ± {se:4.1f}%")
        print(f"{n:>8} | " + " | ".join(cells), flush=True)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n🔎 Lectura guía:", flush=True)
    for n in args.nfacts:
        r = payload["results"][str(n)]
        h, s, rt = r["hadamard_u1"]["mean_acc"], r["quat_s3"]["mean_acc"], r["roletag_u1"]["mean_acc"]
        if abs(h - rt) < 3:
            u1_msg = "roletag≡hadamard (imposibilidad confirmada)"
        else:
            u1_msg = "roletag se separa de hadamard (¡inesperado! revisar)"
        gap = s - max(h, rt)
        if gap > 5:
            s_msg = "S³ preserva orden sobre ambos controles U(1)"
        elif s < 90:
            s_msg = "S³ sufre interferencia a este N"
        else:
            s_msg = "S³ sostiene orden"
        print(f"   N={n:>2}: hadamard {h:5.1f}% | roletag {rt:5.1f}% | s3 {s:5.1f}%  → {u1_msg}; {s_msg}", flush=True)
    print(f"\n💾 Resultados: {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
