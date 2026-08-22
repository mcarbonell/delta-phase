"""
test_double_cover_memory.py
===========================
Frontera E — Falsable #2: el doble recubrimiento como información direccionable.

Física: SU(2) recubre doblemente a SO(3). Un camino de rotaciones que vuelve a la MISMA
orientación física termina en +I o -I del estado espinorial según la paridad de vueltas
(truco del cinturón). Ese signo no es orientación: es memoria topológica del camino que
SO(3) BORRA y SU(2) CONSERVA.

Tarea: caminatas de rotaciones de PI sobre ejes principales {x,y,z}. Los extremos cierran
en el grupo cuaterniónico Q8 = {±I, ±x̂, ±ŷ, ±ẑ}: solo 4 orientaciones físicas (SO(3)
identifica Q~-Q) pero 8 estados espinoriales (SU(2) los separa). Cada paso almacena un
VALOR ÚNICO dirigido por la dirección acumulada:
  su2 : dirección = cuaternión acumulado Q (4 flotantes)
  so3 : dirección = matriz de rotación R(Q) aplanada (9 flotantes); R(Q)==R(-Q) por diseño.

SUBCONJUNTO DISCRIMINANTE (la métrica que importa): pasos cuya dirección espinorial es
única en la caminata PERO cuyo gemelo de signo también fue visitado (misma orientación).
Matemática exacta del resultado esperado:
  su2 : Q y -Q son direcciones antiparalelas -> recuperaciones e1-e2 y -e1+e2 -> NN resuelve
        AMBOS valores al 100%.
  so3 : ambos gemelos escriben en la MISMA dirección -> recuperación e1+e2 -> NN elige
        entre los dos al ~50% (elección aleatoria entre embeddings aleatorios).
Esa brecha (100 vs ~50) ES el doble recubrimiento convertido en canal de información.

Nota de alcance: contra U(1) esto NO es demostrable (con fase libre cuenta a cualquier
periodo). El comparador correcto es orientación SO(3) vs espinor SU(2).

Salida: docs/double_cover_memory_results.json
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

RESULTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "double_cover_memory_results.json"))

DV = 64
N_VALUES_MAX = 256


def qmul(a, b):
    aw, av = a[..., 0:1], a[..., 1:5]
    bw, bv = b[..., 0:1], b[..., 1:5]
    w = aw * bw - (av * bv).sum(-1, keepdim=True)
    ax, ay, az = av[..., 0:1], av[..., 1:2], av[..., 2:3]
    bx, by, bz = bv[..., 0:1], bv[..., 1:2], bv[..., 2:3]
    v = (aw * bv + bw * av
         + torch.cat([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], dim=-1))
    return torch.cat([w, v], dim=-1)


def unit_vecs(n, d, seed):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, d, generator=g)
    return v / v.norm(dim=-1, keepdim=True)


AXIS_QUATS = {
    0: torch.tensor([0.0, 1.0, 0.0, 0.0]),   # pi sobre x
    1: torch.tensor([0.0, 0.0, 1.0, 0.0]),   # pi sobre y
    2: torch.tensor([0.0, 0.0, 0.0, 1.0]),   # pi sobre z
}


def quat_to_matrix(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
    ])


@torch.no_grad()
def run_trial(arm, n_steps, seed, device="cpu"):
    """Memoria SIMBÓLICA: la dirección se cuantiza al estado más cercano del álgebra
    (SO(3): 4 orientaciones / SU(2): 8 espinoriales) y cada símbolo guarda su último valor.
    Elección deliberada: una memoria lineal sobre coordenadas crudas tiene crosstalk
    |<Q,-Q>|=1 entre antípodas (hallazgo registrado) — ciego al doble recubrimiento por
    construcción. La ventaja del signo solo puede manifestarse vía etapa no lineal."""
    g = torch.Generator().manual_seed(seed)
    embs = unit_vecs(N_VALUES_MAX, DV, 55_000 + seed % 50_000).to(device)

    Q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    su2_addrs, so3_addrs = [], []
    for _ in range(n_steps):
        axis = int(torch.randint(0, 3, (1,), generator=g))
        Q = qmul(Q, AXIS_QUATS[axis].to(device))
        su2_addrs.append(Q.clone())
        so3_addrs.append(quat_to_matrix(Q).flatten())

    def uniq_map(addrs):
        m = {}
        for i, a in enumerate(addrs):
            m.setdefault(tuple(torch.round(a, decimals=5).tolist()), []).append(i)
        return m

    su2_map = uniq_map(su2_addrs)
    orient_map = uniq_map(so3_addrs)

    # Subconjunto DISCRIMINANTE: dirección espinorial única + gemelo de signo visitado.
    # Control: espinor único Y orientación única.
    discrim, easy = [], []
    for key, idxs in su2_map.items():
        if len(idxs) != 1:
            continue
        idx = idxs[0]
        okey = tuple(torch.round(so3_addrs[idx], decimals=5).tolist())
        if len(orient_map.get(okey, [])) >= 2:
            discrim.append(idx)
        else:
            easy.append(idx)

    if len(discrim) < 2:
        return None

    # Memoria simbólica: dict símbolo -> último valor escrito en ese símbolo
    addrs = su2_addrs if arm == "su2" else so3_addrs
    sym_store = {}
    for idx in range(n_steps):
        sym_store[tuple(torch.round(addrs[idx], decimals=5).tolist())] = embs[idx]

    def acc_on(idxs):
        if not idxs:
            return float("nan")
        hits = 0
        for idx in idxs:
            stored = sym_store[tuple(torch.round(addrs[idx], decimals=5).tolist())]
            pred = int((embs @ stored).argmax())
            hits += int(pred == idx)
        return 100.0 * hits / len(idxs)

    return {"acc_discriminating": acc_on(discrim), "n_discriminating": len(discrim),
            "acc_easy": acc_on(easy), "n_easy": len(easy)}


def eval_arm(arm, n_steps, trials, device="cpu"):
    disc, easy, n_used = [], [], 0
    for t in range(trials):
        r = run_trial(arm, n_steps, seed=321_000 + t * 17 + n_steps, device=device)
        if r is None:
            continue
        if not math.isnan(r["acc_discriminating"]):
            disc.append(r["acc_discriminating"])
        if not math.isnan(r["acc_easy"]):
            easy.append(r["acc_easy"])
        n_used += r["n_discriminating"]
    se_d = float(np.std(disc) / math.sqrt(len(disc))) if disc else float("nan")
    return (float(np.mean(disc)) if disc else float("nan"), se_d,
            float(np.mean(easy)) if easy else float("nan"), len(disc))


def main():
    ap = argparse.ArgumentParser(description="Frontera E: doble recubrimiento como memoria direccionable")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--steps", type=int, nargs="+", default=[4, 6, 8, 12, 16])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.trials, args.steps = 6, [8, 32]

    print("=" * 104, flush=True)
    print("🧪 FRONTERA E — DOBLE RECUBRIMIENTO COMO MEMORIA DIRECCIONABLE (truco del cinturón)", flush=True)
    print("=" * 104, flush=True)
    print("  • Caminatas de rotaciones PI sobre ejes principales → extremos en Q8 =", flush=True)
    print("     {±I, ±x̂, ±ŷ, ±ẑ}: 4 orientaciones físicas (SO(3) identifica Q~-Q), 8 espinoriales (SU(2)).", flush=True)
    print("  • Cada paso almacena un valor único dirigido por la dirección acumulada.", flush=True)
    print("  • SUBCONJUNTO DISCRIMINANTE: dirección espinorial única + gemelo de signo visitado.", flush=True)
    print("     Memoria SIMBÓLICA (cuantización al estado más cercano): so3 fusiona gemelos (4 símbolos,", flush=True)
    print("     último-valor-gana → ~50% en consultas discriminantes); su2 separa los 8 signos → ~100%.", flush=True)
    print("  • HALLAZGO registrado en el diseño: una memoria lineal sobre coordenadas crudas tiene", flush=True)
    print("     crosstalk |<Q,-Q>|=1 entre antípodas — la familia delta-rule es ciega al doble", flush=True)
    print("     recubrimiento por construcción; el signo exige etapa de direccionamiento no lineal.", flush=True)
    print(f"  • Fecha UTC: {datetime.datetime.now(datetime.timezone.utc).isoformat()} | device={args.device}", flush=True)
    print("-" * 104 + "\n", flush=True)

    payload = {"date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "purpose": "Frontera E: double cover (SU(2) vs SO(3)) as addressable memory information",
               "protocol": {"steps": args.steps, "trials": args.trials},
               "results": {}}

    print(f"{'pasos':>6} | {'su2 (discrim.)':>16} | {'so3 (mismas consultas)':>22} | {'control fácil su2/so3':>22} | consultas", flush=True)
    print("-" * 100, flush=True)
    for ns in args.steps:
        d_m, d_se, easy_su2, n_d = eval_arm("su2", ns, trials=args.trials, device=args.device)
        o_m, o_se, easy_so3, _ = eval_arm("so3", ns, trials=args.trials, device=args.device)
        payload["results"][str(ns)] = {
            "su2": {"acc_discriminating": round(d_m, 2), "se": round(d_se, 2), "acc_easy": round(easy_su2, 2)},
            "so3": {"acc_discriminating": round(o_m, 2), "se": round(o_se, 2), "acc_easy": round(easy_so3, 2)},
            "n_discriminating_per_walk": n_d}
        print(f"{ns:>6} | {d_m:9.2f} ± {d_se:4.1f}% | {o_m:12.2f} ± {o_se:4.1f}% | "
              f"{easy_su2:9.1f} / {easy_so3:9.1f} | ~{n_d}", flush=True)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    last = str(args.steps[-1])
    d = payload["results"][last]["su2"]["acc_discriminating"]
    o = payload["results"][last]["so3"]["acc_discriminating"]
    print("\n🔎 Lectura guía:", flush=True)
    print(f"   Caminata más larga (N={last}): su2 {d:.1f}% vs so3 {o:.1f}% sobre consultas discriminantes", flush=True)
    if d > 95 and o < 70:
        print("   🟢 El signo espinorial (Q vs -Q) porta información de camino que la orientación física", flush=True)
        print("      BORRA. El doble recubrimiento no es una curiosidad matemática: es un canal de", flush=True)
        print("      información direccionable — y SO(3) lo pierde todo, SU(2) lo conserva todo.", flush=True)
    else:
        print("   🔴 Sin separación conforme a la teoría — registrar y revisar diseño.", flush=True)
    print(f"\n💾 Resultados: {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
