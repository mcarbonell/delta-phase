"""
benchmark_ffn_router_ablation.py
================================
P2-10 Audit Remediation: Learnable Substrate Lerp FFN vs gated MLP of equal budget.

Question (docs/project_audit_2026-08.md #10): does the FWHT/DCT-II/Haar multi-substrate
router actually earn its keep versus a plain gated MLP of comparable parameter budget,
and does the learned softmax router specialize away from uniform (33/33/33)?

Arms (identical DeltaPhaseMQAR backbone; ONLY the block FFN differs):
  - lerpffn_nb4 : LearnableSubstrateLerpFFN(num_banks=4)
                  params/block ~= 3 + 12*4*d_model + (4*d_model)*d_model = 4d^2 + 48d + 3
  - mlp_h2d     : Linear(d -> 2d) + GELU + Linear(2d -> d)
                  params/block ~= 4d^2 + small biases   (iso-parameter budget)

Protocol: identical to the certified Level 2 MQAR suite (same generator, models, early
stopping, OOD length evaluation) via import of tests/benchmark_rigorous_mqar.py.
Config: N_pairs=16, L_train=128, 3 seeds, up to 1500 steps, AdamW lr=3e-3.

After training we also dump the learned router probabilities per layer for the lerp arm
(get_substrate_probabilities) — the scientific payoff: does the router specialize?
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
    if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import benchmark_rigorous_mqar as base

RESULTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "ffn_router_ablation_results.json"))


def make_lerp_arm():
    return base.DeltaPhaseMQAR(vocab_size=base.VOCAB_SIZE, d_model=128, n_heads=4, chunk_size=32, num_layers=2)


def make_mlp_arm_per_layer(hidden_mult: float = 2.0):
    """Each layer gets its OWN gated MLP of ~iso-parametric budget (4d^2 weights)."""
    model = make_lerp_arm()
    d = 128
    hidden = int(d * hidden_mult)
    for block in model.blocks:
        block.ffn = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
        )
    return model


def ffn_param_count(model):
    return sum(p.numel() for n, p in model.named_parameters() if ".ffn." in n or "ffn" in n.lower())


def save(payload):
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_suite(arms, seeds, n_pairs, steps_per_train, early_stop_acc, lr, device):
    t0_global = time.time()
    L_train = 128
    eval_lengths = [L_train, 2 * L_train, 4 * L_train]
    total_runs = len(arms) * len(seeds)

    factories = {
        "lerpffn_nb4": make_lerp_arm,
        "mlp_h2d": make_mlp_arm_per_layer,
    }

    print("=" * 112, flush=True)
    print("🧪 P2-10 ABLATION DEL ROUTER DE SUSTRATOS ESPECTRALES (Audit Remediation)", flush=True)
    print("=" * 112, flush=True)
    print("  • Hipótesis: el FFN Lerp (FWHT+DCT+Haar) rinde >= MLP gated de presupuesto iso-paramétrico,", flush=True)
    print("     y el router softmax aprendido se especializa fuera del uniforme 33/33/33.", flush=True)
    print(f"  • Protocolo certificado MQAR: N_pairs={n_pairs}, L={L_train}, eval={eval_lengths}, max {steps_per_train} pasos, lr={lr}", flush=True)
    print(f"  • Dispositivo: {device.upper()} | Python {platform.python_version()} / PyTorch {torch.__version__}", flush=True)
    print(f"  • Fecha UTC: {datetime.datetime.now(datetime.timezone.utc).isoformat()}", flush=True)
    print(f"  • Brazos: {list(factories.keys())[:len(arms)]} | Semillas: {seeds}", flush=True)
    print("-" * 112, flush=True)
    inv = {a: factories[a]() for a in arms}
    total = {a: sum(p.numel() for p in m.parameters()) for a, m in inv.items()}
    ffn = {a: ffn_param_count(m) for a, m in inv.items()}
    for a in arms:
        print(f"     • {a:<12} | params modelo: {total[a]:>10,} | params FFN/bloque x{len(inv[a].blocks)}: {ffn[a]:,}", flush=True)
    print("=" * 112 + "\n", flush=True)

    payload = {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "P2-10 FFN substrate-router ablation vs iso-parametric gated MLP",
        "protocol": {"n_pairs": n_pairs, "L_train": L_train, "eval_lengths": eval_lengths,
                      "steps": steps_per_train, "early_stop_acc": early_stop_acc, "lr": lr,
                      "seeds": seeds, "device": device},
        "arms": {},
    }

    for arm in arms:
        accs = {L: [] for L in eval_lengths}
        s95s, walls = [], []
        router_reports = []

        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            import random; random.seed(seed)

            model = factories[arm]()
            trained, metrics = base.train_mqar_model(
                name=f"{arm} [Seed {seed}]", model=model,
                total_steps=steps_per_train, batch_size=32,
                seq_len=L_train, num_pairs=n_pairs, vocab_size=base.VOCAB_SIZE,
                lr=lr, device=device, log_interval=50, early_stop_acc=early_stop_acc,
                global_model_idx=len(s95s) + 1 + list(factories).index(arm) * len(seeds),
                total_models=total_runs, global_start_time=t0_global,
            )
            s95s.append(metrics["step_to_95"]); walls.append(metrics["wallclock"])
            row = {}
            for L_eval in eval_lengths:
                acc = base.evaluate_mqar_accuracy(trained, num_eval_batches=20, batch_size=32,
                                                  seq_len=L_eval, num_pairs=n_pairs,
                                                  vocab_size=base.VOCAB_SIZE, device=device)
                accs[L_eval].append(acc); row[str(L_eval)] = acc
                print(f"      • L={L_eval}: {acc:6.2f}%", flush=True)

            if arm == "lerpffn_nb4":
                probs = [trained.blocks[i].ffn.get_substrate_probabilities() for i in range(len(trained.blocks))]
                router_reports.append(probs)
                for i, (pf, pd, ph) in enumerate(probs):
                    print(f"      🎛️ Router capa {i+1}: FWHT {pf*100:5.2f}% | DCT {pd*100:5.2f}% | Haar {ph*100:5.2f}%", flush=True)

            payload.setdefault("arms", {}).setdefault(arm, {"seeds": {}})
            payload["arms"][arm]["seeds"][str(seed)] = {
                "length_accs": row, "step_to_95": metrics["step_to_95"],
                "wallclock": metrics["wallclock"], "early_stopped": metrics["early_stopped"],
            }
            if router_reports:
                payload["arms"][arm]["router_probs_per_seed"] = router_reports
            save(payload)

        summary = {
            "mean_accs": {str(L): float(np.mean(accs[L])) for L in eval_lengths},
            "se_accs": {str(L): float(np.std(accs[L]) / math.sqrt(len(accs[L]))) if len(accs[L]) > 1 else 0.0 for L in eval_lengths},
            "mean_steps_to_95": float(np.mean(s95s)),
            "mean_wallclock_sec": float(np.mean(walls)),
        }
        payload["arms"][arm]["summary"] = summary
        save(payload)

    print("\n" + "=" * 112, flush=True)
    print("📊 RESUMEN ABLATION FFN (media ± SE):", flush=True)
    for arm in arms:
        s = payload["arms"][arm]["summary"]
        cols = " | ".join([f"L={L}: {s['mean_accs'][str(L)]:6.2f} ± {s['se_accs'][str(L)]:4.2f}%" for L in eval_lengths])
        print(f"  {arm:<12} | {cols} | s95 medio: {s['mean_steps_to_95']:.0f} pasos", flush=True)
    print("=" * 112, flush=True)
    print(f"\n✅ Resultados guardados en: {RESULTS_PATH}", flush=True)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P2-10 FFN substrate router ablation")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 2024])
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--early-stop-acc", type=float, default=99.5)
    parser.add_argument("--arms", type=str, nargs="+", default=["lerpffn_nb4", "mlp_h2d"])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        run_suite(["lerpffn_nb4", "mlp_h2d"], seeds=[42], n_pairs=8,
                  steps_per_train=40, early_stop_acc=args.early_stop_acc,
                  lr=args.lr, device=args.device)
    else:
        run_suite(args.arms, seeds=args.seeds, n_pairs=args.pairs,
                  steps_per_train=args.steps, early_stop_acc=args.early_stop_acc,
                  lr=args.lr, device=args.device)
