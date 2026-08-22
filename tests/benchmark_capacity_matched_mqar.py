"""
benchmark_capacity_matched_mqar.py
==================================
P0-1 Audit Remediation: Capacity-Matched Control for the MQAR Headline Claim.

Context (docs/project_audit_2026-08.md, R1):
The certified benchmark (benchmark_rigorous_mqar.py) reported DeltaPhase (C^{32x32})
98.81% vs real Gated DeltaNet (R^{32x32}) 75.99% at N_pairs=32. But the complex state
stores 2*d_k^2 = 2048 real floats per head vs 1024 for the real baseline, so the gap
conflates phasor geometry with doubled raw capacity.

Control design — capacity accounting (real floats of recurrent state, total):
  - deltaphase_c32_h4    : 4 heads x C^{32x32} = 4 x 2048 = 8192 floats
  - realdeltanet_r32_h4  : 4 heads x R^{32x32} = 4 x 1024 = 4096 floats  (certified anchor)
  - realdeltanet_r64_h2  : 2 heads x R^{64x64} = 2 x 4096 = 8192 floats  <== CAPACITY-MATCHED

The capacity-matched arm keeps d_model=128 (2 heads x d_k=64), so embeddings, projections,
conv, FFN and optimizer budget are identical in shape to the other arms; ONLY the head
geometry (and hence total state floats) changes. If the phasor-geometry hypothesis is
correct, DeltaPhase (8192) should still beat the capacity-matched real arm (8192); if the
gap closes, the certified headline was driven by raw capacity, not geometry.

Protocol: identical to the certified Level 2 suite (same on-the-fly Zoology generator,
dense query supervision, early stopping at >= 99.5%, OOD length eval 2x/4x).
Reuses the EXACT model classes and data generator via import (no reimplementation drift).
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

try:
    if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Reuse the certified benchmark's generator and model classes verbatim (no drift).
import benchmark_rigorous_mqar as base

RESULTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "capacity_matched_mqar_results.json"))

# (name, factory, state_floats_per_head, n_heads, d_k, description)
def build_arms():
    return {
        "deltaphase_c32_h4": {
            "factory": lambda: base.DeltaPhaseMQAR(vocab_size=base.VOCAB_SIZE, d_model=128, n_heads=4, chunk_size=32, num_layers=2),
            "space": "C^{32x32} x 4 heads",
            "state_floats": 4 * 2 * 32 * 32,
        },
        "realdeltanet_r32_h4": {
            "factory": lambda: base.RealGatedDeltaNetMQAR(vocab_size=base.VOCAB_SIZE, d_model=128, n_heads=4, chunk_size=32, num_layers=2),
            "space": "R^{32x32} x 4 heads (certified anchor)",
            "state_floats": 4 * 32 * 32,
        },
        "realdeltanet_r64_h2": {
            "factory": lambda: base.RealGatedDeltaNetMQAR(vocab_size=base.VOCAB_SIZE, d_model=128, n_heads=2, chunk_size=32, num_layers=2),
            "space": "R^{64x64} x 2 heads (CAPACITY-MATCHED)",
            "state_floats": 2 * 64 * 64,
        },
    }


def save_results(payload):
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_suite(arm_names, seeds, n_pairs, steps_per_train, early_stop_acc, lr, device):
    global_start_time = time.time()
    arms = build_arms()
    for name in arm_names:
        assert name in arms, f"Unknown arm: {name}"

    L_train = 256 if n_pairs > 16 else 128
    eval_lengths = [L_train, 2 * L_train, 4 * L_train]
    total_runs = len(arm_names) * len(seeds)
    run_counter = 0

    print("=" * 112, flush=True)
    print("🧪 P0-1 CAPACITY-MATCHED MQAR CONTROL (Audit Remediation — docs/project_audit_2026-08.md R1)", flush=True)
    print("=" * 112, flush=True)
    print(f"  • Hipótesis bajo test:  ¿la ventaja de DeltaPhase sobre Gated DeltaNet real es geometría fasorial o solo 2x capacidad bruta?", flush=True)
    print(f"  • Protocolo:            idéntico al certificado Nivel 2 (generador Zoology on-the-fly, early stop >= {early_stop_acc}%)", flush=True)
    print(f"  • Dispositivo:          {device.upper()} ({platform.processor() or 'Multicore'})", flush=True)
    print(f"  • Fecha UTC:            {datetime.datetime.now(datetime.timezone.utc).isoformat()}", flush=True)
    print(f"  • Python/PyTorch:       {platform.python_version()} / {torch.__version__}", flush=True)
    print(f"  • Config:               N_pairs={n_pairs}, L_train={L_train}, eval={eval_lengths}, max {steps_per_train} pasos, lr={lr}", flush=True)
    print(f"  • Semillas ({len(seeds)}):        {seeds}  |  Total ejecuciones: {total_runs}", flush=True)
    print("-" * 112, flush=True)
    print("  📋 CONTABILIDAD DE CAPACIDAD DE ESTADO (flotantes reales):", flush=True)
    for name in arm_names:
        a = arms[name]
        print(f"     • {name:<22} | {a['space']:<42} | {a['state_floats']:,} floats", flush=True)
    print("=" * 112 + "\n", flush=True)

    payload = {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "P0-1 capacity-matched MQAR control (audit remediation)",
        "protocol": {"n_pairs": n_pairs, "L_train": L_train, "eval_lengths": eval_lengths,
                     "steps": steps_per_train, "early_stop_acc": early_stop_acc, "lr": lr,
                     "seeds": seeds, "device": device},
        "arms": {name: {"space": arms[name]["space"],
                         "state_floats": arms[name]["state_floats"],
                         "seeds": {}} for name in arm_names},
    }

    for name in arm_names:
        arm = arms[name]
        accs = {L: [] for L in eval_lengths}
        wallclocks, steps_to_50, steps_to_95, final_steps = [], [], [], []

        for seed in seeds:
            run_counter += 1
            torch.manual_seed(seed)
            np.random.seed(seed)
            import random
            random.seed(seed)

            model = arm["factory"]()
            trained, metrics = base.train_mqar_model(
                name=f"{name} [Seed {seed}]",
                model=model,
                total_steps=steps_per_train,
                batch_size=32,
                seq_len=L_train,
                num_pairs=n_pairs,
                vocab_size=base.VOCAB_SIZE,
                lr=lr,
                device=device,
                log_interval=50,
                early_stop_acc=early_stop_acc,
                global_model_idx=run_counter,
                total_models=total_runs,
                global_start_time=global_start_time,
            )
            wallclocks.append(metrics["wallclock"])
            steps_to_50.append(metrics["step_to_50"])
            steps_to_95.append(metrics["step_to_95"])
            final_steps.append(metrics["final_step"])

            for L_eval in eval_lengths:
                acc = base.evaluate_mqar_accuracy(
                    trained, num_eval_batches=20, batch_size=32,
                    seq_len=L_eval, num_pairs=n_pairs, vocab_size=base.VOCAB_SIZE, device=device
                )
                accs[L_eval].append(acc)
                print(f"      • L={L_eval:4d}: {acc:6.2f}%", flush=True)

            # Persistencia incremental (crash-safe)
            payload["arms"][name]["seeds"][str(seed)] = {
                "wallclock": metrics["wallclock"],
                "final_step": metrics["final_step"],
                "step_to_50": metrics["step_to_50"],
                "step_to_95": metrics["step_to_95"],
                "length_accs": {L: accs[L][-1] for L in eval_lengths},
            }
            save_results(payload)

        payload["arms"][name]["summary"] = {
            "mean_accs": {L: float(np.mean(accs[L])) for L in eval_lengths},
            "se_accs": {L: float(np.std(accs[L]) / math.sqrt(len(accs[L]))) if len(accs[L]) > 1 else 0.0 for L in eval_lengths},
            "mean_wallclock_sec": float(np.mean(wallclocks)),
            "mean_steps_to_50": float(np.mean(steps_to_50)),
            "mean_steps_to_95": float(np.mean(steps_to_95)),
        }
        save_results(payload)

    print("\n" + "=" * 112, flush=True)
    print("📊 RESUMEN CAPACIDAD-IGUALADA (media ± SE):", flush=True)
    print("-" * 112, flush=True)
    header = f"{'Arm':<24} | {'State floats':<12} | " + " | ".join([f"L={L:<5}" for L in eval_lengths]) + " |  Tiempo/run"
    print(header, flush=True)
    print("-" * 112, flush=True)
    for name in arm_names:
        s = payload["arms"][name]["summary"]
        cols = " | ".join([f"{s['mean_accs'][L]:6.2f} ± {s['se_accs'][L]:4.2f}%" for L in eval_lengths])
        print(f"{name:<24} | {arms[name]['state_floats']:>12,} | {cols} | {s['mean_wallclock_sec']:8.1f}s", flush=True)
    print("=" * 112, flush=True)
    print(f"\n✅ Resultados guardados en: {RESULTS_PATH}", flush=True)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P0-1 Capacity-Matched MQAR Control")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 2024])
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--early-stop-acc", type=float, default=99.5)
    parser.add_argument("--arms", type=str, nargs="+",
                        default=["deltaphase_c32_h4", "realdeltanet_r32_h4", "realdeltanet_r64_h2"])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        run_suite(arm_names=args.arms, seeds=[42], n_pairs=8, steps_per_train=60,
                  early_stop_acc=args.early_stop_acc, lr=args.lr, device=args.device)
    else:
        run_suite(arm_names=args.arms, seeds=args.seeds, n_pairs=args.pairs,
                  steps_per_train=args.steps, early_stop_acc=args.early_stop_acc,
                  lr=args.lr, device=args.device)
