"""
benchmark_niah_e2e.py
=====================
P0-2 Audit Remediation: End-to-End NIAH with RANDOMIZED Needles and LEARNED Selective Gating.

Fixes the two methodological weaknesses identified in docs/project_audit_2026-08.md (R2):
  1. tests/test_needle_in_haystack.py used a FIXED needle identity (15 -> 85) across all
     trials, allowing a degenerate static shortcut that never exercises memory.
     -> Here the needle (key, value) pair is re-randomized on EVERY sequence, both in
        training and evaluation, so no static input->output mapping can solve the task.
  2. tests/test_selective_gating_niah.py was an oracle simulation (salience profile given).
     -> Here beta_t is produced by the TRAINED model via its data-dependent gate
        (w_beta), and a control arm with beta_t fixed to 1.0 everywhere (uniform writes)
        is trained under an identical budget. Comparing both isolates the value of
        learned selective retention.

Task structure per sequence (vocab=129):
  - Noise tokens (97..127) everywhere.
  - 1 needle pair (K,V): keys 1..32, values 33..96, inserted at a controlled depth.
  - D distractor pairs (K',V') with keys != needle key at random positions.
  - Query at the end: [QUERY_MARKER, K] -> target V (same recipe as the certified MQAR
    benchmark; v1 of this script omitted the marker and both arms stayed at chance,
    an undertrained-task artifact, not a gating finding).
  - Dense supervision: predict the value after EVERY pair key + at the final query.
  - Model has NO positional embeddings (relies on causal conv + recurrent state),
    enabling zero-shot length extrapolation.

Metric: exact-match accuracy at the final position over randomized trials.
Diagnostic: mean learned beta_t at the needle-key position vs noise positions (last layer).
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
from delta_phase import DeltaPhaseConfig, DeltaPhaseModel

RESULTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "niah_e2e_results.json"))

VOCAB_SIZE = 129
QUERY_MARKER = 128
KEY_LO, KEY_HI = 1, 33      # needle/distractor keys: 1..32
VAL_LO, VAL_HI = 33, 97     # values: 33..96
NOISE_LO, NOISE_HI = 97, VOCAB_SIZE


def generate_niah_batch(batch_size, seq_len, num_distractors=4, device='cpu', depth=None, rng=None):
    """Randomized-needle NIAH batch. depth=None -> uniform random depth per sequence."""
    rng = rng or torch
    tokens = torch.randint(NOISE_LO, NOISE_HI, (batch_size, seq_len), device=device)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)

    max_pos = seq_len - 3                      # pair occupies (pos, pos+1), query at seq_len-1
    candidate_slots = torch.arange(0, max_pos, 2)  # even slots guarantee non-overlap

    for b in range(batch_size):
        perm = torch.randperm(len(candidate_slots))[:num_distractors + 1]
        positions = candidate_slots[perm].tolist()

        # Needle key/value (random per sequence); distractor keys distinct from needle key
        all_keys = torch.randperm(KEY_HI - KEY_LO)[:num_distractors + 1] + KEY_LO
        vals = torch.randint(VAL_LO, VAL_HI, (num_distractors + 1,))
        needle_k, needle_v = int(all_keys[0]), int(vals[0])

        # Force needle depth if requested (replace first slot)
        if depth is not None:
            forced = min(max(int(round(depth * max_pos)), 0), max_pos)
            forced -= forced % 2
            positions[0] = forced

        pairs = [(int(all_keys[i]), int(vals[i]), positions[i]) for i in range(num_distractors + 1)]
        pairs[0] = (needle_k, needle_v, positions[0])  # index 0 = needle

        for k, v, pos in pairs:
            tokens[b, pos] = k
            tokens[b, pos + 1] = v
            targets[b, pos] = v                   # dense supervision on every pair key

        tokens[b, -2] = QUERY_MARKER              # query marker (MQAR recipe)
        tokens[b, -1] = needle_k                  # query key at final position
        targets[b, -1] = needle_v

    return tokens, targets


def build_model(beta_mode, device='cpu'):
    config = DeltaPhaseConfig(
        dim=128, emb_dim=0, n_layers=2, n_heads=4,
        vocab_size=VOCAB_SIZE, chunk_size=64, num_banks=2,
        beta_mode=beta_mode
    )
    return DeltaPhaseModel(config).to(device)


def final_position_accuracy(model, seq_len, num_batches=5, batch_size=16, num_distractors=4, device='cpu'):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = generate_niah_batch(batch_size, seq_len, num_distractors, device=device)
            logits = model(x)
            preds = logits[:, -1, :].argmax(dim=-1)
            gold = y[:, -1]
            correct += (preds == gold).sum().item()
            total += batch_size
    model.train()
    return 100.0 * correct / max(total, 1)


def train_arm(beta_mode, seed, steps, train_len, batch_size, lr, early_stop_acc, device, log_interval=50):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(beta_mode, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    t0 = time.time()
    final_step = steps
    for step in range(1, steps + 1):
        x, y = generate_niah_batch(batch_size, train_len, num_distractors=4, device=device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % log_interval == 0 or step == steps:
            acc = final_position_accuracy(model, train_len, num_batches=4, batch_size=batch_size, device=device)
            print(f"   [{beta_mode:>7} | seed {seed}] paso {step:4d}/{steps} | loss {loss.item():6.4f} | acc final {acc:6.2f}% | {time.time()-t0:7.1f}s", flush=True)
            if acc >= early_stop_acc:
                final_step = step
                print(f"   🎯 Early stop ({acc:.2f}% >= {early_stop_acc}%) en paso {step}", flush=True)
                break

    wallclock = time.time() - t0
    in_dist_acc = final_position_accuracy(model, train_len, num_batches=10, batch_size=32, device=device)
    print(f"   ✨ [{beta_mode} | seed {seed}] entrenado en {wallclock:.1f}s | acc in-dist {in_dist_acc:.2f}%", flush=True)
    return model, {"wallclock": wallclock, "final_step": final_step, "in_dist_acc": in_dist_acc}


def evaluate_length_matrix(model, lengths, depths, trials, num_distractors, device):
    results = {}
    for L in lengths:
        results[L] = {}
        for d in depths:
            hits = 0
            for _ in range(trials):
                x, y = generate_niah_batch(1, L, num_distractors, device=device, depth=d)
                with torch.no_grad():
                    pred = model(x)[0, -1, :].argmax().item()
                hits += int(pred == y[0, -1].item())
            results[L][d] = 100.0 * hits / trials
        row = " | ".join(f"{results[L][d]:5.1f}%" for d in depths)
        print(f"      L={L:>6,}: {row}", flush=True)
    return results


def beta_diagnostic(model, seq_len, depth, num_distractors, device):
    """Mean learned beta_t at the needle-key position vs noise positions (last layer)."""
    model.eval()
    betas_needle, betas_noise = [], []
    with torch.no_grad():
        for _ in range(8):
            x, _ = generate_niah_batch(4, seq_len, num_distractors, device=device, depth=depth)
            model(x)
            beta = model.blocks[-1].last_beta          # (B, H, L)
            needle_pos = min(max(int(round(depth * (seq_len - 3))), 0), seq_len - 3)
            betas_needle.append(beta[:, :, needle_pos].mean().item())
            mask = torch.ones(seq_len, dtype=torch.bool)
            mask[needle_pos:needle_pos + 2] = False    # exclude needle pair
            mask[-1] = False                            # exclude query
            betas_noise.append(beta[:, :, mask].mean().item())
    model.train()
    return float(np.mean(betas_needle)), float(np.mean(betas_noise))


def save_payload(payload):
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def apply_noise_mode(mode):
    """Diagnostic switch: 'pad' replaces random noise filler with PAD tokens (id 0)."""
    global NOISE_LO, NOISE_HI
    if mode == "pad":
        NOISE_LO, NOISE_HI = 0, 1


def run_suite(arms, seeds, steps, train_len, batch_size, lr, early_stop_acc,
              eval_lengths, depths, trials, num_distractors, device):
    print("=" * 112, flush=True)
    print("🧪 P0-2 END-TO-END NIAH: AGUJA ALEATORIA + GATING APRENDIDO (Audit Remediation — R2)", flush=True)
    print("=" * 112, flush=True)
    print(f"  • Hipótesis: el gating selectivo APRENDIDO (beta_t data-dependent) retiene mejor la aguja que escritura uniforme (beta=1)", flush=True)
    print(f"  • Aguja aleatoria por trial (keys 1..32, values 33..96) + {num_distractors} distractores | Sin embeddings posicionales (extrapolación)", flush=True)
    print(f"  • Train L={train_len}, batch={batch_size}, lr={lr}, max {steps} pasos, early stop >= {early_stop_acc}%", flush=True)
    print(f"  • Eval: longitudes {eval_lengths} x profundidades {[int(d*100) for d in depths]}% x {trials} trials", flush=True)
    print(f"  • Dispositivo: {device.upper()} | Python {platform.python_version()} / PyTorch {torch.__version__}", flush=True)
    print(f"  • Fecha UTC: {datetime.datetime.now(datetime.timezone.utc).isoformat()}", flush=True)
    print(f" • Brazos: {arms} | Semillas: {seeds}", flush=True)
    print("=" * 112 + "\n", flush=True)

    payload = {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "P0-2 end-to-end NIAH: randomized needles, learned gating vs fixed-beta control",
        "protocol": {"train_len": train_len, "steps": steps, "batch_size": batch_size, "lr": lr,
                      "early_stop_acc": early_stop_acc, "eval_lengths": eval_lengths,
                      "depths": depths, "trials": trials, "num_distractors": num_distractors,
                      "seeds": seeds, "device": device},
        "arms": {},
    }

    for arm in arms:
        payload["arms"][arm] = {"seeds": {}}
        acc_matrices, in_dist_accs, diag_needle, diag_noise = [], [], [], []

        for seed in seeds:
            print(f"\n🚀 Entrenando brazo '{arm}' [seed {seed}]...", flush=True)
            model, tmetrics = train_arm(arm, seed, steps, train_len, batch_size, lr, early_stop_acc, device)
            in_dist_accs.append(tmetrics["in_dist_acc"])

            print(f"   🔍 Matriz de recuperación (aguja aleatoria por trial):", flush=True)
            matrix = evaluate_length_matrix(model, eval_lengths, depths, trials, num_distractors, device)
            acc_matrices.append(matrix)

            b_n, b_no = beta_diagnostic(model, min(train_len, 2048), 0.5, num_distractors, device)
            diag_needle.append(b_n)
            diag_noise.append(b_no)
            print(f"   📊 Beta diagnóstica (última capa, d=50%): media en aguja {b_n:.4f} | media en ruido {b_no:.4f}", flush=True)

            payload["arms"][arm]["seeds"][str(seed)] = {
                **tmetrics,
                "matrix": {str(L): {str(d): matrix[L][d] for d in depths} for L in eval_lengths},
                "beta_needle_mean": b_n,
                "beta_noise_mean": b_no,
            }
            save_payload(payload)

        # Agregado sobre semillas
        agg = {}
        for L in eval_lengths:
            agg[str(L)] = {str(d): {
                "mean": float(np.mean([m[L][d] for m in acc_matrices])),
                "se": float(np.std([m[L][d] for m in acc_matrices]) / math.sqrt(len(acc_matrices))) if len(acc_matrices) > 1 else 0.0,
            } for d in depths}
        payload["arms"][arm]["summary"] = {
            "mean_in_dist_acc": float(np.mean(in_dist_accs)),
            "aggregate_matrix": agg,
            "beta_needle_mean": float(np.mean(diag_needle)),
            "beta_noise_mean": float(np.mean(diag_noise)),
        }
        save_payload(payload)

    print("\n" + "=" * 112, flush=True)
    print("📊 MATRIZ AGREGADA (media ± SE sobre semillas) — precisión exacta en posición final:", flush=True)
    for arm in arms:
        s = payload["arms"][arm]["summary"]
        print(f"\n  Brazo: {arm}  |  In-dist acc: {s['mean_in_dist_acc']:.2f}%  |  β aguja {s['beta_needle_mean']:.4f} vs ruido {s['beta_noise_mean']:.4f}")
        header = "     " + "".join([f"{int(float(d)*100):>9}%" for d in depths])
        print(header)
        for L in eval_lengths:
            cells = "".join([f"{s['aggregate_matrix'][str(L)][str(d)]['mean']:>7.1f}%±{s['aggregate_matrix'][str(L)][str(d)]['se']:<3.1f}" for d in depths])
            print(f"     L={L:>7,}: {cells}")
    print("\n" + "=" * 112, flush=True)
    print(f"✅ Resultados guardados en: {RESULTS_PATH}", flush=True)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P0-2 End-to-End NIAH (randomized needles, learned gating)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--train-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--early-stop-acc", type=float, default=99.5)
    parser.add_argument("--lengths", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    parser.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 0.9])
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--distractors", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 2024])
    parser.add_argument("--arms", type=str, nargs="+", default=["learned", "fixed"])
    parser.add_argument("--long", action="store_true", help="Añade 16384 y 65536 a las longitudes de evaluación")
    parser.add_argument("--noise-mode", type=str, default="noise", choices=["noise", "pad"],
                        help="Diagnostic: 'pad' replaces random noise tokens with PAD (no state writes "
                             "from distractor content) to isolate noise-crosstalk as the bottleneck.")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    lengths = list(args.lengths)
    if args.long:
        lengths += [16384, 65536]

    if args.noise_mode == "pad":
        # Diagnostic mode: filler = PAD (id 0) instead of random noise tokens.
        global NOISE_LO, NOISE_HI
        NOISE_LO, NOISE_HI = 0, 1

    if args.quick:
        run_suite(arms=["learned"], seeds=[42], steps=30, train_len=128, batch_size=8,
                  lr=args.lr, early_stop_acc=200.0,  # disable early stop in smoke test
                  eval_lengths=[128], depths=[0.5], trials=2, num_distractors=2,
                  device=args.device)
    else:
        run_suite(arms=args.arms, seeds=args.seeds, steps=args.steps, train_len=args.train_len,
                  batch_size=args.batch_size, lr=args.lr, early_stop_acc=args.early_stop_acc,
                  eval_lengths=lengths, depths=args.depths, trials=args.trials,
                  num_distractors=args.distractors, device=args.device)
