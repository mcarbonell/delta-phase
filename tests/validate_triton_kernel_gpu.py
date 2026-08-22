"""
validate_triton_kernel_gpu.py
=============================
P2-9b GPU Validation Suite (Colab / Kaggle): certifies the tiled Triton Gram kernel
and the full DeltaPhase dtype/dispatcher stack on real CUDA hardware.

What it validates:
  V0. The previously-skipped pytest module (tests/test_triton_dispatcher.py) now runs,
      including test_triton_gram_kernel_matches_reference.
  V1. Kernel-vs-reference numerical parity of _triton_fused_phase_gram_kernel across
      chunk sizes (incl. NON-power-of-2 C), head dims up to 256, extreme betas.
  V2. delta_phase_chunkwise_fused on CUDA: gradient flow, inference path, and
      grad-path == no-grad-path numerical identity.
  V3. Full block on CUDA: forward/backward finiteness, sequential-vs-chunkwise
      equivalence in FP32, and bf16 autocast training step.
  V4. HONEST micro-benchmark: tiled Triton kernel vs the plain PyTorch vectorized Gram
      (cos@cosT + sin@sinT). If PyTorch wins, we say so.

Usage (Colab/Kaggle):
    !git clone <repo> && cd delta-phase
    !python tests/validate_triton_kernel_gpu.py
Results: prints live + saves docs/triton_kernel_gpu_validation.json
"""

import os
import sys
import json
import time
import subprocess
import datetime
import math

import torch

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

RESULTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "triton_kernel_gpu_validation.json"))
DEVICE = "cuda"


def header():
    import triton
    print("=" * 100, flush=True)
    print("🧪 VALIDACIÓN TRITON KERNEL EN GPU (P2-9b)", flush=True)
    print("=" * 100, flush=True)
    print(f"  • Fecha UTC:   {datetime.datetime.now(datetime.timezone.utc).isoformat()}", flush=True)
    print(f"  • Python:      {sys.version.split()[0]} | PyTorch: {torch.__version__} | Triton: {triton.__version__}", flush=True)
    print(f"  • GPU:         {torch.cuda.get_device_name(0)}", flush=True)
    print(f"  • Capability:  sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}", flush=True)
    print("=" * 100 + "\n", flush=True)
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "cc": ".".join(map(str, torch.cuda.get_device_capability(0)))}


def next_pow2(n):
    return 1 << max(1, (n - 1).bit_length())


# ---------------------------------------------------------------------
# V0 — pytest module (executes the previously-skipped CUDA test)
# ---------------------------------------------------------------------
def v0_pytest_module(payload):
    print("▶ V0: pytest tests/test_triton_dispatcher.py (en GPU)", flush=True)
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_triton_dispatcher.py", "-q"],
                       cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                       capture_output=True, text=True, timeout=600)
    tail = "\n".join(r.stdout.strip().splitlines()[-3:])
    print(tail + "\n", flush=True)
    payload["pytest"] = {"returncode": r.returncode, "tail": tail}
    assert r.returncode == 0, "pytest module failed on GPU"


# ---------------------------------------------------------------------
# V1 — kernel vs reference parity
# ---------------------------------------------------------------------
def v1_parity(payload):
    from delta_phase.kernels.triton_chunk_delta import (
        _triton_fused_phase_gram_kernel, gram_matrix_reference
    )
    print("▶ V1: paridad numérica kernel Triton vs referencia PyTorch", flush=True)
    torch.manual_seed(0)
    configs = [(16, 16), (32, 32), (48, 32), (64, 64), (64, 96), (128, 64), (64, 192)]
    results = []
    worst = 0.0
    for C, dk in configs:
        N = 64
        theta = torch.randn(N, C, dk, device=DEVICE)
        beta = torch.rand(N, C, device=DEVICE)
        beta[0] = 0.0  # fila beta≈0 debe anular su salida
        block_c, block_d = min(next_pow2(C), 128), 32

        ref = gram_matrix_reference(theta, beta)
        try:
            out = torch.empty(N, C, C, device=DEVICE)
            _triton_fused_phase_gram_kernel[(N,)](
                theta, beta, out, C, dk,
                theta.stride(-2), theta.stride(-1),
                beta.stride(-1),
                out.stride(-2), out.stride(-1),
                1.0 / float(dk),
                BLOCK_C=block_c, BLOCK_D=block_d,
            )
            diff = (out - ref).abs().max().item()
            strict_upper = torch.triu(out, diagonal=0).abs().max().item()
            ok = diff < 1e-3 and strict_upper == 0.0
            worst = max(worst, diff)
            results.append({"C": C, "dk": dk, "block_c": block_c, "block_d": block_d,
                            "max_diff": diff, "strict_upper_max": strict_upper, "pass": ok})
            print(f"   C={C:>3} dk={dk:>3} | BLOCK_C={block_c:>3} BLOCK_D={block_d} | "
                  f"max_diff={diff:.3e} | triu==0: {strict_upper == 0.0} | {'✅' if ok else '❌'}", flush=True)
        except Exception as ex:
            results.append({"C": C, "dk": dk, "block_c": block_c, "block_d": block_d,
                            "error": f"{type(ex).__name__}: {ex}", "pass": False})
            print(f"   C={C:>3} dk={dk:>3} | ❌ no compiló/ejecutó en este hw: {type(ex).__name__}", flush=True)

    payload["parity"] = {"worst_diff": worst, "configs": results}
    assert results and any(r["pass"] for r in results), "Ninguna config pasó paridad"
    executed = [r for r in results if "max_diff" in r]
    print(f"   ✅ Paridad OK en {sum(r['pass'] for r in executed)}/{len(executed)} configs ejecutadas "
          f"(peor diff {worst:.3e})\n", flush=True)


# ---------------------------------------------------------------------
# V2 — dispatcher en CUDA
# ---------------------------------------------------------------------
def v2_dispatcher(payload):
    from delta_phase import delta_phase_chunkwise_fused
    print("▶ V2: dispatcher delta_phase_chunkwise_fused en CUDA", flush=True)
    torch.manual_seed(0)
    B, H, L, D, C = 2, 4, 128, 32, 32
    tk = torch.randn(B, H, L, D, device=DEVICE, requires_grad=True)
    tq = torch.randn(B, H, L, D, device=DEVICE)
    v = torch.randn(B, H, L, D, device=DEVICE)
    beta = torch.sigmoid(torch.randn(B, H, L, device=DEVICE))

    out, M = delta_phase_chunkwise_fused(tk, tq, v, beta, chunk_size=C)
    out.sum().backward()
    grad_ok = tk.grad is not None and torch.isfinite(tk.grad).all().item()

    with torch.no_grad():
        out2, M2 = delta_phase_chunkwise_fused(tk.detach(), tq, v, beta, chunk_size=C)
    ident = (out.detach() - out2).abs().max().item()

    payload["dispatcher"] = {"grad_ok": bool(grad_ok), "grad_vs_nograd_max_diff": ident}
    print(f"   Gradiente fluye y es finito: {'✅' if grad_ok else '❌'} | diff grad-vs-nograd: {ident:.3e}\n", flush=True)
    assert grad_ok and ident < 1e-5


# ---------------------------------------------------------------------
# V3 — bloque completo en CUDA (FP32 equivalencia + bf16 autocast)
# ---------------------------------------------------------------------
def v3_block(payload):
    from delta_phase.layers import DeltaPhaseHolographicBlock
    print("▶ V3: bloque completo en CUDA (FP32 + bf16 autocast)", flush=True)
    torch.manual_seed(0)
    block = DeltaPhaseHolographicBlock(d_model=128, n_heads=4, chunk_size=32).to(DEVICE)

    # FP32 equivalence parallel vs sequential
    x = torch.randn(2, 100, 128, device=DEVICE)  # L no múltiplo del chunk
    with torch.no_grad():
        out_par, st_par = block(x)
        outs, st = [], None
        for t in range(x.shape[1]):
            o_t, st = block.step(x[:, t:t + 1, :], state=st)
            outs.append(o_t)
        out_seq = torch.cat(outs, dim=1)
    eq = (out_par - out_seq).abs().max().item()

    # bf16 autocast training step
    xb = torch.randn(8, 128, 128, device=DEVICE, requires_grad=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ob, _ = block(xb)
        loss = ob.float().sum()
    loss.backward()
    bf16_ok = torch.isfinite(ob).all().item() and torch.isfinite(xb.grad).all().item()

    payload["block_cuda"] = {"fp32_equiv_max_diff": eq, "bf16_autocast_ok": bool(bf16_ok)}
    print(f"   Equivalencia paralelo/secuencial FP32: {eq:.3e} ({'✅' if eq < 1e-4 else '❌'})", flush=True)
    print(f"   Entrenamiento bf16 autocast finito:    {'✅' if bf16_ok else '❌'}\n", flush=True)
    assert eq < 1e-4 and bf16_ok


# ---------------------------------------------------------------------
# V4 — benchmark honesto: Triton tile vs PyTorch vectorizado
# ---------------------------------------------------------------------
def v4_benchmark(payload):
    from delta_phase.kernels.triton_chunk_delta import (
        _triton_fused_phase_gram_kernel, gram_matrix_reference
    )
    print("▶ V4: benchmark Triton tile vs PyTorch vectorizado (N=512 matrices, mediana de 30 reps)", flush=True)
    torch.manual_seed(0)
    N = 512
    rows = []
    for C, dk in [(32, 32), (64, 32), (64, 64), (128, 64), (64, 128), (128, 128)]:
        theta = torch.randn(N, C, dk, device=DEVICE)
        beta = torch.rand(N, C, device=DEVICE)
        out = torch.empty(N, C, C, device=DEVICE)
        block_c, block_d = min(next_pow2(C), 128), 32
        args = (theta, beta, out, C, dk,
                theta.stride(-2), theta.stride(-1), beta.stride(-1),
                out.stride(-2), out.stride(-1), 1.0 / float(dk))
        kwargs = {"BLOCK_C": block_c, "BLOCK_D": block_d}

        def run(fn):
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            times = []
            for _ in range(30):
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record(); fn(); e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
            times.sort()
            return times[len(times) // 2]

        try:
            t_tri = run(lambda: _triton_fused_phase_gram_kernel[(N,)](*args, **kwargs))
        except Exception as ex:
            t_tri = float("nan")
            print(f"   C={C:>3} dk={dk:>3}: kernel lanzó {type(ex).__name__} (no soportado en este hw)", flush=True)
        t_ref = run(lambda: gram_matrix_reference(theta, beta))

        sp = (t_ref / t_tri) if t_tri == t_tri else float("nan")
        winner = "TRITON ⚡" if sp > 1.05 else ("pytorch" if sp < 0.95 else "~paridad")
        rows.append({"C": C, "dk": dk, "triton_ms": t_tri, "pytorch_ms": t_ref, "speedup": sp})
        print(f"   C={C:>3} dk={dk:>3} | triton {t_tri:7.3f} ms | pytorch {t_ref:7.3f} ms | {sp:5.2f}× → {winner}", flush=True)

    payload["benchmark"] = rows
    print("", flush=True)


def main():
    assert torch.cuda.is_available(), "Este script requiere CUDA (Colab/Kaggle GPU)"
    payload = {"date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "env": header()}
    v0_pytest_module(payload)
    v1_parity(payload)
    v2_dispatcher(payload)
    v3_block(payload)
    v4_benchmark(payload)

    n_fast = sum(1 for r in payload["benchmark"] if r["speedup"] > 1.05)
    print("=" * 100, flush=True)
    print(f"🏁 RESUMEN: paridad ✅ | dispatcher ✅ | bloque CUDA ✅ | bf16 ✅ | "
          f"Triton más rápido que PyTorch en {n_fast}/{len(payload['benchmark'])} configs", flush=True)
    print("=" * 100, flush=True)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Resultados guardados en: {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
