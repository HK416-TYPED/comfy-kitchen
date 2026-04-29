"""Benchmark + correctness harness for W4A4 research kernels.

Each Phase has its own shared library at kernels/phaseN_*.so with the entry
point `launch_w4a4_gemm_phaseN(act, wgt, ascales, wscales, out, M, N, K, stream)`.

Workload:
  Qwen-Image-Edit-2511 production attention/MLP shapes at M=4096:
    QKV     M=4096 K=3072  N=3072
    OUT     M=4096 K=3072  N=3072
    MLP_FC1 M=4096 K=3072  N=12288
    MLP_FC2 M=4096 K=12288 N=3072
    MOD     M=4096 K=3072  N=18432   (handled by AWQ in production, included for shape coverage)

Reports TFLOPs achieved, ms/forward, and rel error vs eager fp32 reference.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import time
from pathlib import Path

import torch

KERNELS_DIR = Path(__file__).parent.parent / "kernels"


def load_phase_lib(phase: int) -> ctypes.CDLL:
    so_path = next(KERNELS_DIR.glob(f"phase{phase}_*.so"), None)
    if so_path is None:
        raise FileNotFoundError(f"No compiled kernel for phase {phase} in {KERNELS_DIR}")
    lib = ctypes.CDLL(str(so_path))
    fn = getattr(lib, f"launch_w4a4_gemm_phase{phase}")
    fn.restype = None
    fn.argtypes = [
        ctypes.c_void_p,  # act
        ctypes.c_void_p,  # wgt
        ctypes.c_void_p,  # ascales
        ctypes.c_void_p,  # wscales
        ctypes.c_void_p,  # out
        ctypes.c_int,     # M
        ctypes.c_int,     # N
        ctypes.c_int,     # K
        ctypes.c_void_p,  # stream
    ]
    return fn


_GROUP = 64


def make_synthetic_inputs(m: int, n: int, k: int, device: str = "cuda"):
    """Create kitchen-native row-major SVDQuant W4A4 tensors with known
    dequant values so we can compute fp32 reference and check rel error.
    """
    g = _GROUP
    assert k % g == 0
    torch.manual_seed(42)

    # Random signed int4 weight in [-7, 7]
    wgt_int = torch.randint(-7, 8, (n, k), dtype=torch.int8, device=device)
    # Pack two int4 per byte: low nibble = col 2k, high nibble = col 2k+1
    lo = (wgt_int[..., 0::2].to(torch.int32) & 0x0F)
    hi = (wgt_int[..., 1::2].to(torch.int32) & 0x0F)
    wgt = (lo | (hi << 4)).to(torch.int8)

    act_int = torch.randint(-7, 8, (m, k), dtype=torch.int8, device=device)
    lo = (act_int[..., 0::2].to(torch.int32) & 0x0F)
    hi = (act_int[..., 1::2].to(torch.int32) & 0x0F)
    act = (lo | (hi << 4)).to(torch.int8)

    # Per-group scales: small values to keep accumulator in range
    ascales = (torch.rand(k // g, m, dtype=torch.float32, device=device) * 0.1 + 0.01).bfloat16()
    wscales = (torch.rand(k // g, n, dtype=torch.float32, device=device) * 0.1 + 0.01).bfloat16()

    return act, act_int, wgt, wgt_int, ascales, wscales


def fp32_reference(act_int: torch.Tensor, wgt_int: torch.Tensor,
                   ascales: torch.Tensor, wscales: torch.Tensor) -> torch.Tensor:
    """fp32 oracle: dequant act + wgt, group sum × ascale × wscale.

    K-group-by-K-group reduction to keep the (M, N, K/G) intermediate from
    blowing up at large shapes (e.g. M=4096 N=18432 K=3072 → 84 GB einsum).
    """
    m, k = act_int.shape
    n = wgt_int.shape[0]
    g = _GROUP
    n_groups = k // g
    out = torch.zeros(m, n, dtype=torch.float32, device=act_int.device)
    for gi in range(n_groups):
        a_seg = act_int[:, gi * g : (gi + 1) * g].float()      # (M, G)
        w_seg = wgt_int[:, gi * g : (gi + 1) * g].float()      # (N, G)
        int_dot = a_seg @ w_seg.t()                            # (M, N)
        out += int_dot * ascales[gi].float().unsqueeze(1) * wscales[gi].float().unsqueeze(0)
    return out.bfloat16()


def run_kernel(launch_fn, act, wgt, ascales, wscales, m, n, k):
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream
    launch_fn(
        act.data_ptr(), wgt.data_ptr(),
        ascales.data_ptr(), wscales.data_ptr(),
        out.data_ptr(),
        m, n, k, stream,
    )
    return out


def bench(launch_fn, m: int, n: int, k: int, n_warmup=5, n_iter=20):
    act, act_int, wgt, wgt_int, ascales, wscales = make_synthetic_inputs(m, n, k)

    # Warmup
    for _ in range(n_warmup):
        out = run_kernel(launch_fn, act, wgt, ascales, wscales, m, n, k)
    torch.cuda.synchronize()

    # Time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        out = run_kernel(launch_fn, act, wgt, ascales, wscales, m, n, k)
    end.record()
    torch.cuda.synchronize()
    ms_per = start.elapsed_time(end) / n_iter

    # Correctness
    ref = fp32_reference(act_int, wgt_int, ascales, wscales)
    diff = (out.float() - ref.float()).abs()
    rel = diff.max().item() / (ref.float().abs().max().item() + 1e-9)

    flops = 2.0 * m * n * k  # multiply+add per output element × K reduce
    tflops = flops / 1e12 / (ms_per / 1000.0)
    return ms_per, tflops, rel


SHAPES = [
    ("QKV     ", 4096, 3072, 3072),
    ("OUT     ", 4096, 3072, 3072),
    ("MLP_FC1 ", 4096, 3072, 12288),
    ("MLP_FC2 ", 4096, 12288, 3072),
    ("MOD     ", 4096, 3072, 18432),
    # smaller M (modulation runs at M=8 on the inner block during Sampling)
    ("M=8 KQV ", 8, 3072, 3072),
    ("M=64 QKV", 64, 3072, 3072),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True, help="phase number to bench")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    print(f"Phase {args.phase} W4A4 GEMM bench on {torch.cuda.get_device_name(0)}")
    print()
    print(f"  {'shape':<30} {'ms':>8}   {'TFLOPs':>8}   {'rel err':>10}   status")
    print(f"  {'-'*30} {'-'*8}   {'-'*8}   {'-'*10}   ------")

    launch_fn = load_phase_lib(args.phase)

    for name, m, n, k in SHAPES:
        try:
            ms, tflops, rel = bench(launch_fn, m, n, k)
            ok = "OK" if rel < 1e-2 else f"BAD"
            print(f"  {name} M={m:5d} K={k:5d} N={n:5d}  {ms:8.3f}   {tflops:8.1f}   {rel:10.2e}   {ok}")
        except Exception as e:
            print(f"  {name} M={m:5d} K={k:5d} N={n:5d}  FAILED: {e}")


if __name__ == "__main__":
    main()
