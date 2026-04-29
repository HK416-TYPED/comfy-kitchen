"""Phase 4 bench harness — needs tile-packed weight repack.

Repacks the synthetic row-major weight to the Phase 4 layout:
  wgt[cta_n, cta_k, n_stripe (=BLOCK_N/4), k_byte (=BLOCK_KH), n_within (=4)]
"""
from __future__ import annotations

import argparse
import ctypes
import time
from pathlib import Path

import torch

KERNELS_DIR = Path(__file__).parent.parent / "kernels"
_GROUP = 64
BLOCK_M = 32
BLOCK_N = 128
BLOCK_KH = 32          # = BLOCK_K / 2 = 32 packed-int4 bytes
N_STRIPES = BLOCK_N // 4


def load_phase4():
    so = KERNELS_DIR / "phase4_tile_packed.so"
    lib = ctypes.CDLL(str(so))
    fn = lib.launch_w4a4_gemm_phase4
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    return fn


def repack_weight(wgt_row_major: torch.Tensor) -> torch.Tensor:
    """(N, K/2) row-major int8 → (N/BLOCK_N, K/BLOCK_K, BLOCK_N/4, BLOCK_KH, 4) flat int8.

    Each (cta_n, cta_k) tile is BLOCK_N × BLOCK_KH = 4 KB contiguous.
    Inside: BLOCK_N/4 stripes, each = BLOCK_KH bytes × 4 N rows interleaved.
    """
    N, Kh = wgt_row_major.shape
    assert N % BLOCK_N == 0 and Kh % BLOCK_KH == 0, f"N={N} Kh={Kh} not aligned"
    cta_n = N // BLOCK_N
    cta_k = Kh // BLOCK_KH
    # (cta_n, BLOCK_N, cta_k, BLOCK_KH)
    x = wgt_row_major.view(cta_n, BLOCK_N, cta_k, BLOCK_KH)
    # Permute → (cta_n, cta_k, BLOCK_N, BLOCK_KH)
    x = x.permute(0, 2, 1, 3).contiguous()
    # Split BLOCK_N=128 into (BLOCK_N/4=32 stripes, 4 rows per stripe)
    x = x.view(cta_n, cta_k, BLOCK_N // 4, 4, BLOCK_KH)
    # Permute (4, BLOCK_KH) → (BLOCK_KH, 4) so within stripe layout is
    #   k_byte 0..BLOCK_KH-1, then n_within 0..3 (innermost contiguous)
    x = x.permute(0, 1, 2, 4, 3).contiguous()
    return x.view(-1).view(torch.int8)


def make_inputs(m: int, n: int, k: int, device: str = "cuda"):
    g = _GROUP
    torch.manual_seed(42)
    wgt_int = torch.randint(-7, 8, (n, k), dtype=torch.int8, device=device)
    lo = (wgt_int[..., 0::2].to(torch.int32) & 0xF)
    hi = (wgt_int[..., 1::2].to(torch.int32) & 0xF)
    wgt_row = (lo | (hi << 4)).to(torch.int8)
    wgt_packed = repack_weight(wgt_row)

    act_int = torch.randint(-7, 8, (m, k), dtype=torch.int8, device=device)
    lo = (act_int[..., 0::2].to(torch.int32) & 0xF)
    hi = (act_int[..., 1::2].to(torch.int32) & 0xF)
    act = (lo | (hi << 4)).to(torch.int8)

    ascales = (torch.rand(k // g, m, dtype=torch.float32, device=device) * 0.1 + 0.01).bfloat16()
    wscales = (torch.rand(k // g, n, dtype=torch.float32, device=device) * 0.1 + 0.01).bfloat16()
    return act, act_int, wgt_packed, wgt_int, ascales, wscales


def fp32_reference(act_int, wgt_int, ascales, wscales):
    m, k = act_int.shape
    n = wgt_int.shape[0]
    g = _GROUP
    out = torch.zeros(m, n, dtype=torch.float32, device=act_int.device)
    for gi in range(k // g):
        a_seg = act_int[:, gi * g : (gi + 1) * g].float()
        w_seg = wgt_int[:, gi * g : (gi + 1) * g].float()
        out += a_seg @ w_seg.t() * ascales[gi].float().unsqueeze(1) * wscales[gi].float().unsqueeze(0)
    return out.bfloat16()


def bench_one(fn, m, n, k, n_warmup=5, n_iter=20):
    act, act_int, wgt_packed, wgt_int, ascales, wscales = make_inputs(m, n, k)
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    for _ in range(n_warmup):
        fn(act.data_ptr(), wgt_packed.data_ptr(), ascales.data_ptr(), wscales.data_ptr(),
           out.data_ptr(), m, n, k, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter):
        fn(act.data_ptr(), wgt_packed.data_ptr(), ascales.data_ptr(), wscales.data_ptr(),
           out.data_ptr(), m, n, k, torch.cuda.current_stream().cuda_stream)
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / n_iter
    ref = fp32_reference(act_int, wgt_int, ascales, wscales)
    diff = (out.float() - ref.float()).abs()
    rel = diff.max().item() / (ref.abs().max().item() + 1e-9)
    flops = 2.0 * m * n * k
    return ms, flops / 1e12 / (ms / 1000), rel


def main():
    print(f"Phase 4 W4A4 GEMM bench on {torch.cuda.get_device_name(0)}")
    print(f"{'shape':<35} {'ms':>8}  {'TFLOPs':>8}  {'rel err':>10}  status")
    print(f"{'-'*35} {'-'*8}  {'-'*8}  {'-'*10}  ------")

    fn = load_phase4()
    for name, m, n, k in [
        ("QKV     M=4096 K=3072 N=3072", 4096, 3072, 3072),
        ("MLP_FC1 M=4096 K=12288 N=3072", 4096, 3072, 12288),
        ("MLP_FC2 M=4096 K=3072 N=12288", 4096, 12288, 3072),
        ("MOD     M=4096 K=18432 N=3072", 4096, 18432, 3072),
    ]:
        try:
            ms, tflops, rel = bench_one(fn, m, n, k)
            ok = "OK" if rel < 1e-2 else "BAD"
            print(f"{name:<35} {ms:8.3f}  {tflops:8.1f}  {rel:10.2e}  {ok}")
        except Exception as ex:
            print(f"{name:<35}  FAIL: {ex}")


if __name__ == "__main__":
    main()
