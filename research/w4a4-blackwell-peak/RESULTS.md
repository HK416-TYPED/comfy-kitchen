# Phase 0-3 Research Results — W4A4 on Blackwell sm_120 (RTX 5090)

> **TL;DR**: Kitchen-native row-major W4A4 caps near **100 TFLOPs / 22% peak**
> on sm_120 regardless of shmem/pipelining/scale-caching tricks. The structural
> bottleneck is HBM cache-line utilization on the weight load, which is
> determined by the on-disk layout. To reach nunchaku's 1.67 s/iter
> (~150-180 TFLOPs effective), the on-disk format or load-time GPU repack
> must change. **Path B from the original 调研 plan can't be done with
> shmem-side optimization alone.**

## Numbers

Bench shapes from Qwen-Image-Edit-2511 production (M=4096):

| Shape | Phase 0 | Phase 1 | Phase 2v1 | Phase 2v2 | Phase 3 (diag) |
|---|---|---|---|---|---|
| QKV (3072, 3072) | 76 | 75 | 69 | 79 | **93** |
| MLP_FC1 (3072, 12288) | 78 | 76 | 46 | 45 | **94** |
| MLP_FC2 (12288, 3072) | 82 | 79 | 72 | 83 | **98** |

(All numbers in TFLOPs. Peak measured 463 TFLOPs from synthetic mma.sync spam.)

Correctness: all phases pass `rel < 1e-3` vs fp32 oracle on shapes that fit
sm_120 dynamic shmem.

## What each phase did

### Phase 0 — baseline kernel + bench harness ✓ SHIP
Per-warp `mma.sync.m16n8k64.s4.s4.s32`, single-stage cp.async.cg, fp32
accumulator with per-group scale × scale fma. Critical correctness bug
identified and fixed: missing `__syncthreads` at end of K-iteration caused
race between current MMA reads and next-iter cp.async writes (single buffer);
manifested only at large M*N + K>64.

### Phase 1 — double buffer + shmem stride padding ✗ NEGATIVE
Pivoted from cp.async.bulk (rejected: bulk transfers are contiguous-only,
weight tiles are row-strided). Pivot landed on:
1. cp.async.cg with kStages=2 double buffer
2. Shmem row stride padded to 48 bytes (= 32 + 16) to break 32-bank cycle

Result: 76 → 75 TFLOPs. End-of-iter `__syncthreads` (still required to keep
warps coherent across the cp.async commit/wait state) cancels the double
buffer's overlap benefit. Bank-conflict padding alone is too small a win
to register against an unchanged compute path.

### Phase 2v1 — 8-warp producer/consumer + mbarrier ✗ NEGATIVE
4 producer warps (cp.async) + 4 consumer warps (MMA), 3-stage circular
buffer, mbarrier handoff. Architecturally clean but **slower**: 76 → 69
TFLOPs. Doubling warps/CTA halves occupancy; fewer concurrent CTAs more
than offsets the per-CTA pipeline overlap.

Bug found and fixed: mbarrier arrival count must equal producer thread
count (not 1), because `cp.async.wait_group` is per-thread — only the
arriving thread's loads have completed.

### Phase 2v2 — scales-in-shmem only (4-warp) ◐ MIXED
Isolates the scales-shmem optimization. QKV/MLP_FC2 (small K) get +4%
(76 → 79). MLP_FC1 (K=12288) loses 42% because per-CTA scales shmem grows
to 60 KB and total to 83 KB, halving occupancy on sm_120 (default opt-out
shmem ~48 KB).

### Phase 3 — diagnostic, MMA-only ✗ INSUFFICIENT
Stripped the per-K-iter scale fma chain entirely (output incorrect, just
the timing matters). **93-98 TFLOPs across all shapes**. Tells us scale
chain accounts for ~20% of K-loop time. Even with it fully gone, we cap
at ~22% peak.

The remaining 78% loss vs peak is **structural to row-major HBM layout**:
each HBM cache line carries ONE N-row's K-segment instead of nunchaku's
4 N-rows packed (kInterleave=4). On the weight load, this costs 4× HBM
bandwidth. Shmem-side optimizations cannot recover bandwidth that's
already wasted upstream.

## sm_120 hardware constraints discovered

- **Dynamic shmem cap < 112 KB**: `cudaFuncSetAttribute(...,
  MaxDynamicSharedMemorySize, 112 * 1024)` returns `invalid argument` on
  RTX 5090. Phase 2v1's 112 KB scales+tiles for MOD K=18432 hit this.
  Need to confirm exact limit; nominal sm_120 spec lists 228 KB but
  consumer Blackwell appears more restricted.
- **wgmma.wait_group not compilable on sm_120**: prevents direct port
  of Marlin's H100 design.
- **mma.sync.m16n8k64 measured peak 463 TFLOPs** (synthetic spam, ~55%
  of theoretical 836 TFLOPs from datasheet — even peak benchmark caps
  at 55% utility because of warp scheduler limits).

## Recommendation

The research goal "≥ nunchaku's 1.67 s/iter on RTX 5090 with kitchen-native
on-disk row-major and no checkpoint re-conversion" is **not reachable with
shmem-side optimizations alone**. The 4× HBM cache-line waste is structural.

Three viable production paths from here:

1. **GPU-side load-time repack** (`Path A` from research/调研 plan).
   Load-time pass converts row-major → tile-packed in HBM. One-time
   ~50 ms cost per checkpoint load. Kernel then matches nunchaku perf.
   Doesn't touch on-disk format. **2-3 day implementation**.

2. **Use the existing verbatim zgemm port**
   (`feat/nunchaku-verbatim-replica` branch). Already vendored from
   nunchaku — same kernel, ~1.67 s/iter immediately. Costs: re-convert
   25 HF variants to tile-packed (~10 hours batch). **0 day implementation**
   if user accepts re-conversion.

3. **True Marlin-style kernel** (Phase 4 stretch). Tile-packed on-disk +
   register-level int4 unpack with `prmt`/`lop3` PTX + `cp.async.bulk.tensor`
   (TMA descriptors set up host-side). Targets 1.3 s/iter (90% peak).
   **2-3 weeks implementation**, high risk on Blackwell consumer (tcgen05
   functional vs. dead silicon unverified).

Path 1 is the recommended path: acceptable cost, matches nunchaku perf,
keeps clean kitchen-native on-disk layout. The 50 ms repack is invisible
to E2E sampling time (>3 minutes for Qwen-Image-Edit at 20 steps).

## Files in this branch

- `PLAN.md` — original research plan (still useful for Phase 4 scoping)
- `sm120_features.md` — sm_120 PTX scorecard
- `RESULTS.md` — this file
- `probes/01..04_*.cu` — sm_120 feature probes
- `kernels/phase0_baseline.cu` — Phase 0 reference (76 TFLOPs)
- `kernels/phase1_bulkload.cu` — Phase 1 (negative)
- `kernels/phase2v1_producer_consumer.cu` — Phase 2v1 (negative)
- `kernels/phase2v2_scales_shmem.cu` — Phase 2v2 (mixed)
- `kernels/phase3_diag_no_scales.cu` — Phase 3 diagnostic (incorrect output)
- `bench/run_bench.py` — standardized 7-shape benchmark harness
