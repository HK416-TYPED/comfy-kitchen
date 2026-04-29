# W4A4 Peak Performance on Blackwell sm_120 (RTX 5090) — Research Plan

> **Status**: Investigation phase. Output is a design doc + micro-benchmarks of
> candidate techniques, not a finished kernel. Branch is `research/w4a4-blackwell-peak`
> isolated from PR-track branches; nothing here ships until benched against
> nunchaku 1.67 s/it baseline.

## Goal

W4A4 SVDQuant kernel on RTX 5090 reaching **≥80% int4 tensor-core peak** for
Qwen-Image-Edit-2511 production shapes (M=4096, K=3072–12288, N=3072–18432).

Baseline numbers (real-world, kitchen current row-major):
- 3.9 s/iter @ 20 steps cfg=2.5 (≈ 0.25 it/s)
- ≈ 30% effective tensor-core utilization

Targets:
- **1.7 s/iter** (match nunchaku tile-packed vendor) — minimum acceptable
- **1.3 s/iter** (≥80% peak) — research goal
- **<1.0 s/iter** — stretch, would beat anything published on RTX 5090

## sm_120 Hardware/PTX Survey (measured locally)

### Available
| Feature | Status | Use |
|---|---|---|
| `mma.sync.aligned.m16n8k64.s4.s4.s32` | ✓ compiles, 463 TFLOPs measured peak | Core int4 MMA |
| `cp.async.cg` (Ampere style) | ✓ | Already used in W4A4 v1 |
| `cp.async.bulk` + `mbarrier::complete_tx::bytes` | ✓ | TMA-style 256+ byte async loads |
| `mbarrier.init / arrive / expect_tx` (Hopper-style) | ✓ | Producer/consumer sync primitives |
| `tcgen05.alloc` (Blackwell tensor memory namespace) | ✓ compiles | Datacenter-class TC, may not be live |
| `prmt`, `lop3` byte-permutation | ✓ | Register-level int4 unpack |

### Missing / Crippled
| Feature | Status | Implication |
|---|---|---|
| `wgmma.mma_async` | ✓ instruction compiles | — |
| **`wgmma.wait_group`** | ✗ ptxas error on sm_120/sm_120a | **wgmma async pipeline unusable**; can't port Marlin H100 path directly |
| `cp.async.bulk.tensor` (full TMA) | ⚠ untested | Likely available via tcgen05 namespace; verify |

**Key conclusion**: Blackwell consumer (sm_120) is "Hopper-minus" — it has the
async primitives (mbarrier, cp.async.bulk) but NOT the warp-group async MMA
pipeline (wgmma.wait_group missing). Marlin's exact H100 design pattern won't
port. Need a custom design that uses per-warp `mma.sync` + cp.async.bulk +
mbarrier for pipelining.

## Why we lose to nunchaku at 30% peak

Profile of current kitchen row-major kernel (kitchen `feat/svdquant-w4a4-kitchen-native`):

| Loss source | % of peak lost | Explanation |
|---|---|---|
| ldmatrix bank conflicts (8–16 way) | -30% to -50% | row-major stride aligns to 32-bank cycle |
| Cache-line waste in HBM→shmem | -20% to -40% | 1 cache line carries 1 N-row's K-segment, not 4 |
| Scalar dequant→shmem→ldmatrix path | -15% to -25% | Dequantized W stored to shmem, then re-loaded to registers |
| No producer/consumer split | -10% to -15% | Same warps doing copy + dequant + MMA, no overlap |
| `__shfl_xor` for wscale broadcast | -5% to -10% | Extra register movement vs lane-resident scales |

**Compounded**: ~30% peak, matches measurement.

Nunchaku tile-packed kernel hits ~70% peak by design — its on-disk layout
already mirrors the MMA fragment lane pattern, killing all ldmatrix bank
conflicts and cache-line waste in one go.

## Design Space (ranked by ROI)

### Technique 1: `cp.async.bulk` for HBM → shmem
Replace `cp.async.cg` 16-byte transfers with `cp.async.bulk` 256-byte (or larger)
async TMA-style transfers. Single bulk transaction is one ptx instruction; the
GPU's async memory controller bursts the transfer with optimal DRAM scheduling.
Pairs with `mbarrier::complete_tx` for completion signal.

- **Saves**: -20% pipelining loss + -10% HBM scheduling overhead
- **Cost**: Need to align tile sizes to 256-byte multiples. shmem layout becomes
  more constrained.
- **Effort**: 2–3 days
- **Risk**: Low — ptx confirmed compiles on sm_120.

### Technique 2: Producer/consumer warp split with `mbarrier`
4 producer warps issue cp.async.bulk + dequant; 4 consumer warps issue MMA.
mbarrier signals when shmem buffer is filled. Allows perfect overlap of memory
+ compute. Same idea as Marlin's H100 pattern but using per-warp mma.sync
instead of wgmma (since wgmma.wait_group is missing).

- **Saves**: -10% to -15% (the "no overlap" loss)
- **Cost**: 8-warp CTA, more shmem (3-stage pipeline for cp.async.bulk).
- **Effort**: 4–5 days
- **Risk**: Medium — the actual mbarrier handoff timing requires careful tuning.

### Technique 3: Register-level int4 unpack (Marlin's key trick)
Don't store dequantized bf16 W in shmem. Instead, after `ldmatrix` of int4
weight, use `prmt` + `lop3` PTX to expand 8 packed int4 → 8 int8 directly in
registers, feed those into `mma.sync.m16n8k32.s8.s8.s32` (s8 MMA) or use
`mma.sync.m16n8k64.s4.s4.s32` directly with packed int4 fragments.

For our W4A4 with per-group dequant scale × ascale:
- Unpack int4 × int4 → s32 partial sum via mma.s4.s4.s32
- Multiply per-group scale * ascale → fp32 only at K-group boundary
- Avoids ever materializing dequantized bf16 W in shmem

- **Saves**: -15% to -25% (the dequant→shmem→ldmatrix loss)
- **Cost**: Per-group dequant becomes more complex; need careful fp32 accumulation.
- **Effort**: 5–7 days (this is the hardest technique)
- **Risk**: High — numerical precision must match nunchaku/eager (rel < 1e-3).

### Technique 4: Persistent kernel + work-stealing
Single kernel launch processes all CTAs sequentially. Avoids per-CTA launch
overhead, reuses warmed-up SMs.

- **Saves**: -2% to -5%
- **Cost**: Output write-back becomes more complex.
- **Effort**: 2 days
- **Risk**: Low.

### Technique 5: Split-K with fp32 reduction
Tile across K dim into multiple CTAs that produce partial sums; reduce in a
second kernel. For Qwen K=3072 split-K is 3–6 ways. Saves register pressure,
allows wider M tiles.

- **Saves**: marginal at our shapes (K=3072 not large enough for split-K to
  dominate).
- **Effort**: 2 days
- **Risk**: Low, but limited upside.

### Technique 6: tcgen05.mma (datacenter Blackwell tensor cores)
sm_120 has the `tcgen05.mma` instruction available; if functionally live (not
just compiles, but actually runs), this could give 2× over `mma.sync.m16n8k64`
for int4. **Highly speculative**: NVIDIA typically gates sm_100 (datacenter
Blackwell) features on consumer Blackwell.

- **Saves**: potentially +50% to +100% if functional
- **Cost**: Tensor memory namespace requires tcgen05.alloc + dealloc; debugging
  is harder due to limited public docs.
- **Effort**: 1 week of probing + a working baseline first
- **Risk**: Very high — instruction compiles but may silently no-op or give
  wrong results on consumer cards.

## Phased Implementation Plan

> Each phase ships an independent micro-benchmark vs the previous phase. Reject
> if speedup < 80% of predicted.

### Phase 0: Baseline + measurement infrastructure (1 day)
- Port the current row-major kernel into `research/` with isolated Python
  bindings.
- Set up `bench_w4a4.py` that runs the kernel on Qwen-Image production shapes
  (M=4096, K=3072 / 12288, N=3072 / 12288 / 18432) and reports TFLOPs achieved
  + ms/forward.
- Collect baseline: ~150 TFLOPs / ~30% peak.

### Phase 1: cp.async.bulk for weight load (2 days)
Replace per-thread cp.async.cg loads with single cp.async.bulk + mbarrier.
- Predicted: 1.3× kernel speedup → ~200 TFLOPs.
- Validation: rel < 1e-3 vs eager, throughput improves.

### Phase 2: 8-warp CTA + producer/consumer split (4 days)
4 producer warps own cp.async.bulk + dequant; 4 consumer warps own mma.sync.
mbarrier handoff between stages.
- Predicted: 1.5× over Phase 1 → ~300 TFLOPs (~65% peak).

### Phase 3: Register-level int4 unpack (5 days)
Skip shmem dequant. ldmatrix int4 directly into MMA fragments. Per-K-group
fp32 dequant only at output stage.
- Predicted: 1.3× over Phase 2 → ~390 TFLOPs (~85% peak).
- Acceptance: matches nunchaku 1.67 s/iter or beats.

### Phase 4 (optional, exploratory): tcgen05 path (1 week)
Probe whether tcgen05.mma actually executes correctly on sm_120 with synthetic
tests. If yes, port Phase 3 kernel to use tcgen05.
- Predicted: 1.5–2× over Phase 3 → 600–800 TFLOPs (~100%+ of mma.sync peak).
- Acceptance: 1.0 s/iter or better. If tcgen05 is dead silicon on sm_120, abort.

## Out-of-scope for this research

- AWQ W4A16 (covered by separate task #62 follow-up).
- bf16 / fp4 / fp8 paths (not the bottleneck for Qwen-Image-Edit).
- Triton implementations (Phase 5 task #35; would be a port of the final CUDA
  kernel).

## Risk: re-conversion of the 25 HF variants

If Phase 3 succeeds with kitchen row-major on-disk layout (just internal kernel
optimizations + GPU-side load-time repack), no re-conversion needed.

If we end up needing on-disk tile-packed for top performance, ~1 day to update
deepcompressor converter + ~10 hours batch re-conversion + HF re-upload.
Decision deferred until Phase 3 results.

## Reference materials (to read)

- Marlin paper: https://arxiv.org/abs/2408.11743
- Marlin source: https://github.com/IST-DASLab/marlin
- vLLM Marlin port: https://github.com/vllm-project/vllm (csrc/quantization/marlin)
- CUTLASS examples for Blackwell W4A16: https://github.com/NVIDIA/cutlass
- nunchaku zgemm source: /workspace/nunchaku/src/kernels/zgemm/
- PTX ISA 8.5 Blackwell-specific: cp.async.bulk, mbarrier, tcgen05 sections

## Status today (2026-04-29)

- Branch created: `research/w4a4-blackwell-peak` (off main, isolated)
- sm_120 PTX feature survey complete (this doc)
- Phase 0 not yet started.
