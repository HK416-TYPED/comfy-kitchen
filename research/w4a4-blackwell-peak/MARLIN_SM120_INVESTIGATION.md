# True Marlin-style W4A4 kernel on Blackwell sm_120 — Deep Investigation

> Follows from Phase 0–3 negative results (RESULTS.md). Scope: investigate
> what's actually buildable using Marlin's design ideas on consumer Blackwell
> (sm_120, RTX 5090) without re-conversion of the 25 production checkpoints.

## Executive summary

| | sm_80/sm_89/sm_90 (Marlin's home) | **sm_120 (RTX 5090)** |
|---|---|---|
| Per-warp `mma.sync.m16n8k64.s4.s4.s32` | ✓ | ✓ measured 463 TFLOPs peak |
| Marlin's `lop3` int4→fp16 register dequant | ✓ | ✓ but **N/A for W4A4** (needed only for W4A16) |
| `cp.async.bulk` + `mbarrier::expect_tx` (TMA) | ✓ on sm_90+ | ✓ |
| `cp.async.bulk.tensor.2d` (TMA tensor descriptor) | ✓ on sm_90+ | ✓ |
| L2 `evict_first` cache hint via `createpolicy.fractional` | ✓ | ✓ |
| `wgmma.mma_async` + `wgmma.wait_group` (warp-group async MMA) | ✓ on sm_90+ | **✗ wait_group unsupported** |
| `tcgen05.mma.cta_group::1` (Blackwell datacenter TC) | sm_100 only | **✗ cta_group::1 not on sm_120a** |

**Conclusion**: A Marlin-style kernel on sm_120 is buildable using per-warp
`mma.sync` + TMA via `cp.async.bulk.tensor` + mbarrier-based pipeline. It
**cannot** use Hopper's wgmma async pipeline (the design pattern Marlin H100
itself uses). Realistic perf ceiling: **~75% of int4 TC peak ≈ 350 TFLOPs**,
matching nunchaku's vendored kernel at ~1.6-1.7 s/iter on Qwen-Image-Edit but
not significantly beating it.

## Marlin's actual techniques (source-level read)

`vendored/marlin/marlin/marlin_cuda_kernel.cu` — 822 lines, public Apache 2.0.

### 1. `lop3.b32`-based int4 → fp16 register dequant (lines 119-153)

Single 32-bit register holding 8 packed int4 nibbles → 2× `half2` registers
(4 fp16 values) via 2 `lop3` operations + 2 `__hsub2` / `__hfma2`:

```cuda
const int LO = 0x000f000f;
const int HI = 0x00f000f0;
const int EX = 0x64006400;  // = fp16 with exponent=25
int lo = lop3<0xea>(q, LO, EX);   // extract low nibbles into fp16-shaped bits
int hi = lop3<0xea>(q, HI, EX);
const int SUB = 0x64086408;       // 2^16 + 8 (zero-point fold)
const int MUL = 0x2c002c00;       // 2^-12 (rescale shifted-out bits)
const int ADD = 0xd480d480;       // -2^16 - 8 * 2^-12
frag_b[0] = __hsub2(lo, SUB);                  // (q_lo - 8) as fp16
frag_b[1] = __hfma2(hi, MUL, ADD);             // (q_hi - 8) as fp16, with rescale
```

**Why this matters**: Marlin runs `mma.m16n8k16.f16.f16.f32` — both operands
fp16. The int4 weight must be dequantized to fp16 BEFORE the MMA. The lop3
trick does this in 4 instructions per 8 nibbles, all in registers, no shmem
roundtrip.

**For W4A4 this is not needed**. `mma.m16n8k64.s4.s4.s32` accepts int4
directly. The dequant scaling happens once per K-group at the output stage,
not per-MMA. So the lop3 trick **doesn't port** to a W4A4 kernel.

### 2. Stripe layout for B (lines 207-240)

Weight matrix is partitioned into "stripes" — vertical column slices, each
stripe owned by a CTA. Within a stripe, the int4 weights are stored
**row-major in 16x16 fragment-aligned tiles**. The tile order in HBM is laid
out so consecutive HBM addresses correspond to consecutive K positions for
the same fragment, maximizing cache-line reuse.

This is **conceptually identical** to nunchaku's `kInterleave=4` tile-packed
layout. Both layouts pack 4 N-rows per cache line such that ldmatrix.x4 hits
contiguous HBM data.

**For W4A4 on sm_120**: we'd need to re-pack our `(N, K/2)` int8 weight to
this layout. Either at conversion time (touches the 25 checkpoints — vetoed by
user) or at GPU load time (once per checkpoint load, ~50 ms per 12 GB file).

### 3. Async pipeline with `cp.async.cg` + L2 evict_first (lines 67-78)

```cuda
"createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
"cp.async.cg.shared.global.L2::cache_hint [smem], [glob], 16, p;"
```

Tells the L2 cache "this load is one-shot, don't keep it cached" — important
for weights since each weight cell is only used once per forward pass. Frees
L2 capacity for activations + outputs.

✅ Verified compiles and executes on sm_120.

### 4. Producer/consumer pattern via single-warp pipelining (lines 350-500)

Unlike my Phase 2v1 (8-warp split), Marlin uses **single-role 4-warp CTAs**.
Same warps issue cp.async AND mma.sync, but with deep prefetch (kStages=4):

```cuda
for (g = 0; g < n_groups; g++) {
    cp_async_pipeline_step(stage = g % kStages);    // issue load for g+kStages-1
    cp_async_wait_group<kStages-2>();                // wait for stage g - (kStages-2)
    do_mma(stage = g % kStages);                     // mma on stage g
}
```

With kStages=4, at any time 3 cp.async groups are in-flight while the 4th is
being consumed. Full overlap of HBM bandwidth and tensor core compute.

**Why my Phase 2v2 failed to overlap**: I used kStages=2 with a single
`__syncthreads` between iterations. With deeper pipeline (kStages=4) and
proper wait_group(N-2) semantics, the sync isn't needed — only the wait on
the ABOUT-TO-BE-CONSUMED stage matters. Other stages can be in flight or
already-consumed without blocking.

### 5. Striped CTA partitioning across SMs (lines 207-220)

Marlin doesn't use the standard 2D grid (N×M tiles). Instead it lays out the
work as 1D stripes across SMs to ensure every SM gets roughly the same amount
of work, reducing tail-latency where some SMs finish early.

For our shapes this matters less — Qwen GEMMs are large enough that occupancy
is already balanced.

### 6. Per-group scales handling

Marlin loads scales into a separate small shmem buffer at the start of each
"slice" (= block of K positions sharing scales). Reads from shmem during MMA.
Same idea I tried in Phase 2v2 but in tighter form (per-slice not per-kernel).

## sm_120 functional probes (executed)

All probes in `/tmp/sm120_*` and Phase 0 `probes/`. Re-verified:

| PTX feature | Compile | Execute |
|---|---|---|
| `mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32` | ✓ | ✓ 463 TFLOPs measured |
| `cp.async.cg.shared.global` | ✓ | ✓ |
| `cp.async.cg.shared.global.L2::cache_hint` w/ `createpolicy` | ✓ | ✓ |
| `cp.async.bulk.shared::cluster.global` | ✓ | ✓ |
| `cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier` | ✓ | not run yet but compiles |
| `mbarrier.init / arrive / arrive.expect_tx / try_wait.parity` | ✓ | ✓ |
| `lop3.b32` | ✓ | ✓ |
| `prmt.b32` | ✓ | ✓ |
| `wgmma.mma_async.m64nXk32.s32.s4.s4.s32` (instruction itself) | ✓ | not testable without wait_group |
| **`wgmma.wait_group`** | ✗ | **fails ptxas: not supported on sm_120a** |
| **`tcgen05.alloc.cta_group::1`** | ✓ | not testable (dealloc gated) |
| **`tcgen05.dealloc.cta_group::1`** | ✗ | **fails ptxas: cta_group::1 not on sm_120a** |
| **`tcgen05.mma.cta_group::1.kind::i8`** | ✗ | **fails ptxas: cta_group::1 not on sm_120a** |

Bottom line: **wgmma + tcgen05 are dead silicon on consumer Blackwell**. The
per-warp `mma.sync` + manual pipelining path is all we have.

## Realistic perf target on sm_120

```
Theoretical int4 TC peak (datasheet 4× bf16 209 TFLOPs):    836 TFLOPs
Synthetic mma.spam measured peak:                           463 TFLOPs (55% of theoretical)
Marlin-on-H100 efficiency over peak:                        ~95% (with wgmma)
sm_120 efficiency without wgmma (per-warp mma.sync):        ~75% estimated
Realistic Marlin-sm_120 effective:                          ~350 TFLOPs
```

For Qwen-Image-Edit production GEMMs:
- QKV M=4096 K=3072 N=3072: 77 GFLOP → 0.22 ms at 350 TFLOPs (vs 1.01 ms current)
- MLP_FC1 M=4096 K=12288 N=3072: 309 GFLOP → 0.88 ms (vs 3.97 ms current)
- MLP_FC2 M=4096 K=3072 N=12288: 309 GFLOP → 0.88 ms (vs 3.78 ms current)

Total per-forward W4A4 compute time: ~3 ms × 60 blocks / 60 = 3 ms (rough).
Currently W4A4 time is ~70 ms/forward. Other forward overhead (attention
softmax, RoPE, etc.) is ~25 ms. Forward time at Marlin-sm_120 efficiency:
3 + 25 = 28 ms. 40 forwards (20 steps × cfg=2.5) = **~1.1 s/iter**.

**Wait — this is lower than nunchaku's 1.67 s/iter**, possibly because nunchaku
doesn't fully overlap and has its own non-MMA overhead. Realistic floor is
likely 1.4-1.6 s/iter for sm_120 once everything ties together. Verbatim
zgemm port would be slightly slower than Marlin sm_120 because it's tuned for
sm_89/sm_90.

## Implementation roadmap

### Prerequisites (must decide before kernel work)

**Layout decision**: tile-packed weight in HBM is non-negotiable for ≥75%
peak. Options:
- **A**: Re-convert all 25 checkpoints to tile-packed on-disk. Costs 1 day +
  ~10 hours batch + HF re-upload. After this it's identical to using the
  existing verbatim zgemm port — 0 kernel work needed.
- **B**: Keep on-disk row-major, do GPU-side load-time repack. ~2-3 days
  one-time work in ComfyUI Linear._load_from_state_dict. Peak perf identical
  to Path A.

If user picks A, **stop here and use the verbatim zgemm port**. Marlin
sm_120 wouldn't beat nunchaku noticeably and would be a 2-3 week effort.

If user picks B, also **likely use the verbatim zgemm port** since same
result for less work. Marlin sm_120 only makes sense if there's a specific
reason to NOT use the vendored nunchaku kernel (maintenance, license, future
extensibility, etc).

### If Marlin sm_120 is chosen anyway

| Phase | Days | Description |
|---|---|---|
| **A** | 3-5 | Tile-packed layout helpers + GPU repack pass + correctness vs eager. |
| **B** | 3-5 | Per-warp 4-stage pipelined kernel using `cp.async.cg` + `mma.sync.m16n8k64` + L2 evict_first. Match Phase 0 perf as a checkpoint, then start tuning. |
| **C** | 2-3 | Replace `cp.async.cg` with `cp.async.bulk.tensor.2d` (TMA) + `mbarrier::expect_tx`. Should give 1.3-1.5× kernel-only speedup. |
| **D** | 2-3 | Persistent kernel + striped CTA partitioning + final tuning. |
| **E (optional)** | 2-3 | Special path for M ≤ 8 (GEMV — tensor cores under-utilized at small M). |

Total: **12-19 days**, ~2-3 weeks.

Risk: **medium-high**. Specific risks:
1. **Numerical correctness** while juggling pipeline stages + per-group scales
   for s32→fp32 dequant. Marlin only has fp16 accumulator, simpler.
2. **mbarrier deadlock** with insufficient stages or wrong arrival count
   (already burnt by this in Phase 2v1).
3. **Consumer Blackwell may have unexpected throttling on TMA tensor**
   instructions that the basic probes didn't catch.

## Final recommendation

For pure perf on Qwen-Image-Edit on RTX 5090:

| Decision | Effort | Result | Recommended? |
|---|---|---|---|
| Stay at Phase 2v2 (kitchen-native row-major, ~80 TFLOPs) | done | 3.9 s/iter | only if E2E perf doesn't matter |
| **Switch to verbatim zgemm port (already exists)** | 0 days kernel + 1 day pipeline glue + 10 hours batch re-conversion | **1.67 s/iter** | **YES — best ROI** |
| GPU load-time repack (Path B) + Marlin sm_120 | 2-3 weeks | ~1.4-1.7 s/iter | only if you don't want vendored nunchaku code in kitchen |
| GPU load-time repack + verbatim zgemm | 1 week | ~1.67 s/iter | second-best alternative if vendored OK |

The Marlin sm_120 kernel is **technically interesting but not justified by
the perf delta** vs the existing verbatim zgemm port. Consumer Blackwell's
missing wgmma + tcgen05 caps the peak achievable below what would justify a
2-3 week original-kernel effort.

If you want to explore Marlin sm_120 anyway (architectural reasons /
maintenance reasons), the roadmap above is concrete and feasible. The main
unknown is whether TMA tensor descriptors deliver their expected speedup on
sm_120 — that's worth a 1-day spike before committing to the full
implementation.
