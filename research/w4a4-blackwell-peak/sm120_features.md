# sm_120 (RTX 5090, Blackwell consumer) PTX feature scorecard

Test environment: CUDA 12.8.93, RTX 5090, ptxas sm_120 / sm_120a targets.
All probes are tiny `.cu` files, compile-only (some run-only too).

## ✅ Available

| Feature | Probe | Notes |
|---|---|---|
| `mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32` | compiles + runs | **Measured 463 TFLOPs peak**. Datasheet implies up to 836 TFLOPs (= 4× bf16 209 TFLOPs); benchmark capped at ~55% of theoretical. Single-warp instruction. |
| `cp.async.cg.shared.global` | runs in production W4A4 v1 | Standard Ampere async copy. 16-byte transfers per thread. |
| `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes` | compiles | Hopper-style TMA-equivalent. 256+ byte bulk async transfer + completion via mbarrier. |
| `mbarrier.init.shared::cta.b64` | compiles | Initialize barrier in shmem. |
| `mbarrier.arrive.expect_tx.shared::cta.b64` | compiles | Hopper-style expect-N-bytes-then-arrive. Pairs with cp.async.bulk. |
| `wgmma.mma_async.sync.aligned.m64n8k32.s32.s4.s4.s32` | compiles | The async MMA instruction itself. |
| `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32` | compiles | Blackwell tensor memory namespace allocation. **Functional execution NOT verified**. |
| `prmt`, `lop3` byte-permute | always available since Maxwell | Used for register-level int4 unpack tricks. |

## ❌ NOT available / crippled

| Feature | Probe result | Implication |
|---|---|---|
| **`wgmma.wait_group.sync.aligned`** | `Instruction 'wgmma.wait_group' cannot be compiled for architecture 'sm_120a'` | wgmma async pipelining fundamentally broken on sm_120. Marlin's H100 design pattern (which depends on this) cannot be ported as-is. |

## Implications for kernel design

1. **No wgmma pipeline** → can't use Marlin H100 design directly. Must use
   per-warp `mma.sync.m16n8k64` (basic int4 MMA) + manual mbarrier-based
   pipelining.

2. **cp.async.bulk + mbarrier IS available** → can still build a Hopper-style
   producer/consumer pipeline with mbarrier handoff between warps that issue
   bulk loads and warps that issue mma.sync.

3. **tcgen05 is the X-factor** → instructions compile, but unknown if they
   actually execute correctly on consumer Blackwell. NVIDIA gating is opaque.
   Worth probing in Phase 4 once a Phase 3 baseline exists; would unlock
   datacenter-class throughput if functional.

4. **Peak target**: 463 TFLOPs measured, perhaps higher with more careful
   benchmarking. ~370 TFLOPs effective (80%) is realistic ceiling for a
   well-tuned kernel using mma.sync + cp.async.bulk + mbarrier.

## Probe sources

All probes archived at `/tmp/sm120_*.cu` during the investigation; copy here
under `probes/` if we keep them for future reference.
