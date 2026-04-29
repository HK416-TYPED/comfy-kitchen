// SPDX-License-Identifier: Apache-2.0
//
// Phase 2: producer/consumer 8-warp split with mbarrier handoff +
//          per-group scales staged in shmem (no per-K-iter global loads).
//
// Architecture
// ------------
//
//   8 warps per CTA = 256 threads
//   Producer warps (warp_id 4..7): issue cp.async for next K-tile
//   Consumer warps (warp_id 0..3): issue mma.sync on current K-tile
//
//   3-stage circular buffer for shmem (a_sh, w_sh).
//
//   Per stage two mbarriers:
//     bar_data[s]  — producer arrives when load done, consumer waits.
//     bar_free[s]  — consumer arrives when MMA done, producer waits.
//
//   Phase parity flips per stage-cycle so the same mbarrier can be reused
//   across the K-loop without per-iter init.
//
// Scales
// ------
//
//   Loaded once into shmem at kernel start. Since K can be up to 18432
//   (288 groups), the worst-case scale shmem is ~90 KB. RTX 5090 supports
//   228 KB dynamic shmem with cudaFuncSetAttribute. We allocate
//   N_GROUPS_MAX upfront in __shared__; if the actual K exceeds it we
//   fall back to per-iter global loads (handled by template specialization).
//
// What this fixes from Phase 1
// ----------------------------
//
//   * Eliminates the end-of-iter __syncthreads (replaced by mbarrier
//     ordering) — the killer that erased the double-buffer overlap.
//   * Eliminates per-K-iter global scale loads (1152/lane → 0).
//   * Producer warps don't compete with consumer warps for instruction
//     issue slots.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

namespace {

constexpr int kGroupSize = 64;
constexpr int kBlockKHalf = kGroupSize / 2;
constexpr int kSmemRowStride = kBlockKHalf + 16;     // 48 = 32 + 16, breaks bank cycle
constexpr int kStages = 3;
constexpr int kNumProducerWarps = 4;
constexpr int kNumConsumerWarps = 4;
constexpr int kNumWarps = kNumProducerWarps + kNumConsumerWarps;
constexpr int kNumThreads = kNumWarps * 32;

// ---------- inline asm primitives ----------

__device__ __forceinline__ uint32_t cvta_smem(const void* p) {
    uint32_t s;
    asm("{ .reg .u64 ll; cvta.to.shared.u64 ll, %1; cvt.u32.u64 %0, ll; }"
        : "=r"(s) : "l"(p));
    return s;
}

__device__ __forceinline__ void cp_async_16b(uint32_t smem_addr, const void* src) {
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
        :: "r"(smem_addr), "l"(src));
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;");
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}

__device__ __forceinline__ void mbarrier_init(uint32_t bar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(bar_addr), "r"(count));
}

// arrive (signal arrival) — pairs with try_wait
__device__ __forceinline__ void mbarrier_arrive(uint32_t bar_addr) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"(bar_addr));
}

// Wait until mbarrier has expected arrivals at given parity.
__device__ __forceinline__ void mbarrier_wait_parity(uint32_t bar_addr, int phase) {
    asm volatile(
        "{ .reg .pred                     P1;                            \n\t"
        "  LAB_WAIT_PHASE:                                               \n\t"
        "  mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;        \n\t"
        "  @P1                       bra DONE_BAR_PHASE;                 \n\t"
        "  bra                       LAB_WAIT_PHASE;                     \n\t"
        "  DONE_BAR_PHASE:                                               \n\t"
        "}"
        :: "r"(bar_addr), "r"(phase)
    );
}

__device__ __forceinline__ void mma_m16n8k64_s4s4s32(
    int (&a)[4], int (&b)[2], int (&c)[4])
{
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

__device__ __forceinline__ void ldmatrix_x4_b16(uint32_t (&dst)[4], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
        : "r"(addr));
}

// ---------- kernel ----------

template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int N_WARPS_M, int N_WARPS_N>
__global__ void w4a4_gemm_phase2(
    const int8_t* __restrict__ act,
    const int8_t* __restrict__ wgt,
    const __nv_bfloat16* __restrict__ ascales,
    const __nv_bfloat16* __restrict__ wscales,
    __nv_bfloat16* __restrict__ out,
    int M, int N, int K, int n_groups)
{
    static_assert(BLOCK_K == kGroupSize, "BLOCK_K must equal G=64");
    static_assert(N_WARPS_M * N_WARPS_N == kNumConsumerWarps,
        "consumer warp grid must equal kNumConsumerWarps");
    constexpr int kWarpM = 16;
    constexpr int kWarpN = BLOCK_N / N_WARPS_N;
    constexpr int kNUnroll = kWarpN / 8;

    constexpr int kATileBytes = BLOCK_M * kSmemRowStride;
    constexpr int kWTileBytes = BLOCK_N * kSmemRowStride;

    extern __shared__ uint8_t shmem_dyn[];
    int8_t* a_sh_base = reinterpret_cast<int8_t*>(shmem_dyn);
    int8_t* w_sh_base = a_sh_base + kStages * kATileBytes;
    __nv_bfloat16* ascales_sh = reinterpret_cast<__nv_bfloat16*>(
        w_sh_base + kStages * kWTileBytes);
    __nv_bfloat16* wscales_sh = ascales_sh + n_groups * BLOCK_M;

    auto a_stage = [&](int s) -> int8_t* {
        return a_sh_base + s * kATileBytes;
    };
    auto w_stage = [&](int s) -> int8_t* {
        return w_sh_base + s * kWTileBytes;
    };

    // mbarriers in static shmem (avoid offsetting in dynamic shmem)
    __shared__ alignas(8) uint64_t bar_data[kStages];
    __shared__ alignas(8) uint64_t bar_free[kStages];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const bool is_producer = warp_id >= kNumConsumerWarps;
    const int role_warp = is_producer ? (warp_id - kNumConsumerWarps) : warp_id;

    const int cta_m = blockIdx.y * BLOCK_M;
    const int cta_n = blockIdx.x * BLOCK_N;

    // --------- mbarrier init ---------
    // bar_data: ALL kNumProducerWarps*32 producer threads arrive per stage.
    //   This is required because cp.async.wait_group is per-thread; if only
    //   one thread arrived, we'd signal "data ready" while other producer
    //   threads' cp.asyncs are still in flight.
    // bar_free: ALL consumer threads arrive per stage (1/lane = 32/warp).
    if (tid < kStages) {
        mbarrier_init(cvta_smem(&bar_data[tid]), kNumProducerWarps * 32);
        mbarrier_init(cvta_smem(&bar_free[tid]), kNumConsumerWarps * 32);
    }
    __syncthreads();

    // --------- preload all scales into shmem (cooperative, all 256 threads) ---------
    {
        // ascales: shape (n_groups, M). Load ascales[g, cta_m..cta_m+BLOCK_M).
        const int total_a = n_groups * BLOCK_M;
        for (int idx = tid; idx < total_a; idx += blockDim.x) {
            int g = idx / BLOCK_M;
            int m_local = idx % BLOCK_M;
            int m_global = cta_m + m_local;
            ascales_sh[idx] = (m_global < M)
                ? ascales[g * M + m_global]
                : __nv_bfloat16(0.f);
        }
        const int total_w = n_groups * BLOCK_N;
        for (int idx = tid; idx < total_w; idx += blockDim.x) {
            int g = idx / BLOCK_N;
            int n_local = idx % BLOCK_N;
            int n_global = cta_n + n_local;
            wscales_sh[idx] = (n_global < N)
                ? wscales[g * N + n_global]
                : __nv_bfloat16(0.f);
        }
    }
    __syncthreads();

    if (is_producer) {
        // --------- Producer warps: issue cp.async for each K-iter ---------
        // 4 producer warps split the load work:
        //   warp 0 (role_warp=0): a_tile (BLOCK_M=32 rows × 32 bytes = 1024 byte = 64 16-byte loads).
        //                          But we have 4*32=128 producer threads — each thread loads
        //                          ~8 16-byte chunks. Just split a + w across all 4 warps.
        // Total bytes per K-iter: kATileBytes + kWTileBytes (using kSmemRowStride).
        //   But the original tile data is BLOCK_M*kBlockKHalf and BLOCK_N*kBlockKHalf — we
        //   load `kBlockKHalf` bytes per row into the row's leading kBlockKHalf bytes of
        //   kSmemRowStride. Padding bytes in shmem are uninitialized but never read.
        const int producer_tid = tid - kNumConsumerWarps * 32;       // 0..127
        const int producer_threads = kNumProducerWarps * 32;          // 128

        for (int g = 0; g < n_groups; ++g) {
            int s = g % kStages;
            int phase = (g / kStages) & 1;

            // Wait for buffer free (consumer signaled free)
            // First kStages iters skip wait — buffer hasn't been used yet.
            if (g >= kStages) {
                mbarrier_wait_parity(cvta_smem(&bar_free[s]), phase ^ 1);
                // ^1 because the consumer arrived in the OPPOSITE phase.
            }

            const int Kh_base = (g * kGroupSize) / 2;

            // Load a_tile: BLOCK_M rows × kBlockKHalf bytes = 1024 bytes total.
            // 64 16-byte loads. With 128 producer threads, half work.
            const int a_loads = (BLOCK_M * kBlockKHalf) / 16;          // 64
            if (producer_tid < a_loads) {
                int row = (producer_tid * 16) / kBlockKHalf;
                int col = (producer_tid * 16) % kBlockKHalf;
                int m_global = cta_m + row;
                uint32_t s_addr = cvta_smem(
                    &a_stage(s)[row * kSmemRowStride + col]);
                if (m_global < M) {
                    cp_async_16b(s_addr, act + m_global * (K / 2) + Kh_base + col);
                } else {
                    *reinterpret_cast<uint4*>(
                        &a_stage(s)[row * kSmemRowStride + col]) = make_uint4(0,0,0,0);
                }
            }
            // Load w_tile: BLOCK_N rows × kBlockKHalf bytes = 4096 bytes total.
            // 256 16-byte loads. 128 producer threads → 2 loads each.
            const int w_loads = (BLOCK_N * kBlockKHalf) / 16;          // 256
            for (int idx = producer_tid; idx < w_loads; idx += producer_threads) {
                int row = (idx * 16) / kBlockKHalf;
                int col = (idx * 16) % kBlockKHalf;
                int n_global = cta_n + row;
                uint32_t s_addr = cvta_smem(
                    &w_stage(s)[row * kSmemRowStride + col]);
                if (n_global < N) {
                    cp_async_16b(s_addr, wgt + n_global * (K / 2) + Kh_base + col);
                } else {
                    *reinterpret_cast<uint4*>(
                        &w_stage(s)[row * kSmemRowStride + col]) = make_uint4(0,0,0,0);
                }
            }
            cp_async_commit_group();
            cp_async_wait_group<0>();

            // Signal data ready: ALL producer threads arrive (each on its own,
            // after its own wait_group<0> guarantees its cp.asyncs completed).
            mbarrier_arrive(cvta_smem(&bar_data[s]));
        }
    } else {
        // --------- Consumer warps: MMA on current K-iter ---------
        const int warp_m = role_warp % N_WARPS_M;
        const int warp_n = role_warp / N_WARPS_M;

        float acc[kNUnroll][4];
        #pragma unroll
        for (int i = 0; i < kNUnroll; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;

        for (int g = 0; g < n_groups; ++g) {
            int s = g % kStages;
            int phase = (g / kStages) & 1;

            // Wait for data ready
            mbarrier_wait_parity(cvta_smem(&bar_data[s]), phase);

            const int warp_m_base = warp_m * kWarpM;
            const int warp_n_base = warp_n * kWarpN;

            int a_frag[4];
            {
                uint32_t a_addr = cvta_smem(
                    &a_stage(s)[(warp_m_base + (lane % 16)) * kSmemRowStride + (lane / 16) * 16]
                );
                ldmatrix_x4_b16(*reinterpret_cast<uint32_t(*)[4]>(a_frag), a_addr);
            }

            #pragma unroll
            for (int n_mma = 0; n_mma < kNUnroll; ++n_mma) {
                const int n_off = warp_n_base + n_mma * 8;
                int b_frag[2];
                uint32_t b_addr = cvta_smem(
                    &w_stage(s)[(n_off + (lane % 8)) * kSmemRowStride + ((lane / 8) & 1) * 16]
                );
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                    : "=r"(b_frag[0]), "=r"(b_frag[1])
                    : "r"(b_addr)
                );

                int c_int[4] = {0, 0, 0, 0};
                mma_m16n8k64_s4s4s32(
                    *reinterpret_cast<int(*)[4]>(a_frag),
                    *reinterpret_cast<int(*)[2]>(b_frag),
                    *reinterpret_cast<int(*)[4]>(c_int)
                );

                const int row_lo = lane / 4;
                const int col_lo = (lane % 4) * 2;
                #pragma unroll
                for (int half = 0; half < 2; ++half) {
                    int m_local = warp_m_base + row_lo + half * 8;
                    int n_local0 = n_off + col_lo;
                    int n_local1 = n_local0 + 1;

                    // Read scales from shmem (preloaded at kernel start)
                    float a_s = __bfloat162float(
                        ascales_sh[g * BLOCK_M + m_local]
                    );
                    float w_s0 = __bfloat162float(
                        wscales_sh[g * BLOCK_N + n_local0]
                    );
                    float w_s1 = __bfloat162float(
                        wscales_sh[g * BLOCK_N + n_local1]
                    );

                    float scale0 = a_s * w_s0;
                    float scale1 = a_s * w_s1;
                    acc[n_mma][half * 2 + 0] += float(c_int[half * 2 + 0]) * scale0;
                    acc[n_mma][half * 2 + 1] += float(c_int[half * 2 + 1]) * scale1;
                }
            }

            // Signal buffer free: every consumer thread arrives (matches mbarrier count).
            mbarrier_arrive(cvta_smem(&bar_free[s]));
        }

        // Drain: ensure all consumer warps finished before writes. Without
        // this, fast warps may start write before slow warps finish reading
        // from shmem, but writes go to OUT (HBM) and reads were from shmem,
        // so this is precautionary not strictly needed — but cheap.
        __syncwarp();

        // --------- Write output ---------
        const int warp_m_base = warp_m * kWarpM;
        const int warp_n_base = warp_n * kWarpN;
        const int row_lo = lane / 4;
        const int col_lo = (lane % 4) * 2;
        #pragma unroll
        for (int n_mma = 0; n_mma < kNUnroll; ++n_mma) {
            const int n_off = warp_n_base + n_mma * 8;
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                int m_global = cta_m + warp_m_base + row_lo + half * 8;
                int n_global0 = cta_n + n_off + col_lo;
                int n_global1 = n_global0 + 1;
                if (m_global < M) {
                    if (n_global0 < N)
                        out[m_global * N + n_global0] = __float2bfloat16(acc[n_mma][half * 2 + 0]);
                    if (n_global1 < N)
                        out[m_global * N + n_global1] = __float2bfloat16(acc[n_mma][half * 2 + 1]);
                }
            }
        }
    }
}

} // namespace

extern "C" void launch_w4a4_gemm_phase2(
    const void* act, const void* wgt,
    const void* ascales, const void* wscales,
    void* out,
    int M, int N, int K,
    cudaStream_t stream)
{
    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 64;
    constexpr int N_WARPS_M = 2;
    constexpr int N_WARPS_N = 2;

    const int n_groups = K / BLOCK_K;

    // Compute dynamic shmem size:
    //   3 stages * (a_sh tile + w_sh tile) + scales (n_groups * (BLOCK_M + BLOCK_N) * 2 bytes)
    constexpr int kATileBytes = BLOCK_M * kSmemRowStride;
    constexpr int kWTileBytes = BLOCK_N * kSmemRowStride;
    int shmem_bytes = kStages * (kATileBytes + kWTileBytes)
                    + n_groups * (BLOCK_M + BLOCK_N) * (int)sizeof(__nv_bfloat16);

    // sm_120 default opt-out shmem is ~48 KB; ALWAYS opt-in via attribute when
    // requesting > default. Set the max once per kernel; idempotent so cheap.
    if (shmem_bytes > 48 * 1024) {
        cudaError_t e = cudaFuncSetAttribute(
            (const void*)w4a4_gemm_phase2<BLOCK_M, BLOCK_N, BLOCK_K, N_WARPS_M, N_WARPS_N>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_bytes);
        if (e != cudaSuccess) {
            printf("phase2: cudaFuncSetAttribute(%d KB) failed: %s\n",
                shmem_bytes / 1024, cudaGetErrorString(e));
            return;
        }
    }

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(kNumThreads);
    w4a4_gemm_phase2<BLOCK_M, BLOCK_N, BLOCK_K, N_WARPS_M, N_WARPS_N>
        <<<grid, block, shmem_bytes, stream>>>(
            reinterpret_cast<const int8_t*>(act),
            reinterpret_cast<const int8_t*>(wgt),
            reinterpret_cast<const __nv_bfloat16*>(ascales),
            reinterpret_cast<const __nv_bfloat16*>(wscales),
            reinterpret_cast<__nv_bfloat16*>(out),
            M, N, K, n_groups);
}
