// SPDX-License-Identifier: Apache-2.0
//
// Phase 0 baseline: kitchen-native row-major SVDQuant W4A4 GEMM.
//
// Goal of Phase 0: reproduce kitchen's current production kernel as a
// self-contained research starting point. No optimizations beyond what's in
// kitchen's `feat/svdquant-w4a4-kitchen-native` branch.
//
// Layouts (kitchen-native row-major):
//   act     (M, K/2) int8    two signed int4 per byte (range [-7, 7])
//   wgt     (N, K/2) int8    same packing, weight in row-major
//   ascales (K/G, M) bf16    per-row per-group activation scale
//   wscales (K/G, N) bf16    per-col per-group weight scale
//   out     (M, N)   bf16    fp accumulator inside, downcast on store
//
// G = 64 (kitchen / nunchaku group size).
//
// Tile (matching kitchen v1):
//   BLOCK_M = 32, BLOCK_N = 128, BLOCK_K = 64 (= G, so 1 group per K-tile)
//   4 warps in 2x2 grid, each warp owns (16M × 64N) = kMUnroll=1 × kNUnroll=8
//
// Pipeline: 2-stage cp.async.cg double buffer.
//
// Used as the speed baseline (measure TFLOPs achieved) for Phases 1-3.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

namespace {

constexpr int kGroupSize = 64;

// ---------- mma helpers ----------

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

// 16-byte cp.async.cg — Ampere/Hopper async global -> shmem load
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

__device__ __forceinline__ uint32_t cvta_smem(const void* p) {
    uint32_t s;
    asm("{ .reg .u64 ll; cvta.to.shared.u64 ll, %1; cvt.u32.u64 %0, ll; }"
        : "=r"(s) : "l"(p));
    return s;
}

// ldmatrix.x4: 4× (8x8 b8) tiles into 4 regs/lane
__device__ __forceinline__ void ldmatrix_x4_b8(uint32_t (&dst)[4], uint32_t addr) {
    // No b8 ldmatrix exists; we use b16 because 4-bit packed-pair (= 8-bit) maps
    // to 16-bit byte-pair under cuda's view. ptx ldmatrix is dtype-agnostic at
    // the bit level (b16 reads 2 bytes per element).
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
        : "r"(addr));
}

// ---------- kernel ----------

template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int N_WARPS_M, int N_WARPS_N>
__global__ void w4a4_gemm_phase0(
    const int8_t* __restrict__ act,      // (M, K/2)
    const int8_t* __restrict__ wgt,      // (N, K/2)
    const __nv_bfloat16* __restrict__ ascales,   // (K/G, M)
    const __nv_bfloat16* __restrict__ wscales,   // (K/G, N)
    __nv_bfloat16* __restrict__ out,             // (M, N)
    int M, int N, int K)
{
    static_assert(BLOCK_K == 64, "BLOCK_K must equal G=64 for one group/tile");
    static_assert(N_WARPS_M * 16 <= BLOCK_M, "warp tile fits in BLOCK_M");
    static_assert(N_WARPS_N * 8  <= BLOCK_N, "warp tile fits in BLOCK_N");

    constexpr int kWarpsTotal = N_WARPS_M * N_WARPS_N;  // = 4
    constexpr int kThreads = kWarpsTotal * 32;          // = 128
    constexpr int kBlockKHalf = BLOCK_K / 2;             // = 32 bytes per row of qw_tile
    constexpr int kStages = 1;                           // single buffer (sync)

    // Per-warp tile: each warp owns (16M × WARP_N) where WARP_N = BLOCK_N / N_WARPS_N
    constexpr int kWarpM = 16;                           // 1 mma m=16
    constexpr int kWarpN = BLOCK_N / N_WARPS_N;          // = 32 (4 mma cols of 8)
    constexpr int kNUnroll = kWarpN / 8;                 // = 4

    __shared__ alignas(16) int8_t a_sh[kStages][BLOCK_M][kBlockKHalf];
    __shared__ alignas(16) int8_t w_sh[kStages][BLOCK_N][kBlockKHalf];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = warp_id % N_WARPS_M;          // 0..1
    const int warp_n = warp_id / N_WARPS_M;          // 0..1

    const int cta_m = blockIdx.y * BLOCK_M;
    const int cta_n = blockIdx.x * BLOCK_N;
    const int n_groups = K / kGroupSize;

    // Per-warp accumulator: fp32, kNUnroll N-tiles × 4 fp32 per lane
    float acc[kNUnroll][4];
    #pragma unroll
    for (int i = 0; i < kNUnroll; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;

    // Cooperative load helpers: each thread reads 16 bytes per iter, strides
    // by blockDim.x to cover arbitrary tile sizes.
    auto load_a_tile = [&](int stage, int g) {
        const int Kh_base = (g * kGroupSize) / 2;
        const int total_bytes = BLOCK_M * kBlockKHalf;        // 32*32 = 1024
        for (int load_idx = tid; load_idx < total_bytes / 16; load_idx += blockDim.x) {
            int row = (load_idx * 16) / kBlockKHalf;
            int col = (load_idx * 16) % kBlockKHalf;
            int m_global = cta_m + row;
            uint32_t s_addr = cvta_smem(&a_sh[stage][row][col]);
            if (m_global < M) {
                cp_async_16b(s_addr, act + m_global * (K / 2) + Kh_base + col);
            } else {
                *reinterpret_cast<uint4*>(&a_sh[stage][row][col]) = make_uint4(0, 0, 0, 0);
            }
        }
    };

    auto load_w_tile = [&](int stage, int g) {
        const int Kh_base = (g * kGroupSize) / 2;
        const int total_bytes = BLOCK_N * kBlockKHalf;         // 128*32 = 4096
        for (int load_idx = tid; load_idx < total_bytes / 16; load_idx += blockDim.x) {
            int row = (load_idx * 16) / kBlockKHalf;
            int col = (load_idx * 16) % kBlockKHalf;
            int n_global = cta_n + row;
            uint32_t s_addr = cvta_smem(&w_sh[stage][row][col]);
            if (n_global < N) {
                cp_async_16b(s_addr, wgt + n_global * (K / 2) + Kh_base + col);
            } else {
                *reinterpret_cast<uint4*>(&w_sh[stage][row][col]) = make_uint4(0, 0, 0, 0);
            }
        }
    };

    // Synchronous version (no prefetch) — for Phase 0 correctness validation.
    // Phase 1 will reintroduce double buffer + cp.async.bulk pipelining.

    // Main loop
    for (int g = 0; g < n_groups; ++g) {
        int cur_stage = 0;  // single buffer

        load_a_tile(cur_stage, g);
        load_w_tile(cur_stage, g);
        cp_async_commit_group();
        cp_async_wait_group<0>();
        __syncthreads();

        // Per-thread MMA work for this group (1 K-MMA = full BLOCK_K=64)
        // mma.m16n8k64 takes a full 64 K positions, so 1 MMA per K-tile.

        const int warp_m_base = warp_m * kWarpM;
        const int warp_n_base = warp_n * kWarpN;

        // Load A fragment (16M × 64K) - 4 regs per thread.
        // Standard ldmatrix.x4 lane mapping for 16-row × 32-byte:
        //   addr = a_sh + (warp_m_base + lane%16) * kBlockKHalf + (lane/16) * 16
        int a_frag[4];
        {
            uint32_t a_addr = cvta_smem(
                &a_sh[cur_stage][warp_m_base + (lane % 16)][(lane / 16) * 16]
            );
            ldmatrix_x4_b8(*reinterpret_cast<uint32_t(*)[4]>(a_frag), a_addr);
        }

        // For each N-MMA tile (8 cols), load B fragment (8N × 64K) and issue MMA
        #pragma unroll
        for (int n_mma = 0; n_mma < kNUnroll; ++n_mma) {
            const int n_off = warp_n_base + n_mma * 8;
            int b_frag[2];
            // mma B operand for m16n8k64.row.col is 64K × 8N stored as
            // 2 regs/thread; load via ldmatrix.x2.b16 (or x1, depending on layout)
            // Simpler: load B via 2 × ldmatrix.x1 from contiguous shmem.
            // For b8/int4 row-major B = (BLOCK_N, BLOCK_K/2):
            //   B-tile (8N × 64K = 32 bytes per row) lives at:
            //     w_sh[cur_stage][n_off + row][0..32]
            // Each lane reads 2 b16-words from one row at lane% offset.
            // ldmatrix.x2 lane mapping: 16 lanes used (rest dup).
            //   lanes 0..7   → row n_off + 0..7,   col_off 0
            //   lanes 8..15  → row n_off + 0..7,   col_off 16  (= 16 bytes = 16 packed-int4)
            // Each address must be 16-byte aligned (b16 ldmatrix requirement).
            uint32_t b_addr_lo = cvta_smem(
                &w_sh[cur_stage][n_off + (lane % 8)][((lane / 8) & 1) * 16]
            );
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                : "=r"(b_frag[0]), "=r"(b_frag[1])
                : "r"(b_addr_lo)
            );

            // Issue MMA
            int c_int[4] = {0, 0, 0, 0};
            mma_m16n8k64_s4s4s32(
                *reinterpret_cast<int(*)[4]>(a_frag),
                *reinterpret_cast<int(*)[2]>(b_frag),
                *reinterpret_cast<int(*)[4]>(c_int)
            );

            // Per-group dequant: int32 × ascale[g, m] × wscale[g, n] → fp32 acc
            const int row_lo = lane / 4;
            const int col_lo = (lane % 4) * 2;
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                int m_local = row_lo + half * 8;
                int m_global = cta_m + warp_m_base + m_local;
                int n_global0 = cta_n + n_off + col_lo;
                int n_global1 = n_global0 + 1;
                if (m_global >= M) continue;

                float a_s = (m_global < M)
                    ? __bfloat162float(ascales[g * M + m_global])
                    : 0.0f;
                float w_s0 = (n_global0 < N)
                    ? __bfloat162float(wscales[g * N + n_global0])
                    : 0.0f;
                float w_s1 = (n_global1 < N)
                    ? __bfloat162float(wscales[g * N + n_global1])
                    : 0.0f;

                acc[n_mma][half * 2 + 0] += float(c_int[half * 2 + 0]) * a_s * w_s0;
                acc[n_mma][half * 2 + 1] += float(c_int[half * 2 + 1]) * a_s * w_s1;
            }
        }
        // Fence between MMA reads (this iter) and cp.async writes (next iter)
        // on the same single-buffer shmem region.
        __syncthreads();
    }

    // Drain pipeline
    cp_async_wait_group<0>();
    __syncthreads();

    // Write output: each lane stores 4 fp32 to 4 (m, n) positions
    const int row_lo = lane / 4;
    const int col_lo = (lane % 4) * 2;
    const int warp_m_base = warp_m * kWarpM;
    const int warp_n_base = warp_n * kWarpN;
    #pragma unroll
    for (int n_mma = 0; n_mma < kNUnroll; ++n_mma) {
        const int n_off = warp_n_base + n_mma * 8;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            int m_global = cta_m + warp_m_base + row_lo + half * 8;
            int n_global0 = cta_n + n_off + col_lo;
            int n_global1 = n_global0 + 1;
            if (m_global < M) {
                if (n_global0 < N) out[m_global * N + n_global0] = __float2bfloat16(acc[n_mma][half * 2 + 0]);
                if (n_global1 < N) out[m_global * N + n_global1] = __float2bfloat16(acc[n_mma][half * 2 + 1]);
            }
        }
    }
}

} // namespace

extern "C" void launch_w4a4_gemm_phase0(
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
    constexpr int N_THREADS = N_WARPS_M * N_WARPS_N * 32;

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(N_THREADS);
    w4a4_gemm_phase0<BLOCK_M, BLOCK_N, BLOCK_K, N_WARPS_M, N_WARPS_N>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const int8_t*>(act),
            reinterpret_cast<const int8_t*>(wgt),
            reinterpret_cast<const __nv_bfloat16*>(ascales),
            reinterpret_cast<const __nv_bfloat16*>(wscales),
            reinterpret_cast<__nv_bfloat16*>(out),
            M, N, K);
}
