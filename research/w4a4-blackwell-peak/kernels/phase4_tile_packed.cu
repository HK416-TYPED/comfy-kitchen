// SPDX-License-Identifier: Apache-2.0
//
// Phase 4: tile-packed weight + 4-stage cp.async pipeline.
//
// On-disk weight layout (designed to match Marlin/nunchaku's contiguous-CTA-tile
// pattern):
//
//   Outer:  cta_n   = 0..N/BLOCK_N-1
//           cta_k   = 0..(K/2)/BLOCK_KH-1            (BLOCK_KH = BLOCK_K/2)
//   Inside each (cta_n, cta_k) tile (= 4 KB contiguous):
//           n_stripe   = 0..BLOCK_N/4-1              (4-N-row interleave)
//           k_byte     = 0..BLOCK_KH-1
//           n_within   = 0..3
//   Linear byte offset =
//        ((cta_n * cta_k_count + cta_k) * (BLOCK_N/4) + n_stripe)
//          * (BLOCK_KH * 4) + k_byte * 4 + n_within
//
// Why this matters:
//   * Per CTA per K-iter, the entire 4 KB weight tile is one HBM contiguous
//     burst. cp.async (or TMA bulk) saturates HBM bandwidth.
//   * Within each stripe, 4 consecutive N rows share a 128-byte cache line
//     packed by k_byte. One ldmatrix.x4 hits all 4 N rows in one cache line,
//     instead of 4 cache lines (= 4× HBM bandwidth waste in row-major).
//
// Activation A and scales stay row-major / per-group.
//
// Pipeline: 4-stage cp.async.cg with L2::evict_first hint for B (one-shot
// weight reads), default L2 for A (re-used across N-CTAs).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

namespace {

constexpr int kGroupSize = 64;
constexpr int kBlockKHalf = kGroupSize / 2;        // 32
constexpr int kSmemRowStride = kBlockKHalf + 16;   // 48, breaks bank cycle
constexpr int kStages = 4;                          // deep prefetch

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

// Marlin-style L2 evict_first hint for one-shot weight reads.
__device__ __forceinline__ void cp_async_16b_evict(uint32_t smem_addr, const void* src) {
    asm volatile(
        "{ .reg .b64 p;\n"
        "  createpolicy.fractional.L2::evict_first.b64 p, 1.0;\n"
        "  cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, p;\n"
        "}"
        :: "r"(smem_addr), "l"(src));
}

__device__ __forceinline__ void cp_async_commit_group() { asm volatile("cp.async.commit_group;"); }
template<int N>
__device__ __forceinline__ void cp_async_wait_group() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }

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

template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int N_WARPS_M, int N_WARPS_N>
__global__ void w4a4_gemm_phase4(
    const int8_t* __restrict__ act,         // (M, K/2)            row-major
    const int8_t* __restrict__ wgt_packed,  // tile-packed, see header
    const __nv_bfloat16* __restrict__ ascales,
    const __nv_bfloat16* __restrict__ wscales,
    __nv_bfloat16* __restrict__ out,
    int M, int N, int K, int n_groups)
{
    static_assert(BLOCK_K == kGroupSize, "BLOCK_K must equal G=64");
    constexpr int kWarpM = 16;
    constexpr int kWarpN = BLOCK_N / N_WARPS_N;
    constexpr int kNUnroll = kWarpN / 8;

    constexpr int kATileBytes = BLOCK_M * kSmemRowStride;
    // For weight tile in shmem, store it in the SAME tile-packed form as HBM:
    //   stripe_shmem[BLOCK_N/4][BLOCK_KH][4]  contiguous within stripe
    // Plus stride padding to break ldmatrix bank conflict on the last dim.
    constexpr int kStripesPerCta = BLOCK_N / 4;             // 32
    constexpr int kStripeBytes = kBlockKHalf * 4;            // 128
    constexpr int kStripeStride = kStripeBytes + 16;         // 144 (pad)
    constexpr int kWTileBytes = kStripesPerCta * kStripeStride;

    extern __shared__ uint8_t shmem_dyn[];
    int8_t* a_sh_base = reinterpret_cast<int8_t*>(shmem_dyn);
    int8_t* w_sh_base = a_sh_base + kStages * kATileBytes;
    __nv_bfloat16* ascales_sh = reinterpret_cast<__nv_bfloat16*>(
        w_sh_base + kStages * kWTileBytes);
    __nv_bfloat16* wscales_sh = ascales_sh + n_groups * BLOCK_M;

    auto a_stage = [&](int s) -> int8_t* { return a_sh_base + s * kATileBytes; };
    auto w_stage = [&](int s) -> int8_t* { return w_sh_base + s * kWTileBytes; };

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = warp_id % N_WARPS_M;
    const int warp_n = warp_id / N_WARPS_M;

    const int cta_m = blockIdx.y * BLOCK_M;
    const int cta_n = blockIdx.x * BLOCK_N;
    const int cta_n_block = blockIdx.x;     // = cta_n / BLOCK_N
    const int kh_per_cta = kBlockKHalf;     // bytes per K-tile in K dim
    const int cta_k_count = (K / 2) / kh_per_cta;

    // --------- preload all scales into shmem ---------
    {
        const int total_a = n_groups * BLOCK_M;
        for (int idx = tid; idx < total_a; idx += blockDim.x) {
            int g = idx / BLOCK_M;
            int m_local = idx % BLOCK_M;
            int m_global = cta_m + m_local;
            ascales_sh[idx] = (m_global < M)
                ? ascales[g * M + m_global] : __nv_bfloat16(0.f);
        }
        const int total_w = n_groups * BLOCK_N;
        for (int idx = tid; idx < total_w; idx += blockDim.x) {
            int g = idx / BLOCK_N;
            int n_local = idx % BLOCK_N;
            int n_global = cta_n + n_local;
            wscales_sh[idx] = (n_global < N)
                ? wscales[g * N + n_global] : __nv_bfloat16(0.f);
        }
    }
    __syncthreads();

    auto load_a = [&](int stage, int g) {
        const int Kh_base = (g * kGroupSize) / 2;
        for (int load_idx = tid; load_idx < (BLOCK_M * kBlockKHalf) / 16; load_idx += blockDim.x) {
            int row = (load_idx * 16) / kBlockKHalf;
            int col = (load_idx * 16) % kBlockKHalf;
            int m_global = cta_m + row;
            uint32_t s_addr = cvta_smem(&a_stage(stage)[row * kSmemRowStride + col]);
            if (m_global < M) {
                cp_async_16b(s_addr, act + m_global * (K / 2) + Kh_base + col);
            } else {
                *reinterpret_cast<uint4*>(&a_stage(stage)[row * kSmemRowStride + col]) = make_uint4(0,0,0,0);
            }
        }
    };

    // Load tile-packed weight: per CTA per K-iter the entire 4 KB
    // (kStripesPerCta × kStripeBytes = 32 × 128 = 4096) is **contiguous in HBM**.
    // We re-pack it on-the-fly into shmem with stripe stride padding.
    auto load_w = [&](int stage, int g) {
        const int cta_k_idx = (g * kGroupSize / 2) / kh_per_cta;     // = g for our shapes
        // Source base in HBM (packed bytes):
        const int8_t* src_base = wgt_packed
            + ((cta_n_block * cta_k_count + cta_k_idx) * kStripesPerCta * kStripeBytes);
        // Each thread handles 16 bytes. Total 4096 bytes / 16 = 256 loads, with
        // 128 threads → 2 loads per thread.
        const int total_loads = (kStripesPerCta * kStripeBytes) / 16;       // 256
        for (int idx = tid; idx < total_loads; idx += blockDim.x) {
            int stripe = (idx * 16) / kStripeBytes;
            int byte_in_stripe = (idx * 16) % kStripeBytes;
            uint32_t s_addr = cvta_smem(&w_stage(stage)[stripe * kStripeStride + byte_in_stripe]);
            cp_async_16b_evict(s_addr, src_base + stripe * kStripeBytes + byte_in_stripe);
        }
    };

    // Warmup: prefetch the first (kStages-1) iterations
    int load_g = 0;
    for (int s = 0; s < kStages - 1 && load_g < n_groups; ++s, ++load_g) {
        load_a(s, load_g);
        load_w(s, load_g);
        cp_async_commit_group();
    }

    float acc[kNUnroll][4];
    #pragma unroll
    for (int i = 0; i < kNUnroll; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;

    for (int g = 0; g < n_groups; ++g) {
        int cur_stage = g % kStages;

        // Issue the next prefetch
        if (load_g < n_groups) {
            int load_stage = load_g % kStages;
            load_a(load_stage, load_g);
            load_w(load_stage, load_g);
            cp_async_commit_group();
            ++load_g;
        }

        // Wait until at most kStages-1 commits remain in flight (the next ones).
        // The current stage's commit is what we want to wait for: at iter g,
        // we have (kStages-1) prefetches plus 1 just-issued = kStages in flight.
        // Wait for `kStages-1` to remain → current finishes.
        cp_async_wait_group<kStages - 2>();
        __syncthreads();

        const int warp_m_base = warp_m * kWarpM;
        const int warp_n_base = warp_n * kWarpN;

        // Load A fragment from row-major shmem (same as Phase 0/1/2)
        int a_frag[4];
        {
            uint32_t a_addr = cvta_smem(
                &a_stage(cur_stage)[(warp_m_base + (lane % 16)) * kSmemRowStride + (lane / 16) * 16]
            );
            ldmatrix_x4_b16(*reinterpret_cast<uint32_t(*)[4]>(a_frag), a_addr);
        }

        // For each N-MMA tile, load B fragment from tile-packed shmem.
        // Per N-MMA: 8 N rows × 64 K nibbles = 32 bytes. With kInterleave=4,
        // 8 N rows = 2 stripes (4 each). Each stripe is `kStripeBytes=128`
        // bytes contiguous (= 32 K-bytes × 4 N-rows).
        //
        // For mma B operand layout (m16n8k64.row.col, 8N × 64K):
        //   reg 0 (lane k): B[K=(k%4)*16..(k%4)*16+15, N=k/4]
        //   reg 1 (lane k): B[K=(k%4)*16+16..(k%4)*16+31, N=k/4] (oops, that's >16)
        // Actually for s4 m16n8k64: each lane holds 8 packed-int4 = 16 K
        // positions per reg, × 2 regs = 32 K positions. With 8 N cols, total
        // 256 K-N values per warp = 8N × 32K... but spec says 8N × 64K. Hmm.
        //
        // The s4 m16n8k64 lane layout (per PTX docs):
        //   B operand 64K × 8N: 32 lanes × 2 regs × 4 bytes = 256 bytes total
        //   = 64 * 8 / 2 (int4 packing) = 256. ✓
        //   Lane k holds:
        //     reg 0: B[K=(k%4)*16..(k%4)*16+15, N=k/4]    (16 K × 1 N = 8 bytes)
        //     reg 1: B[K=(k%4)*16+0+32..(k%4)*16+15+32, N=k/4]   (16 K × 1 N)
        //
        //   Grouped: 4 K-banks of 16 each (one per (k%4)), each lane sees 32 K total
        //
        // For our tile-packed layout, lane k needs to read 32 K positions for
        // N column k/4. With 8 N cols, each lane covers ONE N column.
        //
        // The 8 N cols of mma B are spread across 2 stripes (4 N each). Lane
        // k/4 = 0..3 → stripe 0; k/4 = 4..7 → stripe 1.
        //
        // Within a stripe (4 N-rows × 32 K-bytes), byte at (k_byte, n_within)
        // is at offset `k_byte * 4 + n_within`.
        //
        // For lane k, n_within = k/4 % 4, K_bytes are split across 2 regs:
        //   reg 0: K_bytes (k%4)*8 .. (k%4)*8+7  (= 16 K positions)
        //   reg 1: K_bytes (k%4)*8+16 .. (k%4)*8+23
        //
        // Each reg = 4 bytes from (k_byte_start + n_within * 4).
        //
        // ldmatrix.x4 isn't a clean fit for this layout — different per-warp
        // lane patterns. For Phase 4 first cut we'll do hand-coded loads from
        // shmem instead of ldmatrix, accepting some perf cost.

        #pragma unroll
        for (int n_mma = 0; n_mma < kNUnroll; ++n_mma) {
            const int n_off = warp_n_base + n_mma * 8;        // N-col base for this MMA
            const int n_stripe_base = n_off / 4;              // stripe index (8N = 2 stripes)
            int b_frag[2];

            // Build b_frag[0] and b_frag[1] for this lane:
            //   lane k (0..31): N_col = k/4 (0..7), K_byte_base = (k%4)*8
            //   reg 0: 4 bytes from k_byte (k%4)*8 .. (k%4)*8+3, n_within
            //   reg 1: 4 bytes from k_byte (k%4)*8+16 .. +19, n_within
            // Each lane reads from stripe ((n_off + (k/4)) / 4)
            // n_within = (n_off + (k/4)) % 4
            const int n_col = n_off + (lane / 4);             // 0..127
            const int n_stripe = n_col / 4;                   // 0..31
            const int n_within = n_col % 4;
            const int kb_base = (lane & 3) * 8;               // 0,8,16,24

            const int8_t* stripe_base = &w_stage(cur_stage)[n_stripe * kStripeStride];
            // mma B operand layout for m16n8k64.row.col.s4 (per PTX 8.5):
            //   Lane t covers N = t/4, K-rows split: reg 0 = K-positions
            //   (t%4)*16..+7, reg 1 = K-positions (t%4)*16+8..+15.
            //   In int4 packing, that's 4 K-bytes per reg, **K-byte adjacent**:
            //     reg 0 = bytes kb_base+0..3
            //     reg 1 = bytes kb_base+4..7
            //   where kb_base = (t%4)*8.
            uint8_t b0[4], b1[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                b0[i] = (uint8_t)stripe_base[(kb_base + i)     * 4 + n_within];
                b1[i] = (uint8_t)stripe_base[(kb_base + i + 4) * 4 + n_within];
            }
            b_frag[0] = (int)((uint32_t)b0[0] | ((uint32_t)b0[1] << 8) |
                              ((uint32_t)b0[2] << 16) | ((uint32_t)b0[3] << 24));
            b_frag[1] = (int)((uint32_t)b1[0] | ((uint32_t)b1[1] << 8) |
                              ((uint32_t)b1[2] << 16) | ((uint32_t)b1[3] << 24));

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

                float a_s = __bfloat162float(ascales_sh[g * BLOCK_M + m_local]);
                float w_s0 = __bfloat162float(wscales_sh[g * BLOCK_N + n_local0]);
                float w_s1 = __bfloat162float(wscales_sh[g * BLOCK_N + n_local1]);
                float scale0 = a_s * w_s0;
                float scale1 = a_s * w_s1;
                acc[n_mma][half * 2 + 0] += float(c_int[half * 2 + 0]) * scale0;
                acc[n_mma][half * 2 + 1] += float(c_int[half * 2 + 1]) * scale1;
            }
        }
        __syncthreads();
    }

    cp_async_wait_group<0>();
    __syncthreads();

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

extern "C" void launch_w4a4_gemm_phase4(
    const void* act, const void* wgt_packed,
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

    const int n_groups = K / BLOCK_K;

    constexpr int kATileBytes = BLOCK_M * kSmemRowStride;
    constexpr int kStripesPerCta = BLOCK_N / 4;
    constexpr int kStripeStride = kBlockKHalf * 4 + 16;
    constexpr int kWTileBytes = kStripesPerCta * kStripeStride;

    int shmem_bytes = kStages * (kATileBytes + kWTileBytes)
                    + n_groups * (BLOCK_M + BLOCK_N) * (int)sizeof(__nv_bfloat16);

    if (shmem_bytes > 48 * 1024) {
        cudaError_t e = cudaFuncSetAttribute(
            (const void*)w4a4_gemm_phase4<BLOCK_M, BLOCK_N, BLOCK_K, N_WARPS_M, N_WARPS_N>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_bytes);
        if (e != cudaSuccess) {
            printf("phase4: shmem(%d KB) attribute set failed: %s\n",
                shmem_bytes / 1024, cudaGetErrorString(e));
            return;
        }
    }

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(N_THREADS);
    w4a4_gemm_phase4<BLOCK_M, BLOCK_N, BLOCK_K, N_WARPS_M, N_WARPS_N>
        <<<grid, block, shmem_bytes, stream>>>(
            reinterpret_cast<const int8_t*>(act),
            reinterpret_cast<const int8_t*>(wgt_packed),
            reinterpret_cast<const __nv_bfloat16*>(ascales),
            reinterpret_cast<const __nv_bfloat16*>(wscales),
            reinterpret_cast<__nv_bfloat16*>(out),
            M, N, K, n_groups);
}
