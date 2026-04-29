// SPDX-License-Identifier: Apache-2.0
//
// Phase 2 v2: keep Phase 1's 4-warp structure, only add scales-in-shmem.
//
// Phase 2 v1 (8-warp producer/consumer) was structurally correct but lost
// per-SM throughput because doubling warps/CTA halved occupancy — fewer
// concurrent CTAs more than offset the per-CTA pipeline overlap. v2 isolates
// the scales-in-shmem win without touching the warp structure.
//
// Specifically: at kernel start, cooperatively load full ascales (BLOCK_M
// per group, all groups) and wscales (BLOCK_N per group, all groups) into
// shmem. The K-loop then reads scales from shmem instead of global,
// eliminating the per-K-iter dependent global loads (8 lookups/lane/iter
// × 48 iters = 384/lane in the hot path).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

namespace {

constexpr int kGroupSize = 64;
constexpr int kBlockKHalf = kGroupSize / 2;
constexpr int kSmemRowStride = kBlockKHalf + 16;     // 48 bytes
constexpr int kStages = 2;

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
__global__ void w4a4_gemm_phase2v2(
    const int8_t* __restrict__ act,
    const int8_t* __restrict__ wgt,
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
    constexpr int kWTileBytes = BLOCK_N * kSmemRowStride;

    extern __shared__ uint8_t shmem_dyn[];
    int8_t* a_sh = reinterpret_cast<int8_t*>(shmem_dyn);
    int8_t* w_sh = a_sh + kStages * kATileBytes;
    __nv_bfloat16* ascales_sh = reinterpret_cast<__nv_bfloat16*>(
        w_sh + kStages * kWTileBytes);
    __nv_bfloat16* wscales_sh = ascales_sh + n_groups * BLOCK_M;

    auto a_stage = [&](int s) -> int8_t* { return a_sh + s * kATileBytes; };
    auto w_stage = [&](int s) -> int8_t* { return w_sh + s * kWTileBytes; };

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = warp_id % N_WARPS_M;
    const int warp_n = warp_id / N_WARPS_M;

    const int cta_m = blockIdx.y * BLOCK_M;
    const int cta_n = blockIdx.x * BLOCK_N;

    // --------- preload all scales into shmem (cooperative, all 128 threads) ---------
    {
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

    auto load_w = [&](int stage, int g) {
        const int Kh_base = (g * kGroupSize) / 2;
        for (int load_idx = tid; load_idx < (BLOCK_N * kBlockKHalf) / 16; load_idx += blockDim.x) {
            int row = (load_idx * 16) / kBlockKHalf;
            int col = (load_idx * 16) % kBlockKHalf;
            int n_global = cta_n + row;
            uint32_t s_addr = cvta_smem(&w_stage(stage)[row * kSmemRowStride + col]);
            if (n_global < N) {
                cp_async_16b(s_addr, wgt + n_global * (K / 2) + Kh_base + col);
            } else {
                *reinterpret_cast<uint4*>(&w_stage(stage)[row * kSmemRowStride + col]) = make_uint4(0,0,0,0);
            }
        }
    };

    if (n_groups > 0) {
        load_a(0, 0);
        load_w(0, 0);
        cp_async_commit_group();
    }

    float acc[kNUnroll][4];
    #pragma unroll
    for (int i = 0; i < kNUnroll; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;

    for (int g = 0; g < n_groups; ++g) {
        int cur_stage = g & 1;
        int next_stage = (g + 1) & 1;

        if (g + 1 < n_groups) {
            load_a(next_stage, g + 1);
            load_w(next_stage, g + 1);
            cp_async_commit_group();
        }

        cp_async_wait_group<1>();
        __syncthreads();

        const int warp_m_base = warp_m * kWarpM;
        const int warp_n_base = warp_n * kWarpN;

        int a_frag[4];
        {
            uint32_t a_addr = cvta_smem(
                &a_stage(cur_stage)[(warp_m_base + (lane % 16)) * kSmemRowStride + (lane / 16) * 16]
            );
            ldmatrix_x4_b16(*reinterpret_cast<uint32_t(*)[4]>(a_frag), a_addr);
        }

        #pragma unroll
        for (int n_mma = 0; n_mma < kNUnroll; ++n_mma) {
            const int n_off = warp_n_base + n_mma * 8;
            int b_frag[2];
            uint32_t b_addr = cvta_smem(
                &w_stage(cur_stage)[(n_off + (lane % 8)) * kSmemRowStride + ((lane / 8) & 1) * 16]
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

                // Read scales from shmem (preloaded at kernel start) — was the
                // K-loop hot-path bottleneck in Phase 1.
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

extern "C" void launch_w4a4_gemm_phase2v2(
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

    const int n_groups = K / BLOCK_K;

    constexpr int kATileBytes = BLOCK_M * kSmemRowStride;
    constexpr int kWTileBytes = BLOCK_N * kSmemRowStride;
    int shmem_bytes = kStages * (kATileBytes + kWTileBytes)
                    + n_groups * (BLOCK_M + BLOCK_N) * (int)sizeof(__nv_bfloat16);

    if (shmem_bytes > 48 * 1024) {
        cudaError_t e = cudaFuncSetAttribute(
            (const void*)w4a4_gemm_phase2v2<BLOCK_M, BLOCK_N, BLOCK_K, N_WARPS_M, N_WARPS_N>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_bytes);
        if (e != cudaSuccess) {
            printf("phase2v2: shmem(%d KB) attribute set failed: %s — falling back\n",
                shmem_bytes / 1024, cudaGetErrorString(e));
            return;
        }
    }

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(N_THREADS);
    w4a4_gemm_phase2v2<BLOCK_M, BLOCK_N, BLOCK_K, N_WARPS_M, N_WARPS_N>
        <<<grid, block, shmem_bytes, stream>>>(
            reinterpret_cast<const int8_t*>(act),
            reinterpret_cast<const int8_t*>(wgt),
            reinterpret_cast<const __nv_bfloat16*>(ascales),
            reinterpret_cast<const __nv_bfloat16*>(wscales),
            reinterpret_cast<__nv_bfloat16*>(out),
            M, N, K, n_groups);
}

// Same launch entry point name pattern as other phases for the bench harness.
extern "C" void launch_w4a4_gemm_phase2(
    const void* act, const void* wgt,
    const void* ascales, const void* wscales,
    void* out,
    int M, int N, int K,
    cudaStream_t stream)
{
    launch_w4a4_gemm_phase2v2(act, wgt, ascales, wscales, out, M, N, K, stream);
}
