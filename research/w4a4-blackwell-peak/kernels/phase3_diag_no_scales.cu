// SPDX-License-Identifier: Apache-2.0
//
// Phase 3 diagnostic: int4 GEMM with NO per-group scale dequant.
//
// Just to find out where the per-K-iter time goes. Replace per-K-iter
// fp32 fma chain with a single fp32 add of c_int into accumulator. If
// throughput jumps significantly, scales dequant was the bottleneck.
// Output is **incorrect** but timing is the data point we care about.
//
// (This is a diagnostic kernel only — not a candidate ship.)
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

namespace {

constexpr int kGroupSize = 64;
constexpr int kBlockKHalf = kGroupSize / 2;
constexpr int kSmemRowStride = kBlockKHalf + 16;
constexpr int kStages = 2;

__device__ __forceinline__ uint32_t cvta_smem(const void* p) {
    uint32_t s; asm("{ .reg .u64 ll; cvta.to.shared.u64 ll, %1; cvt.u32.u64 %0, ll; }" : "=r"(s) : "l"(p));
    return s;
}
__device__ __forceinline__ void cp_async_16b(uint32_t a, const void* p) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;" :: "r"(a), "l"(p));
}
__device__ __forceinline__ void cp_async_commit_group() { asm volatile("cp.async.commit_group;"); }
template<int N>
__device__ __forceinline__ void cp_async_wait_group() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }
__device__ __forceinline__ void mma_m16n8k64_s4s4s32(int (&a)[4], int (&b)[2], int (&c)[4]) {
    asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(c[0]),"=r"(c[1]),"=r"(c[2]),"=r"(c[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),
          "r"(c[0]),"r"(c[1]),"r"(c[2]),"r"(c[3]));
}
__device__ __forceinline__ void ldmatrix_x4_b16(uint32_t (&dst)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(dst[0]),"=r"(dst[1]),"=r"(dst[2]),"=r"(dst[3]) : "r"(addr));
}

template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int N_WARPS_M, int N_WARPS_N>
__global__ void w4a4_diag_no_scales(
    const int8_t* __restrict__ act,
    const int8_t* __restrict__ wgt,
    const __nv_bfloat16* __restrict__ ascales,   // ignored
    const __nv_bfloat16* __restrict__ wscales,   // ignored
    __nv_bfloat16* __restrict__ out,
    int M, int N, int K)
{
    static_assert(BLOCK_K == kGroupSize);
    constexpr int kWarpM = 16;
    constexpr int kWarpN = BLOCK_N / N_WARPS_N;
    constexpr int kNUnroll = kWarpN / 8;

    constexpr int kATileBytes = BLOCK_M * kSmemRowStride;
    constexpr int kWTileBytes = BLOCK_N * kSmemRowStride;
    __shared__ alignas(16) int8_t a_sh[kStages][kATileBytes];
    __shared__ alignas(16) int8_t w_sh[kStages][kWTileBytes];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = warp_id % N_WARPS_M;
    const int warp_n = warp_id / N_WARPS_M;

    const int cta_m = blockIdx.y * BLOCK_M;
    const int cta_n = blockIdx.x * BLOCK_N;
    const int n_groups = K / kGroupSize;

    auto load_a = [&](int s, int g) {
        const int Kh_base = (g * kGroupSize) / 2;
        for (int idx = tid; idx < (BLOCK_M * kBlockKHalf) / 16; idx += blockDim.x) {
            int row = (idx * 16) / kBlockKHalf;
            int col = (idx * 16) % kBlockKHalf;
            int m_global = cta_m + row;
            uint32_t s_addr = cvta_smem(&a_sh[s][row * kSmemRowStride + col]);
            if (m_global < M) cp_async_16b(s_addr, act + m_global * (K / 2) + Kh_base + col);
            else *reinterpret_cast<uint4*>(&a_sh[s][row * kSmemRowStride + col]) = make_uint4(0,0,0,0);
        }
    };
    auto load_w = [&](int s, int g) {
        const int Kh_base = (g * kGroupSize) / 2;
        for (int idx = tid; idx < (BLOCK_N * kBlockKHalf) / 16; idx += blockDim.x) {
            int row = (idx * 16) / kBlockKHalf;
            int col = (idx * 16) % kBlockKHalf;
            int n_global = cta_n + row;
            uint32_t s_addr = cvta_smem(&w_sh[s][row * kSmemRowStride + col]);
            if (n_global < N) cp_async_16b(s_addr, wgt + n_global * (K / 2) + Kh_base + col);
            else *reinterpret_cast<uint4*>(&w_sh[s][row * kSmemRowStride + col]) = make_uint4(0,0,0,0);
        }
    };

    if (n_groups > 0) { load_a(0, 0); load_w(0, 0); cp_async_commit_group(); }

    int acc[kNUnroll][4] = {};
    for (int g = 0; g < n_groups; ++g) {
        int cur = g & 1, nxt = (g + 1) & 1;
        if (g + 1 < n_groups) { load_a(nxt, g + 1); load_w(nxt, g + 1); cp_async_commit_group(); }
        cp_async_wait_group<1>();
        __syncthreads();

        const int warp_m_base = warp_m * kWarpM;
        const int warp_n_base = warp_n * kWarpN;

        int a_frag[4];
        ldmatrix_x4_b16(*reinterpret_cast<uint32_t(*)[4]>(a_frag),
            cvta_smem(&a_sh[cur][(warp_m_base + (lane % 16)) * kSmemRowStride + (lane / 16) * 16]));

        #pragma unroll
        for (int n_mma = 0; n_mma < kNUnroll; ++n_mma) {
            const int n_off = warp_n_base + n_mma * 8;
            int b_frag[2];
            uint32_t b_addr = cvta_smem(&w_sh[cur][(n_off + (lane % 8)) * kSmemRowStride + ((lane / 8) & 1) * 16]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                : "=r"(b_frag[0]), "=r"(b_frag[1]) : "r"(b_addr));
            // Skip per-group scale dequant entirely — accumulate raw int32 in
            // int register. Output meaningless but K-loop is now MMA-only.
            mma_m16n8k64_s4s4s32(*reinterpret_cast<int(*)[4]>(a_frag),
                *reinterpret_cast<int(*)[2]>(b_frag),
                *reinterpret_cast<int(*)[4]>(acc[n_mma]));
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
                if (n_global0 < N) out[m_global * N + n_global0] = __float2bfloat16(float(acc[n_mma][half * 2 + 0]) * 1e-4f);
                if (n_global1 < N) out[m_global * N + n_global1] = __float2bfloat16(float(acc[n_mma][half * 2 + 1]) * 1e-4f);
            }
        }
    }
}

} // namespace

extern "C" void launch_w4a4_gemm_phase3(
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
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(N_WARPS_M * N_WARPS_N * 32);
    w4a4_diag_no_scales<BLOCK_M, BLOCK_N, BLOCK_K, N_WARPS_M, N_WARPS_N>
        <<<grid, block, 0, stream>>>(
            (const int8_t*)act, (const int8_t*)wgt,
            (const __nv_bfloat16*)ascales, (const __nv_bfloat16*)wscales,
            (__nv_bfloat16*)out,
            M, N, K);
}
