#include <cstdint>
// Probe critical sm_120 features for W4A4 peak performance kernel
#include <cuda_runtime.h>
#include <cstdio>

// 1. mma.m16n8k64.s4.s4.s32 (int4 MMA, since sm_80)
__device__ void test_mma_m16n8k64_s4() {
    int a[2], b[1], c[4] = {0};
    asm("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

// 2. cp.async.bulk (Hopper, available on sm_120 via sm_100+ feature parity)
__device__ void test_cp_async_bulk() {
    __shared__ alignas(128) char smem[256];
    extern __shared__ uint64_t bar[];
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 256, [%2];"
        :: "r"((unsigned)__cvta_generic_to_shared(smem)), "l"((const void*)0), "r"((unsigned)__cvta_generic_to_shared(bar))
    );
}

// 3. mbarrier.arrive.expect_tx (Hopper-style async sync)
__device__ void test_mbarrier_expect_tx() {
    __shared__ alignas(8) uint64_t bar;
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
        :: "r"((unsigned)__cvta_generic_to_shared(&bar)), "r"(256));
}

// 4. wgmma (5th-gen tensor core warp-group async MMA, Hopper sm_90+)
__device__ void test_wgmma() {
    // wgmma.mma_async.sync.aligned.m64n8k32.s32.s4.s4.s32 (Hopper)
    // sm_120 (Blackwell consumer) — does it have wgmma?
}

__global__ void probe() {}
int main() { probe<<<1,1>>>(); cudaDeviceSynchronize(); return 0; }
