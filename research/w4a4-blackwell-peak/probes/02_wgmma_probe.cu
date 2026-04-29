#include <cstdint>
#include <cuda_runtime.h>
__device__ void test_wgmma_s4() {
    // wgmma.mma_async.sync.aligned.m64n8k32.s32.s4.s4.s32 (Hopper sm_90+)
    int d[2];
    uint64_t a_desc = 0, b_desc = 0;
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k32.s32.s4.s4.s32 "
        "{%0,%1}, %2, %3, 1, 1, 0;"
        : "=r"(d[0]), "=r"(d[1])
        : "l"(a_desc), "l"(b_desc));
}
__global__ void probe() {}
int main() { probe<<<1,1>>>(); cudaDeviceSynchronize(); return 0; }
