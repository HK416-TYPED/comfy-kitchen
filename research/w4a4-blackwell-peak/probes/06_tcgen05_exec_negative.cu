#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__global__ void test_tcgen05() {
    // Try to allocate tensor memory and check return
    if (threadIdx.x == 0) {
        unsigned tmem_addr = 0;
        // tcgen05.alloc requires a smem ptr to write the allocation handle to
        __shared__ unsigned tmem_alloc[1];
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 1;"
            :: "r"((unsigned)__cvta_generic_to_shared(&tmem_alloc[0])));
        tmem_addr = tmem_alloc[0];
        // Print what we got back
        printf("tcgen05.alloc returned tmem_addr=0x%x\n", tmem_addr);
        // Try to dealloc
        if (tmem_addr != 0) {
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 1;"
                :: "r"(tmem_addr));
            printf("tcgen05.dealloc succeeded\n");
        }
    }
}
int main() {
    test_tcgen05<<<1, 32>>>();
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) printf("ERROR: %s\n", cudaGetErrorString(e));
    return e == cudaSuccess ? 0 : 1;
}
