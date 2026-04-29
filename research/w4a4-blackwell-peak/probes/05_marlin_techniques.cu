#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ int test_lop3(int a, int b, int c) {
    int r;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

__device__ int test_prmt(int a, int b) {
    int r;
    asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

__device__ void test_cp_async_l2hint(uint32_t smem_ptr, const void* glob) {
    asm volatile(
        "{\n"
        "   .reg .b64 p;\n"
        "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;\n"
        "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, p;\n"
        "}\n" :: "r"(smem_ptr), "l"(glob));
}

__global__ void probe(const int* src, int* dst) {
    __shared__ alignas(16) uint8_t buf[64];
    uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(buf);
    test_cp_async_l2hint(s_addr, src);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    int loaded = *reinterpret_cast<int*>(&buf[threadIdx.x * 4]);
    int r = test_lop3(loaded, 0x000F000F, 0x64006400);
    int p = test_prmt(r, 0x12345678);
    dst[threadIdx.x] = p;
}

int main() {
    int* d_src; int* d_dst;
    cudaMalloc(&d_src, 64);
    cudaMalloc(&d_dst, 128);
    int host[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    cudaMemcpy(d_src, host, 64, cudaMemcpyHostToDevice);
    probe<<<1, 16>>>(d_src, d_dst);
    cudaError_t e = cudaDeviceSynchronize();
    return e == cudaSuccess ? 0 : 2;
}
