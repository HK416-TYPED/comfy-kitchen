#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

__global__ void mma_int4_spam(int n_iters) {
    int a[4] = {0,0,0,0}, b[2] = {0,0}, c[4] = {0,0,0,0};
    for (int i = 0; i < n_iters; ++i) {
        asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
            : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]),
              "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
    }
    if (c[0] == 99) printf("");
}

int main() {
    int n_blocks = 170 * 8;
    int n_warps_per_block = 4;
    int n_iters = 10000;
    int total_warps = n_blocks * n_warps_per_block;
    long total_mma = (long)total_warps * n_iters;
    long total_ops = total_mma * 16 * 8 * 64 * 2;

    // warmup
    mma_int4_spam<<<n_blocks, 128>>>(100);
    cudaDeviceSynchronize();

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    mma_int4_spam<<<n_blocks, 128>>>(n_iters);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0;
    cudaEventElapsedTime(&ms, s, e);
    double tflops = total_ops / 1e9 / ms;
    printf("Total MMA ops: %ld   Total int4 madds: %ld\n", total_mma, total_ops);
    printf("Time: %.2f ms\n", ms);
    printf("Throughput: %.1f TFLOPS (int4 mma.m16n8k64)\n", tflops);
    return 0;
}
