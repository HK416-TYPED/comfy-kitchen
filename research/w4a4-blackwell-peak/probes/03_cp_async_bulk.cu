#include <cstdint>
#include <cuda_runtime.h>

__global__ void test_bulk_load() {
    __shared__ alignas(8) uint64_t bar;
    __shared__ alignas(128) char smem_buf[256];
    unsigned bar_addr = (unsigned)__cvta_generic_to_shared(&bar);
    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(smem_buf);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar_addr));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 256;" :: "r"(bar_addr));
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1], 256, [%2];"
        :: "r"(smem_addr), "l"((const void*)0x1000), "r"(bar_addr)
    );
}
int main() { return 0; }
