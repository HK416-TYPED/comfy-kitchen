#include <cstdint>
#include <cuda_runtime.h>
__global__ void test_tma() {
    __shared__ alignas(8) uint64_t bar;
    __shared__ alignas(128) uint8_t buf[1024];
    uint32_t bar_addr = (uint32_t)__cvta_generic_to_shared(&bar);
    uint32_t buf_addr = (uint32_t)__cvta_generic_to_shared(buf);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar_addr));
    // cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
    // Needs a tensor map descriptor (CUtensorMap, 128 bytes, in const memory).
    // Just test that the ptx assembles:
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2,%3}], [%4];"
        :: "r"(buf_addr), "l"((const void*)0x1000), "r"(0), "r"(0), "r"(bar_addr));
}
int main() { return 0; }
