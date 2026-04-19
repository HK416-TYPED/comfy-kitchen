# zgemm — nunchaku verbatim port

Verbatim copy of nunchaku's W4A4 zgemm kernel family, living inside kitchen's
CUDA backend so the runtime math is bit-identical to nunchaku.

## Source

Files in this directory are taken from
[nunchaku-tech/nunchaku](https://github.com/nunchaku-tech/nunchaku) under the
Apache-2.0 license. Paths in the upstream tree:

| file                            | upstream path                              |
|---------------------------------|--------------------------------------------|
| `Tensor.h`                      | `src/Tensor.h`                             |
| `common.h`                      | `src/common.h` *(patched — see below)*     |
| `dispatch_utils.h`              | `src/kernels/dispatch_utils.h`             |
| `utils.cuh`                     | `src/kernels/utils.cuh`                    |
| `zgemm.h`                       | `src/kernels/zgemm/zgemm.h`                |
| `gemm_base.cuh`                 | `src/kernels/zgemm/gemm_base.cuh`          |
| `gemm_utils.cuh`                | `src/kernels/zgemm/gemm_utils.cuh`         |
| `gemm_w4a4.cuh`                 | `src/kernels/zgemm/gemm_w4a4.cuh`          |
| `gemm_w4a4.cu`                  | `src/kernels/zgemm/gemm_w4a4.cu`           |
| `gemm_w4a4_launch.cuh`          | `src/kernels/zgemm/gemm_w4a4_launch.cuh`   |
| `gemm_w4a4_launch_impl.cuh`     | same                                       |
| `gemm_w4a4_launch_bf16_int4.cu` | same                                       |
| `gemm_w4a4_launch_fp16_int4.cu` | same                                       |
| `epilogues.cuh`                 | `src/kernels/zgemm/epilogues.cuh`          |
| `lora.cuh`                      | `src/kernels/zgemm/lora.cuh`               |
| `mma.cuh`                       | `src/kernels/zgemm/mma.cuh`                |
| `mma_earlycuda.cuh`             | `src/kernels/zgemm/mma_earlycuda.cuh`      |

Scope: W4A4 only. Nunchaku's FP4, W8A8, and attention-fused paths are **not**
ported.

## Local patches

Minimal, confined to `common.h`:

- **spdlog** is replaced with no-op inline stubs in a local `namespace spdlog`.
  All call sites (`spdlog::trace(fmt, args...)`) compile away to nothing.
- **`spdlog::fmt_lib::format`** -> `std::format` (C++20). Same `{}`-substitution
  syntax, no behavior change for the surviving error-string formatter.
- **`cublas_v2.h`** include and the unused `CUBLASWrapper` / `getCUBLAS()` /
  `checkCUBLAS()` helpers are dropped. cuBLAS is not used on the W4A4 path.
- `cuda_fp16.h` + `cuda_bf16.h` are added at the top of `common.h` so
  downstream translation units have the half2 / bfloat162 types available
  regardless of include order.

Everything else — all kernel code in `gemm_*.cu*`, `lora.cuh`, `mma*.cuh`,
`epilogues.cuh`, `utils.cuh`, `Tensor.h`, `dispatch_utils.h` — is byte-identical
to the upstream revision.

## Integration

- `CMakeLists.txt` compiles `gemm_w4a4.cu` + the two `gemm_w4a4_launch_*_int4.cu`
  instantiations into the same `_C` nanobind module as the rest of kitchen's
  CUDA ops.
- Python-facing entry point: TBD. The `nunchaku::kernels::gemm_w4a4(...)`
  function takes nunchaku `Tensor` objects; the bridge that adapts DLPack
  pointers coming out of `_wrap_for_dlpack` is under construction.
- Weight layout is nunchaku tile-packed (block-major in N, interleaved MMA
  fragments). The converter at `/workspace/nunchaku/tools/kitchen_native/`
  must be extended with a variant that emits this layout with QKV already
  split (we keep kitchen's convention of separate `to_q` / `to_k` / `to_v`
  layers; the split happens inside the packed representation).

## Why verbatim?

Earlier kitchen-native kernels match nunchaku's algorithm but not its exact
numerics — the tile shapes, accumulator widths, scale-broadcast patterns, and
LoRA fp16/fp32 precision choices all differ slightly. This verbatim port is
the ground-truth reference: when both are loaded with the same weights and
same input, their outputs match nunchaku bit-for-bit.
