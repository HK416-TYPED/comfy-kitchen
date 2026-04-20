// nanobind + DLPack bridge for nunchaku's zgemm W4A4 kernels.
//
// The kernel entry points in ops/zgemm/zgemm.h take `Tensor` objects whose
// storage is owned via `std::shared_ptr<Buffer>`. Kitchen's Python interface
// hands in `nb::ndarray<nb::device::cuda>` views whose storage is owned by
// the caller (PyTorch / ComfyUI). This bridge:
//
//   * wraps caller pointers in `BufferExternal` shells that never allocate
//     or free — they just carry the pointer through Tensor's ownership API
//   * converts DLPack dtype codes to `Tensor::ScalarType`
//   * translates a `None` Python value to an invalid `Tensor{}` sentinel
//     that the kernels already treat as "not provided" via `Tensor::valid()`
//   * pushes the caller stream onto zgemm's thread-local stream stack so
//     `getCurrentCUDAStream()` inside the kernels returns the right stream
//
// Scope: only the W4A4 entry points we actually need — `gemm_w4a4` and
// `quantize_w4a4_act_fuse_lora`. The attention-fused / w8a8 / linear-attn
// entry points are intentionally not exposed; Level 1 is W4A4 only.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <memory>
#include <stdexcept>
#include <vector>

#include "ops/zgemm/Tensor.h"
#include "ops/zgemm/common.h"
#include "ops/zgemm/zgemm.h"

namespace nb = nanobind;

namespace {

// Buffer that borrows an externally-owned pointer. No alloc on construction,
// no free on destruction — the caller guarantees the pointer outlives the
// kernel launch (enforced by the Python layer holding a reference across
// the call).
class BufferExternal : public Buffer {
public:
    BufferExternal(void *p, size_t s, Device d) {
        this->ptr    = p;
        this->size   = s;
        this->device = d;
    }
    ~BufferExternal() override = default;
};

Tensor::ScalarType dlpack_to_scalartype(const nb::dlpack::dtype &dt) {
    using Code = nb::dlpack::dtype_code;
    const uint8_t c = dt.code;
    if (c == (uint8_t)Code::Float) {
        if (dt.bits == 32) return Tensor::FP32;
        if (dt.bits == 16) return Tensor::FP16;
        if (dt.bits == 8)  return Tensor::FP8_E4M3;
    } else if (c == (uint8_t)Code::Bfloat && dt.bits == 16) {
        return Tensor::BF16;
    } else if (c == (uint8_t)Code::Int) {
        if (dt.bits == 8)  return Tensor::INT8;
        if (dt.bits == 16) return Tensor::INT16;
        if (dt.bits == 32) return Tensor::INT32;
        if (dt.bits == 64) return Tensor::INT64;
    } else if (c == (uint8_t)Code::UInt) {
        // nunchaku stores packed int4 bits in uint8 buffers — same byte width
        // as INT8; the kernel bit-casts through packed_act_t/packed_wgt_t.
        if (dt.bits == 8) return Tensor::INT8;
    }
    throw std::runtime_error("zgemm_bridge: unsupported DLPack dtype");
}

// Build a nunchaku Tensor from a DLPack CUDA ndarray. Strides are emitted
// only when the ndarray is non-contiguous, matching TensorShape's "empty
// stride means row-major contiguous" convention.
Tensor to_tensor(nb::ndarray<nb::device::cuda> arr) {
    const size_t nd = arr.ndim();
    std::vector<int> shape;
    shape.reserve(nd);
    for (size_t i = 0; i < nd; i++) {
        shape.push_back(static_cast<int>(arr.shape(i)));
    }

    bool contiguous = true;
    int64_t expected = 1;
    for (int i = static_cast<int>(nd) - 1; i >= 0; i--) {
        const int64_t dim = static_cast<int64_t>(arr.shape(i));
        if (dim > 1 && arr.stride(i) != expected) {
            contiguous = false;
            break;
        }
        expected *= dim;
    }
    std::vector<int> stride;
    if (!contiguous) {
        stride.reserve(nd);
        for (size_t i = 0; i < nd; i++) {
            stride.push_back(static_cast<int>(arr.stride(i)));
        }
    }

    const auto scalar_type = dlpack_to_scalartype(arr.dtype());
    const size_t elem_size = Tensor::scalarSize.at(scalar_type);
    const size_t bytes     = static_cast<size_t>(arr.size()) * elem_size;

    int dev_idx = 0;
    checkCUDA(cudaGetDevice(&dev_idx));
    const Device device{Device::CUDA, dev_idx};

    Tensor t;
    t.shape.dataExtent = std::move(shape);
    t.shape.dataStride = std::move(stride);
    t.shape.offset     = 0;
    t.scalarType       = scalar_type;
    t.buffer           = std::make_shared<BufferExternal>(arr.data(), bytes, device);
    return t;
}

// Accept an optional tensor (Python None or a CUDA ndarray). None →
// empty Tensor{} (Tensor::valid() returns false), which the kernels treat
// as "not provided".
Tensor to_tensor_opt(nb::object obj) {
    if (obj.is_none()) {
        return Tensor{};
    }
    return to_tensor(nb::cast<nb::ndarray<nb::device::cuda>>(obj));
}

// --- gemm_w4a4 -------------------------------------------------------------

// Reduced-surface binding for the Qwen SVDQuant W4A4 fused-QKV GEMM path.
// The full `nunchaku::kernels::gemm_w4a4` has 29 parameters covering FP4,
// attention-GEMM, linear-attention, and fused RMSNorm+RoPE. We expose only
// the W4A4-LoRA-SVDQuant variant (with optional fused next-layer quantize)
// and hard-code the rest. Keeps the nanobind surface small and avoids the
// 30-arg overload-resolution bug we ran into.
void zgemm_gemm_w4a4(
    nb::ndarray<nb::device::cuda> act,
    nb::ndarray<nb::device::cuda> wgt,
    nb::ndarray<nb::device::cuda> out,
    nb::ndarray<nb::device::cuda> ascales,
    nb::ndarray<nb::device::cuda> wscales,
    nb::ndarray<nb::device::cuda> lora_act_in,
    nb::ndarray<nb::device::cuda> lora_up,
    nb::object bias,              // optional (N,)
    nb::object qout,              // optional — fused next-layer packed act
    nb::object oscales,           // optional — fused next-layer ascales
    nb::object lora_down,         // optional — fused next-layer LoRA-down
    nb::object lora_act_out,      // optional — fused next-layer LoRA act
    nb::object smooth_factor,     // optional — fused next-layer smooth
    bool act_unsigned,
    bool fuse_silu,
    nb::list lora_scales_py,
    uintptr_t stream_ptr)
{
    CUDAStreamContext stream_ctx(reinterpret_cast<cudaStream_t>(stream_ptr));

    std::vector<float> lora_scales;
    lora_scales.reserve(lora_scales_py.size());
    for (nb::handle h : lora_scales_py) {
        lora_scales.push_back(nb::cast<float>(h));
    }

    nunchaku::kernels::gemm_w4a4(
        to_tensor(std::move(act)),          // act
        to_tensor(std::move(wgt)),          // wgt
        to_tensor(std::move(out)),          // out
        to_tensor_opt(qout),                // qout (fused next-layer packed act)
        to_tensor(std::move(ascales)),      // ascales
        to_tensor(std::move(wscales)),      // wscales
        to_tensor_opt(oscales),             // oscales (fused next-layer ascales)
        Tensor{},                           // poolout  — unused in Qwen W4A4
        to_tensor(std::move(lora_act_in)),  // lora_act_in
        to_tensor(std::move(lora_up)),      // lora_up
        to_tensor_opt(lora_down),           // lora_down (fused next-layer)
        to_tensor_opt(lora_act_out),        // lora_act_out (fused next-layer)
        Tensor{}, Tensor{}, Tensor{},       // norm_q / norm_k / rotary_emb — attention-GEMM only
        to_tensor_opt(bias),                // bias
        to_tensor_opt(smooth_factor),       // smooth_factor (fused next-layer)
        Tensor{}, Tensor{},                 // out_vk / out_linearattn — linear-attn only
        act_unsigned,
        std::move(lora_scales),
        fuse_silu,
        /* fp4 */ false,                    // FP4 disabled in kitchen port (see gemm_w4a4.cu patch)
        /* alpha */ 1.0f,                   // per-tensor alpha not used on Qwen W4A4 path
        Tensor{},                           // wcscales — FP4 only
        Tensor{}, Tensor{}, Tensor{},       // out_q / out_k / out_v — attention-GEMM only
        /* attn_tokens */ 0);
}

// --- quantize_w4a4_act_fuse_lora ------------------------------------------

void zgemm_quantize_w4a4_act_fuse_lora(
    nb::ndarray<nb::device::cuda> input,
    nb::ndarray<nb::device::cuda> output,
    nb::ndarray<nb::device::cuda> oscales,
    nb::ndarray<nb::device::cuda> lora_down,
    nb::ndarray<nb::device::cuda> lora_act_out,
    nb::object smooth,
    bool fuse_glu,
    bool fp4,
    uintptr_t stream_ptr)
{
    CUDAStreamContext stream_ctx(reinterpret_cast<cudaStream_t>(stream_ptr));

    nunchaku::kernels::quantize_w4a4_act_fuse_lora(
        to_tensor(std::move(input)),
        to_tensor(std::move(output)),
        to_tensor(std::move(oscales)),
        to_tensor(std::move(lora_down)),
        to_tensor(std::move(lora_act_out)),
        to_tensor_opt(smooth),
        fuse_glu,
        fp4);
}

} // namespace

void register_zgemm_bindings(nb::module_ &m) {
    m.def("nunchaku_gemm_w4a4", &zgemm_gemm_w4a4,
          nb::arg("act"),
          nb::arg("wgt"),
          nb::arg("out"),
          nb::arg("ascales"),
          nb::arg("wscales"),
          nb::arg("lora_act_in"),
          nb::arg("lora_up"),
          nb::arg("bias").none(),
          nb::arg("qout").none(),
          nb::arg("oscales").none(),
          nb::arg("lora_down").none(),
          nb::arg("lora_act_out").none(),
          nb::arg("smooth_factor").none(),
          nb::arg("act_unsigned"),
          nb::arg("fuse_silu"),
          nb::arg("lora_scales"),
          nb::arg("stream_ptr"),
          "Nunchaku zgemm W4A4 GEMM (SVDQuant-LoRA variant). Optional next-"
          "layer quantize-fusion via qout/oscales/lora_down/lora_act_out/"
          "smooth_factor; pass None to skip. Attention / linear-attn / FP4 "
          "paths are not exposed in kitchen's port.");

    m.def("nunchaku_quantize_w4a4_act_fuse_lora", &zgemm_quantize_w4a4_act_fuse_lora,
          nb::arg("input"),
          nb::arg("output"),
          nb::arg("oscales"),
          nb::arg("lora_down"),
          nb::arg("lora_act_out"),
          nb::arg("smooth").none(),
          nb::arg("fuse_glu"),
          nb::arg("fp4"),
          nb::arg("stream_ptr"),
          "Nunchaku zgemm W4A4 activation quantize with fused LoRA-down.");
}
