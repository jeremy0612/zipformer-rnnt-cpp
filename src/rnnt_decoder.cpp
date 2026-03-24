#include "rnnt_decoder.hpp"
#include <rknn_api.h>
#include <stdexcept>

namespace zipformer {

struct RNNTDecoder::Impl {
    rknn_context ctx = 0;
    int output_dim   = 512;

    ~Impl() {
        if (ctx) rknn_destroy(ctx);
    }
};

RNNTDecoder::RNNTDecoder(const std::string& model_path, int /*num_threads*/)
    : impl_(new Impl)
{
    FILE* f = fopen(model_path.c_str(), "rb");
    if (!f) throw std::runtime_error("Cannot open decoder model: " + model_path);
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz);
    fread(buf.data(), 1, sz, f);
    fclose(f);

    int ret = rknn_init(&impl_->ctx, buf.data(), sz, 0, nullptr);
    if (ret != RKNN_SUCC)
        throw std::runtime_error("rknn_init decoder failed: " + std::to_string(ret));

    rknn_tensor_attr out_attr{};
    out_attr.index = 0;
    rknn_query(impl_->ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr));
    if (out_attr.n_dims >= 2)
        impl_->output_dim = out_attr.dims[out_attr.n_dims - 1];
}

RNNTDecoder::~RNNTDecoder() = default;

DecoderOutput RNNTDecoder::forward(const std::array<int64_t, 2>& context) {
    // Input: y [1, 2]  int64
    rknn_input inp{};
    inp.index        = 0;
    inp.type         = RKNN_TENSOR_INT64;
    inp.size         = 2 * sizeof(int64_t);
    inp.fmt          = RKNN_TENSOR_UNDEFINED;
    inp.buf          = const_cast<int64_t*>(context.data());
    inp.pass_through = 0;

    int ret = rknn_inputs_set(impl_->ctx, 1, &inp);
    if (ret != RKNN_SUCC)
        throw std::runtime_error("decoder rknn_inputs_set failed: " + std::to_string(ret));

    ret = rknn_run(impl_->ctx, nullptr);
    if (ret != RKNN_SUCC)
        throw std::runtime_error("decoder rknn_run failed: " + std::to_string(ret));

    DecoderOutput result;
    result.dim = impl_->output_dim;
    result.data.resize(impl_->output_dim);

    rknn_output out{};
    out.index        = 0;
    out.want_float   = 1;
    out.is_prealloc  = 1;
    out.buf          = result.data.data();
    out.size         = result.data.size() * sizeof(float);

    ret = rknn_outputs_get(impl_->ctx, 1, &out, nullptr);
    if (ret != RKNN_SUCC)
        throw std::runtime_error("decoder rknn_outputs_get failed: " + std::to_string(ret));

    rknn_outputs_release(impl_->ctx, 1, &out);
    return result;
}

int RNNTDecoder::output_dim() const { return impl_->output_dim; }

} // namespace zipformer
