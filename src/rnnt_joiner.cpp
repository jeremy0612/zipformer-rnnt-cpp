#include "rnnt_joiner.hpp"
#include <rknn_api.h>
#include <stdexcept>

namespace zipformer {

struct RNNTJoiner::Impl {
    rknn_context ctx = 0;
    int vocab_size_  = 2000;

    ~Impl() {
        if (ctx) rknn_destroy(ctx);
    }
};

RNNTJoiner::RNNTJoiner(const std::string& model_path, int /*num_threads*/)
    : impl_(new Impl)
{
    FILE* f = fopen(model_path.c_str(), "rb");
    if (!f) throw std::runtime_error("Cannot open joiner model: " + model_path);
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz);
    fread(buf.data(), 1, sz, f);
    fclose(f);

    int ret = rknn_init(&impl_->ctx, buf.data(), sz, 0, nullptr);
    if (ret != RKNN_SUCC)
        throw std::runtime_error("rknn_init joiner failed: " + std::to_string(ret));

    rknn_tensor_attr out_attr{};
    out_attr.index = 0;
    rknn_query(impl_->ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr));
    if (out_attr.n_dims >= 1)
        impl_->vocab_size_ = out_attr.dims[out_attr.n_dims - 1];
}

RNNTJoiner::~RNNTJoiner() = default;

std::vector<float> RNNTJoiner::forward(const std::vector<float>& encoder_out,
                                       const std::vector<float>& decoder_out) {
    // Inputs: encoder_out [1, 512], decoder_out [1, 512]
    rknn_input inputs[2]{};
    inputs[0].index        = 0;
    inputs[0].type         = RKNN_TENSOR_FLOAT32;
    inputs[0].size         = encoder_out.size() * sizeof(float);
    inputs[0].fmt          = RKNN_TENSOR_UNDEFINED;
    inputs[0].buf          = const_cast<float*>(encoder_out.data());
    inputs[0].pass_through = 0;

    inputs[1].index        = 1;
    inputs[1].type         = RKNN_TENSOR_FLOAT32;
    inputs[1].size         = decoder_out.size() * sizeof(float);
    inputs[1].fmt          = RKNN_TENSOR_UNDEFINED;
    inputs[1].buf          = const_cast<float*>(decoder_out.data());
    inputs[1].pass_through = 0;

    int ret = rknn_inputs_set(impl_->ctx, 2, inputs);
    if (ret != RKNN_SUCC)
        throw std::runtime_error("joiner rknn_inputs_set failed: " + std::to_string(ret));

    ret = rknn_run(impl_->ctx, nullptr);
    if (ret != RKNN_SUCC)
        throw std::runtime_error("joiner rknn_run failed: " + std::to_string(ret));

    std::vector<float> logits(impl_->vocab_size_);

    rknn_output out{};
    out.index        = 0;
    out.want_float   = 1;
    out.is_prealloc  = 1;
    out.buf          = logits.data();
    out.size         = logits.size() * sizeof(float);

    ret = rknn_outputs_get(impl_->ctx, 1, &out, nullptr);
    if (ret != RKNN_SUCC)
        throw std::runtime_error("joiner rknn_outputs_get failed: " + std::to_string(ret));

    rknn_outputs_release(impl_->ctx, 1, &out);
    return logits;
}

int RNNTJoiner::vocab_size() const { return impl_->vocab_size_; }

} // namespace zipformer
