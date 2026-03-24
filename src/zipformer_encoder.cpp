#include "zipformer_encoder.hpp"
#include <rknn_api.h>
#include <stdexcept>
#include <cstring>

namespace zipformer {

void EncoderState::reset() {
    processed_frames = 0;
}

struct ZipformerEncoder::Impl {
    rknn_context ctx = 0;
    int input_dim    = 80;
    int output_dim   = 512;
    int subsampling  = 4;

    ~Impl() {
        if (ctx) rknn_destroy(ctx);
    }
};

ZipformerEncoder::ZipformerEncoder(const std::string& model_path, int num_threads)
    : impl_(new Impl)
{
    // Load model file
    FILE* f = fopen(model_path.c_str(), "rb");
    if (!f) throw std::runtime_error("Cannot open encoder model: " + model_path);
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz);
    fread(buf.data(), 1, sz, f);
    fclose(f);

    int ret = rknn_init(&impl_->ctx, buf.data(), sz, 0, nullptr);
    if (ret != RKNN_SUCC)
        throw std::runtime_error("rknn_init encoder failed: " + std::to_string(ret));

    // Query output shape to confirm output_dim
    rknn_tensor_attr out_attr{};
    out_attr.index = 0;
    rknn_query(impl_->ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr));
    if (out_attr.n_dims >= 3)
        impl_->output_dim = out_attr.dims[2];

    (void)num_threads; // core affinity can be set via rknn_set_core_mask if needed
}

ZipformerEncoder::~ZipformerEncoder() = default;

EncoderOutput ZipformerEncoder::forward(const std::vector<float>& features,
                                        int num_frames,
                                        EncoderState& state)
{
    // Input 0: x       [1, T, 80]  float32
    // Input 1: x_lens  [1]         int64
    rknn_input inputs[2]{};

    inputs[0].index        = 0;
    inputs[0].type         = RKNN_TENSOR_FLOAT32;
    inputs[0].size         = num_frames * impl_->input_dim * sizeof(float);
    inputs[0].fmt          = RKNN_TENSOR_NHWC;
    inputs[0].buf          = const_cast<float*>(features.data());
    inputs[0].pass_through = 0;

    int64_t x_lens = num_frames;
    inputs[1].index        = 1;
    inputs[1].type         = RKNN_TENSOR_INT64;
    inputs[1].size         = sizeof(int64_t);
    inputs[1].fmt          = RKNN_TENSOR_UNDEFINED;
    inputs[1].buf          = &x_lens;
    inputs[1].pass_through = 0;

    int ret = rknn_inputs_set(impl_->ctx, 2, inputs);
    if (ret != RKNN_SUCC)
        throw std::runtime_error("encoder rknn_inputs_set failed: " + std::to_string(ret));

    ret = rknn_run(impl_->ctx, nullptr);
    if (ret != RKNN_SUCC)
        throw std::runtime_error("encoder rknn_run failed: " + std::to_string(ret));

    // Query output shape
    rknn_tensor_attr out_attr{};
    out_attr.index = 0;
    rknn_query(impl_->ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr));
    int T_out = (out_attr.n_dims >= 2) ? out_attr.dims[1] : num_frames / impl_->subsampling;

    EncoderOutput result;
    result.time_steps = T_out;
    result.dim        = impl_->output_dim;
    result.data.resize(T_out * impl_->output_dim);

    rknn_output outputs[1]{};
    outputs[0].index    = 0;
    outputs[0].want_float = 1;
    outputs[0].is_prealloc = 1;
    outputs[0].buf      = result.data.data();
    outputs[0].size     = result.data.size() * sizeof(float);

    ret = rknn_outputs_get(impl_->ctx, 1, outputs, nullptr);
    if (ret != RKNN_SUCC)
        throw std::runtime_error("encoder rknn_outputs_get failed: " + std::to_string(ret));

    rknn_outputs_release(impl_->ctx, 1, outputs);

    state.processed_frames += num_frames;
    return result;
}

int ZipformerEncoder::input_dim()   const { return impl_->input_dim; }
int ZipformerEncoder::output_dim()  const { return impl_->output_dim; }
int ZipformerEncoder::subsampling() const { return impl_->subsampling; }

} // namespace zipformer
