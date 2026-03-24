#pragma once

#include <string>
#include <vector>
#include <memory>

namespace zipformer {

// Encoder state for streaming (cached attention keys/values)
struct EncoderState {
    // Each element corresponds to one encoder stack layer's cache
    // Shape: [num_layers][cache_size] — managed by the encoder
    std::vector<std::vector<float>> cached_key;
    std::vector<std::vector<float>> cached_val;
    std::vector<std::vector<float>> cached_nonlin_attn;
    std::vector<std::vector<float>> cached_conv1;
    std::vector<std::vector<float>> cached_conv2;
    int processed_frames = 0;

    void reset();
};

struct EncoderOutput {
    // encoder_out: [1, T, encoder_dim]
    std::vector<float> data;
    int time_steps = 0;
    int dim        = 0;
};

class ZipformerEncoder {
public:
    explicit ZipformerEncoder(const std::string& model_path, int num_threads = 2);
    ~ZipformerEncoder();

    // features: [1, T+context, 80] log-mel filterbank
    // Returns encoder output and updated state
    EncoderOutput forward(const std::vector<float>& features,
                          int num_frames,
                          EncoderState& state);

    int input_dim()    const;  // 80 (mel bins)
    int output_dim()   const;  // encoder output dim
    int subsampling()  const;  // subsampling factor (e.g. 4 or 8)

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace zipformer
