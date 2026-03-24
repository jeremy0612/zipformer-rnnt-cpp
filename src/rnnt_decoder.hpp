#pragma once

#include <string>
#include <vector>
#include <array>
#include <memory>

namespace zipformer {

struct DecoderOutput {
    // decoder_out: [512]  (flat, batch=1 squeezed)
    std::vector<float> data;
    int dim = 0;
};

class RNNTDecoder {
public:
    explicit RNNTDecoder(const std::string& model_path, int num_threads = 2);
    ~RNNTDecoder();

    // context: last 2 token ids (2-gram), e.g. {blank_id, last_token}
    // decoder_out: [512]
    DecoderOutput forward(const std::array<int64_t, 2>& context);

    int output_dim() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace zipformer
