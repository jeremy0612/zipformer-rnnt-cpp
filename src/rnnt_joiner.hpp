#pragma once

#include <string>
#include <vector>
#include <memory>

namespace zipformer {

class RNNTJoiner {
public:
    explicit RNNTJoiner(const std::string& model_path, int num_threads = 2);
    ~RNNTJoiner();

    // encoder_out: [encoder_dim], decoder_out: [decoder_dim]
    // Returns log-softmax logits: [vocab_size]
    std::vector<float> forward(const std::vector<float>& encoder_out,
                               const std::vector<float>& decoder_out);

    int vocab_size() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace zipformer
