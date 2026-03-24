#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace zipformer {

struct StreamingConfig {
    // RKNN model paths
    std::string encoder_model_path;
    std::string decoder_model_path;
    std::string joiner_model_path;
    std::string bpe_model_path;

    // Streaming parameters
    int sample_rate       = 16000;
    int chunk_size        = 32;   // frames per chunk (encoder subsampling * this)
    int left_context      = 64;   // left context frames
    int right_context     = 4;    // right context frames (look-ahead)

    // Beam search
    int beam_size         = 4;
    float blank_penalty   = 0.0f;

    int num_threads       = 2;
};

// Called with partial transcript as each chunk is decoded
using TranscriptCallback = std::function<void(const std::string& partial, bool is_final)>;

class StreamingASR {
public:
    explicit StreamingASR(const StreamingConfig& cfg);
    ~StreamingASR();

    // Feed raw 16-bit PCM samples (mono, 16kHz)
    void feed(const int16_t* samples, int num_samples);

    // Feed float samples (mono, 16kHz, range [-1, 1])
    void feed(const float* samples, int num_samples);

    // Flush remaining audio and finalize transcript
    std::string flush();

    // Register callback for streaming partial results
    void set_callback(TranscriptCallback cb);

    // Reset decoder state (start new utterance)
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace zipformer
