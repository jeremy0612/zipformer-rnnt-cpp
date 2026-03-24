#include "streaming_asr.hpp"
#include "audio_utils.hpp"
#include "zipformer_encoder.hpp"
#include "rnnt_decoder.hpp"
#include "rnnt_joiner.hpp"
#include "bpe_tokenizer.hpp"
#include <algorithm>
#include <cmath>

namespace zipformer {

struct StreamingASR::Impl {
    StreamingConfig cfg;
    TranscriptCallback callback;

    ZipformerEncoder encoder;
    RNNTDecoder      decoder;
    RNNTJoiner       joiner;
    BPETokenizer     tokenizer;

    // Audio buffer (float samples, 16kHz)
    std::vector<float> audio_buf;

    // Greedy search state: last 2 tokens (2-gram context)
    std::array<int64_t, 2> context;
    std::vector<int> emitted_tokens;

    // Samples per chunk (before fbank)
    int chunk_samples;
    int frame_shift_samples;

    Impl(const StreamingConfig& c)
        : cfg(c)
        , encoder(c.encoder_model_path, c.num_threads)
        , decoder(c.decoder_model_path, c.num_threads)
        , joiner (c.joiner_model_path,  c.num_threads)
        , tokenizer(c.bpe_model_path)
    {
        frame_shift_samples = cfg.sample_rate / 100; // 10ms = 160 samples @16kHz
        // chunk_size frames * frame_shift
        chunk_samples = cfg.chunk_size * frame_shift_samples;
        int blank = tokenizer.blank_id();
        context = {blank, blank};
    }

    // Process one chunk of audio
    void process_chunk(const float* samples, int n) {
        // Compute fbank features: [T, 80]
        std::vector<float> chunk_vec(samples, samples + n);
        auto feats = compute_fbank(chunk_vec, cfg.sample_rate);
        if (feats.empty()) return;

        int T = feats.size() / 80;
        EncoderState enc_state;
        auto enc_out = encoder.forward(feats, T, enc_state);
        if (enc_out.time_steps == 0) return;

        // Greedy transducer search
        for (int t = 0; t < enc_out.time_steps; ++t) {
            // encoder frame: enc_out.data[t * enc_out.dim .. (t+1)*enc_out.dim]
            std::vector<float> enc_frame(
                enc_out.data.begin() + t * enc_out.dim,
                enc_out.data.begin() + (t + 1) * enc_out.dim);

            // Keep emitting tokens until blank
            while (true) {
                auto dec_out = decoder.forward(context);
                auto logits  = joiner.forward(enc_frame, dec_out.data);

                // Argmax
                int best = 0;
                for (int i = 1; i < (int)logits.size(); ++i)
                    if (logits[i] > logits[best]) best = i;

                if (best == tokenizer.blank_id())
                    break;  // blank → advance to next encoder frame

                // Emit token
                emitted_tokens.push_back(best);
                context[0] = context[1];
                context[1] = best;

                if (callback) {
                    std::string partial = tokenizer.decode(emitted_tokens);
                    callback(partial, false);
                }
            }
        }
    }
};

StreamingASR::StreamingASR(const StreamingConfig& cfg)
    : impl_(new Impl(cfg)) {}

StreamingASR::~StreamingASR() = default;

void StreamingASR::feed(const float* samples, int n) {
    auto& buf = impl_->audio_buf;
    buf.insert(buf.end(), samples, samples + n);

    // Process complete chunks
    while ((int)buf.size() >= impl_->chunk_samples) {
        impl_->process_chunk(buf.data(), impl_->chunk_samples);
        buf.erase(buf.begin(), buf.begin() + impl_->chunk_samples);
    }
}

void StreamingASR::feed(const int16_t* samples, int n) {
    auto floats = pcm16_to_float(samples, n);
    feed(floats.data(), n);
}

std::string StreamingASR::flush() {
    // Process any remaining audio
    if (!impl_->audio_buf.empty()) {
        impl_->process_chunk(impl_->audio_buf.data(),
                             static_cast<int>(impl_->audio_buf.size()));
        impl_->audio_buf.clear();
    }

    std::string result = impl_->tokenizer.decode(impl_->emitted_tokens);

    if (impl_->callback)
        impl_->callback(result, true);

    return result;
}

void StreamingASR::set_callback(TranscriptCallback cb) {
    impl_->callback = std::move(cb);
}

void StreamingASR::reset() {
    impl_->audio_buf.clear();
    impl_->emitted_tokens.clear();
    // Init context with blank tokens
    int blank = impl_->tokenizer.blank_id();
    impl_->context = {blank, blank};
}

} // namespace zipformer
