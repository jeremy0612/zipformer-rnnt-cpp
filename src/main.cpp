#include <iostream>
#include <string>
#include <chrono>
#include "streaming_asr.hpp"
#include "audio_utils.hpp"

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options] <wav_file>\n"
              << "\nOptions:\n"
              << "  --encoder   PATH   Encoder RKNN model (default: models/encoder.rknn)\n"
              << "  --decoder   PATH   Decoder RKNN model (default: models/decoder.rknn)\n"
              << "  --joiner    PATH   Joiner  RKNN model (default: models/joiner.rknn)\n"
              << "  --bpe       PATH   BPE model          (default: models/bpe.model)\n"
              << "  --chunk     N      Chunk size in frames (default: 32)\n"
              << "  --beam      N      Beam size            (default: 4)\n"
              << "  --streaming        Print partial results as they arrive\n";
}

int main(int argc, char** argv) {
    zipformer::StreamingConfig cfg;
    cfg.encoder_model_path = "models/encoder.rknn";
    cfg.decoder_model_path = "models/decoder.rknn";
    cfg.joiner_model_path  = "models/joiner.rknn";
    cfg.bpe_model_path     = "models/bpe.model";

    std::string wav_path;
    bool streaming_print = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--encoder" && i + 1 < argc) cfg.encoder_model_path = argv[++i];
        else if (arg == "--decoder" && i + 1 < argc) cfg.decoder_model_path = argv[++i];
        else if (arg == "--joiner"  && i + 1 < argc) cfg.joiner_model_path  = argv[++i];
        else if (arg == "--bpe"     && i + 1 < argc) cfg.bpe_model_path     = argv[++i];
        else if (arg == "--chunk"   && i + 1 < argc) cfg.chunk_size         = std::stoi(argv[++i]);
        else if (arg == "--beam"    && i + 1 < argc) cfg.beam_size          = std::stoi(argv[++i]);
        else if (arg == "--streaming") streaming_print = true;
        else if (arg[0] != '-') wav_path = arg;
        else { print_usage(argv[0]); return 1; }
    }

    if (wav_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    // Load audio
    auto samples = zipformer::load_wav(wav_path, cfg.sample_rate);
    std::cout << "[INFO] Loaded " << samples.size() << " samples ("
              << samples.size() / cfg.sample_rate << "s) from " << wav_path << "\n";

    // Build ASR
    zipformer::StreamingASR asr(cfg);

    if (streaming_print) {
        asr.set_callback([](const std::string& partial, bool is_final) {
            if (is_final)
                std::cout << "\r[FINAL] " << partial << "\n" << std::flush;
            else
                std::cout << "\r[...  ] " << partial << std::flush;
        });
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    asr.feed(samples.data(), static_cast<int>(samples.size()));
    std::string transcript = asr.flush();

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double audio_ms   = 1000.0 * samples.size() / cfg.sample_rate;
    double rtf        = elapsed_ms / audio_ms;

    if (!streaming_print)
        std::cout << transcript << "\n";

    std::cerr << "[PERF] audio=" << audio_ms << "ms  proc=" << elapsed_ms
              << "ms  RTF=" << rtf << "\n";

    return 0;
}
