#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace zipformer {

// Load mono 16kHz WAV (resamples if needed)
std::vector<float> load_wav(const std::string& path, int target_sr = 16000);

// Compute 80-dim log-mel filterbank features
// Returns [num_frames * 80], num_frames = (num_samples - 400) / 160 + 1
std::vector<float> compute_fbank(const std::vector<float>& samples,
                                 int sample_rate = 16000,
                                 int num_mel_bins = 80,
                                 int frame_length_ms = 25,
                                 int frame_shift_ms  = 10);

// Convert int16 PCM to float [-1, 1]
std::vector<float> pcm16_to_float(const int16_t* data, int num_samples);

} // namespace zipformer
