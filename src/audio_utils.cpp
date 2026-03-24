#include "audio_utils.hpp"
#include <sndfile.h>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <cstring>

namespace zipformer {

std::vector<float> load_wav(const std::string& path, int target_sr) {
    SF_INFO info{};
    SNDFILE* sf = sf_open(path.c_str(), SFM_READ, &info);
    if (!sf)
        throw std::runtime_error("Cannot open: " + path);

    std::vector<float> raw(info.frames * info.channels);
    sf_readf_float(sf, raw.data(), info.frames);
    sf_close(sf);

    // Mix down to mono
    std::vector<float> mono(info.frames);
    if (info.channels == 1) {
        mono = std::move(raw);
    } else {
        for (long i = 0; i < info.frames; ++i) {
            float s = 0;
            for (int c = 0; c < info.channels; ++c)
                s += raw[i * info.channels + c];
            mono[i] = s / info.channels;
        }
    }

    if (info.samplerate == target_sr)
        return mono;

    // Simple linear resampling
    double ratio = static_cast<double>(target_sr) / info.samplerate;
    int out_len = static_cast<int>(mono.size() * ratio);
    std::vector<float> resampled(out_len);
    for (int i = 0; i < out_len; ++i) {
        double src = i / ratio;
        int idx = static_cast<int>(src);
        float frac = static_cast<float>(src - idx);
        float a = mono[idx];
        float b = (idx + 1 < (int)mono.size()) ? mono[idx + 1] : a;
        resampled[i] = a + frac * (b - a);
    }
    return resampled;
}

std::vector<float> pcm16_to_float(const int16_t* data, int n) {
    std::vector<float> out(n);
    for (int i = 0; i < n; ++i)
        out[i] = data[i] / 32768.0f;
    return out;
}

// ── Log-mel filterbank ────────────────────────────────────────────────────────

static std::vector<float> hamming_window(int N) {
    std::vector<float> w(N);
    for (int i = 0; i < N; ++i)
        w[i] = 0.54f - 0.46f * std::cos(2.0f * M_PI * i / (N - 1));
    return w;
}

// Naive DFT for frame_len <= 512 (replace with FFT lib if needed)
static void compute_power_spectrum(const float* frame, int frame_len,
                                   std::vector<float>& power, int fft_len) {
    power.assign(fft_len / 2 + 1, 0.0f);
    for (int k = 0; k <= fft_len / 2; ++k) {
        float re = 0, im = 0;
        for (int n = 0; n < frame_len; ++n) {
            float angle = 2.0f * M_PI * k * n / fft_len;
            re += frame[n] * std::cos(angle);
            im -= frame[n] * std::sin(angle);
        }
        power[k] = re * re + im * im;
    }
}

static float hz_to_mel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}
static float mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

std::vector<float> compute_fbank(const std::vector<float>& samples,
                                 int sample_rate,
                                 int num_mel_bins,
                                 int frame_length_ms,
                                 int frame_shift_ms) {
    int frame_len  = sample_rate * frame_length_ms / 1000;  // 400
    int frame_shift = sample_rate * frame_shift_ms  / 1000;  // 160
    int fft_len    = 1;
    while (fft_len < frame_len) fft_len <<= 1;              // 512

    auto window = hamming_window(frame_len);

    // Build mel filterbank
    float mel_low  = hz_to_mel(20.0f);
    float mel_high = hz_to_mel(sample_rate / 2.0f);
    int num_fft_bins = fft_len / 2 + 1;

    std::vector<float> mel_pts(num_mel_bins + 2);
    for (int i = 0; i < num_mel_bins + 2; ++i) {
        float mel = mel_low + (mel_high - mel_low) * i / (num_mel_bins + 1);
        mel_pts[i] = mel_to_hz(mel) * fft_len / sample_rate;
    }

    // filterbank[m][k] = weight of fft bin k for mel filter m
    std::vector<std::vector<float>> fb(num_mel_bins, std::vector<float>(num_fft_bins, 0.0f));
    for (int m = 0; m < num_mel_bins; ++m) {
        float left  = mel_pts[m];
        float center= mel_pts[m + 1];
        float right = mel_pts[m + 2];
        for (int k = 0; k < num_fft_bins; ++k) {
            if (k >= left && k <= center)
                fb[m][k] = (k - left) / (center - left);
            else if (k > center && k <= right)
                fb[m][k] = (right - k) / (right - center);
        }
    }

    int num_frames = ((int)samples.size() - frame_len) / frame_shift + 1;
    if (num_frames <= 0)
        return {};

    std::vector<float> features(num_frames * num_mel_bins);
    std::vector<float> framed(frame_len);
    std::vector<float> power;

    for (int t = 0; t < num_frames; ++t) {
        int offset = t * frame_shift;
        for (int i = 0; i < frame_len; ++i)
            framed[i] = samples[offset + i] * window[i];

        compute_power_spectrum(framed.data(), frame_len, power, fft_len);

        for (int m = 0; m < num_mel_bins; ++m) {
            float energy = 0;
            for (int k = 0; k < num_fft_bins; ++k)
                energy += fb[m][k] * power[k];
            features[t * num_mel_bins + m] = std::log(std::max(energy, 1e-10f));
        }
    }

    return features;
}

} // namespace zipformer
