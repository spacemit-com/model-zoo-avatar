/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "internal.h"

namespace SpacemiT::avatar::detail {

void FeatureStream::Reset() {
    audio_.clear();
    total_ = emitted_ = 0;
    finalized_ = false;
}

FeatureBatch FeatureStream::Push(const std::int16_t* pcm, std::size_t n, bool final) {
    if (finalized_) {
        throw std::runtime_error("Stream finalized; reset before adding audio");
    }
    if (n && !pcm) {
        throw std::invalid_argument("PCM pointer is null");
    }
    if (n > 4 * 1024 * 1024) {
        throw std::invalid_argument("PCM push exceeds 4M samples; split input");
    }
    // A failed inference must not consume PCM or advance the cursor. Callers
    // may retry the whole push, including pushes split into several chunks.
    auto previous_audio = audio_;
    const auto previous_total = total_;
    const auto previous_emitted = emitted_;
    try {
        FeatureBatch out;
        constexpr std::size_t step = 12000;
        for (std::size_t i = 0; i < n; i += step) {
            auto batch = Chunk(pcm + i, std::min(step, n - i), final && i + step >= n);
            out.values.insert(out.values.end(), batch.values.begin(), batch.values.end());
            out.frame_ids.insert(out.frame_ids.end(), batch.frame_ids.begin(), batch.frame_ids.end());
        }
        if (!n) {
            out = Chunk(nullptr, 0, final);
        }
        finalized_ = final;
        return out;
    } catch (...) {
        audio_.swap(previous_audio);
        total_ = previous_total;
        emitted_ = previous_emitted;
        throw;
    }
}

// Runs the model over the latest 20000 samples (1.25 s); Push() splits new
// audio into chunks of at most 12000 samples (0.75 s). Two tokens make one
// model frame, which matches one 40 ms video frame. The window of video frame f
// holds model frames f-4 to f+3. Unless the push is final, the last 4 model
// frames are held back and recomputed by the next call with more context.
FeatureBatch FeatureStream::Chunk(const std::int16_t* pcm, std::size_t n, bool final) {
    for (std::size_t i = 0; i < n; ++i) {
        audio_.push_back(static_cast<float>(pcm[i]) / 32768.0f);
    }
    total_ += static_cast<std::int64_t>(n);
    if (audio_.size() > 20000) {
        audio_.erase(audio_.begin(), audio_.end() - 20000);
    }
    if (audio_.size() < 400) {
        return {};
    }
    double sum = 0;
    for (float x : audio_) {
        sum += x;
    }
    const float mean = static_cast<float>(sum / audio_.size());
    double squares = 0;
    for (float x : audio_) {
        const float d = x - mean;
        squares += d * d;
    }
    const float variance = static_cast<float>(squares / audio_.size());
    const double scale = std::sqrt(static_cast<double>(variance) + 1e-7);
    std::vector<float> normalized(audio_.size());
    for (std::size_t i = 0; i < audio_.size(); ++i) {
        normalized[i] = static_cast<float>((audio_[i] - mean) / scale);
    }
    auto hidden = runner_(normalized);
    if (hidden.size() % 1024) {
        throw std::runtime_error("HuBERT hidden width must be 1024");
    }
    const auto tokens = hidden.size() / 1024;
    const auto frames = static_cast<std::int64_t>((tokens - (tokens % 2)) / 2);
    if (!frames) {
        return {};
    }
    if (std::sqrt(variance) < 1e-3f) {
        std::fill(hidden.begin(), hidden.end(), 0.0f);
    }
    const auto base = (total_ - static_cast<std::int64_t>(audio_.size())) / 640;
    const auto ready = final ? base + frames : std::max(base, base + frames - 4);
    const auto global_end = final ? (total_ + 639) / 640 : total_ / 640;
    const auto end = std::min(ready, global_end);
    const auto begin = std::max(emitted_, base);
    FeatureBatch out;
    for (auto f = begin; f < end; ++f) {
        out.frame_ids.push_back(f);
        for (int w = 0; w < 8; ++w) {
            const auto local = f - base + w - 4;
            if (local < 0 || local >= frames) {
                out.values.insert(out.values.end(), 2048, 0.0f);
            } else {
                out.values.insert(out.values.end(), hidden.begin() + local * 2048, hidden.begin() + (local + 1) * 2048);
            }
        }
    }
    emitted_ = std::max(emitted_, end);
    return out;
}

}  // namespace SpacemiT::avatar::detail
