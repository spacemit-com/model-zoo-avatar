/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef INTERNAL_H
#define INTERNAL_H

#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "avatar_service.h"

namespace SpacemiT::avatar::detail {

using HiddenRunner = std::function<std::vector<float>(const std::vector<float>&)>;

// Turns 16 kHz PCM into 25 fps feature windows. The runner maps normalized
// audio to FeatherHuBERT hidden states [tokens, 1024].
class FeatureStream {
public:
    explicit FeatureStream(HiddenRunner runner) : runner_(std::move(runner)) {}
    FeatureBatch Push(const std::int16_t* pcm, std::size_t samples, bool final);
    void Reset();

private:
    FeatureBatch Chunk(const std::int16_t* pcm, std::size_t samples, bool final);

    HiddenRunner runner_;
    std::vector<float> audio_;
    std::int64_t total_ = 0;
    std::int64_t emitted_ = 0;
    bool finalized_ = false;
};

// FTF1: 64-byte little-endian header (magic, version, sample rate, fps, count,
// unique PCM sample count); fixed 65536-byte float32 feature windows follow.
class FeatureWriter {
public:
    explicit FeatureWriter(const std::string& path);
    void Append(const FeatureBatch& batch);
    void Finish(std::uint64_t samples);

private:
    struct CloseFile {
        void operator()(std::FILE* file) const noexcept { std::fclose(file); }
    };
    std::unique_ptr<std::FILE, CloseFile> stream_;
    std::uint64_t count_ = 0;
};

class FeatureReader {
public:
    explicit FeatureReader(const std::string& path);
    bool Next(std::vector<float>& window);
    std::uint64_t Count() const { return count_; }

private:
    std::ifstream stream_;
    std::uint64_t count_ = 0;
    std::uint64_t read_ = 0;
};

}  // namespace SpacemiT::avatar::detail

#endif  // INTERNAL_H
