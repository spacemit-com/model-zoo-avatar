#pragma once
#include "feathertalk_service.h"
#include <functional>
#include <fstream>
#include <cstdio>
#include <utility>
namespace feathertalk::detail {
using HiddenRunner = std::function<std::vector<float>(const std::vector<float>&)>;
class FeatureStream {
public:
    explicit FeatureStream(HiddenRunner runner) : runner_(std::move(runner)) {}
    FeatureBatch Push(const std::int16_t* pcm, std::size_t samples, bool final);
    void Reset();
private:
    FeatureBatch Chunk(const std::int16_t*, std::size_t, bool);
    HiddenRunner runner_; std::vector<float> audio_;
    std::int64_t total_ = 0, emitted_ = 0;
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
private: std::ifstream stream_; std::uint64_t count_ = 0, read_ = 0;
};
}
