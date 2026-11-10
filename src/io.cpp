/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "internal.h"

namespace {

constexpr std::uint64_t kMaxWavData = 512 * 1024 * 1024;

std::uint64_t ReadLE(const unsigned char* bytes, int size) {
    std::uint64_t value = 0;
    for (int i = 0; i < size; ++i) {
        value |= static_cast<std::uint64_t>(bytes[i]) << (8 * i);
    }
    return value;
}

void WriteLE(unsigned char* bytes, std::uint64_t value, int size) {
    for (int i = 0; i < size; ++i) {
        bytes[i] = static_cast<unsigned char>(value >> (8 * i));
    }
}

void ReadExact(std::istream& stream, void* data, std::size_t size) {
    if (!stream.read(static_cast<char*>(data), static_cast<std::streamsize>(size))) {
        throw std::runtime_error("Truncated input file");
    }
}

// PCM samples and feature windows are stored little-endian and copied as is.
void RequireLittleEndian() {
    const std::uint16_t probe = 1;
    if (*reinterpret_cast<const unsigned char*>(&probe) != 1) {
        throw std::runtime_error("Little-endian host required");
    }
}

std::uint64_t FrameCount(std::uint64_t samples) {
    return samples / 640 + (samples % 640 != 0);
}

}  // namespace

namespace SpacemiT::avatar {

std::vector<std::int16_t> ReadWav16k(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Cannot open WAV: " + path);
    }
    std::array<unsigned char, 12> head{};
    ReadExact(stream, head.data(), head.size());
    if (std::memcmp(head.data(), "RIFF", 4) || std::memcmp(head.data() + 8, "WAVE", 4)) {
        throw std::runtime_error("Expected RIFF WAVE");
    }
    const auto file_size = std::filesystem::file_size(path);
    const auto riff_end = ReadLE(head.data() + 4, 4) + 8;
    if (riff_end < 12 || riff_end > file_size) {
        throw std::runtime_error("Invalid RIFF length");
    }
    bool has_format = false;
    std::vector<std::int16_t> pcm;
    while (static_cast<std::uint64_t>(stream.tellg()) < riff_end) {
        if (riff_end - static_cast<std::uint64_t>(stream.tellg()) < 8) {
            throw std::runtime_error("Truncated WAV chunk header");
        }
        std::array<unsigned char, 8> chunk{};
        ReadExact(stream, chunk.data(), chunk.size());
        const auto size = ReadLE(chunk.data() + 4, 4);
        const auto at = static_cast<std::uint64_t>(stream.tellg());
        if (size + (size & 1) > riff_end - at) {
            throw std::runtime_error("Invalid WAV chunk length");
        }
        if (!std::memcmp(chunk.data(), "fmt ", 4)) {
            if (size < 16) {
                throw std::runtime_error("Short WAV fmt");
            }
            std::array<unsigned char, 16> format{};
            ReadExact(stream, format.data(), format.size());
            const bool pcm16_mono_16k = ReadLE(format.data(), 2) == 1 && ReadLE(format.data() + 2, 2) == 1 &&
                ReadLE(format.data() + 4, 4) == 16000 && ReadLE(format.data() + 8, 4) == 32000 &&
                ReadLE(format.data() + 12, 2) == 2 && ReadLE(format.data() + 14, 2) == 16;
            if (!pcm16_mono_16k) {
                throw std::runtime_error("WAV must be 16kHz mono PCM16");
            }
            has_format = true;
            stream.seekg(static_cast<std::streamoff>(size - 16), std::ios::cur);
        } else if (!std::memcmp(chunk.data(), "data", 4)) {
            if (!has_format || size % 2 || size > kMaxWavData || pcm.size() * 2 + size > kMaxWavData) {
                throw std::runtime_error("Invalid or excessive WAV data");
            }
            const auto before = pcm.size();
            pcm.resize(before + size / 2);
            RequireLittleEndian();
            ReadExact(stream, pcm.data() + before, size);
        } else {
            stream.seekg(static_cast<std::streamoff>(size), std::ios::cur);
        }
        if (size & 1) {
            stream.seekg(1, std::ios::cur);
        }
    }
    if (!has_format || pcm.empty()) {
        throw std::runtime_error("WAV contains no PCM audio");
    }
    return pcm;
}

}  // namespace SpacemiT::avatar

namespace SpacemiT::avatar::detail {

FeatureWriter::FeatureWriter(const std::string& path) {
    RequireLittleEndian();
    stream_.reset(std::fopen(path.c_str(), "wbx"));
    if (!stream_) {
        throw std::runtime_error("Cannot exclusively create feature output: " + path);
    }
    // Placeholder header; Finish() writes the real one once the count is known.
    std::array<char, 64> header{};
    if (std::fwrite(header.data(), 1, header.size(), stream_.get()) != header.size()) {
        throw std::runtime_error("Feature header write failed");
    }
}

void FeatureWriter::Append(const FeatureBatch& batch) {
    if (!stream_) {
        throw std::runtime_error("Feature output already finalized");
    }
    if (batch.values.size() != batch.size() * kWindowElements) {
        throw std::runtime_error("Invalid feature batch");
    }
    for (std::size_t i = 0; i < batch.size(); ++i) {
        if (batch.frame_ids[i] != static_cast<std::int64_t>(count_ + i)) {
            throw std::runtime_error("Feature stream has missing or duplicate frame ids");
        }
    }
    if (!batch.values.empty() &&
        std::fwrite(batch.values.data(), sizeof(float), batch.values.size(), stream_.get()) != batch.values.size()) {
        throw std::runtime_error("Failed to write features");
    }
    count_ += batch.size();
}

void FeatureWriter::Finish(std::uint64_t samples) {
    if (!stream_ || count_ != FrameCount(samples)) {
        throw std::runtime_error("Invalid feature timeline");
    }
    std::array<unsigned char, 64> header{};
    std::memcpy(header.data(), "FTF1", 4);
    WriteLE(header.data() + 4, 1, 4);
    WriteLE(header.data() + 8, 16000, 4);
    WriteLE(header.data() + 12, 25, 4);
    WriteLE(header.data() + 16, count_, 8);
    WriteLE(header.data() + 24, samples, 8);
    if (std::fseek(stream_.get(), 0, SEEK_SET) != 0 ||
        std::fwrite(header.data(), 1, header.size(), stream_.get()) != header.size() ||
        std::fflush(stream_.get()) != 0) {
        throw std::runtime_error("Feature header write failed");
    }
    stream_.reset();
}

FeatureReader::FeatureReader(const std::string& path) : stream_(path, std::ios::binary) {
    RequireLittleEndian();
    if (!stream_) {
        throw std::runtime_error("Cannot open feature file");
    }
    std::array<unsigned char, 64> header{};
    ReadExact(stream_, header.data(), header.size());
    if (std::memcmp(header.data(), "FTF1", 4) || ReadLE(header.data() + 4, 4) != 1 ||
        ReadLE(header.data() + 8, 4) != 16000 || ReadLE(header.data() + 12, 4) != 25) {
        throw std::runtime_error("Invalid FTF contract");
    }
    count_ = ReadLE(header.data() + 16, 8);
    constexpr std::uint64_t kWindowBytes = kWindowElements * 4;
    if (count_ > (std::numeric_limits<std::uint64_t>::max() - 64) / kWindowBytes ||
        std::filesystem::file_size(path) != 64 + count_ * kWindowBytes) {
        throw std::runtime_error("FTF file size mismatch");
    }
    if (count_ != FrameCount(ReadLE(header.data() + 24, 8))) {
        throw std::runtime_error("Invalid FTF timeline");
    }
}

bool FeatureReader::Next(std::vector<float>& window) {
    if (read_ == count_) {
        return false;
    }
    window.resize(kWindowElements);
    ReadExact(stream_, window.data(), window.size() * 4);
    ++read_;
    return true;
}

}  // namespace SpacemiT::avatar::detail
