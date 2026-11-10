/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * avatar_file_demo: render a persona speaking a WAV file with the public API.
 *
 * Models and assets default to the downloader cache:
 *   ~/.cache/models/avatar/{feather_hubert_dynamic.onnx,<persona>_int8.onnx}
 *   ~/.cache/assets/avatar/<persona>/{full_body_img,landmarks}
 *   ~/.cache/assets/avatar/audio/<persona>_test.wav
 *
 * The demo runs frame by frame for clarity; avatar_worker renders the same
 * frames with a pipeline.
 */

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "avatar_service.h"

namespace fs = std::filesystem;
namespace avatar = SpacemiT::avatar;

namespace {

struct Arguments {
    std::string persona = "feifan";
    std::string audio;
    std::string output;
    std::string provider = "spacemit";
    std::size_t max_frames = 0;
};

void PrintUsage() {
    std::cout << "Usage: avatar_file_demo [--persona feifan|zihan] [--audio WAV] [--output DIR]\n"
        "                        [--provider spacemit|cpu] [--max-frames N]\n";
}

bool ParseArguments(int argc, char** argv, Arguments* args) {
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "-h" || key == "--help") {
            PrintUsage();
            return false;
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("Missing value for " + key);
        }
        const std::string value = argv[++i];
        if (key == "--persona") {
            args->persona = value;
        } else if (key == "--audio") {
            args->audio = value;
        } else if (key == "--output") {
            args->output = value;
        } else if (key == "--provider") {
            args->provider = value;
        } else if (key == "--max-frames") {
            args->max_frames = std::stoul(value);
        } else {
            throw std::invalid_argument("Unknown option " + key);
        }
    }
    return true;
}

fs::path CacheRoot() {
    const char* home = std::getenv("HOME");
    if (!home) {
        throw std::runtime_error("HOME unset");
    }
    return fs::path(home) / ".cache";
}

std::vector<std::pair<fs::path, fs::path>> ListFrames(const fs::path& dataset) {
    std::vector<std::pair<std::uint64_t, std::pair<fs::path, fs::path>>> numbered;
    for (const auto& entry : fs::directory_iterator(dataset / "full_body_img")) {
        const auto stem = entry.path().stem().string();
        if (entry.path().extension() != ".jpg" || stem.empty() ||
            !std::all_of(stem.begin(), stem.end(), [](unsigned char c) { return c >= '0' && c <= '9'; })) {
            continue;
        }
        numbered.push_back({std::stoull(stem), {entry.path(), dataset / "landmarks" / (stem + ".lms")}});
    }
    std::sort(numbered.begin(), numbered.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<std::pair<fs::path, fs::path>> frames;
    for (auto& item : numbered) {
        frames.push_back(std::move(item.second));
    }
    if (frames.empty()) {
        throw std::runtime_error("No frames in " + dataset.string());
    }
    return frames;
}

// Feature windows for the whole file; the last video ticks are padded with
// closed-mouth (zero) windows so the frame count covers the audio.
std::vector<std::vector<float>> ExtractWindows(avatar::FeatherHubertExtractor* extractor,
    const std::vector<std::int16_t>& pcm) {
    std::vector<std::vector<float>> windows;
    for (std::size_t offset = 0; offset < pcm.size(); offset += 12000) {
        const auto size = std::min<std::size_t>(12000, pcm.size() - offset);
        const auto batch = extractor->PushPcm(pcm.data() + offset, size, offset + size == pcm.size());
        for (std::size_t i = 0; i < batch.size(); ++i) {
            const auto begin = batch.values.begin() + i * avatar::kWindowElements;
            windows.emplace_back(begin, begin + avatar::kWindowElements);
        }
    }
    const auto needed = (pcm.size() + 639) / 640;
    while (windows.size() < needed) {
        windows.emplace_back(avatar::kWindowElements, 0.0f);
    }
    return windows;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Arguments args;
        if (!ParseArguments(argc, argv, &args)) {
            return 0;
        }
        const auto cache = CacheRoot();
        const auto models = cache / "models" / "avatar";
        const auto assets = cache / "assets" / "avatar";
        if (args.audio.empty()) {
            args.audio = (assets / "audio" / (args.persona + "_test.wav")).string();
        }
        if (args.output.empty()) {
            args.output = args.persona + "-frames";
        }
        fs::create_directories(args.output);
        if (!fs::is_empty(args.output)) {
            throw std::runtime_error("Output directory must be empty: " + args.output);
        }

        avatar::RuntimeOptions hubert_options;  // FeatherHuBERT runs on the CPU.
        avatar::RuntimeOptions unet_options;
        unet_options.provider = args.provider;
        avatar::FeatherHubertExtractor hubert((models / "feather_hubert_dynamic.onnx").string(), hubert_options);
        avatar::UnetLipGenerator unet((models / (args.persona + "_int8.onnx")).string(), unet_options);

        const auto start = std::chrono::steady_clock::now();
        const auto windows = ExtractWindows(&hubert, avatar::ReadWav16k(args.audio));
        const auto frames = ListFrames(assets / args.persona);
        const auto total = args.max_frames ? std::min(args.max_frames, windows.size()) : windows.size();
        const auto period = frames.size() == 1 ? 1 : 2 * frames.size() - 2;

        std::unique_ptr<avatar::PreparedFrame> frame;
        for (std::size_t i = 0; i < total; ++i) {
            const auto at = i % period;
            const auto& paths = frames[at < frames.size() ? at : period - at];
            const auto image = cv::imread(paths.first.string(), cv::IMREAD_COLOR);
            const auto landmarks = avatar::ReadLandmarks(paths.second.string());
            if (!frame) {
                frame = std::make_unique<avatar::PreparedFrame>(image, landmarks);
            } else {
                frame->SetFrame(image, landmarks);
            }
            const auto face = unet.Infer(frame->ImageTensor(), windows[i]);
            std::ostringstream name;
            name << std::setw(8) << std::setfill('0') << i << ".png";
            if (!cv::imwrite((fs::path(args.output) / name.str()).string(), frame->Compose(face))) {
                throw std::runtime_error("Image write failed");
            }
        }
        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << total << " frames saved to " << args.output << " in " << std::fixed << std::setprecision(1)
            << seconds << " s (ORT " << avatar::RuntimeVersion() << ")\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "avatar_file_demo: " << e.what() << '\n';
        return 2;
    }
}
