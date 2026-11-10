/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * avatar_worker: offline two-stage rendering.
 *   extract  WAV -> FTF1 feature file (FeatherHuBERT)
 *   render   FTF1 -> PNG frames (person UNet, pipelined)
 */

#include <fcntl.h>
#include <sched.h>
#include <sys/file.h>
#include <unistd.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <deque>
#include <exception>
#include <filesystem>  // NOLINT(build/c++17)
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "avatar_service.h"
#include "internal.h"

namespace fs = std::filesystem;
namespace avatar = SpacemiT::avatar;

namespace {

volatile std::sig_atomic_t stopped = 0;

void OnSignal(int) {
    stopped = 1;
}

void CheckCancel() {
    if (stopped) {
        throw std::runtime_error("Worker cancelled; inference returned before releasing resources");
    }
}

class OutputDirectory {
public:
    explicit OutputDirectory(const fs::path& path) {
        fs::create_directories(path);
        fd_ = open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (fd_ < 0) {
            throw std::runtime_error("Cannot open output directory");
        }
        try {
            if (flock(fd_, LOCK_EX | LOCK_NB) != 0) {
                throw std::runtime_error("Output directory is in use");
            }
            if (!fs::is_empty(path)) {
                throw std::runtime_error("Output directory must be empty");
            }
        } catch (...) {
            close(fd_);
            throw;
        }
    }
    ~OutputDirectory() { close(fd_); }
    OutputDirectory(const OutputDirectory&) = delete;
    OutputDirectory& operator=(const OutputDirectory&) = delete;

private:
    int fd_;
};

// Bounded FIFO between render stages. Close() wakes every waiter; Pop() still
// drains queued items before it reports the end of the stream.
template <typename T>
class Channel {
public:
    explicit Channel(std::size_t capacity) : capacity_(capacity) {}
    bool Push(T value) {
        std::unique_lock<std::mutex> lock(mutex_);
        space_.wait(lock, [&] { return closed_ || items_.size() < capacity_; });
        if (closed_) {
            return false;
        }
        items_.push_back(std::move(value));
        ready_.notify_one();
        return true;
    }
    bool Pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [&] { return closed_ || !items_.empty(); });
        if (items_.empty()) {
            return false;
        }
        value = std::move(items_.front());
        items_.pop_front();
        space_.notify_one();
        return true;
    }
    void Close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        ready_.notify_all();
        space_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable ready_, space_;
    std::deque<T> items_;
    std::size_t capacity_;
    bool closed_ = false;
};

struct SourceFrame {
    std::uint64_t index;
    cv::Mat image;
    std::vector<cv::Point2f> landmarks;
    std::vector<float> input;
};

struct GeneratedFrame {
    std::uint64_t index;
    cv::Mat image;
    std::vector<cv::Point2f> landmarks;
    std::vector<float> output;
};

struct ComposedFrame {
    std::uint64_t index;
    cv::Mat image;
};

using FramePaths = std::vector<std::pair<fs::path, fs::path>>;

// Keeps the EP busy while X100 threads decode, composite and encode the
// neighbouring frames:
//   prefetch (decode, model input) -> UNet on this thread -> compose -> PNG writers.
// Compose runs in frame order on a single PreparedFrame, whose post-processing
// state carries across frames exactly as in a serial loop, so the frames are
// unchanged. Model inputs come from a separate PreparedFrame, which keeps no
// state that affects them.
std::uint64_t Render(avatar::UnetLipGenerator& model, avatar::detail::FeatureReader& reader, std::uint64_t total,
    const FramePaths& frames, const avatar::RenderOptions& options, const fs::path& output) {
    constexpr std::size_t kDepth = 4, kWriters = 3;
    Channel<SourceFrame> sources(kDepth);
    Channel<GeneratedFrame> generated(kDepth);
    Channel<ComposedFrame> composed(kDepth);
    std::atomic<bool> failed{false};
    std::atomic<std::uint64_t> written{0};
    std::mutex error_mutex;
    std::exception_ptr error;
    auto fail = [&](std::exception_ptr cause) {
        {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!error) {
                error = cause;
            }
        }
        failed = true;
        sources.Close();
        generated.Close();
        composed.Close();
    };
    const auto period = frames.size() == 1 ? 1 : 2 * frames.size() - 2;
    std::vector<std::thread> threads;
    threads.emplace_back([&] {
        try {
            std::unique_ptr<avatar::PreparedFrame> frame;
            for (std::uint64_t i = 0; i < total && !failed; ++i) {
                const auto at = i % period;
                const auto& paths = frames[at < frames.size() ? at : period - at];
                SourceFrame item{i, cv::imread(paths.first.string(), cv::IMREAD_COLOR),
                    avatar::ReadLandmarks(paths.second.string()), {}};
                if (!frame) {
                    frame = std::make_unique<avatar::PreparedFrame>(item.image, item.landmarks, options);
                } else {
                    frame->SetFrame(item.image, item.landmarks, options);
                }
                item.input = frame->ImageTensor();
                if (!sources.Push(std::move(item))) {
                    return;
                }
            }
            sources.Close();
        } catch (...) {
            fail(std::current_exception());
        }
    });
    threads.emplace_back([&] {
        try {
            std::unique_ptr<avatar::PreparedFrame> frame;
            GeneratedFrame item;
            while (generated.Pop(item)) {
                if (failed) {
                    return;
                }
                if (!frame) {
                    frame = std::make_unique<avatar::PreparedFrame>(item.image, item.landmarks, options);
                } else {
                    frame->SetFrame(item.image, item.landmarks, options);
                }
                if (!composed.Push({item.index, frame->Compose(item.output)})) {
                    return;
                }
            }
            composed.Close();
        } catch (...) {
            fail(std::current_exception());
        }
    });
    for (std::size_t i = 0; i < kWriters; ++i) {
        threads.emplace_back([&] {
            try {
                ComposedFrame item;
                while (composed.Pop(item)) {
                    if (failed) {
                        return;
                    }
                    std::ostringstream name;
                    name << std::setw(8) << std::setfill('0') << item.index << ".png";
                    if (!cv::imwrite((output / name.str()).string(), item.image)) {
                        throw std::runtime_error("Image write failed");
                    }
                    ++written;
                }
            } catch (...) {
                fail(std::current_exception());
            }
        });
    }
    try {
        std::vector<float> window;
        SourceFrame item;
        while (!failed && sources.Pop(item)) {
            CheckCancel();
            if (!reader.Next(window)) {
                throw std::runtime_error("Feature file ended early");
            }
            auto face = model.Infer(item.input, window);
            CheckCancel();
            if (!generated.Push({item.index, std::move(item.image), std::move(item.landmarks), std::move(face)})) {
                break;
            }
        }
        generated.Close();
    } catch (...) {
        fail(std::current_exception());
    }
    for (auto& thread : threads) {
        thread.join();
    }
    if (error) {
        std::rethrow_exception(error);
    }
    return written;
}

std::string ResolvePath(const std::string& value, const fs::path& folder) {
    if (value.empty()) {
        throw std::invalid_argument("Empty resource path");
    }
    if (value.rfind("~/", 0) == 0) {
        const char* home = std::getenv("HOME");
        if (!home) {
            throw std::runtime_error("HOME unset");
        }
        return (fs::path(home) / value.substr(2)).string();
    }
    fs::path path(value);
    return (path.is_absolute() ? path : folder / path).lexically_normal().string();
}

avatar::RuntimeOptions ReadRuntime(const YAML::Node& node, const fs::path& folder) {
    avatar::RuntimeOptions options;
    if (node["provider"]) {
        options.provider = node["provider"].as<std::string>();
    }
    if (node["threads"]) {
        options.threads = node["threads"].as<int>();
    }
    if (node["ep_threads"]) {
        options.ep_threads = node["ep_threads"].as<int>();
    }
    if (node["profile_prefix"]) {
        options.profile_prefix = ResolvePath(node["profile_prefix"].as<std::string>(), folder);
    }
    return options;
}

avatar::RenderOptions ReadRender(const YAML::Node& node) {
    avatar::RenderOptions options;
    if (!node) {
        return options;
    }
    if (node["native_pasteback"]) {
        options.native_pasteback = node["native_pasteback"].as<bool>();
    }
    if (node["mouth_only_blend"]) {
        options.mouth_only_blend = node["mouth_only_blend"].as<bool>();
    }
    if (node["feather"]) {
        options.feather = node["feather"].as<bool>();
    }
    if (node["mouth_sharpen"]) {
        options.mouth_sharpen = node["mouth_sharpen"].as<float>();
    }
    if (node["temporal_smooth"]) {
        options.temporal_smooth = node["temporal_smooth"].as<bool>();
    }
    if (node["smooth_alpha"]) {
        options.smooth_alpha = node["smooth_alpha"].as<float>();
    }
    if (node["patch_lowpass"]) {
        options.patch_lowpass = node["patch_lowpass"].as<float>();
    }
    return options;
}

void ApplyAffinity(const YAML::Node& node) {
    if (!node["cpu_affinity"]) {
        return;
    }
    cpu_set_t mask;
    CPU_ZERO(&mask);
    const auto ids = node["cpu_affinity"].as<std::vector<int>>();
    if (ids.empty()) {
        throw std::invalid_argument("Empty CPU affinity");
    }
    for (int id : ids) {
        if (id < 0 || id >= CPU_SETSIZE) {
            throw std::invalid_argument("CPU id out of range");
        }
        CPU_SET(id, &mask);
    }
    if (sched_setaffinity(0, sizeof(mask), &mask)) {
        throw std::runtime_error("Cannot apply CPU affinity");
    }
}

FramePaths ListFrames(const fs::path& dataset) {
    FramePaths frames;
    for (const auto& entry : fs::directory_iterator(dataset / "full_body_img")) {
        if (entry.path().extension() != ".jpg") {
            continue;
        }
        const auto stem = entry.path().stem().string();
        const auto is_digit = [](unsigned char c) { return c >= '0' && c <= '9'; };
        if (stem.empty() || !std::all_of(stem.begin(), stem.end(), is_digit)) {
            continue;
        }
        const auto landmarks = dataset / "landmarks" / (stem + ".lms");
        if (!fs::is_regular_file(landmarks)) {
            throw std::runtime_error("Frame missing landmarks");
        }
        frames.emplace_back(entry.path(), landmarks);
    }
    std::sort(frames.begin(), frames.end(), [](const auto& a, const auto& b) {
        return std::stoull(a.first.stem().string()) < std::stoull(b.first.stem().string());
    });
    if (frames.empty()) {
        throw std::runtime_error("No paired JPG/landmark frames");
    }
    return frames;
}

std::uint64_t Extract(const YAML::Node& node, const fs::path& folder, const std::string& audio,
    const std::string& output) {
    ApplyAffinity(node);
    avatar::FeatherHubertExtractor model(ResolvePath(node["model"].as<std::string>(), folder),
        ReadRuntime(node, folder));
    auto pcm = avatar::ReadWav16k(audio);
    avatar::detail::FeatureWriter writer(output);
    std::uint64_t count = 0;
    for (std::size_t offset = 0; offset < pcm.size(); offset += 12000) {
        CheckCancel();
        const auto size = std::min<std::size_t>(12000, pcm.size() - offset);
        auto batch = model.PushPcm(pcm.data() + offset, size, offset + size == pcm.size());
        CheckCancel();
        count += batch.size();
        writer.Append(batch);
    }
    if (!count) {
        throw std::runtime_error("Audio is too short to produce feature windows");
    }
    // The extractor preserves the original server's model-frame count.
    // Pad the final video ticks with closed-mouth features to cover the PCM duration.
    const auto needed = (pcm.size() + 639) / 640;
    if (count > needed) {
        throw std::runtime_error("Feature count exceeds audio timeline");
    }
    if (count < needed) {
        avatar::FeatureBatch tail;
        for (auto i = count; i < needed; ++i) {
            tail.frame_ids.push_back(static_cast<std::int64_t>(i));
        }
        tail.values.resize(tail.size() * avatar::kWindowElements, 0);
        writer.Append(tail);
        count = needed;
    }
    writer.Finish(pcm.size());
    return count;
}

std::uint64_t ParseLimit(const std::map<std::string, std::string>& args) {
    std::uint64_t limit = 0;
    const auto found = args.find("--max-frames");
    if (found != args.end()) {
        const auto& value = found->second;
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), limit);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
            throw std::invalid_argument("max-frames must be nonnegative");
        }
    }
    return limit;
}

std::uint64_t RenderFrames(const YAML::Node& config, const fs::path& folder, const std::string& features,
    const std::string& output, std::uint64_t limit) {
    const auto node = config["unet"];
    ApplyAffinity(node);
    avatar::UnetLipGenerator model(ResolvePath(node["model"].as<std::string>(), folder), ReadRuntime(node, folder));
    const auto frames = ListFrames(fs::path(ResolvePath(config["assets"]["person"].as<std::string>(), folder)));
    const auto options = ReadRender(config["render"]);
    const OutputDirectory output_directory(output);
    avatar::detail::FeatureReader reader(features);
    const auto total = limit ? std::min<std::uint64_t>(limit, reader.Count()) : reader.Count();
    return Render(model, reader, total, frames, options, fs::path(output));
}

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGTERM, OnSignal);
    std::signal(SIGINT, OnSignal);
    try {
        if (argc < 2) {
            throw std::invalid_argument(
                "Usage: avatar_worker extract|render --config FILE --audio WAV --features FTF --output PATH "
                "[--max-frames N]");
        }
        const std::string mode = argv[1];
        std::map<std::string, std::string> args;
        for (int i = 2; i < argc; i += 2) {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing option value");
            }
            const std::string key = argv[i];
            if (key != "--config" && key != "--audio" && key != "--features" && key != "--output" &&
                key != "--max-frames") {
                throw std::invalid_argument("Unknown option");
            }
            args[key] = argv[i + 1];
        }
        if (!args.count("--config") || !args.count("--output")) {
            throw std::invalid_argument("config and output required");
        }
        const auto file = fs::absolute(args.at("--config"));
        const auto folder = file.parent_path();
        const auto config = YAML::LoadFile(file.string());
        if (!config["schema"] || config["schema"].as<int>() != 1) {
            throw std::runtime_error("Expected config schema=1");
        }
        const auto start = std::chrono::steady_clock::now();
        std::uint64_t count = 0;
        if (mode == "extract") {
            if (!args.count("--audio")) {
                throw std::invalid_argument("audio required");
            }
            count = Extract(config["hubert"], folder, args.at("--audio"), args.at("--output"));
        } else if (mode == "render") {
            if (!args.count("--features")) {
                throw std::invalid_argument("features required");
            }
            count = RenderFrames(config, folder, args.at("--features"), args.at("--output"), ParseLimit(args));
        } else {
            throw std::invalid_argument("Unknown mode");
        }
        const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::cout << "{\"ok\":true,\"frames\":" << count << ",\"elapsed_ms\":" << ms << "}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "avatar_worker: " << e.what() << '\n';
        return 2;
    }
}
