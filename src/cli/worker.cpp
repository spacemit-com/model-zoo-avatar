/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * avatar_worker: offline two-stage rendering.
 *   extract  WAV -> FTF1 feature file (FeatherHuBERT)
 *   render   FTF1 -> PNG frames, or an H.264 MP4 encoded through ffmpeg (person UNet, pipelined)
 * Each stage prints a performance report on stderr and a JSON result line on stdout.
 */

#include <fcntl.h>
#include <sched.h>
#include <spawn.h>
#include <sys/file.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <cinttypes>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
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

extern char** environ;

namespace fs = std::filesystem;
namespace avatar = SpacemiT::avatar;

namespace {

using Clock = std::chrono::steady_clock;
constexpr double kFps = 25.0;
constexpr double kSampleRate = 16000.0;

volatile std::sig_atomic_t stopped = 0;

void OnSignal(int) {
    stopped = 1;
}

void CheckCancel() {
    if (stopped) {
        throw std::runtime_error("Worker cancelled; inference returned before releasing resources");
    }
}

double Seconds(Clock::duration duration) {
    return std::chrono::duration<double>(duration).count();
}

struct Arguments {
    std::string mode;
    std::string config;
    std::string audio;
    std::string features;
    std::string output;
    std::uint64_t max_frames = 0;
};

// Frame count, elapsed time and further JSON members of the result line.
struct Outcome {
    std::uint64_t frames = 0;
    double elapsed_ms = 0;
    std::string json;
};

std::string Number(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << value;
    return out.str();
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

// Busy time of one pipeline stage, summed over frames and threads.
class StageTimer {
public:
    template <class Work>
    void Measure(Work&& work) {
        const auto start = Clock::now();
        work();
        nanoseconds_ += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
    }
    double MillisecondsPerFrame(std::uint64_t frames) const {
        return frames ? static_cast<double>(nanoseconds_) / 1e6 / static_cast<double>(frames) : 0.0;
    }

private:
    std::atomic<std::int64_t> nanoseconds_{0};
};

struct RenderStats {
    StageTimer decode, landmarks, preprocess, unet, compose, write;
    Clock::time_point start;
    std::vector<Clock::time_point> done;  // when the sink took each frame
};

// Destination of composed frames, fed by Threads() writer threads.
class FrameSink {
public:
    virtual ~FrameSink() = default;
    virtual std::size_t Threads() const = 0;
    virtual const char* WriteLabel() const = 0;
    virtual void Write(const ComposedFrame& frame) = 0;
    virtual void Finish() {}
};

// Numbered PNG frames in an empty, locked directory.
class PngSink : public FrameSink {
public:
    explicit PngSink(const fs::path& directory) : directory_(directory), lock_(directory) {}
    std::size_t Threads() const override { return 3; }
    const char* WriteLabel() const override { return "PNG"; }
    void Write(const ComposedFrame& frame) override {
        std::ostringstream name;
        name << std::setw(8) << std::setfill('0') << frame.index << ".png";
        if (!cv::imwrite((directory_ / name.str()).string(), frame.image)) {
            throw std::runtime_error("Image write failed");
        }
    }

private:
    fs::path directory_;
    OutputDirectory lock_;
};

// Runs a command with stdin, stdout and stderr on /dev/null and returns its exit status, or -1.
int RunQuietly(std::vector<std::string> args) {
    std::vector<char*> argv;
    for (auto& arg : args) {
        argv.push_back(arg.data());
    }
    argv.push_back(nullptr);
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    pid_t pid = -1;
    const int error = posix_spawn(&pid, argv[0], &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    int status = 0;
    if (error || waitpid(pid, &status, 0) < 0 || !WIFEXITED(status)) {
        return -1;
    }
    return WEXITSTATUS(status);
}

// H.264 encoder for MP4 output. On K3 the video engine behind the ffmpeg
// encoder h264_stcodec encodes 1280x720 at about 250 fps for 4.4 ms of CPU per
// frame; libx264 veryfast needs about 107 ms of CPU per frame on the X100
// cores and is the fallback on machines without the engine.
struct VideoEncoder {
    std::string name;
    bool hardware = false;
    std::string label;  // for the report
};

// Encodes one black frame of the output size; this takes about 0.13 s on K3.
VideoEncoder ChooseVideoEncoder(const std::string& ffmpeg, cv::Size size) {
    const auto geometry = std::to_string(size.width) + "x" + std::to_string(size.height);
    if (RunQuietly({ffmpeg, "-hide_banner", "-nostdin", "-f", "lavfi", "-i", "color=c=black:s=" + geometry + ":r=25",
            "-frames:v", "1", "-pix_fmt", "yuv420p", "-c:v", "h264_stcodec", "-f", "null", "-"}) == 0) {
        return {"h264_stcodec", true, "h264_stcodec (K3 hardware)"};
    }
    return {"libx264", false, "libx264 veryfast (software, h264_stcodec unavailable)"};
}

// Streams frames to ffmpeg, which encodes H.264 and adds the audio. The
// hardware encoder takes I420 converted here with OpenCV, about 1 ms per frame
// against 20 ms for the conversion inside ffmpeg. libx264 takes BGR and keeps
// the ffmpeg chroma subsampling, which it encodes at higher quality. The video
// gets its final name only after ffmpeg succeeds; a failed or cancelled render
// leaves no partial file.
class Mp4Sink : public FrameSink {
public:
    Mp4Sink(fs::path video, std::string audio, std::string ffmpeg, VideoEncoder encoder)
        : video_(std::move(video)), audio_(std::move(audio)), ffmpeg_(std::move(ffmpeg)),
        encoder_(std::move(encoder)), partial_(video_.string() + "." + std::to_string(getpid()) + ".part"),
        log_(partial_ + ".log") {}
    ~Mp4Sink() override { Stop(); }
    Mp4Sink(const Mp4Sink&) = delete;
    Mp4Sink& operator=(const Mp4Sink&) = delete;

    std::size_t Threads() const override { return 1; }  // ffmpeg needs the frames in order
    const char* WriteLabel() const override { return "to ffmpeg"; }

    void Write(const ComposedFrame& frame) override {
        const cv::Mat image = frame.image.isContinuous() ? frame.image : frame.image.clone();
        if (pid_ < 0) {
            Start(image.cols, image.rows);
        } else if (image.cols != width_ || image.rows != height_) {
            throw std::runtime_error("Base frames differ in size");
        }
        const cv::Mat* pixels = &image;
        if (encoder_.hardware) {
            cv::cvtColor(image, yuv_, cv::COLOR_BGR2YUV_I420);
            pixels = &yuv_;
        }
        const unsigned char* data = pixels->data;
        std::size_t left = pixels->total() * pixels->elemSize();
        while (left) {
            const auto written = write(input_, data, left);
            if (written < 0 && errno == EINTR) {
                continue;
            }
            if (written < 0) {
                const int error = errno;
                CheckCancel();
                if (error == EPIPE) {
                    throw std::runtime_error("ffmpeg stopped reading frames" + LogTail());
                }
                throw std::runtime_error(std::string("Cannot write to ffmpeg: ") + std::strerror(error) + LogTail());
            }
            data += written;
            left -= static_cast<std::size_t>(written);
        }
    }

    void Finish() override {
        close(input_);
        input_ = -1;
        const pid_t pid = pid_;
        pid_ = -1;
        int status = 0;
        rusage usage{};
        if (pid < 0 || wait4(pid, &status, 0, &usage) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            throw std::runtime_error("ffmpeg (" + encoder_.name + ") failed to encode " + video_.string() + LogTail());
        }
        cpu_seconds_ = TimevalSeconds(usage.ru_utime) + TimevalSeconds(usage.ru_stime);
        if (fs::exists(video_)) {
            throw std::runtime_error("Video already exists: " + video_.string());
        }
        fs::rename(partial_, video_);
        finished_ = true;
    }

    double EncoderCpuSeconds() const { return cpu_seconds_; }
    const VideoEncoder& Encoder() const { return encoder_; }

private:
    static double TimevalSeconds(const timeval& value) {
        return static_cast<double>(value.tv_sec) + static_cast<double>(value.tv_usec) / 1e6;
    }

    void Start(int width, int height) {
        width_ = width;
        height_ = height;
        std::vector<std::string> args = {ffmpeg_, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
            "-pix_fmt", encoder_.hardware ? "yuv420p" : "bgr24", "-s",
            std::to_string(width) + "x" + std::to_string(height), "-framerate", "25", "-i", "pipe:0"};
        if (!audio_.empty()) {
            args.insert(args.end(), {"-i", audio_});
        }
        if (encoder_.hardware) {
            args.insert(args.end(), {"-c:v", "h264_stcodec"});
        } else {
            args.insert(args.end(), {"-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p"});
        }
        if (!audio_.empty()) {
            args.insert(args.end(), {"-c:a", "aac"});
        }
        args.insert(args.end(), {"-shortest", "-f", "mp4", partial_});
        std::vector<char*> argv;
        for (auto& arg : args) {
            argv.push_back(arg.data());
        }
        argv.push_back(nullptr);

        int fds[2];
        if (pipe2(fds, O_CLOEXEC) != 0) {
            throw std::runtime_error(std::string("Cannot create a pipe to ffmpeg: ") + std::strerror(errno));
        }
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_adddup2(&actions, fds[0], STDIN_FILENO);
        posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
        // The hardware encoder logs on stderr even at -loglevel error. Keep that out
        // of the report; an error message quotes the end of it.
        posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, log_.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        // The worker ignores SIGPIPE to report a dead ffmpeg; ffmpeg itself keeps the default.
        posix_spawnattr_t attributes;
        posix_spawnattr_init(&attributes);
        sigset_t defaults;
        sigemptyset(&defaults);
        sigaddset(&defaults, SIGPIPE);
        posix_spawnattr_setsigdefault(&attributes, &defaults);
        posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETSIGDEF);
        pid_t pid = -1;
        const int error = posix_spawn(&pid, ffmpeg_.c_str(), &actions, &attributes, argv.data(), environ);
        posix_spawn_file_actions_destroy(&actions);
        posix_spawnattr_destroy(&attributes);
        close(fds[0]);
        if (error) {
            close(fds[1]);
            throw std::runtime_error(std::string("Cannot start ffmpeg: ") + std::strerror(error));
        }
        pid_ = pid;
        input_ = fds[1];
    }

    // Ends an unfinished ffmpeg and removes its partial video.
    void Stop() noexcept {
        if (input_ >= 0) {
            close(input_);
            input_ = -1;
        }
        if (pid_ > 0) {
            kill(pid_, SIGTERM);
            waitpid(pid_, nullptr, 0);
            pid_ = -1;
        }
        std::error_code ignored;
        if (!finished_) {
            fs::remove(partial_, ignored);
        }
        fs::remove(log_, ignored);
    }

    // The last lines ffmpeg wrote on stderr, for an error message.
    std::string LogTail() const {
        std::ifstream stream(log_);
        std::deque<std::string> lines;
        for (std::string line; std::getline(stream, line);) {
            if (!line.empty()) {
                lines.push_back(line);
            }
            if (lines.size() > 3) {
                lines.pop_front();
            }
        }
        std::string tail;
        for (const auto& line : lines) {
            tail += (tail.empty() ? ": " : " | ") + line;
        }
        return tail;
    }

    fs::path video_;
    std::string audio_;
    std::string ffmpeg_;
    VideoEncoder encoder_;
    std::string partial_;
    std::string log_;
    cv::Mat yuv_;
    int width_ = 0;
    int height_ = 0;
    pid_t pid_ = -1;
    int input_ = -1;
    bool finished_ = false;
    double cpu_seconds_ = 0;
};

// Keeps the EP busy while X100 threads decode, composite and write the
// neighbouring frames:
//   prefetch (decode, model input) -> UNet on this thread -> compose -> sink writers.
// Compose runs in frame order on a single PreparedFrame, whose post-processing
// state carries across frames exactly as in a serial loop, so the frames are
// unchanged. Model inputs come from a separate PreparedFrame, which keeps no
// state that affects them.
std::uint64_t Render(avatar::UnetLipGenerator& model, avatar::detail::FeatureReader& reader, std::uint64_t total,
    const FramePaths& frames, const avatar::RenderOptions& options, FrameSink& sink, RenderStats& stats) {
    constexpr std::size_t kDepth = 4;
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
    stats.start = Clock::now();
    stats.done.assign(total, stats.start);
    const auto period = frames.size() == 1 ? 1 : 2 * frames.size() - 2;
    std::vector<std::thread> threads;
    threads.emplace_back([&] {
        try {
            std::unique_ptr<avatar::PreparedFrame> frame;
            for (std::uint64_t i = 0; i < total && !failed; ++i) {
                const auto at = i % period;
                const auto& paths = frames[at < frames.size() ? at : period - at];
                SourceFrame item{i, {}, {}, {}};
                stats.decode.Measure([&] { item.image = cv::imread(paths.first.string(), cv::IMREAD_COLOR); });
                stats.landmarks.Measure([&] { item.landmarks = avatar::ReadLandmarks(paths.second.string()); });
                stats.preprocess.Measure([&] {
                    if (!frame) {
                        frame = std::make_unique<avatar::PreparedFrame>(item.image, item.landmarks, options);
                    } else {
                        frame->SetFrame(item.image, item.landmarks, options);
                    }
                    item.input = frame->ImageTensor();
                });
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
                cv::Mat image;
                stats.compose.Measure([&] {
                    if (!frame) {
                        frame = std::make_unique<avatar::PreparedFrame>(item.image, item.landmarks, options);
                    } else {
                        frame->SetFrame(item.image, item.landmarks, options);
                    }
                    image = frame->Compose(item.output);
                });
                if (!composed.Push({item.index, std::move(image)})) {
                    return;
                }
            }
            composed.Close();
        } catch (...) {
            fail(std::current_exception());
        }
    });
    for (std::size_t i = 0; i < sink.Threads(); ++i) {
        threads.emplace_back([&] {
            try {
                ComposedFrame item;
                while (composed.Pop(item)) {
                    if (failed) {
                        return;
                    }
                    stats.write.Measure([&] { sink.Write(item); });
                    stats.done[item.index] = Clock::now();
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
            std::vector<float> face;
            stats.unet.Measure([&] { face = model.Infer(item.input, window); });
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

// Absolute path of an executable found on PATH, or empty.
std::string FindOnPath(const std::string& name) {
    const char* path = std::getenv("PATH");
    std::istringstream directories(path ? path : "");
    std::string directory;
    while (std::getline(directories, directory, ':')) {
        const auto candidate = fs::absolute(fs::path(directory.empty() ? "." : directory) / name);
        std::error_code error;
        if (fs::is_regular_file(candidate, error) && access(candidate.c_str(), X_OK) == 0) {
            return candidate.string();
        }
    }
    return {};
}

bool IsVideoPath(const std::string& output) {
    auto extension = fs::path(output).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return extension == ".mp4";
}

void CreateParent(const std::string& output) {
    const auto parent = fs::path(output).parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent);
    }
}

Outcome Extract(const YAML::Node& node, const fs::path& folder, const Arguments& args, Clock::time_point start) {
    // Read the audio and check the output before loading the model.
    const auto pcm = avatar::ReadWav16k(args.audio);
    if (fs::exists(args.output)) {
        throw std::runtime_error("Feature file already exists: " + args.output);
    }
    ApplyAffinity(node);
    const auto load_start = Clock::now();
    avatar::FeatherHubertExtractor model(ResolvePath(node["model"].as<std::string>(), folder),
        ReadRuntime(node, folder));
    const auto extract_start = Clock::now();
    CreateParent(args.output);
    auto writer = std::make_unique<avatar::detail::FeatureWriter>(args.output);
    std::uint64_t count = 0;
    try {
        for (std::size_t offset = 0; offset < pcm.size(); offset += 12000) {
            CheckCancel();
            const auto size = std::min<std::size_t>(12000, pcm.size() - offset);
            auto batch = model.PushPcm(pcm.data() + offset, size, offset + size == pcm.size());
            CheckCancel();
            count += batch.size();
            writer->Append(batch);
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
            writer->Append(tail);
            count = needed;
        }
        writer->Finish(pcm.size());
    } catch (...) {
        writer.reset();
        std::error_code ignored;
        fs::remove(args.output, ignored);  // An incomplete feature file would block the next run.
        throw;
    }
    const auto end = Clock::now();
    const double audio_s = static_cast<double>(pcm.size()) / kSampleRate;
    const double load_s = Seconds(extract_start - load_start);
    const double extract_s = Seconds(end - extract_start);
    const double total_s = Seconds(end - start);
    std::fprintf(stderr, "avatar_worker extract: %.2f s of audio -> %" PRIu64 " feature windows -> %s\n", audio_s,
        count, args.output.c_str());
    std::fprintf(stderr, "  total      %.2f s  (model load %.2f s, FeatherHuBERT %.2f s), RTF %.3f\n", total_s, load_s,
        extract_s, extract_s / audio_s);
    return {count, total_s * 1000,
        ",\"load_ms\":" + Number(load_s * 1000) + ",\"extract_ms\":" + Number(extract_s * 1000) +
            ",\"audio_s\":" + Number(audio_s) + ",\"rtf\":" + Number(extract_s / audio_s)};
}

std::uint64_t ParseLimit(const std::string& value) {
    std::uint64_t limit = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), limit);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
        throw std::invalid_argument("max-frames must be nonnegative");
    }
    return limit;
}

struct RenderSummary {
    std::uint64_t frames = 0;
    bool video = false;
    double load_s = 0, render_s = 0, finish_s = 0, total_s = 0, encoder_cpu_s = 0;
    double fps = 0, steady_fps = 0, rtf = 0;
    double decode = 0, landmarks = 0, preprocess = 0, unet = 0, compose = 0, write = 0;  // ms per frame
    std::string bottleneck;
    double bottleneck_fps = 0;
    VideoEncoder encoder;
};

RenderSummary Summarize(const RenderStats& stats, const FrameSink& sink, std::uint64_t frames) {
    RenderSummary s;
    s.frames = frames;
    s.decode = stats.decode.MillisecondsPerFrame(frames);
    s.landmarks = stats.landmarks.MillisecondsPerFrame(frames);
    s.preprocess = stats.preprocess.MillisecondsPerFrame(frames);
    s.unet = stats.unet.MillisecondsPerFrame(frames);
    s.compose = stats.compose.MillisecondsPerFrame(frames);
    s.write = stats.write.MillisecondsPerFrame(frames);
    const auto [first, last] = std::minmax_element(stats.done.begin(), stats.done.end());
    s.steady_fps = frames > 1 ? static_cast<double>(frames - 1) / Seconds(*last - *first) : 0.0;
    // Throughput each stage could sustain on its own threads; the lowest one limits the render.
    const auto rate = [](double ms, std::size_t threads) { return ms > 0 ? 1000.0 * threads / ms : 1e9; };
    const std::vector<std::pair<std::string, double>> limits = {
        {"prefetch (decode + landmarks + preprocess)", rate(s.decode + s.landmarks + s.preprocess, 1)},
        {"UNet", rate(s.unet, 1)},
        {"compose", rate(s.compose, 1)},
        {sink.WriteLabel(), rate(s.write, sink.Threads())},
    };
    const auto slowest = std::min_element(limits.begin(), limits.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });
    s.bottleneck = slowest->first;
    s.bottleneck_fps = slowest->second;
    return s;
}

void PrintRenderReport(const RenderSummary& s, const FrameSink& sink, const Arguments& args) {
    std::fprintf(stderr, "avatar_worker render: %" PRIu64 " frames, %.2f s of video -> %s\n", s.frames,
        static_cast<double>(s.frames) / kFps, args.output.c_str());
    if (s.video) {
        std::fprintf(stderr, "  total      %.2f s  (model load %.2f s, render %.2f s, ffmpeg finish %.2f s)\n",
            s.total_s, s.load_s, s.render_s, s.finish_s);
    } else {
        std::fprintf(stderr, "  total      %.2f s  (model load %.2f s, render %.2f s)\n", s.total_s, s.load_s,
            s.render_s + s.finish_s);
    }
    std::fprintf(stderr, "  speed      %.1f fps, steady %.1f fps, RTF %.3f (model load excluded)\n", s.fps,
        s.steady_fps, s.rtf);
    std::fprintf(stderr,
        "  per frame  decode %.1f | landmarks %.1f | preprocess %.1f | UNet %.1f | compose %.1f | %s %.1f ms\n",
        s.decode, s.landmarks, s.preprocess, s.unet, s.compose, sink.WriteLabel(), s.write);
    if (s.video) {
        std::fprintf(stderr, "  encoder    %s, ffmpeg %.1f ms CPU per frame%s\n", s.encoder.label.c_str(),
            s.encoder_cpu_s * 1000 / static_cast<double>(s.frames), args.audio.empty() ? ", no audio track" : "");
    }
    std::fprintf(stderr, "  bottleneck %s, at most %.1f fps\n", s.bottleneck.c_str(), s.bottleneck_fps);
}

std::string RenderJson(const RenderSummary& s) {
    std::string json = ",\"load_ms\":" + Number(s.load_s * 1000) + ",\"render_ms\":" + Number(s.render_s * 1000) +
        ",\"finish_ms\":" + Number(s.finish_s * 1000) + ",\"fps\":" + Number(s.fps) +
        ",\"steady_fps\":" + Number(s.steady_fps) + ",\"rtf\":" + Number(s.rtf) +
        ",\"stage_ms\":{\"decode\":" + Number(s.decode) + ",\"landmarks\":" + Number(s.landmarks) +
        ",\"preprocess\":" + Number(s.preprocess) + ",\"unet\":" + Number(s.unet) +
        ",\"compose\":" + Number(s.compose) + ",\"write\":" + Number(s.write) + "}";
    if (s.video) {
        json += ",\"encoder\":\"" + s.encoder.name + "\",\"encoder_cpu_ms\":" +
            Number(s.encoder_cpu_s * 1000 / static_cast<double>(s.frames));
    }
    return json;
}

Outcome RenderFrames(const YAML::Node& config, const fs::path& folder, const Arguments& args, Clock::time_point start) {
    // Check every input before loading the model: mistakes fail fast and leave no output behind.
    const bool video = IsVideoPath(args.output);
    if (!video && !args.audio.empty()) {
        throw std::invalid_argument("--audio is only used with an .mp4 output");
    }
    std::string ffmpeg;
    if (video) {
        if (fs::exists(args.output)) {
            throw std::runtime_error("Video already exists: " + args.output);
        }
        ffmpeg = FindOnPath("ffmpeg");
        if (ffmpeg.empty()) {
            throw std::runtime_error(
                "ffmpeg not found; install it (sudo apt install ffmpeg) or pass --output DIR for PNG frames");
        }
    } else if (fs::exists(args.output) && (!fs::is_directory(args.output) || !fs::is_empty(args.output))) {
        throw std::runtime_error("Output directory must be empty: " + args.output);
    }
    if (!fs::is_regular_file(args.features)) {
        throw std::runtime_error("Feature file not found: " + args.features + "; run avatar_worker extract first");
    }
    avatar::detail::FeatureReader reader(args.features);
    if (!args.audio.empty()) {
        const auto samples = avatar::ReadWav16k(args.audio).size();
        if (samples != reader.Samples()) {
            throw std::runtime_error("Audio does not match the features: " + args.audio + " has " +
                std::to_string(samples) + " samples, the features cover " + std::to_string(reader.Samples()));
        }
    }
    const auto frames = ListFrames(fs::path(ResolvePath(config["assets"]["person"].as<std::string>(), folder)));
    VideoEncoder encoder;
    if (video) {
        const auto size = cv::imread(frames.front().first.string(), cv::IMREAD_COLOR).size();
        if (size.empty() || size.width % 2 || size.height % 2) {
            throw std::runtime_error("MP4 output needs base frames with an even width and height: " +
                frames.front().first.string());
        }
        encoder = ChooseVideoEncoder(ffmpeg, size);
    }
    const auto options = ReadRender(config["render"]);
    const auto total = args.max_frames ? std::min<std::uint64_t>(args.max_frames, reader.Count()) : reader.Count();
    if (!total) {
        throw std::runtime_error("Feature file has no windows: " + args.features);
    }

    const auto node = config["unet"];
    ApplyAffinity(node);
    const auto load_start = Clock::now();
    avatar::UnetLipGenerator model(ResolvePath(node["model"].as<std::string>(), folder), ReadRuntime(node, folder));
    const double load_s = Seconds(Clock::now() - load_start);

    std::unique_ptr<FrameSink> sink;
    Mp4Sink* mp4_sink = nullptr;
    if (video) {
        CreateParent(args.output);
        auto mp4 = std::make_unique<Mp4Sink>(args.output, args.audio, ffmpeg, encoder);
        mp4_sink = mp4.get();
        sink = std::move(mp4);
    } else {
        sink = std::make_unique<PngSink>(args.output);
    }
    RenderStats stats;
    const auto count = Render(model, reader, total, frames, options, *sink, stats);
    const auto finish_start = Clock::now();
    sink->Finish();
    const auto end = Clock::now();

    auto summary = Summarize(stats, *sink, count);
    summary.video = video;
    summary.load_s = load_s;
    summary.render_s = Seconds(finish_start - stats.start);
    summary.finish_s = Seconds(end - finish_start);
    summary.total_s = Seconds(end - start);
    if (mp4_sink) {
        summary.encoder_cpu_s = mp4_sink->EncoderCpuSeconds();
        summary.encoder = mp4_sink->Encoder();
    }
    const double busy_s = summary.render_s + summary.finish_s;
    summary.fps = static_cast<double>(count) / busy_s;
    summary.rtf = busy_s * kFps / static_cast<double>(count);
    PrintRenderReport(summary, *sink, args);
    return {count, summary.total_s * 1000, RenderJson(summary)};
}

Arguments ParseArguments(int argc, char** argv) {
    Arguments args;
    args.mode = argv[1];
    std::map<std::string, std::string*> fields = {{"--config", &args.config}, {"--audio", &args.audio},
        {"--features", &args.features}, {"--output", &args.output}};
    std::string limit;
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) {
            throw std::invalid_argument("Missing option value");
        }
        const std::string key = argv[i];
        if (key == "--max-frames") {
            limit = argv[i + 1];
        } else if (fields.count(key)) {
            *fields.at(key) = argv[i + 1];
        } else {
            throw std::invalid_argument("Unknown option");
        }
    }
    if (args.config.empty() || args.output.empty()) {
        throw std::invalid_argument("config and output required");
    }
    if (!limit.empty()) {
        args.max_frames = ParseLimit(limit);
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGTERM, OnSignal);
    std::signal(SIGINT, OnSignal);
    std::signal(SIGPIPE, SIG_IGN);  // A dead ffmpeg surfaces as EPIPE instead of killing the worker.
    if (argc < 2) {
        std::cerr << "Usage: avatar_worker extract --config FILE --audio WAV --output FEATURES.ftf\n"
            "       avatar_worker render --config FILE --features FEATURES.ftf --output DIR|VIDEO.mp4\n"
            "                            [--audio WAV] [--max-frames N]\n";
        return 2;
    }
    try {
        const auto args = ParseArguments(argc, argv);
        const auto file = fs::absolute(args.config);
        const auto folder = file.parent_path();
        const auto config = YAML::LoadFile(file.string());
        if (!config["schema"] || config["schema"].as<int>() != 1) {
            throw std::runtime_error("Expected config schema=1");
        }
        const auto start = Clock::now();
        Outcome outcome;
        if (args.mode == "extract") {
            if (args.audio.empty()) {
                throw std::invalid_argument("audio required");
            }
            outcome = Extract(config["hubert"], folder, args, start);
        } else if (args.mode == "render") {
            if (args.features.empty()) {
                throw std::invalid_argument("features required");
            }
            outcome = RenderFrames(config, folder, args, start);
        } else {
            throw std::invalid_argument("Unknown mode");
        }
        std::cout << "{\"ok\":true,\"frames\":" << outcome.frames << ",\"elapsed_ms\":" << Number(outcome.elapsed_ms)
            << outcome.json << "}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "avatar_worker: " << e.what() << '\n';
        return 2;
    }
}
