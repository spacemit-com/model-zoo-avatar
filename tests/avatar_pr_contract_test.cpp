/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Avatar PR contract test. Runs without model weights.
 *
 *   --core-contract              feature timeline, masking, compositing, FTF1 files
 *   --config-contract            option defaults, linked runtime, shipped configs
 *   --invalid-input-error-path   invalid inputs fail fast with an exception
 *
 * Without arguments every contract runs.
 */

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "avatar_service.h"
#include "internal.h"

namespace fs = std::filesystem;
namespace avatar = SpacemiT::avatar;

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <class Error = std::exception>
void ExpectThrows(const std::function<void()>& action, const std::string& message) {
    try {
        action();
    } catch (const Error&) {
        return;
    }
    throw std::runtime_error("Accepted invalid input: " + message);
}

std::vector<float> HalfRunner(const std::vector<float>& samples) {
    return std::vector<float>((samples.size() / 320) * 1024, 0.5f);
}

cv::Mat GradientImage() {
    cv::Mat image(240, 240, CV_8UC3);
    for (int y = 0; y < 240; ++y) {
        for (int x = 0; x < 240; ++x) {
            image.at<cv::Vec3b>(y, x) = {static_cast<unsigned char>(x), static_cast<unsigned char>(y),
                static_cast<unsigned char>((x + y) / 2)};
        }
    }
    return image;
}

std::vector<cv::Point2f> FaceLandmarks() {
    std::vector<cv::Point2f> landmarks(110, {120, 120});
    landmarks[1] = {30, 120};
    landmarks[31] = {210, 120};
    landmarks[52] = {120, 30};
    for (int i = 90; i < 110; ++i) {
        landmarks[i] = {100.0f + (i - 90) * 2.0f, 150.0f + static_cast<float>(i % 3)};
    }
    return landmarks;
}

fs::path TempDirectory(const std::string& tag) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = fs::temp_directory_path() / ("avatar-" + tag + "-" + std::to_string(stamp));
    Require(fs::create_directory(path), "Test directory exists");
    return path;
}

void WriteWav(const fs::path& path, std::uint32_t rate, std::uint16_t channels, std::size_t samples) {
    auto put = [](std::ofstream& out, std::uint32_t value, int bytes) {
        for (int i = 0; i < bytes; ++i) {
            out.put(static_cast<char>((value >> (8 * i)) & 0xff));
        }
    };
    const std::uint32_t data_bytes = static_cast<std::uint32_t>(samples * channels * 2);
    std::ofstream out(path, std::ios::binary);
    out.write("RIFF", 4);
    put(out, 36 + data_bytes, 4);
    out.write("WAVEfmt ", 8);
    put(out, 16, 4);
    put(out, 1, 2);
    put(out, channels, 2);
    put(out, rate, 4);
    put(out, rate * channels * 2, 4);
    put(out, channels * 2, 2);
    put(out, 16, 2);
    out.write("data", 4);
    put(out, data_bytes, 4);
    for (std::size_t i = 0; i < samples * channels; ++i) {
        put(out, 0, 2);
    }
}

void CoreContract() {
    avatar::detail::FeatureStream stream(HalfRunner);
    std::vector<std::int16_t> silence(24000, 0);
    auto first = stream.Push(silence.data(), 12000, false);
    auto last = stream.Push(silence.data() + 12000, 12000, true);
    Require(first.size() > 0 && last.size() > 0, "No silent frames");
    Require(std::all_of(first.values.begin(), first.values.end(), [](float v) { return v == 0; }), "Silence amplified");
    Require(first.frame_ids.front() == 0 && last.frame_ids.front() == first.frame_ids.back() + 1,
        "Frame ids not continuous");

    stream.Reset();
    std::vector<std::int16_t> speech(12000);
    for (std::size_t i = 0; i < speech.size(); ++i) {
        speech[i] = static_cast<std::int16_t>(12000 * std::sin(i * 0.1));
    }
    auto voice = stream.Push(speech.data(), speech.size(), true);
    Require(voice.frame_ids.front() == 0, "Reset cursor failed");
    Require(std::any_of(voice.values.begin(), voice.values.end(), [](float v) { return v != 0; }), "Speech discarded");

    int calls = 0;
    avatar::detail::FeatureStream retry([&](const std::vector<float>& samples) {
        if (++calls == 2) {
            throw std::runtime_error("Injected inference failure");
        }
        return HalfRunner(samples);
    });
    ExpectThrows<std::runtime_error>([&] { retry.Push(silence.data(), silence.size(), true); },
        "injected inference failure");
    const auto retried = retry.Push(silence.data(), silence.size(), true);
    Require(retried.size() == first.size() + last.size() && retried.frame_ids.front() == 0,
        "Failed multi-chunk push consumed audio or advanced the cursor");

    const auto image = GradientImage();
    avatar::PreparedFrame frame(image, FaceLandmarks());
    const auto tensor = frame.ImageTensor();
    Require(tensor.size() == 6 * 160 * 160, "Image tensor shape");
    const auto plane = 160 * 160;
    Require(tensor[3 * plane + 5 * 160 + 5] == 0, "Mask start wrong");
    Require(tensor[3 * plane + 149 * 160 + 154] == 0, "Mask end wrong");
    Require(tensor[3 * plane + 150 * 160 + 155] > 0, "Mask extended past boundary");
    const auto composed = frame.Compose(std::vector<float>(tensor.begin(), tensor.begin() + 3 * plane));
    Require(composed.type() == CV_8UC3 && composed.size() == image.size(), "Composite shape");
    Require(composed.at<cv::Vec3b>(0, 0) == image.at<cv::Vec3b>(0, 0), "Modified outside face");

    const auto temp = TempDirectory("core");
    const auto file = temp / "features.ftf";
    {
        avatar::detail::FeatureWriter writer(file.string());
        avatar::FeatureBatch batch;
        batch.frame_ids = {0};
        batch.values.resize(avatar::kWindowElements, 0.25f);
        writer.Append(batch);
        writer.Finish(640);
    }
    {
        avatar::detail::FeatureReader reader(file.string());
        std::vector<float> window;
        Require(reader.Count() == 1 && reader.Samples() == 640 && reader.Next(window) &&
            window.size() == avatar::kWindowElements &&
            window[0] == 0.25f && !reader.Next(window), "Feature file roundtrip failed");
    }
    fs::remove_all(temp);
    std::cout << "core contract: feature cursor, final/reset, retry, mask bounds, composite, FTF1: passed\n";
}

void ConfigContract() {
    const avatar::RuntimeOptions runtime;
    Require(runtime.provider == "cpu" && runtime.threads == 4 && runtime.ep_threads == 2 &&
        runtime.profile_prefix.empty(), "RuntimeOptions defaults changed");
    const avatar::RenderOptions render;
    Require(render.feather && render.native_pasteback && !render.mouth_only_blend && !render.temporal_smooth &&
        render.mouth_sharpen == 0.15f && render.smooth_alpha == 0.9f && render.patch_lowpass == 0.0f,
        "RenderOptions defaults changed");
    Require(avatar::kWindowElements == 16 * 32 * 32, "Feature window size changed");
    Require(!avatar::RuntimeVersion().empty(), "Linked ONNX Runtime has no version");

    for (const std::string persona : {"feifan", "zihan"}) {
        const auto config = YAML::LoadFile((fs::path(AVATAR_SOURCE_DIR) / "configs" / (persona + ".yaml")).string());
        const auto where = "configs/" + persona + ".yaml: ";
        Require(config["schema"].as<int>() == 1, where + "schema must be 1");
        Require(config["hubert"]["provider"].as<std::string>() == "cpu", where + "FeatherHuBERT must use the CPU");
        Require(config["unet"]["provider"].as<std::string>() == "spacemit", where + "UNet must use the SpaceMIT EP");
        Require(config["hubert"]["model"].as<std::string>().find("feather_hubert_dynamic.onnx") != std::string::npos,
            where + "unexpected FeatherHuBERT model");
        Require(config["unet"]["model"].as<std::string>().find(persona + "_int8.onnx") != std::string::npos,
            where + "UNet does not match the persona");
        const auto person = config["assets"]["person"].as<std::string>();
        Require(person.size() >= persona.size() && person.compare(person.size() - persona.size(), persona.size(),
            persona) == 0, where + "assets do not match the persona");
        Require(!config["unet"]["ort_library"] && !config["unet"]["ep_library"], where + "uses removed runtime keys");
    }
    std::cout << "config contract: option defaults, linked runtime, shipped configs: passed\n";
}

void InvalidInputErrorPath() {
    avatar::detail::FeatureStream stream(HalfRunner);
    std::vector<std::int16_t> silence(12000, 0);
    stream.Push(silence.data(), silence.size(), true);
    ExpectThrows<std::runtime_error>([&] { stream.Push(silence.data(), silence.size(), false); }, "push after final");

    const auto image = GradientImage();
    const auto landmarks = FaceLandmarks();
    avatar::PreparedFrame frame(image, landmarks);
    ExpectThrows<std::invalid_argument>([&] { frame.Compose({0}); }, "wrong output size");
    std::vector<float> nonfinite(3 * 160 * 160, 0.5f);
    nonfinite[7] = std::numeric_limits<float>::quiet_NaN();
    ExpectThrows<std::invalid_argument>([&] { frame.Compose(nonfinite); }, "non-finite output");
    ExpectThrows<std::invalid_argument>([&] { avatar::PreparedFrame invalid(cv::Mat(), landmarks); }, "empty image");
    ExpectThrows<std::invalid_argument>([&] {
        avatar::PreparedFrame invalid(image, std::vector<cv::Point2f>(landmarks.begin(), landmarks.begin() + 100));
    }, "too few landmarks");
    auto bad_points = landmarks;
    bad_points[3].x = std::numeric_limits<float>::infinity();
    ExpectThrows<std::invalid_argument>([&] { avatar::PreparedFrame invalid(image, bad_points); },
        "non-finite landmark");
    avatar::RenderOptions bad_render;
    bad_render.mouth_sharpen = -1.0f;
    ExpectThrows<std::invalid_argument>([&] { avatar::PreparedFrame invalid(image, landmarks, bad_render); },
        "negative mouth_sharpen");

    avatar::RuntimeOptions no_threads;
    no_threads.threads = 0;
    ExpectThrows<std::invalid_argument>([&] { avatar::FeatherHubertExtractor invalid("unused.onnx", no_threads); },
        "zero threads");
    avatar::RuntimeOptions bad_provider;
    bad_provider.provider = "gpu";
    ExpectThrows<std::invalid_argument>([&] { avatar::UnetLipGenerator invalid("unused.onnx", bad_provider); },
        "unknown provider");
    avatar::RuntimeOptions ep_for_hubert;
    ep_for_hubert.provider = "spacemit";
    ExpectThrows<std::invalid_argument>([&] { avatar::FeatherHubertExtractor invalid("unused.onnx", ep_for_hubert); },
        "FeatherHuBERT on the EP");

    const auto temp = TempDirectory("invalid");
    const auto features = temp / "features.ftf";
    {
        avatar::detail::FeatureWriter writer(features.string());
        avatar::FeatureBatch batch;
        batch.frame_ids = {0};
        batch.values.resize(avatar::kWindowElements, 0.0f);
        writer.Append(batch);
        writer.Finish(640);
    }
    ExpectThrows<std::runtime_error>([&] { avatar::detail::FeatureWriter writer(features.string()); },
        "overwriting a feature file");
    const auto bad_magic = temp / "bad.ftf";
    std::ofstream(bad_magic, std::ios::binary).write(std::string(64, '\0').data(), 64);
    ExpectThrows([&] { avatar::detail::FeatureReader reader(bad_magic.string()); }, "feature file without FTF1 header");
    fs::resize_file(features, 64 + 100);
    ExpectThrows([&] { avatar::detail::FeatureReader reader(features.string()); }, "truncated feature file");
    const auto rate_8k = temp / "8k.wav";
    WriteWav(rate_8k, 8000, 1, 800);
    ExpectThrows([&] { avatar::ReadWav16k(rate_8k.string()); }, "8 kHz WAV");
    const auto stereo = temp / "stereo.wav";
    WriteWav(stereo, 16000, 2, 800);
    ExpectThrows([&] { avatar::ReadWav16k(stereo.string()); }, "stereo WAV");
    ExpectThrows([&] { avatar::ReadWav16k((temp / "missing.wav").string()); }, "missing WAV");
    ExpectThrows([&] { avatar::ReadLandmarks((temp / "missing.lms").string()); }, "missing landmarks");
    fs::remove_all(temp);
    std::cout << "invalid input error path: streams, frames, options, runtime, files: passed\n";
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    try {
        if (mode.empty() || mode == "--core-contract") {
            CoreContract();
        }
        if (mode.empty() || mode == "--config-contract") {
            ConfigContract();
        }
        if (mode.empty() || mode == "--invalid-input-error-path") {
            InvalidInputErrorPath();
        }
        if (!mode.empty() && mode != "--core-contract" && mode != "--config-contract" &&
            mode != "--invalid-input-error-path") {
            std::cerr << "Unknown mode " << mode << '\n';
            return 2;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
