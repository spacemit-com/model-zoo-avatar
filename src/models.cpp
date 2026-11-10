/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "internal.h"
#include "non_reentrant.h"
#include "runtime.h"

namespace SpacemiT::avatar {

namespace {

constexpr std::size_t kFaceElements = 3 * 160 * 160;

template <class T>
void RequireFinite(const T& values, const char* message) {
    for (float x : values) {
        if (!std::isfinite(x)) {
            throw std::invalid_argument(message);
        }
    }
}

}  // namespace

struct FeatherHubertExtractor::Impl {
    std::mutex mutex;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<detail::FeatureStream> stream;

    Impl(const std::string& model, const RuntimeOptions& options) {
        if (options.provider != "cpu") {
            throw std::invalid_argument("FeatherHuBERT baseline requires CPU provider");
        }
        auto session_options = detail::SessionOptions(options);
        session = std::make_unique<Ort::Session>(detail::Environment(), model.c_str(), session_options);
        if (session->GetInputCount() != 1 || session->GetOutputCount() != 1) {
            throw std::runtime_error("Invalid HuBERT graph signature");
        }
        stream = std::make_unique<detail::FeatureStream>(
            [this](const std::vector<float>& audio) { return Hidden(audio); });
    }

    // waveform [1, samples] -> hidden [1, tokens, 1024]
    std::vector<float> Hidden(const std::vector<float>& audio) {
        const std::array<std::int64_t, 2> shape{1, static_cast<std::int64_t>(audio.size())};
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input = Ort::Value::CreateTensor<float>(memory, const_cast<float*>(audio.data()), audio.size(),
            shape.data(), shape.size());
        const char* input_names[] = {"waveform"};
        const char* output_names[] = {"hidden"};
        auto values = session->Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);
        auto info = values[0].GetTensorTypeAndShapeInfo();
        const auto dims = info.GetShape();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || dims.size() != 3 || dims[0] != 1 ||
            dims[2] != 1024) {
            throw std::runtime_error("Invalid HuBERT hidden shape");
        }
        const auto count = info.GetElementCount();
        const float* data = values[0].GetTensorData<float>();
        for (std::size_t i = 0; i < count; ++i) {
            if (!std::isfinite(data[i])) {
                throw std::runtime_error("Nonfinite HuBERT features");
            }
        }
        return std::vector<float>(data, data + count);
    }
};

FeatherHubertExtractor::FeatherHubertExtractor(const std::string& model, const RuntimeOptions& options)
    : impl_(std::make_unique<Impl>(model, options)) {}

FeatherHubertExtractor::~FeatherHubertExtractor() = default;

FeatureBatch FeatherHubertExtractor::PushPcm(const std::int16_t* pcm, std::size_t samples, bool final) {
    auto gate = detail::Enter(impl_->mutex);
    return impl_->stream->Push(pcm, samples, final);
}

void FeatherHubertExtractor::Reset() {
    auto gate = detail::Enter(impl_->mutex);
    impl_->stream->Reset();
}

struct UnetLipGenerator::Impl {
    std::mutex mutex;
    std::unique_ptr<Ort::Session> session;

    Impl(const std::string& model, const RuntimeOptions& options) {
        auto session_options = detail::SessionOptions(options);
        session = std::make_unique<Ort::Session>(detail::Environment(), model.c_str(), session_options);
        if (session->GetInputCount() != 2 || session->GetOutputCount() != 1) {
            throw std::runtime_error("Invalid UNet graph signature");
        }
    }
};

UnetLipGenerator::UnetLipGenerator(const std::string& model, const RuntimeOptions& options)
    : impl_(std::make_unique<Impl>(model, options)) {}

UnetLipGenerator::~UnetLipGenerator() = default;

std::vector<float> UnetLipGenerator::Infer(const std::vector<float>& image, const std::vector<float>& audio) {
    auto gate = detail::Enter(impl_->mutex);
    if (image.size() != 6 * 160 * 160 || audio.size() != kWindowElements) {
        throw std::invalid_argument("UNet expects image [1,6,160,160], audio [1,16,32,32]");
    }
    RequireFinite(image, "Nonfinite image tensor");
    RequireFinite(audio, "Nonfinite audio tensor");
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<std::int64_t, 4> image_shape{1, 6, 160, 160};
    const std::array<std::int64_t, 4> audio_shape{1, 16, 32, 32};
    std::array<Ort::Value, 2> inputs{
        Ort::Value::CreateTensor<float>(memory, const_cast<float*>(image.data()), image.size(), image_shape.data(),
            image_shape.size()),
        Ort::Value::CreateTensor<float>(memory, const_cast<float*>(audio.data()), audio.size(), audio_shape.data(),
            audio_shape.size())};
    const char* input_names[] = {"input", "audio"};
    const char* output_names[] = {"output"};
    auto result = impl_->session->Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), 2, output_names, 1);
    detail::CheckTensor(result[0], {1, 3, 160, 160});
    const auto* data = result[0].GetTensorData<float>();
    for (std::size_t i = 0; i < kFaceElements; ++i) {
        if (!std::isfinite(data[i])) {
            throw std::runtime_error("Nonfinite UNet output");
        }
    }
    return {data, data + kFaceElements};
}

}  // namespace SpacemiT::avatar
