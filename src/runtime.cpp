/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime.h"

#include <stdexcept>
#include <string>
#include <unordered_map>

#ifdef AVATAR_HAVE_SPACEMIT_EP
#include <spacemit_ort_env.h>
#endif

namespace SpacemiT::avatar::detail {

Ort::Env& Environment() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "avatar");
    return env;
}

Ort::SessionOptions SessionOptions(const RuntimeOptions& options) {
    if (options.threads < 1 || options.threads > 256 || options.ep_threads < 1 || options.ep_threads > 64) {
        throw std::invalid_argument("Invalid thread count");
    }
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(options.threads);
    session_options.SetInterOpNumThreads(1);
    session_options.SetExecutionMode(ORT_SEQUENTIAL);
    if (options.provider == "spacemit") {
#ifdef AVATAR_HAVE_SPACEMIT_EP
        const std::unordered_map<std::string, std::string> provider_options{
            {"SPACEMIT_EP_INTRA_THREAD_NUM", std::to_string(options.ep_threads)},
            {"SPACEMIT_EP_INTER_THREAD_NUM", "1"},
            {"SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD", "0"},
        };
        Ort::Status status = Ort::SessionOptionsSpaceMITEnvInit(session_options, provider_options);
        if (!status.IsOK()) {
            throw std::runtime_error("SpaceMIT EP init failed: " + status.GetErrorMessage());
        }
#else
        throw std::invalid_argument("This build has no SpaceMIT EP; install spacemit-onnxruntime and rebuild");
#endif
    } else if (options.provider != "cpu") {
        throw std::invalid_argument("provider must be cpu or spacemit");
    }
    if (!options.profile_prefix.empty()) {
        session_options.EnableProfiling(options.profile_prefix.c_str());
    }
    return session_options;
}

void CheckTensor(const Ort::Value& value, const std::vector<std::int64_t>& shape) {
    if (!value.IsTensor()) {
        throw std::runtime_error("Expected tensor");
    }
    auto info = value.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || info.GetShape() != shape) {
        throw std::runtime_error("Unexpected tensor dtype or shape");
    }
}

}  // namespace SpacemiT::avatar::detail

namespace SpacemiT::avatar {

std::string RuntimeVersion() {
    return Ort::GetVersionString();
}

}  // namespace SpacemiT::avatar
