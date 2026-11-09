#pragma once
#include "avatar_service.h"
#include <onnxruntime_cxx_api.h>
namespace SpacemiT::avatar::detail {
void InitRuntime(const std::string& library);
Ort::Env& Environment();
Ort::SessionOptions SessionOptions(const RuntimeOptions& options);
void CheckTensor(const Ort::Value& tensor, const std::vector<std::int64_t>& shape);
}
