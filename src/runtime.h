/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RUNTIME_H
#define RUNTIME_H

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <vector>

#include "avatar_service.h"

namespace SpacemiT::avatar::detail {

Ort::Env& Environment();
Ort::SessionOptions SessionOptions(const RuntimeOptions& options);
void CheckTensor(const Ort::Value& tensor, const std::vector<std::int64_t>& shape);

}  // namespace SpacemiT::avatar::detail

#endif  // RUNTIME_H
