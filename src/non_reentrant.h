/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NON_REENTRANT_H
#define NON_REENTRANT_H

#include <mutex>
#include <stdexcept>

namespace SpacemiT::avatar::detail {

// Rejects a second concurrent call on the same instance instead of blocking it.
inline std::unique_lock<std::mutex> Enter(std::mutex& mutex) {
    std::unique_lock<std::mutex> gate(mutex, std::try_to_lock);
    if (!gate.owns_lock()) {
        throw std::runtime_error("Concurrent operation on one model/stream instance is not supported");
    }
    return gate;
}

}  // namespace SpacemiT::avatar::detail

#endif  // NON_REENTRANT_H
