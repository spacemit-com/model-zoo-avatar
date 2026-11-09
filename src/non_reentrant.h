#pragma once
#include <mutex>
#include <stdexcept>
namespace SpacemiT::avatar::detail {
inline std::unique_lock<std::mutex> Enter(std::mutex& mutex){
    std::unique_lock<std::mutex> gate(mutex,std::try_to_lock);
    if(!gate.owns_lock())throw std::runtime_error("Concurrent operation on one model/stream instance is not supported");
    return gate;
}
}
