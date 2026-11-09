#include "runtime.h"
#include <dlfcn.h>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <map>

namespace feathertalk::detail {
namespace {
std::mutex lock;
std::string selected;
std::string version;
std::map<std::string, void*> providers;
}
void InitRuntime(const std::string& library) {
    std::lock_guard<std::mutex> guard(lock);
    if (library.empty()) throw std::invalid_argument("ORT library must not be empty");
    const auto identity = std::filesystem::exists(library) ? std::filesystem::canonical(library).string() : library;
    if (!selected.empty()) {
        if (selected != identity) throw std::runtime_error("One ORT binary per process: use the isolated pipeline for different runtimes");
        return;
    }
    // Keep runtime handles alive for process lifetime: ORT owns TLS and static state.
    void* handle = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) throw std::runtime_error(std::string("Cannot load ORT: ") + dlerror());
    using GetBase = const OrtApiBase* (*)();
    auto get = reinterpret_cast<GetBase>(dlsym(handle, "OrtGetApiBase"));
    if (!get) {
        dlclose(handle);
        throw std::runtime_error("ORT missing OrtGetApiBase");
    }
    const auto* api = get()->GetApi(ORT_API_VERSION);
    if (!api) {
        dlclose(handle);
        throw std::runtime_error("ORT C API version does not match build headers");
    }
    // SpaceMIT EP resolves ORT symbols globally. Publish the handle only after
    // validation so a rejected runtime cannot affect the next load attempt.
    if (!dlopen(library.c_str(), RTLD_NOLOAD | RTLD_NOW | RTLD_GLOBAL)) {
        dlclose(handle);
        throw std::runtime_error("Cannot promote validated ORT runtime");
    }
    Ort::InitApi(api); selected = identity; version = get()->GetVersionString();
    dlclose(handle);
}
Ort::Env& Environment() { static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "feathertalk"); return env; }
Ort::SessionOptions SessionOptions(const RuntimeOptions& o) {
    if (o.threads < 1 || o.threads > 256 || o.ep_threads < 1 || o.ep_threads > 64) throw std::invalid_argument("Invalid thread count");
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(o.threads); so.SetInterOpNumThreads(1); so.SetExecutionMode(ORT_SEQUENTIAL);
    if (o.provider == "spacemit") {
        std::lock_guard<std::mutex> guard(lock);
        auto existing = providers.find(o.ep_library);
        void* ep = existing == providers.end() ? dlopen(o.ep_library.c_str(), RTLD_NOW | RTLD_LOCAL) : existing->second;
        if (!ep) throw std::runtime_error(std::string("Cannot load SpaceMIT EP: ") + dlerror());
        using Init = OrtStatus* (*)(OrtSessionOptions*, const char* const*, const char* const*, std::size_t);
        auto init = reinterpret_cast<Init>(dlsym(ep, "OrtSessionOptionsSpaceMITEnvInit"));
        if (!init) {
            if (existing == providers.end()) dlclose(ep);
            throw std::runtime_error("SpaceMIT EP missing initialization entry");
        }
        providers.emplace(o.ep_library, ep);
        std::map<std::string,std::string> values{{"SPACEMIT_EP_INTRA_THREAD_NUM",std::to_string(o.ep_threads)},
            {"SPACEMIT_EP_INTER_THREAD_NUM","1"},{"SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD","0"}};
        if (!o.ep_affinity.empty()) values["SPACEMIT_EP_INTRA_THREAD_AFFINITY"] = o.ep_affinity;
        std::vector<const char*> keys, vals;
        for (const auto& v : values) { keys.push_back(v.first.c_str()); vals.push_back(v.second.c_str()); }
        Ort::ThrowOnError(init(so, keys.data(), vals.data(), keys.size()));
    } else if (o.provider != "cpu") throw std::invalid_argument("provider must be cpu or spacemit");
    if (!o.profile_prefix.empty()) so.EnableProfiling(o.profile_prefix.c_str());
    return so;
}
void CheckTensor(const Ort::Value& v, const std::vector<std::int64_t>& shape) {
    if (!v.IsTensor()) throw std::runtime_error("Expected tensor");
    auto info = v.GetTensorTypeAndShapeInfo();
    if (info.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || info.GetShape()!=shape)
        throw std::runtime_error("Unexpected tensor dtype or shape");
}
}
namespace feathertalk {
std::string RuntimeVersion() { std::lock_guard<std::mutex> guard(detail::lock); return detail::selected.empty()?"not-loaded":detail::version; }
}
