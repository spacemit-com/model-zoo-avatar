#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace SpacemiT::avatar {
constexpr std::size_t kWindowElements = 16 * 32 * 32;
struct RuntimeOptions {
    std::string ort_library = "libonnxruntime.so";
    std::string provider = "cpu";
    std::string ep_library = "libspacemit_ep.so";
    std::string ep_affinity;
    std::string profile_prefix;
    int threads = 4;
    int ep_threads = 2;
};
struct FeatureBatch {
    std::vector<float> values; // C order [N,8,2,32,32] == [N,16,32,32]
    std::vector<std::int64_t> frame_ids;
    std::size_t size() const { return frame_ids.size(); }
};
// Stateful and non-reentrant. One extractor per stream; reset before reusing it.
class FeatherHubertExtractor {
public:
    FeatherHubertExtractor(const std::string& model, const RuntimeOptions& options = {});
    ~FeatherHubertExtractor();
    FeatureBatch PushPcm(const std::int16_t* pcm, std::size_t samples, bool final = false);
    void Reset();
private:
    struct Impl; std::unique_ptr<Impl> impl_;
};
class UnetLipGenerator {
public:
    UnetLipGenerator(const std::string& model, const RuntimeOptions& options = {});
    ~UnetLipGenerator();
    std::vector<float> Infer(const std::vector<float>& image, const std::vector<float>& audio);
private:
    struct Impl; std::unique_ptr<Impl> impl_;
};
struct RenderOptions {
    bool feather = true;
    bool mouth_only_blend = false;
    bool native_pasteback = true;
    bool temporal_smooth = false;
    float mouth_sharpen = 0.15f;
    float smooth_alpha = 0.9f;
    float patch_lowpass = 0.0f;
};
// Owns the reference frame and temporal postprocessing state. Returned images own memory.
class PreparedFrame {
public:
    PreparedFrame(const cv::Mat& bgr, const std::vector<cv::Point2f>& landmarks,
                  const RenderOptions& options = {});
    ~PreparedFrame();
    std::vector<float> ImageTensor() const;
    void SetFrame(const cv::Mat& bgr, const std::vector<cv::Point2f>& landmarks,
                  const RenderOptions& options = {});
    cv::Rect FaceRect() const;
    cv::Mat Compose(const std::vector<float>& output);
    void Reset();
private:
    struct Impl; std::unique_ptr<Impl> impl_;
};
std::vector<cv::Point2f> ReadLandmarks(const std::string& path);
std::vector<std::int16_t> ReadWav16k(const std::string& path);
std::string RuntimeVersion();
} // namespace SpacemiT::avatar
