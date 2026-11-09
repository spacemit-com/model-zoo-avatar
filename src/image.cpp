// Extracted from bianbu-avatar; exact upstream blocks tracked in config/source-baseline.json.
#include "avatar_service.h"
#include "non_reentrant.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
namespace SpacemiT::avatar::detail {
namespace fs=std::filesystem;
constexpr int kFaceSize=168,kInnerOffset=4,kInnerSize=160;
struct FrameAsset {cv::Mat face;cv::Rect bbox,mouthRoi;};
std::vector<cv::Point2f> readLandmarks(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("failed to open landmarks: " + path.string());
    }
    std::vector<cv::Point2f> points;
    float x = 0.0f;
    float y = 0.0f;
    while (stream >> x >> y) {
        points.emplace_back(x, y);
    }
    if (points.size() <= 52) {
        throw std::runtime_error("not enough landmarks: " + path.string());
    }
    return points;
}

cv::Rect computeFaceRect(const std::vector<cv::Point2f>& landmarks, const cv::Size& size) {
    const int xmin = static_cast<int>(landmarks[1].x);
    const int ymin = static_cast<int>(landmarks[52].y);
    const int xmax = static_cast<int>(landmarks[31].x);
    const int side = xmax - xmin;
    cv::Rect rect(xmin, ymin, side, side);
    rect &= cv::Rect(0, 0, size.width, size.height);
    if (rect.width <= 0 || rect.height <= 0) {
        throw std::runtime_error("invalid face rectangle");
    }
    return rect;
}

float computeMouthOpenRatio(const std::vector<cv::Point2f>& landmarks) {
    constexpr std::size_t mouthBegin = 90;
    constexpr std::size_t mouthEnd = 110;
    if (landmarks.size() < mouthEnd) {
        return 1.0f;
    }
    float xmin = landmarks[mouthBegin].x;
    float xmax = xmin;
    float ymin = landmarks[mouthBegin].y;
    float ymax = ymin;
    for (std::size_t index = mouthBegin + 1; index < mouthEnd; ++index) {
        xmin = std::min(xmin, landmarks[index].x);
        xmax = std::max(xmax, landmarks[index].x);
        ymin = std::min(ymin, landmarks[index].y);
        ymax = std::max(ymax, landmarks[index].y);
    }
    return (ymax - ymin) / std::max(xmax - xmin, 1.0f);
}

cv::Rect computeMouthRoi(
    const std::vector<cv::Point2f>& landmarks, const cv::Rect& bbox)
{
    constexpr std::size_t mouthBegin = 90;
    constexpr std::size_t mouthEnd = 110;
    float xmin = landmarks[mouthBegin].x;
    float xmax = xmin;
    float ymin = landmarks[mouthBegin].y;
    float ymax = ymin;
    for (std::size_t index = mouthBegin + 1; index < mouthEnd; ++index) {
        xmin = std::min(xmin, landmarks[index].x);
        xmax = std::max(xmax, landmarks[index].x);
        ymin = std::min(ymin, landmarks[index].y);
        ymax = std::max(ymax, landmarks[index].y);
    }
    const float width = std::max(xmax - xmin, 1.0f);
    const float height = std::max(ymax - ymin, 1.0f);
    // Keep the post-processing ROI tightly around the lip contour.  The
    // previous 30%/70% margins extended into the moustache, cheeks and chin;
    // any residual colour mismatch there is perceived as a second (blurred)
    // mouth.  A small horizontal/vertical guard is sufficient for the teeth
    // and lip corners while preserving the source video's skin detail.
    const float marginX = width * 0.18f;
    const float marginY = height * 0.42f;
    const float scaleX = static_cast<float>(kFaceSize) / bbox.width;
    const float scaleY = static_cast<float>(kFaceSize) / bbox.height;
    cv::Rect roi(
        cvFloor((xmin - marginX - bbox.x) * scaleX),
        cvFloor((ymin - marginY - bbox.y) * scaleY),
        cvCeil((width + marginX * 2.0f) * scaleX),
        cvCeil((height + marginY * 2.0f) * scaleY));
    roi &= cv::Rect(0, 0, kFaceSize, kFaceSize);
    if (roi.width <= 0 || roi.height <= 0) {
        throw std::runtime_error("invalid mouth ROI");
    }
    return roi;
}

cv::Mat createAlpha(
    const std::vector<cv::Point2f>& landmarks,
    const cv::Rect& bbox,
    bool feather,
    bool mouthOnlyBlend)
{
    cv::Mat alpha(bbox.size(), CV_32FC1, cv::Scalar(0.0f));
    if (mouthOnlyBlend) {
        // Match the training/inference pipeline: all 20 mouth landmarks
        // (90..109) define a soft ellipse.  A polygon made from only the
        // outer points produces the visible white/red hexagonal occluder and
        // its hard edge flashes after H.264 chroma subsampling.
        constexpr std::size_t mouthBegin = 90;
        constexpr std::size_t mouthEnd = 110;  // exclusive
        if (landmarks.size() < mouthEnd) {
            throw std::runtime_error("mouth-only blend requires mouth landmarks 90..109");
        }
        float xmin = landmarks[mouthBegin].x;
        float xmax = xmin;
        float ymin = landmarks[mouthBegin].y;
        float ymax = ymin;
        for (std::size_t index = mouthBegin + 1; index < mouthEnd; ++index) {
            xmin = std::min(xmin, landmarks[index].x);
            xmax = std::max(xmax, landmarks[index].x);
            ymin = std::min(ymin, landmarks[index].y);
            ymax = std::max(ymax, landmarks[index].y);
        }
        const float width = std::max(xmax - xmin, 1.0f);
        const float height = std::max(ymax - ymin, 1.0f);
        const cv::Point center(
            cvRound((xmin + xmax) * 0.5f) - bbox.x,
            cvRound((ymin + ymax) * 0.5f + height * 0.02f) - bbox.y);
        const int radiusX = std::max(1, cvRound(width * 0.72f));
        // The old vertical radius used height*1.15.  When the mouth opened,
        // 160/300 production frames became a vertical oval (worst aspect
        // ratio 1.29), exposing the residual boundary from nose to chin.
        // Cover the lip contour and teeth, but cap the mask to a horizontal
        // shape whose vertical radius never exceeds 78% of its horizontal
        // radius.  This still covers the maximum observed 116px mouth height.
        const int radiusY = std::max(1, cvRound(std::min(
            std::max(height * 0.78f, width * 0.34f),
            static_cast<float>(radiusX) * 0.78f)));
        // Build a smoothstep feather rather than a hard filled ellipse.  The
        // direct replacement reaches full weight across the complete lip
        // contour and fades only over the outer 18%.
        // The entire landmark lip contour must be in the full-replacement
        // core.  A 0.65 core left the lip extrema in the translucent feather
        // band, which retained a second outline from the source frame.
        constexpr float innerRadius = 0.82f;
        for (int y = 0; y < alpha.rows; ++y) {
            auto* row = alpha.ptr<float>(y);
            const float dy = static_cast<float>(y - center.y) /
                static_cast<float>(radiusY);
            for (int x = 0; x < alpha.cols; ++x) {
                const float dx = static_cast<float>(x - center.x) /
                    static_cast<float>(radiusX);
                const float radius = std::sqrt(dx * dx + dy * dy);
                if (radius <= innerRadius) {
                    row[x] = 1.0f;
                } else if (radius < 1.0f) {
                    const float t = (1.0f - radius) / (1.0f - innerRadius);
                    row[x] = t * t * (3.0f - 2.0f * t);
                }
            }
        }
    } else {
        std::vector<cv::Point> faceContour;
        faceContour.reserve(31);
        for (std::size_t index = 1; index <= 31; ++index) {
            faceContour.emplace_back(
                cvRound(landmarks[index].x) - bbox.x,
                cvRound(landmarks[index].y) - bbox.y);
        }
        cv::fillPoly(alpha, std::vector<std::vector<cv::Point>>{faceContour}, cv::Scalar(1.0f));
        const cv::Mat erosionKernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::erode(alpha, alpha, erosionKernel);
    }
    if (feather) {
        const int kernelSize = mouthOnlyBlend ? 9 : 15;
        const double sigma = mouthOnlyBlend ? 3.15 : 4.0;
        cv::GaussianBlur(
            alpha, alpha, cv::Size(kernelSize, kernelSize), sigma, sigma, cv::BORDER_REPLICATE);
    }
    cv::min(alpha, 1.0f, alpha);
    cv::max(alpha, 0.0f, alpha);
    return alpha;
}

std::vector<float> createImageInput(const cv::Mat& face) {
    const cv::Mat inner = face(cv::Rect(kInnerOffset, kInnerOffset, kInnerSize, kInnerSize));
    cv::Mat masked = inner.clone();
    // Match the original preprocessing exactly. cv::Rect uses an
    // exclusive right/bottom edge, so this covers x=[5,154], y=[5,149]. The
    // previous Point overload covered only x=[5,150], y=[5,145], changing the
    // model input distribution and producing unstable first mouth frames.
    cv::rectangle(masked, cv::Rect(5, 5, 150, 145), cv::Scalar(0, 0, 0), cv::FILLED);
    std::vector<float> input(6 * kInnerSize * kInnerSize);
    const std::size_t plane = static_cast<std::size_t>(kInnerSize) * kInnerSize;
    for (int y = 0; y < kInnerSize; ++y) {
        const auto* originalRow = inner.ptr<cv::Vec3b>(y);
        const auto* maskedRow = masked.ptr<cv::Vec3b>(y);
        for (int x = 0; x < kInnerSize; ++x) {
            const std::size_t offset = static_cast<std::size_t>(y) * kInnerSize + x;
            for (int channel = 0; channel < 3; ++channel) {
                input[channel * plane + offset] = originalRow[x][channel] / 255.0f;
                input[(channel + 3) * plane + offset] = maskedRow[x][channel] / 255.0f;
            }
        }
    }
    return input;
}

struct FacePostprocessScratch {
    cv::Mat face;
    cv::Mat decodedFloat;
    cv::Mat blurredMouth;
    cv::Mat lowCurrent;
    cv::Mat lowPrevious;
    cv::Mat coreMask;
    cv::Mat blendedFace;
    cv::Mat previousFace;
    // Reusable ROI buffers.  The hot path is called once per rendered frame;
    // keeping these allocations alive avoids allocator/OpenCV thread-pool
    // jitter that otherwise shows up as periodic low-FPS bursts.
    cv::Mat mouthScratch;
    cv::Mat mouthBlur;
    // Reused by the software residual compositor.  Allocating two full-size
    // resize targets for every frame causes allocator/page-fault jitter on K3.
    cv::Mat residualGenerated;
    cv::Mat residualReference;
    cv::Mat scaledAlpha;
    cv::Mat nativeAlpha;
    // nativeAlphaMask() is a pure function of the box size (margin derives from
    // it) and a speaking turn keeps that size for long stretches, so rebuilding
    // it every frame is wasted work.  Track the size the cached mask is for.
    int nativeAlphaWidth = -1;
    int nativeAlphaHeight = -1;
    bool hasPrevious = false;
    // Last known-good generated face.  The SpaceMIT EP can occasionally
    // return a corrupted tensor under sustained load; retaining a valid
    // residual prevents a single bad inference from becoming a red block in
    // the encoded stream.
    cv::Mat previousGeneratedFace;
    bool hasPreviousGenerated = false;
    std::size_t rejectedGeneratedFrames = 0;
};

bool generatedFaceLooksValid(const cv::Mat& face, const cv::Mat& reference)
{
    if (face.empty() || face.type() != CV_8UC3 || face.size() != reference.size()) {
        return false;
    }
    // Reject NaN/Inf before conversion and obvious saturated colour bands
    // (large contiguous regions where one RGB channel is clipped).  Natural
    // skin/lips do not contain tens of thousands of such pixels.
    // One walk for the moments and the saturation count.  cv::meanStdDev plus a
    // second loop double-reads every pixel; the population standard deviation
    // below is what meanStdDev returns.
    double sum[3] = {0.0, 0.0, 0.0};
    double sumSq[3] = {0.0, 0.0, 0.0};
    std::size_t saturated = 0;
    std::size_t total = 0;
    for (int y = 0; y < face.rows; ++y) {
        const auto* row = face.ptr<cv::Vec3b>(y);
        for (int x = 0; x < face.cols; ++x) {
            const int b = row[x][0], g = row[x][1], r = row[x][2];
            sum[0] += b; sum[1] += g; sum[2] += r;
            sumSq[0] += static_cast<double>(b) * b;
            sumSq[1] += static_cast<double>(g) * g;
            sumSq[2] += static_cast<double>(r) * r;
            const int hi = std::max({b, g, r});
            const int lo = std::min({b, g, r});
            if (hi >= 250 && hi - lo >= 90) {
                ++saturated;
            }
            ++total;
        }
    }
    if (total == 0) {
        return false;
    }
    const double inv = 1.0 / static_cast<double>(total);
    bool flat = false;
    for (int channel = 0; channel < 3; ++channel) {
        const double mean = sum[channel] * inv;
        const double variance = std::max(0.0, sumSq[channel] * inv - mean * mean);
        if (std::sqrt(variance) <= 2.0) {
            flat = true;
        }
    }
    return saturated <= static_cast<std::size_t>(static_cast<double>(total) * 0.08) && !flat;
}

const cv::Mat& createOutputFace(const float* output, const FrameAsset& asset,
    float mouthSharpen, bool temporalSmooth, float smoothAlpha, float patchLowpass,
    FacePostprocessScratch& scratch)
{
    asset.face.copyTo(scratch.face);
    cv::Mat inner = scratch.face(cv::Rect(kInnerOffset, kInnerOffset, kInnerSize, kInnerSize));
    // Merge the three planes straight into the 8-bit BGR patch.  The INT8 EP can
    // transiently return NaN/Inf or a slightly out-of-range value when a decode
    // frame is invalid; letting that reach the pixels creates saturated
    // white/primary blocks which flash in the mouth ROI.  Sanitising and
    // converting in the same walk replaces merge + patchNaNs + min + max +
    // convertTo, which read the same 3 x 160 x 160 floats five times.
    const std::size_t plane = static_cast<std::size_t>(kInnerSize) * kInnerSize;
    const float* planes[3] = {output, output + plane, output + 2 * plane};
    for (int y = 0; y < kInnerSize; ++y) {
        auto* dst = inner.ptr<cv::Vec3b>(y);
        const std::size_t rowOffset = static_cast<std::size_t>(y) * kInnerSize;
        for (int x = 0; x < kInnerSize; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                float value = planes[channel][rowOffset + static_cast<std::size_t>(x)];
                if (!std::isfinite(value)) {
                    value = 0.0f;
                }
                value = std::clamp(value, 0.0f, 1.0f);
                dst[x][channel] = static_cast<std::uint8_t>(value * 255.0f + 0.5f);
            }
        }
    }
    if (!generatedFaceLooksValid(scratch.face, asset.face)) {
        if (scratch.hasPreviousGenerated &&
            scratch.previousGeneratedFace.size() == scratch.face.size()) {
            scratch.previousGeneratedFace.copyTo(scratch.face);
            ++scratch.rejectedGeneratedFrames;
        } else {
            asset.face.copyTo(scratch.face);
        }
    } else {
        scratch.face.copyTo(scratch.previousGeneratedFace);
        scratch.hasPreviousGenerated = true;
    }
    if (temporalSmooth && asset.mouthRoi.width > 0 && asset.mouthRoi.height > 0) {
        // Smooth only the central mouth ellipse.  Keeping the surrounding
        // face on the current frame avoids a full-face ghosting/latency effect.
        scratch.coreMask.create(asset.face.size(), CV_32FC1);
        scratch.coreMask.setTo(cv::Scalar(0.0f));
        const cv::Point center(
            asset.mouthRoi.x + asset.mouthRoi.width / 2,
            asset.mouthRoi.y + asset.mouthRoi.height / 2);
        const cv::Size axes(
            std::max(1, cvRound(asset.mouthRoi.width * 0.30)),
            std::max(1, cvRound(asset.mouthRoi.height * 0.30)));
        cv::ellipse(scratch.coreMask, center, axes, 0.0, 0.0, 360.0,
            cv::Scalar(1.0f), cv::FILLED);
        cv::GaussianBlur(scratch.coreMask, scratch.coreMask, cv::Size(0, 0), 2.0, 2.0,
            cv::BORDER_REPLICATE);
        if (!scratch.hasPrevious || scratch.previousFace.size() != scratch.face.size()) {
            scratch.face.copyTo(scratch.previousFace);
            scratch.hasPrevious = true;
        } else {
            const float alpha = std::clamp(smoothAlpha, 0.0f, 1.0f);
            // EMA the low-frequency colour only.  Add back current-frame
            // detail so lip contours and teeth do not become soft.
            cv::GaussianBlur(scratch.face, scratch.lowCurrent, cv::Size(0, 0),
                1.0, 1.0, cv::BORDER_REPLICATE);
            cv::GaussianBlur(scratch.previousFace, scratch.lowPrevious,
                cv::Size(0, 0), 1.0, 1.0, cv::BORDER_REPLICATE);
            double motionSum = 0.0;
            std::size_t motionCount = 0;
            for (int y = 0; y < scratch.face.rows; ++y) {
                const auto* maskRow = scratch.coreMask.ptr<float>(y);
                const auto* currentLow = scratch.lowCurrent.ptr<cv::Vec3b>(y);
                const auto* previousLow = scratch.lowPrevious.ptr<cv::Vec3b>(y);
                for (int x = 0; x < scratch.face.cols; ++x) {
                    if (maskRow[x] > 0.35f) {
                        const cv::Vec3b& a = currentLow[x];
                        const cv::Vec3b& b = previousLow[x];
                        motionSum += (std::abs(int(a[0]) - int(b[0])) +
                            std::abs(int(a[1]) - int(b[1])) +
                            std::abs(int(a[2]) - int(b[2]))) / 3.0;
                        ++motionCount;
                    }
                }
            }
            const float motion = motionCount > 0 ?
                static_cast<float>(motionSum / motionCount / 255.0) : 0.0f;
            const float effectiveAlpha = std::clamp(
                alpha + 0.35f * std::clamp(motion * 4.0f, 0.0f, 1.0f),
                alpha, 1.0f);
            // Keep the destination allocation stable across frames.  clone()
            // allocates a new image on every speaking frame and can briefly
            // starve the renderer/MPP producer under long conversations.
            scratch.face.copyTo(scratch.blendedFace);
            for (int y = 0; y < scratch.face.rows; ++y) {
                const auto* maskRow = scratch.coreMask.ptr<float>(y);
                const auto* currentRow = scratch.face.ptr<cv::Vec3b>(y);
                [[maybe_unused]] const auto* previousRow = scratch.previousFace.ptr<cv::Vec3b>(y);
                const auto* currentLow = scratch.lowCurrent.ptr<cv::Vec3b>(y);
                const auto* previousLow = scratch.lowPrevious.ptr<cv::Vec3b>(y);
                auto* blendedRow = scratch.blendedFace.ptr<cv::Vec3b>(y);
                for (int x = 0; x < scratch.face.cols; ++x) {
                    const float weight = maskRow[x];
                    for (int channel = 0; channel < 3; ++channel) {
                        const float emaLow = effectiveAlpha * currentLow[x][channel] +
                            (1.0f - effectiveAlpha) * previousLow[x][channel];
                        const float detail = currentRow[x][channel] - currentLow[x][channel];
                        const float filtered = std::clamp(emaLow + 0.95f * detail, 0.0f, 255.0f);
                        blendedRow[x][channel] = cv::saturate_cast<std::uint8_t>(
                            currentRow[x][channel] * (1.0f - weight) + filtered * weight);
                    }
                }
            }
            scratch.blendedFace.copyTo(scratch.face);
            scratch.face.copyTo(scratch.previousFace);
        }
    } else if (!temporalSmooth) {
        scratch.hasPrevious = false;
    }
    if (mouthSharpen > 0.0f && asset.mouthRoi.width > 0 && asset.mouthRoi.height > 0) {
        // Fast luminance-only unsharp mask.  The previous implementation did
        // cvtColor -> split -> several floating-point Mats -> merge -> cvtColor
        // for every frame.  The mouth ROI is small, so a direct 3x3 box filter
        // gives the same edge-preserving behaviour without heap allocations or
        // colour-space conversions.  Chroma is left untouched, avoiding teeth
        // halos and keeping the generated lip colour stable.
        const cv::Rect bounds(0, 0, scratch.face.cols, scratch.face.rows);
        const cv::Rect roi = asset.mouthRoi & bounds;
        if (roi.width > 2 && roi.height > 2) {
            scratch.mouthScratch.create(roi.size(), CV_8UC1);
            scratch.mouthBlur.create(roi.size(), CV_8UC1);
            cv::Mat mouth = scratch.face(roi);
            for (int y = 0; y < roi.height; ++y) {
                const auto* src = mouth.ptr<cv::Vec3b>(y);
                auto* lum = scratch.mouthScratch.ptr<std::uint8_t>(y);
                for (int x = 0; x < roi.width; ++x) {
                    const auto& p = src[x];
                    lum[x] = static_cast<std::uint8_t>(
                        (77 * static_cast<int>(p[2]) +
                         150 * static_cast<int>(p[1]) +
                         29 * static_cast<int>(p[0]) + 128) >> 8);
                }
            }
            cv::blur(scratch.mouthScratch, scratch.mouthBlur, cv::Size(3, 3),
                cv::Point(-1, -1), cv::BORDER_REPLICATE);
            const float gain = std::clamp(mouthSharpen, 0.0f, 1.0f);
            for (int y = 0; y < roi.height; ++y) {
                auto* dst = mouth.ptr<cv::Vec3b>(y);
                const auto* lum = scratch.mouthScratch.ptr<std::uint8_t>(y);
                const auto* low = scratch.mouthBlur.ptr<std::uint8_t>(y);
                for (int x = 0; x < roi.width; ++x) {
                    const float high = static_cast<float>(lum[x]) - low[x];
                    const float edgeGain = std::min(1.0f, std::abs(high) / 12.0f);
                    const float whiteGuard = 1.0f - 0.65f * std::clamp(
                        (static_cast<float>(low[x]) - 218.0f) / 37.0f, 0.0f, 1.0f);
                    const int delta = cvRound(high * edgeGain * whiteGuard * gain);
                    for (int channel = 0; channel < 3; ++channel) {
                        dst[x][channel] = cv::saturate_cast<std::uint8_t>(
                            static_cast<int>(dst[x][channel]) + delta);
                    }
                }
            }
        }
    }
    if (patchLowpass > 0.0f) {
        // The historical composite pasted the network's 160px output
        // *upscaled* into the face box (1.3x-1.8x), and that interpolation
        // attenuated the top of the patch's spectrum -- which is where the
        // model's frame-to-frame synthesis noise lives.  A ~1:1 paste has no
        // such filter, and the pasted face then reads as jittering.  This is
        // the same attenuation, made explicit and tunable: 0 keeps the sharp
        // 1:1 paste, a sigma of ~1 trades a little softness for stability.
        cv::GaussianBlur(scratch.face, scratch.face, cv::Size(0, 0),
            patchLowpass, patchLowpass, cv::BORDER_REPLICATE);
    }
    return scratch.face;
}

const cv::Mat& nativeAlphaMask(const cv::Rect& bbox, FacePostprocessScratch& scratch)
{
    // The whole face box comes from the model here, but its border must not be
    // visible.  The box follows *this frame's* landmarks, so a hard cut moves by
    // a pixel or two every frame and the eye reads that as a jittering
    // rectangle around the face -- which is exactly what the booth reported.
    // Ramp the outermost ~10% instead: the fully generated area still covers the
    // face, the transition blends into the base, and 1-2 px of box movement is
    // invisible because it happens inside a soft ramp.
    if (!scratch.nativeAlpha.empty() &&
        scratch.nativeAlphaWidth == bbox.width &&
        scratch.nativeAlphaHeight == bbox.height &&
        scratch.nativeAlpha.type() == CV_32FC1) {
        return scratch.nativeAlpha;
    }
    scratch.nativeAlpha.create(bbox.size(), CV_32FC1);
    scratch.nativeAlphaWidth = bbox.width;
    scratch.nativeAlphaHeight = bbox.height;
    const int margin = std::clamp(
        static_cast<int>(std::lround(std::min(bbox.width, bbox.height) * 0.10)), 2, 64);
    for (int y = 0; y < bbox.height; ++y) {
        auto* row = scratch.nativeAlpha.ptr<float>(y);
        const float edgeY = std::min(static_cast<float>(y) + 0.5f,
            static_cast<float>(bbox.height) - 0.5f - static_cast<float>(y));
        const float weightY = std::clamp(edgeY / margin, 0.0f, 1.0f);
        for (int x = 0; x < bbox.width; ++x) {
            const float edgeX = std::min(static_cast<float>(x) + 0.5f,
                static_cast<float>(bbox.width) - 0.5f - static_cast<float>(x));
            row[x] = weightY * std::clamp(edgeX / margin, 0.0f, 1.0f);
        }
    }
    return scratch.nativeAlpha;
}

void pasteResidualFace(const cv::Mat& generatedFace, const cv::Mat& referenceFace,
    const cv::Rect& bbox, const cv::Mat& alphaMask, float alphaScale,
    bool directBlend, cv::Mat& frame, FacePostprocessScratch& scratch)
{
    cv::Mat destination = frame(bbox);
    // Linear interpolation is sufficient for the residual path (the source
    // frame already supplies high-frequency detail) and is substantially
    // cheaper than cubic interpolation at 1080p.  Reuse the resize targets so
    // that long runs do not periodically hit the allocator.
    scratch.residualGenerated.create(bbox.size(), generatedFace.type());
    scratch.residualReference.create(bbox.size(), referenceFace.type());
    // The network output is only 160x160 while the production face ROI is
    // roughly 480x480.  Cubic interpolation avoids exposing block pixels when
    // the first speaking frames are blended at a low alpha.
    cv::resize(generatedFace, scratch.residualGenerated, bbox.size(),
        0.0, 0.0, cv::INTER_CUBIC);
    cv::resize(referenceFace, scratch.residualReference, bbox.size(),
        0.0, 0.0, cv::INTER_CUBIC);
    const cv::Mat& generated = scratch.residualGenerated;
    const cv::Mat& reference = scratch.residualReference;
    for (int y = 0; y < destination.rows; ++y) {
        auto* destinationRow = destination.ptr<cv::Vec3b>(y);
        const auto* generatedRow = generated.ptr<cv::Vec3b>(y);
        const auto* referenceRow = reference.ptr<cv::Vec3b>(y);
        const auto* alphaRow = alphaMask.ptr<float>(y);
        for (int x = 0; x < destination.cols; ++x) {
            const float alpha = alphaRow[x] * alphaScale;
            for (int channel = 0; channel < 3; ++channel) {
                // The residual compositor preserves useful source skin
                // detail, but it also adds the source lip edge back on top of
                // the generated lip.  Mouth-only mode therefore performs a
                // standard alpha replacement; the face-contour mode retains
                // the legacy detail-preserving residual behaviour.
                const float referenceValue = directBlend
                    ? static_cast<float>(destinationRow[x][channel])
                    : static_cast<float>(referenceRow[x][channel]);
                destinationRow[x][channel] = cv::saturate_cast<std::uint8_t>(
                    destinationRow[x][channel] + alpha *
                    (generatedRow[x][channel] - referenceValue));
            }
        }
    }
}

} // namespace detail
namespace SpacemiT::avatar {
struct PreparedFrame::Impl {
    std::mutex mutex;detail::FrameAsset asset; cv::Mat base,alpha;
    std::vector<float> input;RenderOptions options;detail::FacePostprocessScratch scratch;
};
PreparedFrame::PreparedFrame(const cv::Mat& bgr,const std::vector<cv::Point2f>& points,const RenderOptions& o):impl_(std::make_unique<Impl>()){SetFrame(bgr,points,o);}
PreparedFrame::~PreparedFrame()=default;
void PreparedFrame::SetFrame(const cv::Mat& bgr,const std::vector<cv::Point2f>& points,const RenderOptions& o){
    auto gate=detail::Enter(impl_->mutex);
    if(bgr.empty()||bgr.type()!=CV_8UC3||bgr.cols>16384||bgr.rows>16384)throw std::invalid_argument("Expected nonempty uint8 BGR image");
    if(points.size()<110)throw std::invalid_argument("Expected at least 110 landmarks");
    for(const auto& p:points)if(!std::isfinite(p.x)||!std::isfinite(p.y)||std::abs(p.x)>1e6f||std::abs(p.y)>1e6f)throw std::invalid_argument("Invalid landmark coordinate");
    if(!std::isfinite(o.mouth_sharpen)||!std::isfinite(o.smooth_alpha)||!std::isfinite(o.patch_lowpass)||o.mouth_sharpen<0||o.mouth_sharpen>4||o.smooth_alpha<0||o.smooth_alpha>1||o.patch_lowpass<0||o.patch_lowpass>4)throw std::invalid_argument("Invalid render options");
    detail::FrameAsset next;next.bbox=detail::computeFaceRect(points,bgr.size());
    cv::resize(bgr(next.bbox),next.face,{168,168},0,0,cv::INTER_AREA);next.mouthRoi=detail::computeMouthRoi(points,next.bbox);
    auto input=detail::createImageInput(next.face);auto alpha=detail::createAlpha(points,next.bbox,o.feather,o.mouth_only_blend);
    impl_->base=bgr.clone();impl_->asset=std::move(next);impl_->input=std::move(input);impl_->alpha=std::move(alpha);impl_->options=o;
}
std::vector<float> PreparedFrame::ImageTensor()const{auto gate=detail::Enter(impl_->mutex);return impl_->input;}
cv::Rect PreparedFrame::FaceRect()const{auto gate=detail::Enter(impl_->mutex);return impl_->asset.bbox;}
void PreparedFrame::Reset(){auto gate=detail::Enter(impl_->mutex);impl_->scratch=detail::FacePostprocessScratch{};}
cv::Mat PreparedFrame::Compose(const std::vector<float>& output){
    auto gate=detail::Enter(impl_->mutex);
    if(output.size()!=3*160*160)throw std::invalid_argument("Expected output [1,3,160,160]");
    for(float v:output)if(!std::isfinite(v))throw std::invalid_argument("Nonfinite generated tensor");
    const auto& o=impl_->options;auto& s=impl_->scratch;const auto& a=impl_->asset;
    const auto& face=detail::createOutputFace(output.data(),a,o.native_pasteback?0:o.mouth_sharpen,o.native_pasteback?false:o.temporal_smooth,o.smooth_alpha,o.patch_lowpass,s);
    const auto& alpha=o.native_pasteback?detail::nativeAlphaMask(a.bbox,s):impl_->alpha;
    auto result=impl_->base.clone();detail::pasteResidualFace(face,a.face,a.bbox,alpha,1.0f,o.mouth_only_blend,result,s);return result;
}
std::vector<cv::Point2f> ReadLandmarks(const std::string& path){return detail::readLandmarks(path);}
}
