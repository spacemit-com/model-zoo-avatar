# Avatar API

数字人口型推理框架：由语音生成人物说话的逐帧画面，支持 C++ 和 Python。

## 功能特性

- **流式特征提取**: FeatherHuBERT 按块输入 PCM，输出逐帧音频特征窗口
- **唇形生成**: 人物专属 UNet 根据人脸输入与音频窗口生成嘴部
- **人脸预处理与贴回**: 由关键点裁剪人脸、构造模型输入，并把生成结果贴回原帧
- **离线文件渲染**: `avatar_worker` / `AvatarService` 从 WAV 生成 PNG 帧序列，流水线并行
- **推理后端**: FeatherHuBERT 使用 CPU，人物 UNet 使用 SpaceMIT EP（系统 `spacemit-onnxruntime`）

---

## C++ API

```cpp
namespace SpacemiT::avatar {

// 每帧音频窗口元素数：8 帧 × 2 × 32 × 32 = [16,32,32]
constexpr std::size_t kWindowElements = 16 * 32 * 32;

// =============================================================================
// RuntimeOptions - 推理运行时
// =============================================================================

struct RuntimeOptions {
    std::string provider = "cpu";   // "cpu" 或 "spacemit"；FeatherHuBERT 仅支持 "cpu"
    std::string profile_prefix;     // 非空时输出 ORT profile
    int threads = 4;                // ORT 线程数；EP 模式下为 CPU 回退线程数
    int ep_threads = 2;             // EP 计算线程数
};

// =============================================================================
// FeatureBatch - 一次 PushPcm 输出的特征窗口
// =============================================================================

struct FeatureBatch {
    std::vector<float> values;           // C 顺序 [N,16,32,32]
    std::vector<std::int64_t> frame_ids; // 全局帧号，严格连续
    std::size_t size() const;            // 窗口数 N
};

// =============================================================================
// FeatherHubertExtractor - 流式音频特征（有状态，不可重入）
// =============================================================================

class FeatherHubertExtractor {
public:
    explicit FeatherHubertExtractor(const std::string& model, const RuntimeOptions& options = {});

    // 输入 16 kHz 单声道 PCM16；final=true 时输出保留的尾部帧。
    // 每 640 个样本对应 1 帧（25 fps）。final 之后须 Reset() 才能继续输入。
    FeatureBatch PushPcm(const std::int16_t* pcm, std::size_t samples, bool final = false);
    void Reset();
};

// =============================================================================
// UnetLipGenerator - 人物 UNet（不可重入）
// =============================================================================

class UnetLipGenerator {
public:
    explicit UnetLipGenerator(const std::string& model, const RuntimeOptions& options = {});

    // image: [1,6,160,160]（PreparedFrame::ImageTensor），audio: [1,16,32,32]（一个特征窗口）
    // 返回 [1,3,160,160]；输出非有限值时抛出异常
    std::vector<float> Infer(const std::vector<float>& image, const std::vector<float>& audio);
};

// =============================================================================
// RenderOptions - 贴回与后处理
// =============================================================================

struct RenderOptions {
    bool feather = true;            // 贴回边缘羽化
    bool mouth_only_blend = false;  // 仅嘴部掩码（关闭原生贴回时生效）
    bool native_pasteback = true;   // 原生贴回：直接替换人脸区域，与 bianbu-avatar 一致
    bool temporal_smooth = false;   // 时间平滑（原生贴回时不生效）
    float mouth_sharpen = 0.15f;    // 嘴部锐化（原生贴回时不生效）
    float smooth_alpha = 0.9f;      // 时间平滑系数
    float patch_lowpass = 0.0f;     // 生成人脸低通
};

// =============================================================================
// PreparedFrame - 单帧预处理与贴回（保存跨帧后处理状态，不可重入）
// =============================================================================

class PreparedFrame {
public:
    // bgr: uint8 BGR 底帧；landmarks: 至少 110 个关键点
    PreparedFrame(const cv::Mat& bgr, const std::vector<cv::Point2f>& landmarks,
                  const RenderOptions& options = {});

    std::vector<float> ImageTensor() const;  // UNet 图像输入 [1,6,160,160]
    cv::Rect FaceRect() const;                // 人脸框
    void SetFrame(const cv::Mat& bgr, const std::vector<cv::Point2f>& landmarks,
                  const RenderOptions& options = {});  // 切换底帧，保留跨帧状态
    cv::Mat Compose(const std::vector<float>& output);  // 贴回，返回整帧 BGR
    void Reset();                             // 清除跨帧状态
};

// =============================================================================
// 工具函数
// =============================================================================

std::vector<cv::Point2f> ReadLandmarks(const std::string& path);  // 读取 .lms
std::vector<std::int16_t> ReadWav16k(const std::string& path);     // 读取 16 kHz 单声道 PCM16 WAV
std::string RuntimeVersion();                                      // 链接的 ORT 版本

}  // namespace SpacemiT::avatar
```

### C++ 示例

```cpp
#include "avatar_service.h"
#include <opencv2/imgcodecs.hpp>
using namespace SpacemiT::avatar;

int main() {
    // FeatherHuBERT 使用 CPU，人物 UNet 使用系统 SpaceMIT EP
    RuntimeOptions hubert_options;
    RuntimeOptions unet_options;
    unet_options.provider = "spacemit";
    FeatherHubertExtractor hubert("feather_hubert_dynamic.onnx", hubert_options);
    UnetLipGenerator unet("feifan_int8.onnx", unet_options);

    // 音频特征
    auto pcm = ReadWav16k("feifan_test.wav");
    auto batch = hubert.PushPcm(pcm.data(), pcm.size(), /*final=*/true);

    // 逐帧生成并贴回
    PreparedFrame frame(cv::imread("feifan/full_body_img/0.jpg"),
                        ReadLandmarks("feifan/landmarks/0.lms"));
    for (std::size_t i = 0; i < batch.size(); ++i) {
        std::vector<float> window(batch.values.begin() + i * kWindowElements,
                                  batch.values.begin() + (i + 1) * kWindowElements);
        auto face = unet.Infer(frame.ImageTensor(), window);
        cv::imwrite(std::to_string(batch.frame_ids[i]) + ".png", frame.Compose(face));
    }
    return 0;
}
```

示例固定使用一张底帧并串行执行。`examples/avatar_file_demo.cpp` 按帧轮换底帧并补齐结尾的静音特征，输出与 `avatar_worker` 逐字节一致；`avatar_worker` 另外把解码、推理、贴回与写 PNG 流水线并行。

链接：`libavatar.a` 为静态库，应用还需链接 OpenCV（core、imgproc、imgcodecs）、`libonnxruntime.so` 与 `libspacemit_ep.so`；通过 CMake `add_subdirectory` 引入时这些依赖随 `avatar` 目标传递。

---

## Python API

```python
"""
Avatar Python API
"""
import spacemit_avatar
from spacemit_avatar import (AvatarService, RenderResult, RuntimeOptions, RenderOptions,
                             FeatherHubertExtractor, UnetLipGenerator, PreparedFrame)

# =============================================================================
# AvatarService - 离线文件渲染（推荐）
# =============================================================================

class AvatarService:
    def __init__(self, config: str, worker: str | None = None):
        """
        Args:
            config: YAML 配置路径（schema: 1）
            worker: avatar_worker 路径；默认依次使用 AVATAR_WORKER、wheel 内置、PATH
        """

    def render_file(self, audio_path: str, output_dir: str, max_frames: int = 0) -> RenderResult:
        """在两个独立 worker 进程中提取特征并渲染 PNG；output_dir 须不存在或为空"""

    def cancel(self) -> None: ...   # 协作式取消，回收 worker 进程
    def close(self) -> None: ...    # 关闭服务；支持 with 语句

@dataclass(frozen=True)
class RenderResult:
    output_dir: Path
    frames: int
    extract_metrics: dict   # {"ok", "frames", "elapsed_ms"}
    render_metrics: dict

# =============================================================================
# RuntimeOptions / RenderOptions - 字段与 C++ 同名
# =============================================================================

options = RuntimeOptions()       # provider, profile_prefix, threads, ep_threads
render = RenderOptions()         # feather, mouth_only_blend, native_pasteback,
                                 # temporal_smooth, mouth_sharpen, smooth_alpha, patch_lowpass

# =============================================================================
# 低层模型接口
# =============================================================================

class FeatherHubertExtractor:
    def __init__(self, model: str, options: RuntimeOptions = RuntimeOptions()): ...
    def push_pcm(self, pcm: np.ndarray, final: bool = False) -> FeatureBatch:
        """pcm: 一维 int16；返回 FeatureBatch，windows 为 [N,16,32,32] float32，frame_ids 为帧号列表"""
    def reset(self) -> None: ...

class UnetLipGenerator:
    def __init__(self, model: str, options: RuntimeOptions = RuntimeOptions()): ...
    def infer(self, image: np.ndarray, audio: np.ndarray) -> np.ndarray:
        """image: [1,6,160,160] float32；audio: [1,16,32,32] float32；返回 [1,3,160,160]"""

class PreparedFrame:
    def __init__(self, bgr: np.ndarray, landmarks: np.ndarray, options: RenderOptions = RenderOptions()): ...
    image_tensor: np.ndarray                     # [1,6,160,160]
    face_rect: tuple[int, int, int, int]         # (x, y, w, h)
    def set_frame(self, bgr, landmarks, options=RenderOptions()) -> None: ...
    def compose(self, output: np.ndarray) -> np.ndarray:  # 返回整帧 uint8 BGR
    def reset(self) -> None: ...

def runtime_version() -> str: ...
```

### Python 示例

```python
import spacemit_avatar

# 离线文件渲染
with spacemit_avatar.AvatarService("configs/feifan.yaml") as engine:
    result = engine.render_file("feifan_test.wav", "output/frames")
    print(f"{result.frames} 帧，渲染 {result.render_metrics['elapsed_ms']} ms")
```

### 低层逐帧示例

```python
import cv2
import numpy as np
import spacemit_avatar as avatar

hubert = avatar.FeatherHubertExtractor("feather_hubert_dynamic.onnx")
options = avatar.RuntimeOptions()
options.provider = "spacemit"
unet = avatar.UnetLipGenerator("feifan_int8.onnx", options)

pcm = np.zeros(16000, dtype=np.int16)            # 1 秒 16 kHz PCM16
batch = hubert.push_pcm(pcm, final=True)

bgr = cv2.imread("feifan/full_body_img/0.jpg")
landmarks = np.loadtxt("feifan/landmarks/0.lms", dtype=np.float32)
frame = avatar.PreparedFrame(bgr, landmarks)
for window in batch.windows:
    face = unet.infer(frame.image_tensor, window[None])
    image = frame.compose(face)                   # uint8 BGR，可直接 cv2.imwrite
```

---

## 命令行 avatar_worker

```bash
avatar_worker extract --config CONFIG --audio WAV --output FEATURES.ftf
avatar_worker render  --config CONFIG --features FEATURES.ftf --output DIR [--max-frames N]
```

成功时向标准输出打印 `{"ok":true,"frames":N,"elapsed_ms":T}`，失败或收到 SIGTERM/SIGINT 时返回 2。`render` 输出 `00000000.png` 起的连续帧，目录须不存在或为空。

---

## 数据格式

- **音频**: RIFF WAVE，16000 Hz，单声道，PCM16 小端
- **帧率**: 25 fps，每帧 640 个样本；视频帧数为 `ceil(样本数 / 640)`
- **人物素材**: `full_body_img/<n>.jpg` 与 `landmarks/<n>.lms` 按编号配对；`.lms` 每行一个 `x y` 坐标，至少 110 行
- **特征文件（FTF1）**: 64 字节小端文件头（魔数 `FTF1`、版本 1、采样率 16000、帧率 25、窗口数、PCM 样本数），随后每窗口 16384 个 float32（`[16,32,32]`）
- **输出**: PNG，uint8 BGR，与底帧同尺寸

```bash
# 其他格式转换为 16 kHz 单声道 PCM16
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```
