# Avatar 组件

## 1. 项目简介

本组件从 bianbu-avatar 数字人中抽取**音频驱动口型**能力：输入一段语音，输出人物说话的逐帧画面。它包含 FeatherHuBERT 音频特征和人物专属 UNet 唇形生成两个模型及其前后处理，提供 C++ 接口、Python 绑定和离线示例。ASR、LLM、TTS、WebRTC 和实时调度不在本组件内。

| 类别 | 支持 |
| --- | --- |
| 部署方式 | **本地**（ONNX 推理），离线两阶段：先提取音频特征，再逐帧渲染 |
| 推理后端 | FeatherHuBERT：CPU；人物 UNet：SpaceMIT EP（系统 `spacemit-onnxruntime`） |
| 人物 | `zihan`（函姐）、`feifan`（进迭小哥） |
| 输入 / 输出 | 16 kHz 单声道 PCM16 WAV → 25 fps 的 PNG 帧序列，K3 上渲染约 29 fps |
| 接口 | C++（`include/avatar_service.h`，命名空间 `SpacemiT::avatar`）、Python（`spacemit_avatar`）、命令行 `avatar_worker` |
| 平台 | SpacemiT K3 / Bianbu 4.0 / Python 3.14。Windows 仅支持下载器与主机测试 |

模型：

| 模型 | 文件 | 输入 | 输出 |
| --- | --- | --- | --- |
| FeatherHuBERT | `feather_hubert_dynamic.onnx` | FP32 `waveform [1,samples]`，16 kHz | FP32 `hidden [1,tokens,1024]` |
| 人物 UNet | `zihan_int8.onnx`、`feifan_int8.onnx` | FP32 `input [1,6,160,160]`、`audio [1,16,32,32]` | FP32 `output [1,3,160,160]` |

UNet 的 INT8 是图内量化，公开张量仍为 FP32。每个人物的 UNet 必须与该人物的 750 帧底图和关键点配套使用，不支持直接换成任意人像。

处理流程：

```text
WAV ──avatar_worker extract──▶ 特征文件（FTF1）──avatar_worker render──▶ PNG 帧 ──ffmpeg──▶ mp4
       FeatherHuBERT（CPU）                     UNet（SpaceMIT EP）+ 人脸预处理 + 贴回
```

实现细节见 [实现与运行边界](docs/ARCHITECTURE.md)。

## 2. 验证模型

按以下顺序完成依赖安装、模型准备与示例运行。

### 2.1. 安装依赖

- **编译环境**：CMake ≥ 3.18，C++17 编译器，Linux。
- **必选**：OpenCV（core、imgproc、imgcodecs）、yaml-cpp、`spacemit-onnxruntime`（ONNX Runtime 与 SpaceMIT EP 的头文件和运行库）。
- **可选**：pybind11（Python 绑定）、zstd 或 python3-zstandard（解压资源）、FFmpeg（把帧合成视频）。

K3 / Bianbu 4.0 上全部使用系统软件包：

```bash
sudo apt-get install -y --no-install-recommends cmake ninja-build pkg-config \
  libopencv-core-dev libopencv-imgproc-dev libopencv-imgcodecs-dev libyaml-cpp-dev \
  python3-dev pybind11-dev python3-pybind11 python3-build python3-wheel python3-setuptools \
  python3-numpy python3-yaml zstd ffmpeg spacemit-onnxruntime
```

运行时按配置加载系统的 `libonnxruntime.so` 与 `libspacemit_ep.so`，构建时不链接。头文件不在默认路径时，配置时加 `-DONNXRUNTIME_INCLUDE_DIR=/path/to/include`。

### 2.2. 下载模型

模型、人物素材和测试音频通过下载器获取，按 `config/resources.json` 校验大小和 SHA256，支持断点续传。

**资源源**：进迭时空镜像 <https://archive.spacemit.com/spacemit-ai/model_zoo/avatar/>，测试音频位于 <https://archive.spacemit.com/spacemit-ai/model_zoo/assets/audio/>

```bash
# FeatherHuBERT + 人物 UNet
bash scripts/download_models.sh --persona feifan
bash scripts/download_models.sh --persona zihan
# 人物底帧与关键点（--frames 0 表示全部 750 帧，默认只取前 32 帧）及测试音频
bash scripts/download_assets.sh --persona feifan --frames 0
bash scripts/download_assets.sh --persona zihan --frames 0
```

| 内容 | 默认位置 |
| --- | --- |
| 模型 | `~/.cache/models/avatar/`：`feather_hubert_dynamic.onnx`、`zihan_int8.onnx`、`feifan_int8.onnx` |
| 人物素材 | `~/.cache/assets/avatar/<人物>/full_body_img/*.jpg`、`landmarks/*.lms` |
| 测试音频 | `~/.cache/assets/avatar/audio/`：`feifan_test.wav`（9.0 s）、`zihan_test.wav`（5.6 s）、本地生成的 `silence.wav` |

其他用法：`--model hubert` 只下载特征模型；`--dry-run` 预览；`--offline` 只使用已缓存的归档；`--models-dir`、`--assets-dir`、`--cache-dir` 指定位置（修改后同步修改配置文件中的路径）。每个人物的归档约 64 MB，同时包含权重和素材，下载器只解出选中的文件，模型和素材共享缓存。已存在且内容不同的文件会被保留，加 `--force` 才替换。

### 2.3. 测试

先构建（在 ai-sdk 中集成的方式见 [3.1](#31-构建与集成产物)）：

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure
export AVATAR_WORKER="$PWD/build/avatar_worker"
```

为两个人物生成口型视频：

```bash
# 进迭小哥
bash examples/cpp/run_pipeline.sh configs/feifan.yaml \
  ~/.cache/assets/avatar/audio/feifan_test.wav output/feifan-frames
ffmpeg -framerate 25 -i output/feifan-frames/%08d.png \
  -i ~/.cache/assets/avatar/audio/feifan_test.wav \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest output/feifan.mp4

# 函姐
bash examples/cpp/run_pipeline.sh configs/zihan.yaml \
  ~/.cache/assets/avatar/audio/zihan_test.wav output/zihan-frames
ffmpeg -framerate 25 -i output/zihan-frames/%08d.png \
  -i ~/.cache/assets/avatar/audio/zihan_test.wav \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest output/zihan.mp4
```

输出目录必须不存在或为空。视频帧数为 `ceil(PCM 样本数 / 640)`，底帧按 0→749→0 往返使用。

**核分配**：FeatherHuBERT 在 X100 0–7 用 CPU 4 线程。UNet 使用 SpaceMIT EP 的 2 个计算线程；worker 进程、ORT 处理 EP 不支持节点的 4 个 CPU 线程以及渲染流水线线程都在 X100 0–7。系统 EP 2.0.8 自行放置 A100 线程：每个 A100 核各起一个，计算只在 A100 8、9 上进行，`ep_affinity` 不生效，因此配置中不设置。

**渲染流水线**：`avatar_worker render` 把每帧拆成预取（解码底帧、计算模型输入）、UNet 推理、按帧序贴回合成、多线程写 PNG 四段并行执行，推理期间 X100 同时处理前后帧。贴回在同一份后处理状态上按帧序进行，输出与逐帧串行处理逐字节一致。

更完整的测试入口（与 `test.yaml` 一致）：

```bash
bash tests/run_sdk_test.sh core        # C++ 合约测试
bash tests/run_sdk_test.sh python      # 下载器与 worker 生命周期
bash tests/run_sdk_test.sh installed   # 安装到临时前缀后加载绑定、启动 worker
```

真实模型差分测试 `bash tests/run_sdk_test.sh parity` 需设置 `AVATAR_TEST_CPU_ORT`、`AVATAR_TEST_HUBERT`、`AVATAR_TEST_UNET`，见 [验证方法](docs/VALIDATION.md)。

## 3. 应用开发

### 3.1. 构建与集成产物

在 ai-sdk 中，本组件位于 `avatar/`，与 `vision/`、`tts/` 同级：

```bash
cd ai-sdk
source build/envsetup.sh
cd avatar && mm -py
```

| 产物 | 位置 |
| --- | --- |
| 共享库 | `output/staging/lib/libavatar.so*` |
| 头文件 | `output/staging/include/avatar_service.h` |
| 命令行 | `output/staging/bin/avatar_worker`（使用时将该 `bin` 加入 `PATH`） |
| Python 包 | `output/staging/lib/python3.14/site-packages/spacemit_avatar/`（可用 `AVATAR_PYTHON_INSTALL_DIR` 调整相对路径） |
| wheel | `output/wheels/avatar/` |

单独构建时使用上面的 CMake 命令，`cmake --install build --prefix <前缀>` 得到相同布局；只要 C++ 时加 `-DBUILD_PYTHON_BINDINGS=OFF`。单独打 wheel：

```bash
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
python -m build --wheel --no-isolation
python -m pip install --no-deps dist/spacemit_avatar-*.whl
```

wheel 标签为 `cp314-cp314-linux_riscv64`，运行时使用系统 OpenCV、yaml-cpp 和 `spacemit-onnxruntime`。

### 3.2. API 使用

**Python（推荐）**：`AvatarService` 用两个独立的 `avatar_worker` 子进程完成离线渲染，支持超时、取消和关闭。

```python
import os
from spacemit_avatar import AvatarService

audio = os.path.expanduser("~/.cache/assets/avatar/audio/feifan_test.wav")
with AvatarService("configs/feifan.yaml") as engine:
    result = engine.render_file(audio, "output/python-frames")
    print(result.frames, result.extract_metrics, result.render_metrics)
```

worker 默认从包内 `bin/` 或 `AVATAR_WORKER` 查找。取消是协作式的：在当前推理返回后退出；取消或失败时已写出的部分 PNG 会保留，不能当作完整结果。

**Python 低层接口**：直接调用模型，适合自行编排。

```python
import numpy as np
from spacemit_avatar import RuntimeOptions, FeatherHubertExtractor

options = RuntimeOptions()  # FeatherHuBERT 使用默认的 CPU、4 线程、libonnxruntime.so
extractor = FeatherHubertExtractor("/path/to/feather_hubert_dynamic.onnx", options)
batch = extractor.push_pcm(np.zeros(16000, dtype=np.int16), final=True)
print(batch.windows.shape, batch.frame_ids)  # [N,16,32,32]
extractor.reset()
```

其余低层类：`UnetLipGenerator.infer(image, audio)`、`PreparedFrame(bgr, landmarks, options)` 的 `image_tensor`、`face_rect`、`set_frame`、`compose(output)`。UNet 的 `RuntimeOptions` 设置 `provider = "spacemit"`。

**C++**：

```cpp
#include "avatar_service.h"
#include <opencv2/imgcodecs.hpp>
using namespace SpacemiT::avatar;

int main() {
    RuntimeOptions hubert_options;          // CPU，4 线程
    RuntimeOptions unet_options;
    unet_options.provider = "spacemit";     // 系统 libspacemit_ep.so
    unet_options.ep_threads = 2;
    FeatherHubertExtractor hubert("feather_hubert_dynamic.onnx", hubert_options);
    UnetLipGenerator unet("feifan_int8.onnx", unet_options);

    auto pcm = ReadWav16k("feifan_test.wav");
    auto batch = hubert.PushPcm(pcm.data(), pcm.size(), /*final=*/true);

    PreparedFrame frame(cv::imread("feifan/full_body_img/0.jpg"),
                        ReadLandmarks("feifan/landmarks/0.lms"));
    for (std::size_t i = 0; i < batch.size(); ++i) {
        std::vector<float> window(batch.values.begin() + i * kWindowElements,
                                  batch.values.begin() + (i + 1) * kWindowElements);
        auto face = unet.Infer(frame.ImageTensor(), window);
        cv::imwrite(std::to_string(batch.frame_ids[i]) + ".png", frame.Compose(face));
    }
}
```

示例为简洁起见固定使用一张底帧并串行执行；`avatar_worker` 会按帧轮换底帧、用流水线并行处理，并在结尾补齐静音特征。

使用约束：

- 模型实例不可并发调用；`final` 之后须 `Reset()` 才能继续输入。
- 同一进程只能加载一套 ORT，选择不同路径会报错；两个模型使用同一个 `libonnxruntime.so`。`AvatarService` 的两个 worker 进程相互隔离，可用 `AVATAR_HUBERT_LIBRARY_DIR`、`AVATAR_UNET_LIBRARY_DIR` 仅为对应子进程设置库搜索路径。
- Python 绑定在推理时释放 GIL，输入输出数组拥有独立内存。

### 3.3. 配置参数

`avatar_worker` 与 `AvatarService` 读取 YAML 配置。路径支持 `~/` 和相对于配置文件的相对路径。

| 段 | 字段 | 说明 |
| --- | --- | --- |
| — | `schema` | 固定为 `1` |
| `hubert` / `unet` | `model` | 模型路径 |
| | `provider` | FeatherHuBERT 为 `cpu`；人物 UNet 为 `spacemit` |
| | `ort_library` | ONNX Runtime 库，默认系统 `libonnxruntime.so` |
| | `ep_library` | SpaceMIT EP 库，默认系统 `libspacemit_ep.so` |
| | `threads` | ORT 线程数；UNet 中为处理 EP 不支持节点的 CPU 线程数 |
| | `ep_threads` | EP 计算线程数，默认 2 |
| | `cpu_affinity` | 进程绑核列表，如 `[0, 1, 2, 3, 4, 5, 6, 7]` |
| | `timeout_s` | `AvatarService` 中该阶段的超时，默认 3600 |
| `assets` | `person` | 人物素材目录，含 `full_body_img/` 与 `landmarks/` |
| `render` | `native_pasteback` | 原生贴回，默认开启，与 bianbu-avatar 一致 |
| | `feather` | 贴回边缘羽化，默认开启 |
| | `mouth_only_blend` | 仅嘴部掩码；关闭原生贴回时生效 |
| | `mouth_sharpen` / `temporal_smooth` / `smooth_alpha` / `patch_lowpass` | 后处理选项；原生贴回时锐化与时间平滑不生效 |

自带配置：`configs/feifan.yaml`、`configs/zihan.yaml`。

## 4. 常见问题

| 现象 | 可能原因 | 处理 |
| --- | --- | --- |
| 下载报 404 或校验失败 | 资源地址变更或本地仓库较旧 | 更新仓库后重新下载；`--force` 替换内容不同的旧文件。 |
| 提示 WAV 格式不支持 | 采样率或声道不符 | 用 `ffmpeg -i in.wav -ar 16000 -ac 1 -c:a pcm_s16le out.wav` 转换。 |
| 提示输出目录非空 | 防止混入旧帧 | 使用新的或空的输出目录。 |
| `One ORT binary per process` | 同一进程选择了两套 ORT | 两个模型使用同一 `ort_library`，或改用 `AvatarService`。 |
| 8–15 号核上都有 EP 线程 | 系统 EP 2.0.8 在每个 A100 核各起一个线程 | 正常现象：只有 2 个线程（A100 8、9）在计算，其余空闲。 |
| 无法加载 `libspacemit_ep.so` | 未安装 `spacemit-onnxruntime` | 安装该软件包，或在配置中写明库的绝对路径。 |
| 无法解压 `.tar.zst` | 缺少 zstd | 安装 `zstd` 或 `python3-zstandard`。 |
| Windows 构建失败 | 原生推理只支持 Linux | Windows 仅可运行下载器与 `tests/python` 主机测试。 |

## 5. 版本与发布

版本以根目录 `VERSION` 为准，CMake 和 wheel 从中读取；发版时同步更新 `package.xml`。资源版本（`20261021-r1`）与软件版本分开固定，模型和素材通过 SHA256 绑定。

| 版本 | 说明 |
| --- | --- |
| 0.1.0 | 提供 FeatherHuBERT 与 zihan、feifan 两个人物的 UNet，C++ / Python 接口，系统 SpaceMIT EP 推理，流水线渲染、离线示例和人物测试音频。 |

## 6. 贡献方式

欢迎参与贡献：提交 Issue 反馈问题，或通过 Pull Request 提交代码。

- **编码规范**：C++ 代码遵循 [Google C++ 风格指南](https://google.github.io/styleguide/cppguide.html)。
- **提交前检查**：运行 `bash tests/run_sdk_test.sh core`、`python`、`installed`；改动模型前后处理时，在 K3 上运行 `parity`。
- **来源对照**：图像前后处理来自 bianbu-avatar，上游版本与代码块哈希记录在 `config/source-baseline.json`。

## 7. License

本组件代码以 Apache-2.0 发布，见 `LICENSE` 与 `NOTICE`。模型权重、人物素材和测试音频不在源码中，由下载器从资源服务器获取，不受本代码许可覆盖；其说明见 [模型说明](docs/MODEL_CARD.md)。

## 8. 附录：性能指标

以下数据基于 K3 实测（Bianbu 4.0，系统 `spacemit-onnxruntime` 2.0.8，`configs/*.yaml` 默认配置），渲染时间包含写 PNG。

| 人物 | 音频 | 帧数 | 特征提取（CPU） | 渲染（含进程启动与模型加载） |
| --- | --- | --- | --- | --- |
| zihan | `zihan_test.wav`（5.6 s） | 140 | 1.0 s | 5.5 s（25.3 fps） |
| feifan | `feifan_test.wav`（9.0 s） | 225 | 1.3 s | 8.3 s（26.9 fps） |

去掉启动开销后渲染稳定在约 29 fps。单帧各环节耗时（串行测量）：

| 解码底帧 | 关键点 | 人脸预处理 | UNet 推理（EP） | 贴回合成 | 写 PNG |
| --- | --- | --- | --- | --- | --- |
| 9.5 ms | 0.6 ms | 3.2 ms | 33.9 ms | 3.7 ms | 27.0 ms |

串行处理每帧约 78 ms（约 13 fps）；流水线让解码、预处理、贴回和写 PNG 与 UNet 推理并行，吞吐接近推理上限（约 29.5 fps）。
