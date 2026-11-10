# Avatar 组件

## 1. 项目简介

本组件为 bianbu-avatar 数字人的口型推理封装，提供统一的 C++ 接口与 Python 绑定：输入一段语音，离线生成人物说话的逐帧画面，便于集成到数字人、AI Agent 等应用中。当前包含 FeatherHuBERT 音频特征与人物专属 UNet 唇形生成两个模型及其前后处理；ASR、LLM、TTS、WebRTC 与实时调度不在本组件内。功能特性如下：

| 类别     | 支持                                                                 |
| -------- | -------------------------------------------------------------------- |
| 部署方式 | **本地**（ONNX 推理）：FeatherHuBERT 使用 CPU，人物 UNet 使用 SpaceMIT EP |
| 处理方式 | 流式特征提取 `PushPcm()`；逐帧渲染 `Infer()` + `Compose()`；离线文件渲染 `avatar_worker` / `AvatarService.render_file()` |
| 人物     | `zihan`（函姐）、`feifan`（进迭小哥）                                 |
| 输入输出 | 16 kHz 单声道 PCM16 WAV → 25 fps PNG 帧序列；K3 上渲染约 29 fps       |
| 接口     | C++（`include/avatar_service.h`）、Python（`spacemit_avatar`）、命令行（`avatar_worker`） |

支持的模型：

| 模型 | 文件 | 输入 | 输出 | 推理 |
|------|------|------|------|------|
| FeatherHuBERT | `feather_hubert_dynamic.onnx` | `waveform [1,samples]`，16 kHz | `hidden [1,tokens,1024]` | CPU，4 线程 |
| 人物 UNet | `zihan_int8.onnx`、`feifan_int8.onnx` | `input [1,6,160,160]`、`audio [1,16,32,32]` | `output [1,3,160,160]` | SpaceMIT EP，2 个 A100 核 |

张量均为 FP32，UNet 的 INT8 为图内量化。每个人物的 UNet 须与该人物的 750 帧底图和关键点配套使用，不支持直接换成任意人像。处理流程：

```text
WAV ──avatar_worker extract──▶ 特征文件（FTF1）──avatar_worker render──▶ mp4（或 PNG 帧）
       FeatherHuBERT（CPU）                     UNet（SpaceMIT EP）+ 人脸预处理 + 贴回 + ffmpeg 编码
```

接口详见 [API.md](API.md)，实现细节见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 2. 验证模型

按以下顺序完成依赖安装、模型准备与示例运行。以下命令均在 **K3 板端（Bianbu Linux）终端**、组件目录下执行，脚本可直接运行（如 `./scripts/download_models.sh`）；Windows 只能下载资源与运行主机测试，见 [2.2](#22-下载模型)。

### 2.1. 安装依赖

- **运行平台**：SpacemiT K3 / Bianbu 4.0，Python 3.14。Windows 仅支持下载器与主机测试。
- **编译环境**：CMake ≥ 3.18，C++17 编译器。
- **必选**：OpenCV（core、imgproc、imgcodecs）、yaml-cpp、`spacemit-onnxruntime`（ONNX Runtime 与 SpaceMIT EP）。
- **可选**：pybind11（Python 绑定）、zstd 或 python3-zstandard（解压资源）、FFmpeg（帧合成视频）。

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends cmake ninja-build pkg-config \
  libopencv-core-dev libopencv-imgproc-dev libopencv-imgcodecs-dev libyaml-cpp-dev \
  python3-dev pybind11-dev python3-pybind11 python3-build python3-wheel python3-setuptools \
  python3-numpy python3-yaml zstd ffmpeg spacemit-onnxruntime
```

构建时直接链接 `spacemit-onnxruntime` 提供的 `libonnxruntime.so` 与 `libspacemit_ep.so`。头文件不在默认路径时，配置时加 `-DONNXRUNTIME_INCLUDE_DIR=/path/to/include`；找不到 SpaceMIT EP 时 CMake 给出警告并构建仅 CPU 的库，此时 UNet 无法使用 `spacemit`。

### 2.2. 下载模型

模型、人物素材与测试音频统一缓存到 `~/.cache/models/avatar/` 与 `~/.cache/assets/avatar/`，下载器按 `config/resources.json` 校验大小与 SHA256，支持断点续传。

**模型源**：进迭时空镜像 <https://archive.spacemit.com/spacemit-ai/model_zoo/avatar/>；测试音频 <https://archive.spacemit.com/spacemit-ai/model_zoo/assets/audio/>

```bash
# FeatherHuBERT + 人物 UNet
./scripts/download_models.sh --persona feifan
./scripts/download_models.sh --persona zihan
# 人物底帧与关键点（--frames 0 为全部 750 帧，默认前 32 帧）及测试音频
./scripts/download_assets.sh --persona feifan --frames 0
./scripts/download_assets.sh --persona zihan --frames 0
```

| 内容 | 默认位置 |
| ---- | ---- |
| 模型 | `~/.cache/models/avatar/`：`feather_hubert_dynamic.onnx`、`zihan_int8.onnx`、`feifan_int8.onnx` |
| 人物素材 | `~/.cache/assets/avatar/<人物>/full_body_img/*.jpg`、`landmarks/*.lms` |
| 测试音频 | `~/.cache/assets/avatar/audio/`：`feifan_test.wav`（9.0 s）、`zihan_test.wav`（5.6 s）、本地生成的 `silence.wav` |

其他选项：`--model hubert` 只下载特征模型；`--dry-run` 预览；`--offline` 仅用已缓存归档；`--models-dir`、`--assets-dir`、`--cache-dir` 指定位置（修改后同步修改配置文件中的路径）；`--force` 替换内容不同的已有文件。每个人物的归档约 64 MB，同时包含权重与素材，下载器只解出选中的文件。

**Windows**：只需下载资源时，可直接用 Python 调用下载器，参数与上面相同：

```bat
python scripts\download_resources.py models --persona feifan
python scripts\download_resources.py assets --persona feifan --frames 0
```

### 2.3. 测试

本节提供示例程序的编译与运行方式，便于开发者快速验证效果。使用前需先按下列两种方式之一完成编译，再运行对应示例。

- **在 SDK 中验证**（2.3.1）：在已拉取的 SpacemiT AI SDK 工程内用 `mm` 编译，产物部署到 `output/staging`，适合整机集成或与 ASR、TTS 等模块联调。
- **独立构建下验证**（2.3.2）：在 Avatar 组件目录下用 CMake 本地编译，不依赖完整 SDK，适合快速体验或只克隆本仓库时使用。

示例默认读取 [2.2](#22-下载模型) 的下载位置，渲染 PNG 帧后用 ffmpeg 合成带音频的 mp4；未安装 ffmpeg 时只输出帧。换人物时把 `feifan` 换成 `zihan`。

#### 2.3.1. 在 SDK 中验证

**编译**：在 SDK 根目录下执行。本组件在 ai-sdk 中位于 `avatar/`，在 SpacemiT Robot SDK 中位于 `components/model_zoo/avatar`。

```bash
source build/envsetup.sh
cd avatar
mm
```

构建产物会安装到 `output/staging`，wheel 包输出到 SDK 根目录下的 `output/dist/`。

**运行**：在 `avatar/` 目录下执行。

**C++ 文件渲染：**
```bash
avatar_file_demo --persona feifan     # 输出 output/feifan-frames/ 与 output/feifan.mp4
```

**Python 文件渲染**（需先安装 wheel；Bianbu 系统 Python 需在虚拟环境中安装）：
```bash
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install --no-deps ../output/dist/spacemit_avatar-*.whl
python python/examples/avatar_file_demo.py --persona feifan   # 输出 output/feifan-python.mp4
```

#### 2.3.2. 独立构建下验证

**C++ 文件渲染：**
```bash
cd /path/to/avatar
cmake -S . -B build && cmake --build build -j4
./build/avatar_file_demo --persona feifan
```

**Python 文件渲染**（同样在虚拟环境中安装）：
```bash
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
python -m build --wheel --no-isolation python
pip install --no-deps python/dist/spacemit_avatar-*.whl
python python/examples/avatar_file_demo.py --persona feifan
```

两个示例的常用参数相同：`--audio` 指定其他 16 kHz WAV，`--output` 指定帧目录（须为空），`--video` 指定 mp4 路径，`--no-video` 只输出帧，`--max-frames` 限制帧数。

**契约测试**（不需要模型）：`./tests/run_sdk_test.sh core|config|invalid-input|python|installed`；真实模型差分测试见 [docs/VALIDATION.md](docs/VALIDATION.md)。

## 3. 应用开发

本章说明如何在自有工程中**集成 Avatar 并调用 API**。环境与依赖见 [2.1](#21-安装依赖)，模型准备见 [2.2](#22-下载模型)，编译与运行示例见 [2.3](#23-测试)。

### 3.1. 构建与集成产物

无论通过 [2.3.1](#231-在-sdk-中验证)（SDK）或 [2.3.2](#232-独立构建下验证)（独立构建）哪种方式编译，完成后**应用开发所需**的库与头文件如下，集成时只需**包含头文件并链接对应库**：

| 产物 | 说明 |
| ---- | ---- |
| `include/avatar_service.h` | **C++ API 头文件**，应用侧只需包含此头文件并链接下方库即可调用 |
| `libavatar.a` | C++ 核心库（SDK：`output/staging/lib/`；独立构建：`build/`），链接时还需 OpenCV 与 `spacemit-onnxruntime` |
| `avatar_worker` | 离线文件渲染命令行（SDK：`output/staging/bin/`；独立构建：`build/`） |
| `spacemit_avatar-0.1.0-*.whl` | Python wheel 包（SDK：`output/dist/`；独立构建：`python -m build --wheel --no-isolation python` 输出到 `python/dist/`），`pip install` 后 `import spacemit_avatar` |

示例程序（非集成必需）：`examples/avatar_file_demo.cpp`（编译为 `avatar_file_demo`）、`python/examples/avatar_file_demo.py`。运行与验证步骤见 [2.3.1](#231-在-sdk-中验证) 或 [2.3.2](#232-独立构建下验证)。

### 3.2. API 使用

**C++**：头文件 `include/avatar_service.h` 为唯一 API 入口，实现为 PIMPL。在业务代码中 `#include "avatar_service.h"`，链接 `libavatar.a`（及 OpenCV、ONNX Runtime、SpaceMIT EP），即可使用。完整的逐帧示例见 `examples/avatar_file_demo.cpp`。

```cpp
#include "avatar_service.h"
#include <opencv2/imgcodecs.hpp>
using namespace SpacemiT::avatar;

RuntimeOptions hubert_options;          // CPU，4 线程
RuntimeOptions unet_options;
unet_options.provider = "spacemit";     // 系统 libspacemit_ep.so
FeatherHubertExtractor hubert("feather_hubert_dynamic.onnx", hubert_options);
UnetLipGenerator unet("feifan_int8.onnx", unet_options);

auto pcm = ReadWav16k("feifan_test.wav");
auto batch = hubert.PushPcm(pcm.data(), pcm.size(), /*final=*/true);

PreparedFrame frame(cv::imread("feifan/full_body_img/0.jpg"), ReadLandmarks("feifan/landmarks/0.lms"));
std::vector<float> window(batch.values.begin(), batch.values.begin() + kWindowElements);
cv::Mat image = frame.Compose(unet.Infer(frame.ImageTensor(), window));
```

**Python**：安装后 `import spacemit_avatar`，完整示例见 `python/examples/avatar_file_demo.py`，接口见 [API.md](API.md)。

```python
import spacemit_avatar
with spacemit_avatar.AvatarService("configs/feifan.yaml") as engine:
    result = engine.render_file("feifan_test.wav", "output/frames")
    print(result.frames, result.render_metrics)
# 或使用低层接口逐帧处理
extractor = spacemit_avatar.FeatherHubertExtractor("feather_hubert_dynamic.onnx")
batch = extractor.push_pcm(pcm, final=True)   # batch.windows: [N,16,32,32]
```

**CMake 集成**：将本组件作为子目录引入，并链接 `avatar` 即可，头文件路径随目标传递。
```cmake
add_subdirectory(avatar)
target_link_libraries(your_target PRIVATE avatar)
```

使用约束：模型实例不可并发调用，`final` 之后须 `Reset()` 才能继续输入；`AvatarService` 的两个 worker 进程相互隔离，取消或失败时已写出的部分 PNG 不能当作完整结果。

### 3.3. 命令行渲染

`avatar_worker` 分两阶段离线渲染：`extract` 用 FeatherHuBERT 把 WAV 转成特征文件，`render` 流水线渲染并直接输出带音频的 mp4（需已安装 ffmpeg）。Python 的 `AvatarService` 内部也调用它。在组件目录下执行：

```bash
avatar_worker extract --config configs/feifan.yaml \
  --audio ~/.cache/assets/avatar/audio/feifan_test.wav --output output/feifan.ftf
avatar_worker render --config configs/feifan.yaml --features output/feifan.ftf \
  --audio ~/.cache/assets/avatar/audio/feifan_test.wav --output output/feifan-worker.mp4
```

- `--output` 以 `.mp4` 结尾时，帧不写盘，直接送入 ffmpeg 编码为 H.264：K3 上用视频硬件编码器（`h264_stcodec`），开始前先试编码一帧，不可用时自动改用软件编码 libx264 veryfast，报告的 `encoder` 行会注明。`--audio` 加入音轨，须与 `extract` 使用的音频相同，省略时生成无声视频。
- `--output` 为目录时输出 `00000000.png` 起的 PNG 帧序列。
- 输出的上级目录会自动创建；已存在的 mp4 或特征文件不会被覆盖；失败或按 Ctrl+C 取消时不留下不完整的文件。

两个命令结束时都会打印性能报告。K3 上 `render` 的输出示例：

```text
avatar_worker render: 225 frames, 9.00 s of video -> output/feifan-worker.mp4
  total      8.23 s  (model load 0.31 s, render 7.73 s, ffmpeg finish 0.02 s)
  speed      29.0 fps, steady 29.9 fps, RTF 0.861 (model load excluded)
  per frame  decode 9.1 | landmarks 0.2 | preprocess 2.7 | UNet 34.2 | compose 5.7 | to ffmpeg 3.2 ms
  encoder    h264_stcodec (K3 hardware), ffmpeg 5.9 ms CPU per frame
  bottleneck UNet, at most 29.3 fps
```

| 字段 | 含义 |
|------|------|
| `total` | 命令总耗时，括号内为模型加载、流水线渲染、ffmpeg 收尾（编码剩余帧并写入文件） |
| `speed` | 渲染速度（不含模型加载）；`steady` 为第一帧之后的稳态帧率；`RTF` 为处理时间 ÷ 视频时长，小于 1 即快于实时 |
| `per frame` | 各环节每帧平均耗时：解码底帧、读关键点、人脸预处理、UNet 推理、贴回合成、送入 ffmpeg（输出目录时为写 PNG） |
| `encoder` | 所用编码器，以及 ffmpeg 每帧消耗的 CPU 时间（多线程累计） |
| `bottleneck` | 各环节按自身线程数能达到的最高帧率中最低的一项，即当前的速度上限 |

报告输出到 stderr；stdout 的最后一行是同样指标的 JSON（字段见 [API.md](API.md)），`AvatarService` 的 `render_metrics` 即来自这一行。

### 3.4. 配置参数

`avatar_worker` 与 `AvatarService` 读取 YAML 配置（`schema: 1`），自带 `configs/feifan.yaml`、`configs/zihan.yaml`。路径支持 `~/` 与相对配置文件的相对路径。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hubert.model` / `unet.model` | `string` | — | 模型路径（必填） |
| `*.provider` | `string` | `cpu` | FeatherHuBERT 为 `cpu`；人物 UNet 为 `spacemit` |
| `*.threads` | `int` | `4` | ORT 线程数；UNet 中为处理 EP 不支持节点的 CPU 线程数 |
| `unet.ep_threads` | `int` | `2` | EP 计算线程数 |
| `*.profile_prefix` | `string` | 不记录 | ORT 性能分析输出前缀，用于确认 EP 实际执行 |
| `*.cpu_affinity` | `int[]` | 不绑核 | 进程绑核，自带配置为 `[0, 1, 2, 3, 4, 5, 6, 7]` |
| `*.timeout_s` | `float` | `3600` | `AvatarService` 中该阶段超时 |
| `assets.person` | `string` | — | 人物素材目录，含 `full_body_img/` 与 `landmarks/`（必填） |
| `render.native_pasteback` | `bool` | `true` | 原生贴回，与 bianbu-avatar 一致 |
| `render.feather` | `bool` | `true` | 贴回边缘羽化 |
| `render.mouth_only_blend` | `bool` | `false` | 仅嘴部掩码，关闭原生贴回时生效 |
| `render.mouth_sharpen` | `float` | `0.15` | 嘴部锐化，原生贴回时不生效 |
| `render.temporal_smooth` / `smooth_alpha` | `bool` / `float` | `false` / `0.9` | 时间平滑，原生贴回时不生效 |
| `render.patch_lowpass` | `float` | `0.0` | 生成人脸低通 |

详见 [API.md](API.md)。

## 4. 常见问题

| 现象 | 可能原因 | 处理 |
| --- | --- | --- |
| 下载报 404 或校验失败 | 资源地址变更或本地仓库较旧 | 更新仓库后重新下载；`--force` 替换内容不同的旧文件。 |
| 提示 WAV 格式不支持 | 采样率或声道不符 | 用 `ffmpeg -i in.wav -ar 16000 -ac 1 -c:a pcm_s16le out.wav` 转换。 |
| 提示输出目录非空、`Video already exists` 或 `Feature file already exists` | 不覆盖上次的结果 | 删除旧的 `output/<人物>-frames/` 与 mp4，或用 `--output`、`--video` 指定新位置。 |
| 只有帧，提示 `ffmpeg not found` | 未安装 ffmpeg | `sudo apt install ffmpeg` 后重新运行示例。 |
| `This build has no SpaceMIT EP` | 构建时未找到 SpaceMIT EP | 安装 `spacemit-onnxruntime` 后重新构建。 |
| 8–15 号核上都有 EP 线程 | 系统 EP 2.0.8 在每个 A100 核各起一个线程 | 正常现象：只有 A100 8、9 上的 2 个线程在计算，其余空闲。 |
| `pip install` 提示 externally-managed-environment | 系统 Python 受保护 | 在 `python3 -m venv --system-site-packages` 创建的虚拟环境中安装。 |
| 无法解压 `.tar.zst` | 缺少 zstd | 安装 `zstd` 或 `python3-zstandard`。 |

## 5. 版本与发布

版本以本组件文档或仓库 tag 为准；CMake 与 wheel 从 `VERSION` 读取，发版时同步更新 `package.xml`。模型资源版本（`20261021-r1`）与软件版本分开固定，通过 SHA256 绑定。

| 版本   | 说明 |
| ------ | ---- |
| 0.1.0  | 提供 C++ / Python 接口，支持 FeatherHuBERT 与 zihan、feifan 人物 UNet，SpaceMIT EP 推理与流水线渲染。 |

## 6. 贡献方式

欢迎参与贡献：提交 Issue 反馈问题，或通过 Pull Request 提交代码。

- **编码规范**：C++ 代码遵循 [Google C++ 风格指南](https://google.github.io/styleguide/cppguide.html)（4 空格缩进、120 列），提交时由 `code-style` 工作流运行 cpplint、shellcheck 与 ruff。
- **提交前检查**：运行 `./tests/run_sdk_test.sh core`、`config`、`invalid-input`、`python`、`installed`；改动模型前后处理时在 K3 上运行 `parity`。图像前后处理来自 bianbu-avatar，上游版本、代码块哈希与本地改动见 `config/source-baseline.json`。

## 7. License

本组件源码以 Apache-2.0 发布，最终以本目录 `LICENSE` 文件为准。模型权重、人物素材与测试音频由下载器从资源服务器获取，不在源码中，也不受本代码许可覆盖，说明见 [docs/MODEL_CARD.md](docs/MODEL_CARD.md)。

## 8. 附录：性能指标

以下数据基于 K3 平台（Bianbu 4.0，系统 `spacemit-onnxruntime` 2.0.8，自带配置）实测：用 [3.3](#33-命令行渲染) 的两条命令直接生成带音频的 mp4，数字取自命令打印的性能报告。

| 人物 | 测试音频 | 帧数 | 特征提取（含模型加载） | 渲染为 mp4（含模型加载与编码） | 渲染速度 | 稳态 |
|------|----------|------|------------------------|--------------------------------|----------|------|
| zihan | `zihan_test.wav`（5.6 s） | 140 | 0.77 s（RTF 0.11） | 5.35 s | 28.8 fps | 30.1 fps |
| feifan | `feifan_test.wav`（9.0 s） | 225 | 1.06 s（RTF 0.10） | 8.23 s | 29.0 fps | 29.9 fps |

feifan 单帧各环节耗时（流水线中各线程实测）：

| 解码底帧 | 关键点 | 人脸预处理 | UNet 推理（EP） | 贴回合成 | 转 I420 并送入 ffmpeg | ffmpeg 编码（硬件） |
|----------|--------|------------|-----------------|----------|------------------------|---------------------|
| 9.1 ms | 0.2 ms | 2.7 ms | 34.2 ms | 5.7 ms | 3.2 ms | 5.9 ms CPU |

UNet 推理是瓶颈，其余环节在 X100 上与它并行。H.264 由 K3 视频硬件编码器完成，ffmpeg 每帧只占约 6 ms CPU，生成 mp4 与输出 PNG 帧（约 29 fps）速度相同。没有硬件编码器时改用 libx264 veryfast，ffmpeg 每帧约 109 ms CPU，与 UNet 的 CPU 部分争用 X100，渲染约 26 fps。`avatar_file_demo` 为便于阅读逐帧串行执行，不用于测速。

**说明**：`avatar_worker render` 将每帧拆为预取（解码底帧、计算模型输入）、UNet 推理、按帧序贴回、写出（多线程写 PNG，或按帧序送入 ffmpeg）四段并行执行，输出与逐帧串行处理逐字节一致。FeatherHuBERT 及 worker 的 CPU 线程运行在 X100 0–7；系统 EP 2.0.8 自行放置 A100 线程，在每个 A100 核各起一个、计算只在 A100 8、9 上进行。
