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
WAV ──avatar_worker extract──▶ 特征文件（FTF1）──avatar_worker render──▶ PNG 帧 ──ffmpeg──▶ mp4
       FeatherHuBERT（CPU）                     UNet（SpaceMIT EP）+ 人脸预处理 + 贴回
```

接口详见 [API.md](API.md)，实现细节见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 2. 验证模型

按以下顺序完成依赖安装、模型准备与示例运行。

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

运行时按配置加载系统的 `libonnxruntime.so` 与 `libspacemit_ep.so`，构建时不链接。头文件不在默认路径时，配置时加 `-DONNXRUNTIME_INCLUDE_DIR=/path/to/include`。

### 2.2. 下载模型

模型、人物素材与测试音频统一缓存到 `~/.cache/models/avatar/` 与 `~/.cache/assets/avatar/`，下载器按 `config/resources.json` 校验大小与 SHA256，支持断点续传。

**模型源**：进迭时空镜像 <https://archive.spacemit.com/spacemit-ai/model_zoo/avatar/>；测试音频 <https://archive.spacemit.com/spacemit-ai/model_zoo/assets/audio/>

```bash
# FeatherHuBERT + 人物 UNet
bash scripts/download_models.sh --persona feifan
bash scripts/download_models.sh --persona zihan
# 人物底帧与关键点（--frames 0 为全部 750 帧，默认前 32 帧）及测试音频
bash scripts/download_assets.sh --persona feifan --frames 0
bash scripts/download_assets.sh --persona zihan --frames 0
```

| 内容 | 默认位置 |
| ---- | ---- |
| 模型 | `~/.cache/models/avatar/`：`feather_hubert_dynamic.onnx`、`zihan_int8.onnx`、`feifan_int8.onnx` |
| 人物素材 | `~/.cache/assets/avatar/<人物>/full_body_img/*.jpg`、`landmarks/*.lms` |
| 测试音频 | `~/.cache/assets/avatar/audio/`：`feifan_test.wav`（9.0 s）、`zihan_test.wav`（5.6 s）、本地生成的 `silence.wav` |

其他选项：`--model hubert` 只下载特征模型；`--dry-run` 预览；`--offline` 仅用已缓存归档；`--models-dir`、`--assets-dir`、`--cache-dir` 指定位置（修改后同步修改配置文件中的路径）；`--force` 替换内容不同的已有文件。每个人物的归档约 64 MB，同时包含权重与素材，下载器只解出选中的文件。

### 2.3. 测试

本节提供示例程序的编译与运行方式，便于开发者快速验证效果。使用前需先按下列两种方式之一完成编译，再运行对应示例。

- **在 SDK 中验证**（2.3.1）：在已拉取的 SpacemiT AI SDK 工程内用 `mm` 编译，产物部署到 `output/staging`，适合整机集成或与 ASR、TTS 等模块联调。
- **独立构建下验证**（2.3.2）：在 Avatar 组件目录下用 CMake 本地编译，不依赖完整 SDK，适合快速体验或在不使用 SDK 的环境下使用。

#### 2.3.1. 在 SDK 中验证

**编译**：在 SDK 根目录下执行。本组件在 ai-sdk 中位于 `avatar/`，在 SpacemiT Robot SDK 中位于 `components/model_zoo/avatar`。

```bash
source build/envsetup.sh
cd avatar
mm -py
```

构建产物安装到 `output/staging`，wheel 包输出到 SDK 根目录下的 `output/wheels/avatar/`，例如：

```bash
output/wheels/avatar/spacemit_avatar-0.1.0-cp314-cp314-linux_riscv64.whl
```

**运行**：运行前在 SDK 根目录执行 `source build/envsetup.sh`，使 PATH 与库路径指向 `output/staging`，然后在组件目录下执行：

**C++ 文件渲染：**
```bash
bash examples/cpp/run_pipeline.sh configs/feifan.yaml \
  ~/.cache/assets/avatar/audio/feifan_test.wav output/feifan-frames
ffmpeg -framerate 25 -i output/feifan-frames/%08d.png \
  -i ~/.cache/assets/avatar/audio/feifan_test.wav \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest output/feifan.mp4

bash examples/cpp/run_pipeline.sh configs/zihan.yaml \
  ~/.cache/assets/avatar/audio/zihan_test.wav output/zihan-frames
```

**Python 文件渲染**（推荐先在虚拟环境中安装 wheel 包，wheel 内含 `avatar_worker`）：
```bash
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install --no-deps ../output/wheels/avatar/spacemit_avatar-0.1.0-*.whl
python examples/python/render_file.py --config configs/feifan.yaml \
  --audio ~/.cache/assets/avatar/audio/feifan_test.wav --output output/python-frames
```

#### 2.3.2. 独立构建下验证

在 Avatar 组件目录下完成编译后，运行下列示例。

**C++ 文件渲染：**
```bash
cd /path/to/avatar
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure
AVATAR_WORKER="$PWD/build/avatar_worker" bash examples/cpp/run_pipeline.sh configs/feifan.yaml \
  ~/.cache/assets/avatar/audio/feifan_test.wav output/feifan-frames
```

**Python 文件渲染：**
```bash
cd /path/to/avatar
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
python -m build --wheel --no-isolation
pip install --no-deps dist/spacemit_avatar-*.whl
python examples/python/render_file.py --config configs/feifan.yaml \
  --audio ~/.cache/assets/avatar/audio/feifan_test.wav --output output/python-frames
```

输出目录须不存在或为空；视频帧数为 `ceil(PCM 样本数 / 640)`，底帧按 0→749→0 往返使用。组件测试入口与 `test.yaml` 一致：`bash tests/run_sdk_test.sh core|python|installed`；真实模型差分测试 `parity` 见 [docs/VALIDATION.md](docs/VALIDATION.md)。

**说明**：`avatar_worker render` 将每帧拆为预取（解码底帧、计算模型输入）、UNet 推理、按帧序贴回、多线程写 PNG 四段并行执行，输出与逐帧串行处理逐字节一致。FeatherHuBERT 及 worker 的 CPU 线程运行在 X100 0–7；系统 EP 2.0.8 自行放置 A100 线程，在每个 A100 核各起一个、计算只在 A100 8、9 上进行。

## 3. 应用开发

本章说明如何在自有工程中**集成 Avatar 并调用 API**。环境与依赖见 [2.1](#21-安装依赖)，模型准备见 [2.2](#22-下载模型)，编译与运行示例见 [2.3](#23-测试)。

### 3.1. 构建与集成产物

无论通过 [2.3.1](#231-在-sdk-中验证)（SDK）或 [2.3.2](#232-独立构建下验证)（独立构建）哪种方式编译，完成后**应用开发所需**的库与头文件如下，集成时只需**包含头文件并链接对应库**：

| 产物 | 说明 |
| ---- | ---- |
| `include/avatar_service.h` | **C++ API 头文件**，应用侧只需包含此头文件并链接下方库即可调用 |
| `libavatar.so` | C++ 核心库（SDK：`output/staging/lib/`；独立构建：`build/`） |
| `avatar_worker` | 离线文件渲染命令行（SDK：`output/staging/bin/`；独立构建：`build/`） |
| `spacemit_avatar-0.1.0-*.whl` | Python wheel 包（SDK：`output/wheels/avatar/`；独立构建：`dist/`），`pip install` 后 `import spacemit_avatar` |

示例程序（非集成必需）：`examples/cpp/run_pipeline.sh`、`examples/python/render_file.py`。运行与验证步骤见 [2.3.1](#231-在-sdk-中验证) 或 [2.3.2](#232-独立构建下验证)。

### 3.2. API 使用

**C++**：头文件 `include/avatar_service.h` 为唯一 API 入口，实现为 PIMPL。在业务代码中 `#include "avatar_service.h"`，链接 `libavatar.so`（及 OpenCV），即可使用；ONNX Runtime 与 SpaceMIT EP 在运行时按 `RuntimeOptions` 加载。

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

**Python**：安装后 `import spacemit_avatar`，详见 `examples/python/` 与 [API.md](API.md)。

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

使用约束：模型实例不可并发调用，`final` 之后须 `Reset()` 才能继续输入；同一进程只能加载一套 ONNX Runtime，两个模型使用同一个 `libonnxruntime.so`；`AvatarService` 的两个 worker 进程相互隔离，取消或失败时已写出的部分 PNG 不能当作完整结果。

### 3.3. 配置参数

`avatar_worker` 与 `AvatarService` 读取 YAML 配置（`schema: 1`），自带 `configs/feifan.yaml`、`configs/zihan.yaml`。路径支持 `~/` 与相对配置文件的相对路径。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hubert.model` / `unet.model` | `string` | — | 模型路径（必填） |
| `*.provider` | `string` | `cpu` | FeatherHuBERT 为 `cpu`；人物 UNet 为 `spacemit` |
| `*.ort_library` | `string` | `libonnxruntime.so` | ONNX Runtime 库 |
| `unet.ep_library` | `string` | `libspacemit_ep.so` | SpaceMIT EP 库 |
| `*.threads` | `int` | `4` | ORT 线程数；UNet 中为处理 EP 不支持节点的 CPU 线程数 |
| `unet.ep_threads` | `int` | `2` | EP 计算线程数 |
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
| 提示输出目录非空 | 防止混入旧帧 | 使用新的或空的输出目录。 |
| `One ORT binary per process` | 同一进程选择了两套 ORT | 两个模型使用同一 `ort_library`，或改用 `AvatarService`。 |
| 无法加载 `libspacemit_ep.so` | 未安装 `spacemit-onnxruntime` | 安装该软件包，或在配置中写明库的绝对路径。 |
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

- **编码规范**：C++ 代码遵循 [Google C++ 风格指南](https://google.github.io/styleguide/cppguide.html)。
- **提交前检查**：运行 `bash tests/run_sdk_test.sh core`、`python`、`installed`；改动模型前后处理时在 K3 上运行 `parity`。图像前后处理来自 bianbu-avatar，上游版本与代码块哈希见 `config/source-baseline.json`。

## 7. License

本组件源码以 Apache-2.0 发布，最终以本目录 `LICENSE` 文件为准。模型权重、人物素材与测试音频由下载器从资源服务器获取，不在源码中，也不受本代码许可覆盖，说明见 [docs/MODEL_CARD.md](docs/MODEL_CARD.md)。

## 8. 附录：性能指标

以下数据基于 K3 平台（Bianbu 4.0，系统 `spacemit-onnxruntime` 2.0.8，自带配置）实测，渲染时间包含写 PNG。

| 人物 | 测试音频 | 帧数 | 特征提取（CPU） | 渲染（含进程启动） | 稳态渲染 |
|------|----------|------|-----------------|--------------------|----------|
| zihan | `zihan_test.wav`（5.6 s） | 140 | 1.0 s | 5.5 s（25.3 fps） | 约 29 fps |
| feifan | `feifan_test.wav`（9.0 s） | 225 | 1.3 s | 8.3 s（26.9 fps） | 约 29 fps |

单帧各环节耗时（串行测量）如下；流水线使解码、预处理、贴回与写 PNG 和 UNet 推理并行，吞吐接近推理上限（约 29.5 fps）。

| 解码底帧 | 关键点 | 人脸预处理 | UNet 推理（EP） | 贴回合成 | 写 PNG |
|----------|--------|------------|-----------------|----------|--------|
| 9.5 ms | 0.6 ms | 3.2 ms | 33.9 ms | 3.7 ms | 27.0 ms |

测试命令：`avatar_worker render --config configs/<人物>.yaml --features <人物>.ftf --output <空目录>`
