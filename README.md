# bianbu-avatar 数字人推理组件

独立抽取 bianbu-avatar 的 **FeatherHuBERT 音频特征 + 人物 UNet 唇形生成**。C++17 实现推理和前后处理，pybind11 提供 NumPy 接口；离线完整示例以两个原生子进程隔离 CPU/EP 的 ORT。没有 ASR、LLM、TTS、摄像头检测、WebRTC 或 systemd 服务依赖。

CMake 和 Python wheel 从 `VERSION` 读取版本。发版时同步更新 `package.xml`。当前构建目标是 Linux，首个实机验证平台为 K3/Bianbu/Python 3.14；Windows 可运行下载器与下载安全测试，不宣称支持 Windows 原生推理。

代码以 Apache-2.0 发布，见 `LICENSE` 和 `NOTICE`。项目方已确认当前人物素材用于本次 Model Zoo 提交无需另行许可。模型权重的再分发条款见 `docs/MODEL_CARD.md`。

## 1. 模型与资源

| 模型 | 输入 | 输出 |
|---|---|---|
| FeatherHuBERT | FP32 `waveform [1,samples]`，16kHz | FP32 `hidden [1,tokens,1024]` |
| 人物 UNet | FP32 `input [1,6,160,160]`、`audio [1,16,32,32]` | FP32 `output [1,3,160,160]` |

UNet 的 INT8 是图内部量化，公开张量仍为 FP32。`feifan` 与 `zihan` 权重分别配对应人物的帧和关键点，不支持随意更换任意人像。

```bash
# 默认只下载 FeatherHuBERT + feifan UNet；不下载任何其他模型。
bash scripts/download_models.sh --persona feifan
# 同一人物前32组JPG/关键点 + 公开测试WAV + 本地生成的1秒静音WAV。
bash scripts/download_assets.sh --persona feifan
# 可选：只取特征模型；另一人物；全部配套帧。
bash scripts/download_models.sh --model hubert
bash scripts/download_models.sh --persona zihan
bash scripts/download_assets.sh --persona zihan --frames 0
# 预览、不联网 / 已缓存离线运行
bash scripts/download_models.sh --dry-run
bash scripts/download_models.sh --offline
```

资源写入 `~/.cache/models/avatar/` 与 `~/.cache/assets/avatar/`。通过 `--models-dir`、`--assets-dir` 和 `--cache-dir` 可指定位置；修改位置后同步修改 YAML。

清单 `config/resources.json` 引用资源服务器 `model_zoo/avatar/` 下 bianbu-avatar 的 20261021-r1 发布资源。人物历史 tar 同时包含权重和素材，下载器只解出本次选中的成员；首次下载该人物任一成员仍需下载整个约64MB归档，后续模型/素材共享缓存。音频是驱动嘴型的测试录音，不是克隆音色参考录音。

下载器验证归档和逐文件大小/SHA256，支持HTTP续传及服务端忽略Range时重新下载，拒绝路径逃逸和选中软链接；已有不同内容的文件保留，明确指定 `--force` 才替换。安装 `python3-zstandard` 或 `zstd` 可解压 tar.zst。

## 2. K3 原生构建

```bash
sudo apt-get install -y --no-install-recommends cmake ninja-build pkg-config \
  libopencv-core-dev libopencv-imgproc-dev libopencv-imgcodecs-dev libyaml-cpp-dev \
  python3-dev pybind11-dev python3-pybind11 python3-build python3-wheel python3-setuptools \
  python3-numpy python3-yaml zstd spacemit-onnxruntime
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

需要与目标 ORT 配套的 C/C++ 头文件。非默认路径使用 `-DONNXRUNTIME_INCLUDE_DIR=/path/to/include`；不在构建阶段自动下载模型。纯C++构建可加 `-DBUILD_PYTHON_BINDINGS=OFF`。

## 3. C++ 离线示例

```bash
export AVATAR_WORKER="$PWD/build/avatar_worker"
bash examples/cpp/run_pipeline.sh configs/cpu.yaml \
  ~/.cache/assets/avatar/audio/speech.wav output/demo-frames
# 合成带原音轨的视频（FFmpeg仅是示例输出工具）
ffmpeg -framerate 25 -i output/demo-frames/%08d.png \
  -i ~/.cache/assets/avatar/audio/speech.wav \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest output/demo.mp4
```

WAV 必须为16kHz、单声道、PCM16；其它输入先用 FFmpeg 转换。输出目录必须为空，以免混入旧帧。`extract` 子命令生成 FTF1 特征文件，`render` 子命令读取特征并保存PNG；特征写入和读取逐批进行，不把整段视频帧堆进内存。特征提取器保持原服务的输出；CLI在最终不足的视频拍补零特征，使最终视频帧数为 `ceil(PCM样本数/640)`。少于模型最小输入而没有特征的音频明确失败。

zihan配套示例使用 `configs/zihan.cpu.yaml`。同一人物的UNet与assets需同时下载，不要只改模型路径而留下另一人物的素材。

`configs/k3.yaml` 与 `configs/zihan.k3.yaml` 显式选择 SpaceMIT EP，核分配与 bianbu-avatar 长对话配置一致：FeatherHuBERT 在 X100 0–7 用 CPU 4线程；UNet worker 进程及 ORT 的4个 CPU 回退线程在 X100 0–7，2个 EP 线程绑定 A100 12、13（bianbu-avatar 默认配置为 10、11）。只设置 `ep_affinity` 不能把 EP 限定在这两个核上：EP 会在每个可用的 A100 核上各起一个线程。需像 bianbu-avatar 的 systemd 服务一样，在只允许 X100 0–7 和所选 A100 核的 cgroup 中运行，例如 `sudo systemd-run --scope -p AllowedCPUs=0-7,12-13 bash examples/cpp/run_pipeline.sh configs/k3.yaml …`。K3 实测该方式需配合 bianbu-avatar 随包的 EP 2.0.7 运行库；系统 EP 2.0.8 在受限 cgroup 中仍会在其它 A100 核上起线程。请按实际安装设置其中的 `ort_library`、`ep_library` 和 affinity。需要分别使用不同 ORT 时，Python示例的两个worker进程已隔离；可用 `AVATAR_HUBERT_LIBRARY_DIR` 与 `AVATAR_UNET_LIBRARY_DIR` **仅设置子进程**库搜索路径。系统部署还应保证cpuset允许所选EP核。示例不修改主机核分配，也不自动占用其它推理服务的核心。

## 4. Python wheel 与示例

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m build --wheel --no-isolation
python -m pip install --no-deps dist/spacemit_avatar-*.whl
python examples/python/render_file.py --config configs/cpu.yaml \
  --audio ~/.cache/assets/avatar/audio/speech.wav \
  --output output/python-frames --max-frames 25
```

完整示例是**离线两阶段**处理，不宣称Python层边合成边播放或实时逐帧输出。`AvatarService.render_file` 返回帧目录、帧数及分阶段耗时；`cancel/close` 管理真实worker生命周期。低层API的 `FeatherHubertExtractor.push_pcm` 支持流式输入与final/reset。

```python
import numpy as np
from spacemit_avatar import RuntimeOptions, FeatherHubertExtractor
options = RuntimeOptions()
options.ort_library = "/path/to/matched/cpu/libonnxruntime.so"
extractor = FeatherHubertExtractor("/path/to/feather_hubert_dynamic.onnx", options)
batch = extractor.push_pcm(np.zeros(16000, dtype=np.int16), final=True)
print(batch.windows.shape, batch.frame_ids)  # [N,16,32,32]
extractor.reset()
```

低层模型实例不支持并发操作；误用会报错，不能在推理中reset/换帧。同一进程只能初始化一套 ORT；加载第二套不同路径会明确报错，使用完整分进程示例解决。绑定在推理时释放GIL，入参/返回数组使用独立所有权，不暴露悬空ORT缓冲区。

上述运行库检查只覆盖本组件自己的初始化路径；无法替其它Python扩展管理已加载的ORT。直接调用低层接口时，不要在同进程选择与其它扩展不同的运行库。高层离线接口默认通过独立worker避免这类混装。取消是协作式的：worker收到信号后，在当前推理返回的安全点退出；关闭/超时最终会回收子进程。取消或失败时保留已有的部分PNG，不能把它们当作完整结果。

wheel 采用 CPython 3.14 `linux_riscv64` 标签，运行时使用系统 OpenCV、yaml-cpp 和配置指定的 ORT。CMake 安装到 SDK staging 时，Python 包位于前缀的 `lib/python3.14/site-packages`，可用 `AVATAR_PYTHON_INSTALL_DIR` 设置相对路径。使用该布局时，将前缀的 `bin` 加入 `PATH`。

## 5. 测试与 SDK 集成

SDK 根目录下的 `avatar/` 与 `vision/` 同级。在 SDK 根目录加载 `source build/envsetup.sh`，进入 `avatar/` 后执行 `mm -py`。原生库、头文件、绑定和 worker 安装到 `output/staging`，wheel 写入 `output/wheels/avatar`。C++ 接口位于 `avatar_service.h`（命名空间 `SpacemiT::avatar`），Python 包为 `spacemit_avatar`，命令行程序为 `avatar_worker`。单配置 CMake 默认用 `Release`，也可显式指定 `CMAKE_BUILD_TYPE`。

生产 worker 源码在 `src/cli/`，`examples/` 保留调用示例。独立仓库的 GitHub Actions 与 `test.yaml` 的 PR 项共用 `tests/run_sdk_test.sh`，执行 Python 单测、C++ 合约和安装后的加载检查。真实 ONNX 对照继续在准备好模型和运行库的环境中按需运行，详见 [验证方法](docs/VALIDATION.md)。

`python3 -m unittest discover -s tests/python -p test_resources.py -v` 验证下载校验、续传、路径/软链接、保留旧文件及静音资源。`tests/python/native_parity.py` 读取真实模型，分别比较原Python特征服务、CPU UNet、预处理与独立NumPy/OpenCV贴回参考；命令和限制见 `docs/VALIDATION.md`。

`package.xml`、`test.yaml` 和根 `pyproject.toml` 分别提供 SDK 依赖、测试入口和 PEP 517 构建入口。提交说明见 `docs/MODEL_ZOO_SUBMISSION.md`。
