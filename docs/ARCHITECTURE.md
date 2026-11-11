# 实现与运行边界

## 分层

- `include/avatar_service.h`：RuntimeOptions、FeatureBatch、流式FeatherHubertExtractor、UnetLipGenerator、PreparedFrame。没有暴露Ort::Value或应用服务对象。
- `src/features.cpp`：原Python滚动特征算法的C++迁移。每次推理把最近20000个采样点（1.25 s）送入模型，新音频按12000个采样点（0.75 s）分块送入，相邻两次推理重叠0.5 s。每640个采样点（40 ms）对应一个视频帧；模型每20 ms输出一个token，2个token组成一个模型帧，与视频帧一一对应。视频帧f的特征窗口是模型帧f-4至f+3（前4帧、当前帧、后3帧，共8帧，即[16,32,32]）；1.25 s的输入保证每个窗口的8帧都来自同一次推理，只在音频开头之前和结尾之后补零。非final时，每次推理的最后4个模型帧不输出，由下一次推理用更长的上下文重新计算后再输出。整个模型输入的标准差低于1e-3时，本次输出全部置零（闭嘴）。frame id全局连续。
- `src/models.cpp`：两张ONNX图的执行、类型/shape及非有限值检查。
- `src/runtime.cpp`：用ORT C++ API创建会话选项；provider为 `spacemit` 时通过 `SessionOptionsSpaceMITEnvInit` 启用SpaceMIT EP。构建时链接系统 `spacemit-onnxruntime` 的 `libonnxruntime.so` 与 `libspacemit_ep.so`，不复制运行库到源码或wheel；找不到EP时构建仅CPU的库，并拒绝 `spacemit`。
- `src/image.cpp`：从旧渲染器提取的11段图像函数及拥有输入、缓冲区、逐流状态的新公共封装。配置显式控制native mask、mouth mask、锐化和时间平滑。
- `src/cli/worker.cpp`：原生extract/render工具。render按预取（解码、模型输入）、UNet推理、按帧序贴回、写出四段流水线并行：输出目录时多线程写PNG，输出mp4时按帧序经管道送入ffmpeg：K3上先用OpenCV转成I420，交给视频硬件编码器h264_stcodec；试编码一帧失败时改为送BGR帧给libx264 veryfast；成功后才改为最终文件名；各阶段计时，结束时打印性能报告与瓶颈；贴回在同一份后处理状态上按帧序进行，输出与串行逐帧一致。不直接链接MPP，也不依赖播放器或网络服务。
- `examples/avatar_file_demo.cpp`：只用公共API逐帧串行渲染的C++示例，输出与worker逐字节一致，渲染后调用ffmpeg合成带音频的mp4；`python/examples/avatar_file_demo.py` 通过 `AvatarService` 完成同样流程。
- `python/avatar_bindings.cpp`：仅做dtype/shape、所有权、GIL和接口转换。
- `python/spacemit_avatar/pipeline.py`：两个原生子进程的离线编排、取消和超时；没有第二套Python模型算法。

## 对等与变化

参考服务保存在 `tests/reference/feature_reference.py`（原样保留，注释中的“one-second context”是旧说法，实际为1.25 s上下文、0.75 s步长），来源哈希在 `config/source-baseline.json`。同一分块序列逐组比较，不声称不同外部分块方式的特征数值逐位相同。NumPy和C++归约次序可能导致浮点舍入差异，真实模型误差由回归报告量化。

低层特征提取器保持原帧数与窗口语义；完成final后要求reset再输入，这是新增的明确生命周期约束。CLI只在最后补闭嘴零特征，使文件示例覆盖完整PCM时长，补帧不属于模型结果。PreparedFrame默认与bianbu-avatar一致：原生贴回直接替换人脸区域，仅在 `native_pasteback` 与 `mouth_only_blend` 都关闭时使用残差合成；没有把它等同于MPP硬件编码或整个数字人播放器。

## FTF1

64字节小端头：0..3魔数FTF1，4..7版本1，8..11采样率16000，12..15帧率25，16..23窗口数uint64，24..31原始PCM样本数uint64，其余保留为0。随后每窗口为16384个小端FP32，布局[16,32,32]。固定长度与文件大小严格校验。此格式仅作为同版本原生示例的临时交换协议。

## 构建和发布

C++17、CMake、可选pybind11、PEP517 setuptools后端（`python/setup.py`、`python/pyproject.toml`）。`libavatar.a` 为静态库，链接进扩展、worker与示例。在SDK中构建时（envsetup导出 `SROBOTIS_OUTPUT_ROOT`），CMake的 `avatar-python-wheel` 目标用本次编译的扩展与worker打包wheel并安装到 `output/dist`，不重复编译。VERSION是构建版本来源。模型和assets独立下载；不因import Python包联网。资源版本与软件版本分开固定，模型和素材通过SHA256绑定。代码以Apache-2.0发布；模型权重的再分发条款见MODEL_CARD.md。
