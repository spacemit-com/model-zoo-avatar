# 实现与运行边界

## 分层

- `include/avatar_service.h`：RuntimeOptions、FeatureBatch、流式FeatherHubertExtractor、UnetLipGenerator、PreparedFrame。没有暴露Ort::Value或应用服务对象。
- `src/features.cpp`：原Python滚动特征算法的C++迁移；12000采样点内部分块、20000采样点滚动上下文、640采样点视频拍、左右4帧上下文、1e-3静音门槛、全局frame id。
- `src/models.cpp`：两张ONNX图的执行、类型/shape及非有限值检查。
- `src/runtime.cpp`：手动初始化ORT C API、进程内固定一套运行库、按需初始化SpaceMIT EP；不链接或复制私有运行库到源码。
- `src/image.cpp`：从旧渲染器提取的11段图像函数及拥有输入、缓冲区、逐流状态的新公共封装。配置显式控制native mask、mouth mask、锐化和时间平滑。
- `src/cli/worker.cpp`：原生extract/render工具；输出PNG，无MPP/播放器/网络服务依赖。
- `python/avatar_bindings.cpp`：仅做dtype/shape、所有权、GIL和接口转换。
- `python/spacemit_avatar/pipeline.py`：两个原生子进程的离线编排、取消和超时；没有第二套Python模型算法。

## 对等与变化

参考服务保存在 `tests/reference/feature_reference.py`，来源哈希在 `config/source-baseline.json`。同一分块序列逐组比较，不声称不同外部分块方式的特征数值逐位相同。NumPy和C++归约次序可能导致浮点舍入差异，真实模型误差由回归报告量化。

低层特征提取器保持原帧数与窗口语义；完成final后要求reset再输入，这是新增的明确生命周期约束。CLI只在最后补闭嘴零特征，使文件示例覆盖完整PCM时长，补帧不属于模型结果。PreparedFrame默认与bianbu-avatar一致：原生贴回直接替换人脸区域，仅在 `native_pasteback` 与 `mouth_only_blend` 都关闭时使用残差合成；没有把它等同于MPP硬件编码或整个数字人播放器。

## FTF1

64字节小端头：0..3魔数FTF1，4..7版本1，8..11采样率16000，12..15帧率25，16..23窗口数uint64，24..31原始PCM样本数uint64，其余保留为0。随后每窗口为16384个小端FP32，布局[16,32,32]。固定长度与文件大小严格校验。此格式仅作为同版本原生示例的临时交换协议。

## 构建和发布

C++17、CMake、可选pybind11、PEP517 setuptools后端。VERSION是构建版本来源。模型和assets独立下载；不因import Python包联网。资源版本与软件版本分开固定，模型和素材通过SHA256绑定。代码以Apache-2.0发布；模型权重的再分发条款见MODEL_CARD.md。
