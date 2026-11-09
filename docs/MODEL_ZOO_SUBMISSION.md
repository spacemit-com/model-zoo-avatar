# Avatar Model Zoo 提交说明

建议标题 `feat(avatar): add FeatherHuBERT and UNet inference component`

组件包含 FeatherHuBERT 音频特征提取、人物 UNet 推理、图像前后处理、pybind11 接口和离线示例。模型与配套示例素材通过 `config/resources.json` 下载并校验。模型文件和人物素材由资源服务器分发，源码包及 wheel 交付软件。

## SDK 接入

- 对照 ai-sdk `5a851083e1d9b0f6ad05fefd65e92511cd9c2178` 的 Vision、ASR、TTS 和 `build/python_wheels.sh`。采用 `package.xml`、CMake install、`test.yaml`、pybind11 和 PEP 517。
- 建议注册为独立的 `avatar` 组件，测试清单使用 `components/model_zoo/avatar`。合入时由维护方确定仓库位置和聚合入口。
- CMake 安装提供公共头文件、共享库、worker 和完整 Python 包。wheel 只安装一个共享库 SONAME 和该包的原生 worker。
- 提交审核使用现有 `0.1.0` 版本。K3 候选 wheel 的平台标签为 `cp314-cp314-linux_riscv64`，运行时由 Bianbu 提供 OpenCV、yaml-cpp 与配套 ORT。

## 审查修复

| 问题 | 修复 |
|---|---|
| 多段特征提取中途失败后游标已经前移 | 将一次 PCM push 作为事务，失败恢复滚动窗口和帧游标 |
| C-contiguous NumPy 缓冲区可未对齐 | 用字节复制进入自有对齐存储后再按类型访问 |
| 错误 ORT 在校验前进入进程的全局符号空间 | 先以局部符号加载并校验，再提升为全局。已加载 EP 句柄复用 |
| 特征文件检查与创建之间存在竞争 | 用独占方式创建，检查 FTF 头、文件大小和 PCM 时间线 |
| 并发渲染可能同时通过空目录检查 | 原生 worker 独占输出目录后检查目录内容 |
| 取消与下一阶段启动之间缺少统一状态保护 | 同一把锁保护 worker 与任务状态，超时先请求退出再回收 |
| 同一模型归档的下载共享临时文件 | 按 SHA256 加缓存锁，安装采用原子操作 |
| CMake staging 缺少 Python 包文件与可用 RPATH | 补齐 site-packages 布局，并验证源码目录外运行 |
| PR 测试假定已有原地 build 目录 | 测试入口显式配置并构建目标，真实模型差分默认使用合成输入 |

## 验证与提交条件

K3 执行 CTest、Python 单测、真实模型差分、人物素材下载、SDK staging 安装和 wheel 安装。对比两个安装方式的输出帧，核验源码包和 wheel 文件清单及 SHA256。EP 覆盖通过真实 ORT profile 记录。详见本轮实机验证报告。

项目方于 2026-11-06 确认当前人物素材用于本次 Model Zoo 提交无需另行许可。提取代码以 Apache-2.0 随本组件发布。正式公开提交前还需记录模型权重的再分发条款。原测试音频按源站条款引用。仓库归属和组件登记由 Model Zoo 维护方确定。
