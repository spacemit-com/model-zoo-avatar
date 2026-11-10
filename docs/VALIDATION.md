# 验证方法

在 Linux 或 K3 的组件目录内执行 SDK 测试清单（`test.yaml`）使用的命令，均不需要模型：

```bash
./tests/run_sdk_test.sh core            # ctest：特征游标、final/reset、重试、遮罩边界、贴回、FTF1
./tests/run_sdk_test.sh config          # 默认参数、链接的 ORT 版本、自带配置
./tests/run_sdk_test.sh invalid-input   # 非法输入抛出异常；avatar_worker 非法命令行返回 2
./tests/run_sdk_test.sh python          # 下载器与 worker 进程生命周期
./tests/run_sdk_test.sh installed       # 安装到临时前缀后从源码目录外加载
```

`installed` 将 CMake 产物安装到临时前缀，从源码目录外加载 Python 原生绑定，启动 `avatar_worker` 与 `avatar_file_demo`，并检查头文件与 `libavatar.a`。GitHub Actions 在 Ubuntu 上运行 `python`、`core`、`config`、`invalid-input`、`installed`，代码风格由 `code-style` 工作流检查。Windows 可直接用 `python -m unittest discover -s tests/python -p 'test_*.py' -v` 运行主机测试。下载器测试不加载模型，使用真实本地 HTTP 续传及临时 tar 文件。

## 真实模型差分

在 K3 上设置 `AVATAR_TEST_HUBERT` 和 `AVATAR_TEST_UNET` 后，运行 `./tests/run_sdk_test.sh parity`。设置 `AVATAR_TEST_UNET_ALT` 可顺带检查第二套人物权重。传入的路径应为绝对路径。此入口对应 `test.yaml` 的 `manual` 和 `release` 范围。

```bash
AVATAR_TEST_HUBERT=$HOME/.cache/models/avatar/feather_hubert_dynamic.onnx \
AVATAR_TEST_UNET=$HOME/.cache/models/avatar/feifan_int8.onnx \
AVATAR_TEST_UNET_ALT=$HOME/.cache/models/avatar/zihan_int8.onnx \
./tests/run_sdk_test.sh parity
```

差分测试默认生成合成 PCM、图像和关键点，也可直接调用并用 `--wav`、`--dataset` 指定真实输入。UNet 参考实现使用 Python OpenCV。

```bash
export PYTHONPATH="$PWD/build/sdk-tests/parity${PYTHONPATH:+:$PYTHONPATH}"
python3 tests/python/native_parity.py --mode feature \
  --model /path/to/feather_hubert_dynamic.onnx --wav /path/to/16k-mono-pcm16.wav \
  --output output/feature-parity.json
python3 tests/python/native_parity.py --mode unet \
  --model /path/to/feifan_int8.onnx --dataset /path/to/feifan --output output/unet-cpu-parity.json
```

特征模式在同一进程中导入 Python onnxruntime 作为参考。UNet 参考在独立 Python 子进程中产生，父进程只使用原生扩展链接的系统 ORT。

EP 验证加 `--provider spacemit --profile output/unet-profile`，再用 `python3 scripts/verify_ep_profile.py <实际profile.json>` 解析 profile，检查 EP 实际执行且没有 CPU Resize 事件。2026-11-05 在 66 板实测系统 EP 2.0.8 仍有 4 个 CPU Resize 事件，数值对照可通过，但 Resize 未全部下沉。不依赖版本字符串推断覆盖能力。系统 EP 2.0.8 自行放置 A100 线程。不要为测试抢占正在使用的数字人工程或更改其 unit/cpuset；本组件不自动管理其它服务。

特征误差阈值 atol=2e-5、rtol=1e-5，帧 id 必须严格连续，静音输出必须全零。相同 CPU 运行库的 UNet 误差阈值 1e-5；EP 相对 CPU 的量化图 RMSE 门槛 0.01。默认图像预处理与参考应逐元素一致，CPU 贴回参考允许最多 1 级 uint8 舍入差异，ROI 外必须逐像素不变。阈值适用于所测固定输入，不是任意素材的视觉质量保证。

## 渲染一致性

改动渲染流程或前后处理时，用改动前后的 `avatar_worker` 对同一特征文件渲染，逐帧比较 PNG 的哈希；`avatar_file_demo` 的输出也应与 worker 的前若干帧逐字节一致。2026-11-10 在 66 板上，两个人物的特征文件与全部帧（feifan 225 帧、zihan 140 帧）均与改动前构建逐字节一致。

实例不允许并发操作，final 后必须 reset；输出数组拥有独立存储。完整文件示例默认离线两阶段，渲染耗时包含写 PNG，不等同于纯模型延迟。
