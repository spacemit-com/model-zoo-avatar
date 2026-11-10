# 验证方法

在 Linux 或 K3 的组件目录内执行 SDK 测试清单使用的命令：

```bash
./tests/run_sdk_test.sh core
./tests/run_sdk_test.sh python
./tests/run_sdk_test.sh installed
```

`core` 使用合成输入检查 C++ 合约。`installed` 将 CMake 产物安装到临时前缀，从源码目录外加载 Python 原生绑定并启动 worker，检查动态库布局。GitHub Actions 在 Ubuntu 上运行这三项。Windows 可直接用 `python -m unittest discover -s tests/python -p 'test_*.py' -v` 运行主机测试。

在 K3 上设置 `AVATAR_TEST_CPU_ORT`、`AVATAR_TEST_HUBERT` 和 `AVATAR_TEST_UNET` 后，运行 `./tests/run_sdk_test.sh parity`。设置 `AVATAR_TEST_UNET_ALT` 可顺带检查第二套人物权重。传入的路径应为绝对路径。此入口对应 `test.yaml` 的 `manual` 和 `release` 范围。

构建后 `ctest --test-dir build --output-on-failure` 检查静音、frame id、final/reset、遮罩边界、图像输出形状和非法输入。下载器测试不加载模型，使用真实本地HTTP续传及临时tar文件。

在有真实模型的 Linux 环境运行。差分测试默认生成合成 PCM、图像和关键点，也可用 `--wav` 和 `--dataset` 指定真实输入。UNet 参考实现使用 Python OpenCV。

```bash
export PYTHONPATH="$PWD/build${PYTHONPATH:+:$PYTHONPATH}"
python3 tests/python/native_parity.py --mode feature \
  --model /path/to/feather_hubert_dynamic.onnx \
  --ort-library /path/to/cpu/libonnxruntime.so --wav /path/to/16k-mono-pcm16.wav \
  --output output/feature-parity.json
python3 tests/python/native_parity.py --mode unet \
  --model /path/to/feifan_int8.onnx --dataset /path/to/feifan \
  --ort-library /path/to/cpu/libonnxruntime.so --output output/unet-cpu-parity.json
```

特征模式应选Python onnxruntime自身使用的CPU ORT，避免在同进程装载不同二进制。UNet参考在独立Python子进程中产生，父进程只加载所选原生运行库。

EP验证使用系统库 `--provider spacemit --ep-library /usr/lib/libspacemit_ep.so --profile output/unet-profile`，并解析profile检查真实Resize覆盖。系统EP 2.0.8自行放置A100线程，`--affinity` 不生效。不要为测试抢占正在使用的数字人工程或更改其unit/cpuset；本组件不自动管理其它服务。

用 `python3 scripts/verify_ep_profile.py <实际profile.json>` 进行覆盖门禁。2026-11-05的66板实测系统EP 2.0.8仍有4个CPU Resize事件；已验证优化配套库为0个CPU Resize事件、1个EP子图。两组数值对照都可通过，但仅后者满足“Resize全部下沉”。不依赖版本字符串推断覆盖能力，也不把优化库复制入wheel冒充系统包已支持。

特征误差阈值atol=2e-5、rtol=1e-5，帧id必须严格连续，静音输出必须全零。相同CPU运行库的UNet误差阈值1e-5；EP相对CPU的量化图RMSE门槛0.01。默认图像预处理与参考应逐元素一致，CPU贴回参考允许最多1级uint8舍入差异，ROI外必须逐像素不变。阈值适用于所测固定输入，不是任意素材的视觉质量保证。

实例不允许并发操作，final后必须reset；输出数组拥有独立存储。完整文件示例默认离线两阶段，不把PNG保存耗时叫作纯模型延迟，也不沿用历史数字人的25fps/长对话数据作为新SDK性能结论。

本机实际构建、wheel安装及真实模型结果另见output/validation（被Git忽略）；状态摘要由完成验证后记录。
