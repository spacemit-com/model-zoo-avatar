# 模型与素材说明

本组件提供音频驱动、人物绑定的局部人脸生成，不是任意照片泛化或整脸动作生成模型。

- FeatherHuBERT：waveform FP32 [1,samples] → hidden FP32 [1,tokens,1024]。
- 人物UNet：input FP32 [1,6,160,160] + audio FP32 [1,16,32,32] → output FP32 [1,3,160,160]。
- zihan与feifan是两份不同的人物UNet及配套帧/关键点；不能互换模型与素材。
- 音频为16kHz单声道PCM16。静音门槛和归一化与固定版本参考服务一致。
- 权重、素材来自资源服务器 model_zoo/avatar/ 下的20261021-r1归档；来源、大小和SHA256见config/resources.json。

speech.wav是公开资源服务器的通用测试讲话录音；feifan_test.wav（9.0秒）与zihan_test.wav（5.6秒）分别是两个人物的测试录音，均为16kHz单声道PCM16，用于驱动嘴型；silence.wav在本地确定性生成。它们不是TTS或克隆音色参考录音。

项目方于 2026-11-06 确认，当前人物素材用于本次 Model Zoo 提交无需另行许可。代码以 Apache-2.0 发布；模型权重的再分发条款单独记录，当前状态见 `config/resources.json` 的 `license_status`。模型、人物帧和关键点由下载器按资源清单获取，发布包只包含软件。

本次抽取不重新训练、不改ONNX插值属性、不重新量化。现有Resize优化依赖配套EP能力，使用spacemit配置时需以ORT profile确认真实UNet的CPU Resize回退情况；只显示provider名称不是覆盖率证明。
