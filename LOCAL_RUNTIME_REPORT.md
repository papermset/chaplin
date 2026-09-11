# Mac 本地运行验证 · 2026-09-11

本机可以运行当前 Auto-AVSR + Qwen3:4B 完整链路。CPU 唇语解码已经通过真实预训练模型测试，Qwen 使用 Apple GPU；两段示例的完整处理约需 15～17 秒，适合先做短句实验。尚未验证用户自己的静默唇语准确率和现场按键/摄像头权限。

## 环境与安装

- MacBook Air M4：10 核 CPU、10 核 GPU、16 GB 统一内存，macOS 26.3。
- 项目 `.venv`：Python 3.12.13、torch 2.14.0、MediaPipe 0.10.14；详细依赖见初次交付报告。
- Homebrew 安装 Ollama 0.32.15，并运行 `brew services run ollama`；监听本机 `127.0.0.1:11434`，没有注册登录自启动。Homebrew 同时安装/更新了它需要的 MLX、Python 等依赖。
- Qwen `qwen3:4b`：4.0B 参数，Q4_K_M，下载约 2.5 GB；digest `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`。
- 完整 VSR 和语言模型的四个文件已经放入 `benchmarks/LRS3/`；下载后的两个 checkpoint SHA-256 与 Hugging Face 返回的文件校验值匹配。

| 模型 | checkpoint 字节数 | SHA-256 |
|---|---:|---|
| Amanvir/LRS3_V_WER19.1 | 1,001,908,942 | `e740cef369abeabd0ba2c18e37a0661342e1d94d432d6caa77755a11821d8fe3` |
| Amanvir/lm_en_subword | 215,076,928 | `c75aa39020dec98f432c8689b145d3f4cc407d4daa90a0c64202386c34f83c18` |

## 实测结果

使用[公开项目示例视频](https://github.com/spazewalker/AudioVisualSpeechRecog_OpenCV/blob/master/video.mp4)，分别截取从第 4 秒开始的 3 秒和从第 12 秒开始的 6 秒。按 Chaplin 的保存条件模拟为 213×160、16 FPS、JPEG quality 25、灰度视频；预处理重采样到模型的 25 FPS。真实 MediaPipe 人脸检测、嘴部裁剪、视觉 encoder、ESPnet beam search、语言模型和 Ollama 全部执行，没有使用模拟输出。

| 片段 | CPU VSR | Qwen 重排 | 总处理时间 | 真实候选 | CTC logits |
|---|---:|---:|---:|---:|---|
| 3 秒 | 2.65 秒 | 11.93 秒 | 14.63 秒 | 10 | 75×5049 |
| 6 秒 | 4.09 秒 | 12.35 秒 | 16.53 秒 | 10 | 150×5049 |

CPU 使用 4 个 PyTorch 线程，完整 VSR + LM 加载 1.85 秒。Qwen 在测量前已加载，所以以上重排时间不代表冷启动。总处理时间不含录制时长和最后向应用键入文本。每种长度只测一次最终配置，不代表稳定分位数。

两条原始输出分别是 `THAT'S WHAT I'M GOING TO DO`、`THE CONCLUSION ABOUT THE BELLS`。Qwen 都成功给十个真实候选评分，选择原始 top-1，状态均为 `reranked`。这些文本是模型预测，并非人工确认的 transcript；没有参考标注，因此不能据此计算识别准确率，也没有证明重排改善 WER。四项 context 均进入接口；此次初始 context 为空，第二段带上第一段输出/历史，尚未测量自定义词汇的收益。

结果与原始样本：

- [runtime.json](data/sessions/local_cpu_validated/runtime.json)
- [3 秒样本](data/sessions/local_cpu_validated/20260911T090045243073Z_e18c6c6f/sample.json)
- [6 秒样本](data/sessions/local_cpu_validated/20260911T090059885258Z_d634a049/sample.json)
- 每个样本同目录保留 `video.mp4` 和 `ctc_logits.npz`。
- [CSV/JSON 评测](data/sessions/local_cpu_validated/evaluation/metrics.json)：两条均为未标注而排除，WER/CER 为 null。

进程峰值 RSS 为 2,369,093,632 字节（约 2.21 GiB）；Ollama 报告模型及运行分配约 2.90 GB（2.70 GiB），context 4096、GPU 加载。这两种统计口径不同，不能把它们的和当作整机精确峰值。测试期间系统内存压力等级曾升至 2、交换空间约 3.7 GB，包含用户其他应用和 MPS 尝试，不能归因于本项目全部占用。安装后剩余磁盘约 14 GiB。

## 本轮修改与发现

- `setup.sh`：改用 macOS 自带 curl，支持断点续传；下载完成才改为正式文件名。
- `benchmark/check_local.py`：加入可重复的现有视频验证入口，保存完整样本、时延、模型加载时间、进程 RSS 和 Ollama 状态；不打开摄像头、不触发键盘输入。
- `decoding/reranker.py`：真实 Qwen 默认思考导致两个样本均触发 60 秒超时；仅关闭思考后返回过界分数。最终关闭思考、把输出 schema 限定为 0～1 的 0.1 间隔评分，明确区分输入 log score 与输出评分，精简提示中的 token/scorer 冗余；完整原始信息仍在样本审计中保存。无效回复的原文也随回退结果保存，便于定位问题。
- 请求参数：`think=false`、`temperature=0`、`seed=0`、`num_ctx=4096`、`num_predict=1024`；超时默认 60 秒。离散评分可能增加并列，仍按原始 beam 顺序破同分；上下文过长受窗口限制。
- `tests/test_reranker.py`：补充真实请求协议及有效选择测试，验证异常回复仍保留审计原文。
- `README.md`、本报告、初次交付报告：更新安装与实测状态；`validation/` 保存测试、来源和 MPS 失败证据。

## MPS 的具体限制

完整模型可以加载到 MPS，但第一次 CTC prefix scoring 即失败：

```text
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, mps:0 and cpu!
```

仓库内 `espnet/nets/ctc_prefix_score.py` 的 `CTCPrefixScoreTH.__init__` 使用 `x.is_cuda` 选择设备，非 CUDA 一律设置成 CPU。因此基准索引 `idx_bo` 留在 CPU，而候选 `scoring_ids` 已在 MPS。详见 [MPS 失败日志](validation/mps_failure.log)。这一问题不同于缺少模型或显存不足，也不能靠开启 CPU 算子 fallback 自动解决显式的设备混用。

本轮保持上游 beam 算法不变，主入口继续采用已验证的 CPU 路径；诊断脚本的 MPS 选项不能视作已支持的运行方式。后续若要支持 MPS，需要修正 scorer 的设备继承，并验证后续索引、算子支持、CPU/MPS 候选一致性和实际速度。

真实 N-best 与 CTC pre-softmax logits 均已成功导出。当前 beam 对象只保存累计路径分数，不稳定暴露整个搜索树上 attention/LM 的逐步原始 logits；本次没有把 CTC 张量冒充这些数据。

## 现在怎么运行

在本报告所在项目目录执行：

```sh
# 服务若因重启而停止：
brew services run ollama

# 采集自己的三模式标注数据；默认不向应用键入预测文本。
.venv/bin/python main.py \
  config_filename=./configs/LRS3_V_WER19.1.ini detector=mediapipe \
  benchmark.enabled=true benchmark.session_dir=data/sessions/my_first_test \
  benchmark.manifest=configs/benchmark_manifest.example.jsonl \
  context_file=configs/context.example.json capture_logits=true
```

按 Option/Alt 开始和停止一句话，按 `q` 退出；交互与原项目一致。相机和全局按键可能需要授予运行终端的摄像头、辅助功能/输入监控权限，本轮没有代替用户完成现场授权或实际录制。

普通模式去掉 benchmark/context/logits 参数即可；如需先体验 CPU 原始输出速度，可加 `reranker.enabled=false`。普通模式会按现有行为键入最终结果并删除临时视频。

重新验证本次保留的公开视频，不需要摄像头：

```sh
.venv/bin/python -m benchmark.check_local \
  data/sessions/local_cpu_validated/*/video.mp4 \
  --session-dir data/sessions/recheck_cpu --device cpu
```

采集后评测：

```sh
.venv/bin/python -m benchmark.evaluate data/sessions/my_first_test
```

## 检查结果与剩余验证

- 单元/最小神经网络 smoke tests：**29 passed，1.87 秒**，10 条上游 distutils 弃用警告；[测试输出](validation/tests.txt)。
- Ruff、compileall、`bash -n setup.sh`、`git diff --check`：通过。
- 主入口 Hydra 配置加载：通过。
- 真实 CPU + Qwen 两段视频：通过；每条样本均有真实十候选、原始 score、有效 rerank score、最终选择及 CTC logits。
- 真实 MPS：失败，具体原因见上文；未改动默认 CPU 路径。
- 仍需现场 Option/Alt、摄像头、实际键入与用户自己的 normal/low_voice/silent 标注测试。公开有声说话视频不能替代静默唇语准确率验证。

没有加入任何自动录制功能、配置或预留接口。本报告记录当时的本地运行结果；
后续个人试录暴露的重排无收益和部分输出未完成问题见 README 的当前状态说明。
