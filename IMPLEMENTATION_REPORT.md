# Chaplin P0 + P1 交付报告

基于 [amanvirparhar/chaplin](https://github.com/amanvirparhar/chaplin) 的提交
`7aee1f8fca776ce4f63690063310b53573b7d804`，已在本地仓库实际完成改造。
本报告记录初次本地交付时的状态。已创建本项目专用 `.venv` 并安装测试依赖。

后续已安装完整模型并完成真实 CPU + Qwen 验证，当前结果见
[本机运行报告](LOCAL_RUNTIME_REPORT.md)。以下初次交付测试记录保留其原始口径。

## 约束与完成范围

- Option/Alt 仍是唯一录制触发：按一次开始、再按一次停止；`q` 退出。
- 没有自动 V-SAD、auto-record 的实现、配置、占位接口或预留扩展。
- benchmark 保存每个片段的视频、参考 transcript（`ground_truth`）、三种 mode、speaker、相机元数据、原始 VSR、LLM 最终选择和延迟。
- benchmark 中过短、退出时中断、推理失败的视频也保留并标记；正常模式在 VSR 后清理临时视频。
- 离线评测输出 `metrics.json`、`metrics.csv`、`samples.csv`，包含 raw/LLM WER、两路 CER、总体及按 mode 分组指标、排除样本原因。
- N-best 直接来自仓库内置 ESPnet beam search，默认最多 10 条；不会从单一文本生成候选。
- 保存 token IDs、原始总分、各 scorer 分量及权重；按需保存 CTC 原始 logits 和词表。
- LLM 使用四项 context 给已有候选 ID 打分；程序执行选择，不能输出 LLM 新造的文本。
- 保存原始分数、rerank score、最终 ID/文本和回退原因。LLM 超时、异常或无效回复回退到 raw top-1。

## 修改文件

| 文件 | 作用 |
|---|---|
| `chaplin.py` | 接入采样标签、样本生命周期、唯一 Alt 热键、视频保留/清理、异步输出顺序与退出排空 |
| `main.py` | 初始化新配置、先加载模型再启动线程、传入实际保存的 16 FPS |
| `pipelines/model.py` | 移除结构化路径中的 one-best 截断；直接提取真实 beam hypotheses、score、CTC logits；保留字符串接口 |
| `pipelines/pipeline.py` | 新增结构化 decode 接口；传递 nbest/logits/长度限制；匹配输入与模型帧率 |
| `decoding/hypotheses.py` | DecodeResult/Hypothesis 数据模型及真实 beam 转换 |
| `decoding/reranker.py` | 四项 context、Ollama 结构化评分、严格 ID/score 校验、超时回退 |
| `decoding/session.py` | 推理计时、上下文累积、重排结果保存、普通模式 P1 审计记录 |
| `benchmark/schema.py` | 样本标签和 JSONL manifest 校验 |
| `benchmark/recorder.py` | 唯一样本目录、原子 JSON、视频路径及 logits 文件保存 |
| `benchmark/metrics.py`、`benchmark/evaluate.py` | 编辑距离、统一归一化、micro WER/CER、分组及 CSV/JSON 导出 |
| `pipelines/data/video_io.py` | 使用现有 OpenCV 解码 RGB 视频，兼容移除 read_video 的新版 torchvision |
| `pipelines/data/data_module.py`、两个 detector 的 `detector.py` | 接入同一个 RGB 视频读取函数 |
| `hydra_configs/default.yaml` | nbest、benchmark、reranker、context、logits 和 audit 配置 |
| `configs/benchmark_manifest.example.jsonl`、`configs/context.example.json` | 三模式采样和 context 示例 |
| `requirements.txt`、`requirements-dev.txt`、`pyproject.toml` | MediaPipe 旧 API 兼容范围、统一 OpenCV 依赖、Python 3.12 distutils 兼容、测试/检查配置 |
| `tests/test_benchmark.py`、`test_decoder.py`、`test_reranker.py`、`test_session.py`、`test_capture.py`、`test_video_io.py` | 指标、真实解码链、重排验证、生命周期和 MP4 smoke tests |
| `README.md`、`.gitignore`、各模块的 `__init__.py` | 安装/运行/数据语义说明、忽略本地采样数据、模块入口 |

上游 `espnet/` 算法和模型权重未修改。主入口仍为 `main.py`，原有
`InferencePipeline(video)` 和 `AVSR.infer(data)` 仍返回字符串。

## 运行

在本仓库目录执行。完整现场运行需要先下载上游模型，并启动 Ollama：

```sh
./setup.sh
ollama pull qwen3:4b
# 确保 Ollama 服务已启动。
```

本次已安装 `.venv`；在另一台机器上可重建：

```sh
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-dev.txt
```

采集三种 mode 的示例 benchmark：

```sh
.venv/bin/python main.py \
  config_filename=./configs/LRS3_V_WER19.1.ini detector=mediapipe \
  benchmark.enabled=true benchmark.session_dir=data/sessions/test_01 \
  benchmark.manifest=configs/benchmark_manifest.example.jsonl \
  context_file=configs/context.example.json capture_logits=true
```

按终端提示用 Option/Alt 开始和停止每句话；默认 benchmark 不向光标键入文本。
每次打开片段消耗一个 manifest 条目，包括短片段和中断片段。
普通使用去掉 benchmark 参数即可；仍会保留 P1 分数审计 JSON，但不会保留视频。

评测可以单独使用标准库 Python，无须加载模型：

```sh
python3 -m benchmark.evaluate data/sessions/test_01
```

输出位于 `data/sessions/test_01/evaluation/`。同一批保存的预测可以反复计算，
不会调用 LLM 或覆盖原始样本。`ground_truth=null` 表示待标注，空字符串表示
明确的空参考；零分母指标为 null。LLM 回退样本按实际最终输出计分，回退数量
单独列出。详细归一化、延迟口径及数据字段见 README。

## 实际验证结果

环境：macOS 26.3、Apple Silicon、Python 3.12.13。

- `PYNPUT_BACKEND=dummy .venv/bin/python -m pytest -q`：**28 passed，24.08 秒**。
- 指定新增/修改 Python 文件的 Ruff 检查：**通过**。
- `compileall` 检查 benchmark、decoding、tests、主入口及 pipelines：**通过**。
- `git diff --check`：**通过**。
- `PYNPUT_BACKEND=dummy .venv/bin/python main.py --cfg job`：**配置加载通过**。
- 离线评测 CLI 在测试中连续执行两次，三个导出文件逐字节一致。

真实运行过的神经网络 smoke test 包括：

1. 仓库内置 attention decoder + CTC prefix scorer + BatchBeamSearch，返回十条候选。
2. 每条候选的 token、原始总分和分项分数与原始 beam 对象逐项对照。
3. CTC logits 与 pre-softmax projection 的实际张量数值对照。
4. 创建小型随机 checkpoint，重新由 AVSR 加载；真实 MP4 解码、固定 landmarks 的嘴部裁剪、时间重采样、视觉 encoder、beam 和 logits 全链通过。
5. 真实 MP4 的写入/读取另有独立验证。

录制生命周期测试使用模拟摄像头、全局热键和键盘，实际调用录制代码、线程与
保存/清理逻辑；没有开启用户摄像头或向用户应用键入内容。LLM 测试使用模拟
Ollama 回复验证协议、错误和超时，未调用真实 Qwen。

最终环境关键版本：

```text
torch==2.14.0
torchvision==0.29.0
torchaudio==2.11.0
mediapipe==0.10.14
opencv-contrib-python==5.0.0.93
numpy==2.5.3
ollama==0.6.2
pydantic==2.13.5
hydra-core==1.3.6
pytest==9.1.1
ruff==0.16.7
```

仍有 10 条上游 distutils 版本比较弃用警告；测试无失败。
曾发现 OpenCV/PyAV 同时导入产生重复 AVFoundation 类警告，最终改为共享
OpenCV 读取后，配置启动验证中不再出现该警告。

## 技术限制与未验证风险

**N-best 能真实暴露，但不保证永远凑满十条。** 上游 beam search 返回已到 EOS
的路径集合，数量受终止条件、输入和剪枝影响。保存请求数量、可用数量、返回
数量，候选不足时如实保留；路径对应文本重复时不伪造或替补候选。beam score
是各 scorer 加权累积值，不是已校准的概率。

**实际导出的是 CTC logits，不是整个搜索树的 attention/LM logits。** 当前 CTC
线性层提供稳定的 `[T, vocabulary]` pre-softmax 激活，可以使用同一 encoder
输出提取。Attention scorer 对不同 prefix 返回 log-softmax 分数，beam 对象
最终只保留路径累计分数和内部状态，没有稳定导出所有分支逐步原始 logits 的
接口。本次未侵入搜索循环去保留完整 lattice。没有 CTC projection 时明确记录
`model_has_no_ctc_projection`，没有启用采集时记录 `capture_disabled`。

**尚未验证用户静默唇语的真实识别效果。** 后续已下载并运行完整 LRS3 + LM
权重及真实 Qwen 重排，详见本机运行报告；尚无实际用户标注视频或现场
Option/Alt/摄像头权限测试。
随机权重 smoke test 只能证明数据与解码接口可执行，不能证明 WER 下降。
需要收集三种 mode 的标注视频后再比较 raw 与重排指标。

**帧率和上下文仍有实验边界。** 16 FPS 保存视频被重采样到 25 FPS，无法补回
丢失的视觉信息；相机实际采集速度也可能波动，已保存 observed_fps。
`current_app` 来自用户配置，不自动读取操作系统；上下文历史保存最近二十条
最终输出。LLM 的评分是启发式判断，temperature/seed 不能跨模型版本保证一致。
按顺序重排可能增加排队延迟，相关 queue_ms 已暴露。

后续 MPS 测试已发现上游 CTC 设备兼容问题，主入口继续使用 CPU；
其他操作系统、可选 RetinaFace 依赖及音频/视听模式未现场验证。本次
目标是 Chaplin 当前 visual-only + MediaPipe 使用路径。
