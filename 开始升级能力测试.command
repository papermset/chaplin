#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if ! curl --fail --silent --max-time 3 http://127.0.0.1:11434/api/version >/dev/null; then
    if [[ -x /opt/homebrew/bin/brew ]]; then
        /opt/homebrew/bin/brew services run ollama
    else
        echo "请先启动 Ollama，然后重新打开本文件。"
        exit 1
    fi
fi
session_dir="data/sessions/upgrade_test_$(date +%Y%m%d_%H%M%S)_$$"
echo "本次数据保存到：$session_dir"
echo "共 15 段：正常说话 5 段 → 小声说话 5 段 → 无声口型 5 段。"
echo "每次录制前看 NEXT SAMPLE 提示。Option 开始，再按 Option 停止。"
echo "每段至少 2 秒，建议 3～6 秒。完成后点击摄像头窗口按 q，等待全部处理结束。"
echo "本测试保存视频和预测，但不会向其他应用键入内容。"
.venv/bin/python -u main.py \
    config_filename=./configs/LRS3_V_WER19.1.ini detector=mediapipe \
    benchmark.enabled=true "benchmark.session_dir=$session_dir" \
    benchmark.manifest=configs/upgrade_test.jsonl \
    context_file=configs/upgrade_test_context.json capture_logits=true
printf '\n采集与处理已结束。双击“评测最近一次升级测试.command”查看结果。\n'
