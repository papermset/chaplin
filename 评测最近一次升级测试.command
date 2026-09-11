#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
shopt -s nullglob
sessions=(data/sessions/upgrade_test_*)
if (( ${#sessions[@]} == 0 )); then
    echo "还没有测试数据，请先运行“开始升级能力测试.command”。"
    exit 1
fi
session_dir="${sessions[${#sessions[@]}-1]}"
echo "评测：$session_dir"
.venv/bin/python -m benchmark.evaluate "$session_dir"
echo "评测文件已生成，正在打开结果文件夹。"
open "$session_dir/evaluation"
