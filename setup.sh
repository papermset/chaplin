#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# curl ships with macOS. Retain partial downloads for resuming and never expose
# an unfinished file under the final checkpoint name.
download() {
    local url="$1" destination="$2"
    if [[ -s "$destination" ]]; then
        echo "Already present: $destination"
        return
    fi
    mkdir -p "$(dirname "$destination")"
    echo "Downloading: $destination"
    curl --fail --location --show-error --silent --retry 3 \
        --connect-timeout 20 --speed-time 60 --speed-limit 1024 \
        --continue-at - --output "$destination.part" "$url"
    mv "$destination.part" "$destination"
}

download https://huggingface.co/Amanvir/lm_en_subword/resolve/main/model.json \
    benchmarks/LRS3/language_models/lm_en_subword/model.json
download https://huggingface.co/Amanvir/lm_en_subword/resolve/main/model.pth \
    benchmarks/LRS3/language_models/lm_en_subword/model.pth
download https://huggingface.co/Amanvir/LRS3_V_WER19.1/resolve/main/model.json \
    benchmarks/LRS3/models/LRS3_V_WER19.1/model.json
download https://huggingface.co/Amanvir/LRS3_V_WER19.1/resolve/main/model.pth \
    benchmarks/LRS3/models/LRS3_V_WER19.1/model.pth
