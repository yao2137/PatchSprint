#!/usr/bin/env bash
# Download the CanItEdit benchmark and the Qwen2.5-Coder tokenizer (no model weights).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw/canitedit data/raw/qwen2.5-coder-tokenizer
HF=https://huggingface.co
curl -sfL -o data/raw/canitedit/test-00000-of-00001.parquet \
  "$HF/datasets/nuprl/CanItEdit/resolve/main/data/test-00000-of-00001.parquet"
for f in tokenizer.json tokenizer_config.json; do
  curl -sfL -o "data/raw/qwen2.5-coder-tokenizer/$f" "$HF/Qwen/Qwen2.5-Coder-7B-Instruct/resolve/main/$f"
done
echo "done"
