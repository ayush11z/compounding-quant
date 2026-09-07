#!/usr/bin/env bash
set -uo pipefail
mkdir -p ~/quantization/models
nohup huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct \
  --local-dir ~/quantization/models/Qwen2.5-Coder-7B-Instruct \
  > ~/quantization/download.log 2>&1 &
disown
echo "Download started in background, PID $!"
echo "Check progress with: tail -20 ~/quantization/download.log"
