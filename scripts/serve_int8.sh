#!/usr/bin/env bash
set -uo pipefail
cd ~/quantization
source .venv/bin/activate
export LD_LIBRARY_PATH=$HOME/quantization/mpi_env/lib:${LD_LIBRARY_PATH:-}
nohup trtllm-serve \
  ~/quantization/engines/qwen2.5-coder-7b-int8 \
  --tokenizer ~/quantization/models/Qwen2.5-Coder-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  > ~/quantization/serve_int8.log 2>&1 &
disown
echo "Server started, PID $!"
echo "Check progress with: tail -40 ~/quantization/serve_int8.log"
