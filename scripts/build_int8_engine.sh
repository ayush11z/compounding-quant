#!/usr/bin/env bash
set -uo pipefail
cd ~/quantization
source .venv/bin/activate
export LD_LIBRARY_PATH=$HOME/quantization/mpi_env/lib:${LD_LIBRARY_PATH:-}
nohup trtllm-build \
  --checkpoint_dir ~/quantization/checkpoints/qwen2.5-coder-7b-int8 \
  --output_dir ~/quantization/engines/qwen2.5-coder-7b-int8 \
  --gemm_plugin bfloat16 \
  --max_batch_size 4 \
  --max_seq_len 32768 \
  --max_num_tokens 32768 \
  > ~/quantization/build_int8.log 2>&1 &
disown
echo "Build started, PID $!"
echo "Check progress with: tail -40 ~/quantization/build_int8.log"
