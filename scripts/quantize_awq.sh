#!/usr/bin/env bash
set -uo pipefail
cd ~/quantization
source .venv/bin/activate
export LD_LIBRARY_PATH=$HOME/quantization/mpi_env/lib:${LD_LIBRARY_PATH:-}
nohup python3 TensorRT-LLM-repo/examples/quantization/quantize.py \
  --model_dir ~/quantization/models/Qwen2.5-Coder-7B-Instruct \
  --dtype bfloat16 \
  --qformat int4_awq \
  --awq_block_size 128 \
  --calib_dataset ~/quantization/calib_data \
  --calib_size 400 \
  --output_dir ~/quantization/checkpoints/qwen2.5-coder-7b-awq \
  > ~/quantization/quantize_awq.log 2>&1 &
disown
echo "AWQ quantization started, PID $!"
echo "Check progress with: tail -30 ~/quantization/quantize_awq.log"
