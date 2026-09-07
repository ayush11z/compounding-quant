#!/usr/bin/env bash
set -uo pipefail
cd ~/quantization
source .venv/bin/activate
export LD_LIBRARY_PATH=$HOME/quantization/mpi_env/lib:${LD_LIBRARY_PATH:-}
nohup python3 TensorRT-LLM-repo/examples/qwen/convert_checkpoint.py \
  --model_dir ~/quantization/models/Qwen2.5-Coder-7B-Instruct \
  --output_dir ~/quantization/checkpoints/qwen2.5-coder-7b-gptq \
  --dtype bfloat16 \
  --use_weight_only \
  --weight_only_precision int4_gptq \
  --calib_dataset ~/quantization/calib_data \
  --per_group \
  --group_size 128 \
  > ~/quantization/convert_gptq.log 2>&1 &
disown
echo "Conversion started, PID $!"
echo "Check progress with: tail -30 ~/quantization/convert_gptq.log"
