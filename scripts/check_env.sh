#!/usr/bin/env bash
# Phase 0a: report GPU/driver/CUDA/VRAM and host environment inside a DSMLP pod.
# Run from inside a launched pod (after `launch-scipy-ml.sh -g 1 -l gpu-class=medium`),
# not from dsmlp-login. Output is both printed and saved for the CLAUDE.md record.
set -uo pipefail

OUT="${1:-/home/$USER/quantization_env_report_$(date +%Y%m%d_%H%M%S).txt}"

{
  echo "=== date ==="
  date
  echo "=== hostname / node ==="
  hostname
  echo "=== nvidia-smi ==="
  nvidia-smi
  echo "=== nvcc --version ==="
  nvcc --version 2>&1 || echo "no nvcc"
  echo "=== python3 --version ==="
  python3 --version
  echo "=== torch cuda info ==="
  python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'device', torch.cuda.get_device_name(0), 'cap', torch.cuda.get_device_capability(0))" 2>&1
  echo "=== pip: tensorrt_llm / tensorrt / triton presence ==="
  python3 -m pip list 2>/dev/null | grep -iE "tensorrt|triton" || echo "none installed yet"
  echo "=== disk: overlay root ==="
  df -h /
  echo "=== disk: home ==="
  df -h "/home/$USER" 2>&1
  echo "=== mount points for home (persistence check) ==="
  mount | grep -i "$USER" || mount | grep -i home
  echo "=== memory ==="
  free -h
  echo "=== cpu count ==="
  nproc
} | tee "$OUT"

echo ""
echo "Report saved to: $OUT"
