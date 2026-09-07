#!/usr/bin/env python3
"""Remove unused placeholder tensors from a TRT-LLM checkpoint's rank0.safetensors.

trtllm-build 0.16.0 chokes on Qwen2 checkpoints' unused RoPE placeholder tensors
(embed_positions, rotary_inv_freq, embed_positions_for_gpt_attention) with a
`set_weights_name` TypeError, since they're stored as raw numpy arrays rather than
going through the normal weight-loading path. convert_checkpoint.py itself flags
them as "Provided but not required tensors" -- safe to drop.
"""
import sys
from safetensors.torch import load_file, save_file

UNUSED_KEYS = {"embed_positions", "rotary_inv_freq", "embed_positions_for_gpt_attention"}

def main(ckpt_path):
    print(f"Loading {ckpt_path} ...")
    tensors = load_file(ckpt_path)
    dropped = [k for k in tensors if k in UNUSED_KEYS]
    print(f"Dropping keys: {dropped}")
    for k in dropped:
        del tensors[k]
    print(f"Saving back to {ckpt_path} ({len(tensors)} tensors remain) ...")
    save_file(tensors, ckpt_path)
    print("Done.")

if __name__ == "__main__":
    main(sys.argv[1])
