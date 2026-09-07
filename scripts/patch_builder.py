#!/usr/bin/env python3
"""Patch tensorrt_llm/builder.py to skip known-bad unused Qwen2 placeholder
weights (embed_positions, rotary_inv_freq, embed_positions_for_gpt_attention)
that crash set_weights_name() with a TypeError on a raw numpy array. See
CLAUDE.md gotchas for full context. Idempotent -- safe to re-run.
"""
import sys

path = sys.argv[1]
with open(path) as f:
    content = f.read()

old = """                if not param.set_name(name, network):
                    raise RuntimeError(f'Failed to set weight: {name}')"""

new = """                if name in ('embed_positions', 'rotary_inv_freq', 'embed_positions_for_gpt_attention'):
                    continue
                if not param.set_name(name, network):
                    raise RuntimeError(f'Failed to set weight: {name}')"""

if "embed_positions', 'rotary_inv_freq'" in content:
    print("Already patched, skipping.")
    sys.exit(0)

if old not in content:
    print("ERROR: expected code block not found, aborting (file may differ from expected).")
    sys.exit(1)

content = content.replace(old, new, 1)
with open(path, "w") as f:
    f.write(content)
print("Patched successfully.")
