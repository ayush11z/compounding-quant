#!/usr/bin/env python3
"""Patch tensorrt_llm/builder.py to tolerate set_weights_name() failures.

Two distinct bugs hit the same code path in TRT-LLM 0.16.0's build_engine():
1. Unused Qwen2 RoPE placeholder tensors (embed_positions, rotary_inv_freq,
   embed_positions_for_gpt_attention) are zero-init parameters that reach the
   weight-naming loop despite being unused in forward().
2. INT8 weight-only quantized parameters: _get_weights() does
   `tensor.producer.__class__ = trt.IConstantLayer; return tensor.producer.weights`,
   but for these parameters `tensor.producer` isn't really an IConstantLayer, so the
   unsafe __class__ punning produces a raw numpy array instead of a trt.Weights object.

set_name() is cosmetic (associates a debug-friendly name with an already-embedded
constant in the network graph) -- engine correctness doesn't depend on it succeeding.
Rather than chase the C++ binding bug further, catch the TypeError generically and
skip naming for that one parameter. See CLAUDE.md gotchas for full context.
"""
import sys

path = sys.argv[1]
with open(path) as f:
    content = f.read()

candidates = [
    # Pristine, never-patched file.
    (
        """                if not param.set_name(name, network):
                    raise RuntimeError(f'Failed to set weight: {name}')""",
        """                try:
                    named = param.set_name(name, network)
                except TypeError as e:
                    logger.warning(f"Skipping set_name for '{name}' (cosmetic only): {e}")
                    named = True
                if not named:
                    raise RuntimeError(f'Failed to set weight: {name}')""",
    ),
    # Already has the earlier name-based skip for the RoPE placeholders (Phase 0)
    # -- keep it (harmless) and additionally wrap the remaining call generically,
    # since it turns out both bugs hit the same underlying TypeError.
    (
        """                if name in ('embed_positions', 'rotary_inv_freq', 'embed_positions_for_gpt_attention'):
                    continue
                if not param.set_name(name, network):
                    raise RuntimeError(f'Failed to set weight: {name}')""",
        """                if name in ('embed_positions', 'rotary_inv_freq', 'embed_positions_for_gpt_attention'):
                    continue
                try:
                    named = param.set_name(name, network)
                except TypeError as e:
                    logger.warning(f"Skipping set_name for '{name}' (cosmetic only): {e}")
                    named = True
                if not named:
                    raise RuntimeError(f'Failed to set weight: {name}')""",
    ),
]

if "Skipping set_name for" in content:
    print("Already patched, skipping.")
    sys.exit(0)

for old, new in candidates:
    if old in content:
        break
else:
    print("ERROR: no expected code block found -- file differs from all known states. Aborting.")
    sys.exit(1)

content = content.replace(old, new, 1)
with open(path, "w") as f:
    f.write(content)
print("Patched successfully.")
