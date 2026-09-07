#!/usr/bin/env python3
import inspect
import tensorrt_llm.parameter as p

src = inspect.getsource(p.Parameter)
# print set_name and _get_weights method bodies
import re
for name in ["set_name", "_get_weights", "_regularize_value", "raw_value", "is_managed", "value"]:
    m = re.search(rf"    def {name}\b.*?(?=\n    def |\n    @|\Z)", src, re.DOTALL)
    print(f"=== {name} ===")
    print(m.group(0) if m else "NOT FOUND")
    print()
