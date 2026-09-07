"""Load and filter SWE-bench instances. Uses the official swebench package's loader --
never reimplement dataset parsing (per CLAUDE.md non-goals)."""
from swebench.harness.utils import load_swebench_dataset

# Prefer lighter repos over django/sympy, per CLAUDE.md tech constraints.
DEFAULT_REPOS = ["psf/requests", "pylint-dev/pylint"]


def load_instances(repos=None, limit=None, instance_ids=None):
    repos = repos if repos is not None else DEFAULT_REPOS
    dataset = load_swebench_dataset(
        name="SWE-bench/SWE-bench_Verified",
        split="test",
        instance_ids=instance_ids,
    )
    filtered = [inst for inst in dataset if inst["repo"] in repos]
    if limit is not None:
        filtered = filtered[:limit]
    return filtered
