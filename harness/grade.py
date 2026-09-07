"""Grade a model-produced patch using the official swebench package -- never
reimplement SWE-bench grading (per CLAUDE.md non-goals)."""
import docker
from swebench.harness.run_evaluation import run_instance
from swebench.harness.utils import make_test_spec

from harness.sandbox import IMAGE_PLATFORM
from logging_setup import get_logger

logger = get_logger(__name__)


def grade(instance: dict, model_patch: str, run_id: str, model_name: str = "qwen2.5-coder-7b-bf16"):
    test_spec = make_test_spec(instance)
    client = docker.from_env()
    try:
        client.images.get(test_spec.image)
    except docker.errors.ImageNotFound:
        logger.info(f"Pulling {test_spec.image} for {IMAGE_PLATFORM} (emulated)...")
        client.images.pull(test_spec.image, platform=IMAGE_PLATFORM)

    pred = {
        "instance_id": instance["instance_id"],
        "model_patch": model_patch,
        "model_name_or_path": model_name,
    }
    instance_id, report = run_instance(test_spec, pred, client, run_id)
    resolved = bool(report.get(instance_id, {}).get("resolved", False))
    return resolved, report
