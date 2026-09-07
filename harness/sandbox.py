"""Docker sandbox for one SWE-bench instance, built on the official swebench package's
own container lifecycle helpers (create_container, exec_run_with_timeout, cleanup_container)
so the agent's live environment matches exactly what grading will use later."""
import base64
from pathlib import Path

import docker
from swebench.harness.run_evaluation import (
    CONTAINER_WORKDIR,
    cleanup_container,
    create_container,
    exec_run_with_timeout,
    setup_logger,
)
from swebench.harness.utils import make_test_spec

from logging_setup import get_logger

logger = get_logger(__name__)

# Published swebench images are amd64-only; on Apple Silicon the default
# platform-less pull 404s ("no matching manifest for linux/arm64/v8"). Pre-pull
# under emulation so create_container's `images.get` finds it already cached.
IMAGE_PLATFORM = "linux/amd64"

# Without a cap, one large file read or verbose test run can single-handedly consume
# the whole context window, making longer trajectories structurally impossible
# regardless of precision. This is harness plumbing, not a resolve-rate tweak.
MAX_TOOL_OUTPUT_CHARS = 8000


class Sandbox:
    def __init__(self, instance: dict, run_id: str):
        self.instance = instance
        self.test_spec = make_test_spec(instance)
        self.run_id = run_id
        self.client = docker.from_env()
        self.container = None

    def start(self) -> "Sandbox":
        try:
            self.client.images.get(self.test_spec.image)
        except docker.errors.ImageNotFound:
            logger.info(f"Pulling {self.test_spec.image} for {IMAGE_PLATFORM} (emulated)...")
            self.client.images.pull(self.test_spec.image, platform=IMAGE_PLATFORM)

        log_file = Path("results/logs") / self.run_id / f"{self.test_spec.instance_id}.log"
        self.logger = setup_logger(self.test_spec.instance_id, log_file, add_stdout=False)

        self.container = create_container(self.test_spec, self.client, self.run_id, self.logger)
        self.container.start()
        return self

    def stop(self):
        if self.container is not None:
            cleanup_container(self.client, self.container, self.logger)
            self.container = None

    def _exec_raw(self, cmd: str, timeout: int = 120) -> str:
        # exec_run_with_timeout (unlike container.exec_run) has no workdir param,
        # so cd explicitly instead of relying on the image's default WORKDIR.
        wrapped = f"cd {CONTAINER_WORKDIR} && {cmd}"
        output, timed_out, elapsed = exec_run_with_timeout(
            self.container, ["/bin/bash", "-lc", wrapped], timeout=timeout
        )
        if timed_out:
            output += f"\n[sandbox: command timed out after {elapsed:.0f}s]"
        return output

    def _exec(self, cmd: str, timeout: int = 120) -> str:
        """For agent-facing tool output only -- truncated to protect the context
        window. Never use for get_diff(): the patch must not be truncated."""
        return self._truncate(self._exec_raw(cmd, timeout=timeout))

    @staticmethod
    def _truncate(output: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
        if len(output) <= limit:
            return output
        head, tail = output[: limit // 2], output[-limit // 2 :]
        omitted = len(output) - limit
        return f"{head}\n[sandbox: output truncated, {omitted} chars omitted]\n{tail}"

    def run_bash(self, cmd: str, timeout: int = 120) -> str:
        return self._exec(cmd, timeout=timeout)

    def read_file(self, path: str) -> str:
        return self._exec(f"cat -- {path} 2>&1")

    def write_file(self, path: str, content: str) -> str:
        b64 = base64.b64encode(content.encode()).decode()
        cmd = f"mkdir -p -- \"$(dirname -- {path})\" && echo {b64} | base64 -d > {path}"
        out = self._exec(cmd)
        return out or f"wrote {len(content)} bytes to {path}"

    def run_tests(self) -> str:
        script = "\n".join(self.test_spec.eval_script_list)
        return self._exec(script, timeout=300)

    def get_diff(self) -> str:
        return self._exec_raw("git -c core.fileMode=false diff")
