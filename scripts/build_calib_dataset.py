#!/usr/bin/env python3
"""Build a small, code-heavy AWQ/GPTQ calibration corpus.

Rationale (see CLAUDE.md): the model's actual workload in this study is reading and
editing Python source, not prose -- calibrating on generic web text (the common default,
e.g. cnn_dailymail) would optimize quantization for a distribution the model never
actually sees during the agent study. Source repos are deliberately DISTINCT from the
evaluation repos (psf/requests, pylint-dev/pylint) to avoid calibration-time
contamination on top of the already-known SWE-bench training-data contamination risk.

Output: a local JSONL file with one {"text": <file content>} per line, loadable via
`datasets.load_dataset("json", data_files=<path>, split="train")` -- the format
tensorrt_llm's quantize_by_modelopt.get_calib_dataloader expects for a custom
--calib_dataset (a local dir/file with a "text" column).
"""
import json
import random
import subprocess
import tempfile
from pathlib import Path

# Distinct from evaluation repos (requests, pylint). Chosen for permissive licenses and
# genuinely varied Python style (web framework, data validation, CLI tooling).
CALIB_REPOS = [
    "https://github.com/pallets/flask.git",
    "https://github.com/psf/black.git",
    "https://github.com/pydantic/pydantic.git",
]
MIN_CHARS = 200
MAX_CHARS = 8000
N_SAMPLES = 400
SEED = 0
OUT_PATH = Path(__file__).parent.parent / "calib_data" / "python_code_calib.jsonl"


def collect_files(repo_url: str, tmp: Path) -> list[str]:
    dest = tmp / Path(repo_url).stem
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(dest)], check=True,
                    capture_output=True)
    texts = []
    for f in dest.rglob("*.py"):
        try:
            content = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if MIN_CHARS <= len(content) <= MAX_CHARS:
            texts.append(content)
    return texts


def main():
    random.seed(SEED)
    all_texts = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for repo in CALIB_REPOS:
            print(f"Cloning {repo} ...")
            texts = collect_files(repo, tmp_path)
            print(f"  {len(texts)} eligible .py files")
            all_texts.extend(texts)

    random.shuffle(all_texts)
    sample = all_texts[:N_SAMPLES]
    print(f"Sampled {len(sample)} files from {len(all_texts)} eligible total")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for text in sample:
            f.write(json.dumps({"text": text}) + "\n")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
