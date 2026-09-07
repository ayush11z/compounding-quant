#!/usr/bin/env python3
"""Phase 1 sanity check: run N instances at a given precision, report resolve rate.
Usage: uv run scripts/run_sanity_check.py [--limit 20] [--max-turns 20] [--precision bf16]
"""
import argparse
import time

from harness.dataset import load_instances
from harness.runner import run_and_grade
from logging_setup import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    run_id = args.run_id or f"sanity-{args.precision}-{int(time.time())}"
    instances = load_instances(limit=args.limit)
    logger.info(f"Loaded {len(instances)} instances for run_id={run_id}")

    results = []
    for i, instance in enumerate(instances, 1):
        logger.info(f"=== [{i}/{len(instances)}] {instance['instance_id']} ===")
        result = run_and_grade(instance, run_id, args.precision, max_turns=args.max_turns)
        results.append(result)

    n_resolved = sum(1 for r in results if r["resolved"])
    n_total = len(results)
    print(f"\n=== SUMMARY (run_id={run_id}) ===")
    print(f"Resolved: {n_resolved}/{n_total} ({100 * n_resolved / n_total:.1f}%)" if n_total else "No instances run")
    for r in results:
        print(f"  {r['instance_id']}: resolved={r['resolved']} turns={r['turns']} "
              f"submitted={r['submitted']} tokens={r['prompt_tokens']}/{r['completion_tokens']}")


if __name__ == "__main__":
    main()
