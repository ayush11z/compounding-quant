"""Run one instance end-to-end: agent episode -> official grading -> record result."""
from agent.episode import run_episode
from harness.grade import grade
from logging_setup import get_logger
from results.store import record_episode

logger = get_logger(__name__)


def run_and_grade(instance: dict, run_id: str, precision: str, max_turns: int = 20) -> dict:
    instance_id = instance["instance_id"]
    logger.info(f"[{instance_id}] starting episode (max_turns={max_turns})")
    episode_result = run_episode(instance, run_id, max_turns=max_turns)

    resolved = False
    if not episode_result.get("error") and episode_result["model_patch"].strip():
        try:
            resolved, report = grade(instance, episode_result["model_patch"], run_id)
        except Exception:
            logger.exception(f"[{instance_id}] grading failed")
    else:
        logger.info(f"[{instance_id}] empty patch, skipping grading (resolved=False)")

    record_episode(run_id, precision, episode_result, resolved)
    logger.info(
        f"[{instance_id}] resolved={resolved} turns={episode_result['turns']} "
        f"submitted={episode_result['submitted']} stuck={episode_result.get('stuck', False)} "
        f"tokens(p/c)={episode_result['prompt_tokens']}/{episode_result['completion_tokens']} "
        f"wall_clock={episode_result['wall_clock_seconds']:.1f}s"
    )
    return {**episode_result, "resolved": resolved}
