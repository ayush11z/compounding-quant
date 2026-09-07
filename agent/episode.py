"""Run one agent episode against one SWE-bench instance's sandbox and return the
resulting patch plus trajectory stats. Grading happens separately (harness/grade.py)."""
import time

from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph import build_graph
from agent.llm import build_llm
from agent.tools import TOOL_DESCRIPTIONS, build_tools
from harness.sandbox import Sandbox
from logging_setup import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = f"""You are an autonomous coding agent fixing a bug in a Python repository.
The repo is checked out at /testbed inside your sandbox.

To act, respond with EXACTLY these two lines and nothing else:
ACTION: <tool_name>
ACTION_INPUT: <a single valid JSON object of arguments>

Available tools:
{TOOL_DESCRIPTIONS}

Example:
ACTION: run_bash
ACTION_INPUT: {{"command": "grep -rn 'def some_function' --include=*.py ."}}

Start by exploring with run_bash (e.g. `find . -name '*.py'`, `grep -rn <keyword> .`,
`ls <dir>`) to locate the exact file and lines the problem statement refers to -- do not
guess filenames like README.md or requirements.txt. Once you've found the relevant code,
read it with read_file, make your fix with write_file, then run_tests to check it. Only
call submit once you have run_tests and confirmed it passes. Every response must be only
an ACTION/ACTION_INPUT pair -- no other commentary."""


def run_episode(instance: dict, run_id: str, max_turns: int = 20) -> dict:
    instance_id = instance["instance_id"]
    sandbox = Sandbox(instance, run_id).start()
    try:
        tools = build_tools(sandbox)
        llm = build_llm()
        graph = build_graph(llm, tools, max_turns)

        initial_state = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=instance["problem_statement"]),
            ],
            "turns": 0,
            "submitted": False,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "stuck": False,
        }
        config = {
            "configurable": {"thread_id": instance_id},
            "recursion_limit": max_turns * 2 + 10,
        }

        start = time.time()
        final_state = graph.invoke(initial_state, config=config)
        wall_clock_seconds = time.time() - start

        model_patch = sandbox.get_diff()
        return {
            "instance_id": instance_id,
            "model_patch": model_patch,
            "turns": final_state["turns"],
            "submitted": final_state.get("submitted", False),
            "stuck": final_state.get("stuck", False),
            "prompt_tokens": final_state.get("prompt_tokens", 0),
            "completion_tokens": final_state.get("completion_tokens", 0),
            "wall_clock_seconds": wall_clock_seconds,
        }
    except Exception:
        logger.exception(f"Episode failed for {instance_id}")
        return {
            "instance_id": instance_id,
            "model_patch": "",
            "turns": 0,
            "submitted": False,
            "stuck": False,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "wall_clock_seconds": 0.0,
            "error": True,
        }
    finally:
        sandbox.stop()
