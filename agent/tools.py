"""Agent tools: read_file, write_file, run_bash, run_tests (per CLAUDE.md tech
constraints), plus a `submit` control-flow tool the agent calls when it believes
its fix is complete -- not a capability, just how the harness knows to stop and
extract the diff instead of running until the turn cap.

trtllm-serve has no OpenAI tool-calling support (verified: no tool_call/tool_choice
code anywhere in its serve module), so tools are dispatched via a parsed ReAct-style
text protocol (see agent/graph.py) rather than LangChain's bind_tools/tool_calls --
plain name->callable(args_dict) is all that protocol needs.
"""
from harness.sandbox import Sandbox

TOOL_DESCRIPTIONS = """- read_file: {"path": "<file path>"}
- write_file: {"path": "<file path>", "content": "<new full file content>"}
- run_bash: {"command": "<shell command>"}
- run_tests: {}
- submit: {}"""


def build_tools(sandbox: Sandbox) -> dict:
    return {
        "read_file": lambda args: sandbox.read_file(args["path"]),
        "write_file": lambda args: sandbox.write_file(args["path"], args["content"]),
        "run_bash": lambda args: sandbox.run_bash(args["command"]),
        "run_tests": lambda args: sandbox.run_tests(),
        "submit": lambda args: "submitted",
    }
