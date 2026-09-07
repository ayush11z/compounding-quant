"""LangGraph agent loop: identical across precision arms (per CLAUDE.md non-goals) --
only agent/llm.py's target server changes. Explicit state, turn-capped, checkpointed.

Uses a parsed ReAct-style text protocol instead of OpenAI tool_calls, because
trtllm-serve has no tool-calling support (see agent/tools.py docstring)."""
import json
import re
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

ACTION_RE = re.compile(r"ACTION:\s*(\w+)\s*\n\s*ACTION_INPUT:\s*(\{.*\})", re.DOTALL)


_INVALID_ESCAPE_RE = re.compile(r"\\(?![\"\\/bfnrtu])")


def parse_action(text: str):
    m = ACTION_RE.search(text)
    if not m:
        return None
    name = m.group(1).strip()
    raw = m.group(2)
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        # Small models commonly backslash-escape characters JSON never requires
        # escaping (e.g. \' in a sed one-liner) -- at temperature=0 an identical
        # error just reproduces the identical broken retry forever, so repair the
        # common case instead of relying on the model to self-correct.
        repaired = _INVALID_ESCAPE_RE.sub("", raw)  # drop the stray backslash, keep the char
        try:
            args = json.loads(repaired)
        except json.JSONDecodeError as e:
            return {"name": name, "args": None, "parse_error": str(e)}
    return {"name": name, "args": args}


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    turns: int
    submitted: bool
    prompt_tokens: int
    completion_tokens: int
    stuck: bool


# At temperature=0, an identical observation produces an identical next response --
# so an unbroken run of repeats means the agent is stuck and will never recover on
# its own. Ending early saves the rest of the turn budget rather than silently
# burning it on a dead loop (this affects the study's own cost-per-instance numbers
# equally across arms, not resolve rate, so it's harness hygiene, not a tweak).
STUCK_REPEAT_THRESHOLD = 3


def build_graph(llm, tools: dict, max_turns: int):
    def agent_node(state: AgentState):
        response = llm.invoke(state["messages"])
        usage = getattr(response, "usage_metadata", None) or {}
        messages = state["messages"]
        content = response.content if isinstance(response.content, str) else ""
        repeat_run = 1
        for m in reversed(messages):
            if m.__class__.__name__ != "AIMessage":
                continue
            prev = m.content if isinstance(m.content, str) else ""
            if prev == content:
                repeat_run += 1
            else:
                break
        return {
            "messages": [response],
            "turns": state["turns"] + 1,
            "prompt_tokens": state.get("prompt_tokens", 0) + usage.get("input_tokens", 0),
            "completion_tokens": state.get("completion_tokens", 0) + usage.get("output_tokens", 0),
            "stuck": repeat_run >= STUCK_REPEAT_THRESHOLD,
        }

    def tools_node(state: AgentState):
        last = state["messages"][-1]
        action = parse_action(last.content if isinstance(last.content, str) else "")
        if action is None:
            observation = (
                "ERROR: could not parse an ACTION from your last message. "
                "Respond with exactly:\nACTION: <tool_name>\nACTION_INPUT: <json object>"
            )
            return {"messages": [HumanMessage(content=observation)], "submitted": False}

        name = action["name"]
        if action.get("args") is None:
            observation = (
                f"ERROR: ACTION_INPUT for '{name}' was not valid JSON "
                f"({action.get('parse_error', 'unknown error')}) -- it may have been "
                "cut off by the output length limit. If writing a large file, try a "
                "shorter run_bash edit (e.g. sed/python) instead of write_file."
            )
            return {"messages": [HumanMessage(content=observation)], "submitted": False}
        if name == "submit":
            return {"messages": [HumanMessage(content="submitted")], "submitted": True}

        tool = tools.get(name)
        if tool is None:
            observation = f"ERROR: unknown tool '{name}'. Available: {', '.join(tools)}"
        else:
            try:
                observation = str(tool(action["args"]))
            except Exception as e:
                observation = f"ERROR: {e}"
        return {"messages": [HumanMessage(content=f"OBSERVATION: {observation}")], "submitted": False}

    def route_after_agent(state: AgentState):
        if state["turns"] >= max_turns or state.get("stuck"):
            return END
        return "tools"

    def route_after_tools(state: AgentState):
        return END if state.get("submitted") else "agent"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", route_after_tools, {"agent": "agent", END: END})
    return graph.compile(checkpointer=MemorySaver())
