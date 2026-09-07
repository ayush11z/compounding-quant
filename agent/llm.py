"""LLM client pointed at the model server. The precision arm is purely a config
change here (which engine trtllm-serve/Triton has loaded) -- this file and the
agent graph never change between arms, per CLAUDE.md non-goals."""
import os

from langchain_openai import ChatOpenAI

MODEL_SERVER_BASE_URL = os.environ.get("MODEL_SERVER_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen2.5-coder-7b-bf16")


def build_llm():
    return ChatOpenAI(
        base_url=MODEL_SERVER_BASE_URL,
        api_key="not-needed",
        model=MODEL_NAME,
        temperature=0,
        # trtllm-serve's OpenAI-compatible endpoint rejects `max_completion_tokens`,
        # which langchain_openai always renames `max_tokens=` to internally. `extra_body`
        # bypasses that renaming and merges directly into the request payload.
        # 4096: write_file needs room for a full file's content as escaped JSON
        # (~2x source size); 1024 silently truncated mid-JSON, corrupting the call.
        extra_body={"max_tokens": 4096},
    )
