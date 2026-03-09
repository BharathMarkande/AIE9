"""Model utilities for constructing chat and embedding clients.

Centralizes configuration of the default chat model and embeddings so graphs
and RAG can import a single helper without repeating provider-specific wiring.
Supports OpenAI and Fireworks via LLM_PROVIDER (or inferred from API keys).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


def _get_provider() -> str:
    """Return 'openai' or 'fireworks'. Uses LLM_PROVIDER or infers from API keys."""
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if provider in ("openai", "fireworks"):
        return provider
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("FIREWORKS_API_KEY"):
        return "fireworks"
    raise ValueError(
        "Set LLM_PROVIDER to 'openai' or 'fireworks', or set OPENAI_API_KEY or FIREWORKS_API_KEY"
    )


def get_chat_model(model_name: str | None = None, *, temperature: float = 0) -> Any:
    """Return a configured LangChain ChatOpenAI client (OpenAI or Fireworks)."""
    provider = _get_provider()
    if provider == "openai":
        name = model_name or os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o")
        return ChatOpenAI(
            model=name,
            temperature=temperature,
            openai_api_key=os.environ["OPENAI_API_KEY"],
        )
    # Fireworks
    name = model_name or os.environ.get(
        "FIREWORKS_CHAT_MODEL", "accounts/fireworks/models/gpt-oss-20b"
    )
    return ChatOpenAI(
        model=name,
        temperature=temperature,
        openai_api_key=os.environ["FIREWORKS_API_KEY"],
        openai_api_base=FIREWORKS_BASE_URL,
    )


def get_embedding_model() -> Any:
    """Return a configured embedding model (OpenAI or Fireworks)."""
    provider = _get_provider()
    if provider == "openai":
        model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)
    # Fireworks
    model = os.environ.get(
        "FIREWORKS_EMBEDDING_MODEL", "accounts/fireworks/models/qwen3-embedding-4b"
    )
    return OpenAIEmbeddings(
        model=model,
        openai_api_key=os.environ["FIREWORKS_API_KEY"],
        openai_api_base=FIREWORKS_BASE_URL,
        check_embedding_ctx_length=False,
        dimensions=4096,
    )


def fix_tool_calls(response: AIMessage) -> AIMessage:
    """Fix invalid tool calls caused by models appending extra tokens like <|call|>."""
    if not response.invalid_tool_calls:
        return response

    fixed = list(response.tool_calls)
    remaining_invalid = []

    for tc in response.invalid_tool_calls:
        cleaned = re.sub(r"\s*<\|call\|>\s*$", "", tc["args"])
        try:
            parsed = json.loads(cleaned)
            fixed.append(
                {
                    "name": tc["name"],
                    "args": parsed,
                    "id": tc["id"],
                    "type": "tool_call",
                }
            )
        except (json.JSONDecodeError, TypeError):
            remaining_invalid.append(tc)

    response.tool_calls = fixed
    response.invalid_tool_calls = remaining_invalid
    return response
