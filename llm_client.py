"""
agents/llm_client.py
--------------------
Unified async LLM interface supporting OpenAI and Anthropic.
The rest of the system imports this single module — never a provider SDK directly.

Swap providers at runtime via QUEUE_BACKEND / ANTHROPIC_API_KEY / OPENAI_API_KEY.
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator

log = logging.getLogger(__name__)

_ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
_OPENAI_KEY    = os.getenv("OPENAI_API_KEY", "")
_MODEL_ANTHROPIC = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
_MODEL_OPENAI    = os.getenv("OPENAI_MODEL",    "gpt-4o-mini")
_MAX_TOKENS      = int(os.getenv("LLM_MAX_TOKENS", "1024"))
_TEMPERATURE     = float(os.getenv("LLM_TEMPERATURE", "0.3"))


class LLMClient:
    """
    Thin adapter over OpenAI / Anthropic SDKs.

    Provider selection priority:
      1. ANTHROPIC_API_KEY present  →  Anthropic
      2. OPENAI_API_KEY present     →  OpenAI
      3. Neither                    →  MockLLM (returns deterministic stubs)

    Public API
    ----------
    await complete(system, user, *, temperature, max_tokens) -> str
    stream(system, user, *, temperature, max_tokens) -> AsyncIterator[str]
    await batch_complete(items) -> list[{"task_id", "output"}]
    """

    def __init__(self, client: Any | None = None) -> None:
        """
        Pass an existing SDK client to inject it directly (useful in tests).
        If None, the client is built from environment variables.
        """
        self._client   = client
        self._provider = self._detect_provider()

    def _detect_provider(self) -> str:
        if self._client is not None:
            # Detect from client type
            ctype = type(self._client).__name__
            if "Anthropic" in ctype:
                return "anthropic"
            if "OpenAI" in ctype:
                return "openai"
            return "mock"
        if _ANTHROPIC_KEY:
            return "anthropic"
        if _OPENAI_KEY:
            return "openai"
        log.warning("No LLM API key — using mock responses")
        return "mock"

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self._provider == "anthropic":
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=_ANTHROPIC_KEY)
        elif self._provider == "openai":
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=_OPENAI_KEY)
        else:
            self._client = _MockClient()
        return self._client

    # ── Non-streaming completion ───────────────────────────────────────────────

    async def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = _TEMPERATURE,
        max_tokens: int = _MAX_TOKENS,
    ) -> str:
        client = self._get_client()

        if self._provider == "anthropic":
            resp = await client.messages.create(
                model=_MODEL_ANTHROPIC,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text

        if self._provider == "openai":
            resp = await client.chat.completions.create(
                model=_MODEL_OPENAI,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            return resp.choices[0].message.content or ""

        # mock — call .messages.create and extract .text the same way
        resp = await client.messages.create(
            messages=[{"content": user}], system=system
        )
        return resp.content[0].text

    # ── Streaming completion ───────────────────────────────────────────────────

    async def stream(
        self,
        system: str,
        user: str,
        *,
        temperature: float = _TEMPERATURE,
        max_tokens: int = _MAX_TOKENS,
    ) -> AsyncIterator[str]:
        client = self._get_client()

        if self._provider == "anthropic":
            async with client.messages.stream(
                model=_MODEL_ANTHROPIC,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            ) as s:
                async for text in s.text_stream:
                    yield text
            return

        if self._provider == "openai":
            async with client.chat.completions.stream(
                model=_MODEL_OPENAI,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            ) as s:
                async for chunk in s:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
            return

        # mock: stream word by word
        import asyncio
        text = await self.complete(system, user)
        for word in text.split():
            yield word + " "
            await asyncio.sleep(0)

    # ── Manual batch completion ────────────────────────────────────────────────

    async def batch_complete(
        self,
        items: list[dict],
    ) -> list[dict]:
        """
        items: [{"task_id": str, "system": str, "user": str}, ...]
        returns: [{"task_id": str, "output": str}, ...]

        Sends all requests concurrently (not a single API batch call, which is
        not universally supported). Isolates individual failures so one bad item
        cannot block the whole batch.
        """
        import asyncio

        async def _one(item: dict) -> dict:
            try:
                out = await self.complete(item["system"], item["user"])
                return {"task_id": item["task_id"], "output": out, "error": None}
            except Exception as exc:
                log.error("batch_complete: item %s failed: %s", item["task_id"], exc)
                return {"task_id": item["task_id"], "output": "", "error": str(exc)}

        return await asyncio.gather(*[_one(it) for it in items])


# ── Mock client (no API key required) ─────────────────────────────────────────


class _MockClient:
    """Deterministic stub — useful in tests and local dev without an API key."""

    class _Resp:
        def __init__(self, text: str):
            self.content = [type("B", (), {"text": text})()]

    async def messages_create(self, *, messages: list, system: str = "", **_: Any) -> Any:
        snippet = messages[-1].get("content", "")[:60] if messages else ""
        text = (
            f'{{"chunks": ["Relevant info about {snippet}"], "sources": ["mock_db"], '
            f'"summary": "Mock summary for: {snippet}"}}'
        )
        return self._Resp(text)

    # Make it work whether called as client.messages.create or directly
    class messages:
        @staticmethod
        async def create(*args: Any, **kwargs: Any) -> Any:
            messages = kwargs.get("messages", [])
            snippet = messages[-1].get("content", "")[:60] if messages else ""
            text = (
                f'{{"chunks": ["Relevant info about {snippet}"], '
                f'"sources": ["mock_db"], "summary": "Mock summary for: {snippet}"}}'
            )

            class Resp:
                content = [type("B", (), {"text": text})()]

            return Resp()
