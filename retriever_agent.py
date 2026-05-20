"""
agents/retriever_agent.py
-------------------------
Retrieves relevant context for a query.

In production this would call:
  - A vector DB (Pinecone / Weaviate) for semantic search / RAG
  - A web search API for real-time facts
  - An internal document store

Here we use the LLM to simulate retrieval so the system runs without
external data infrastructure.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from models.schemas import AgentType, TaskMessage, TaskResult, TaskStatus
from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)

_SYSTEM = """You are a Retriever agent. Your only job is to find and surface
relevant factual information for the given query.

Return ONLY a valid JSON object — no markdown fences, no preamble:
{
  "sources": ["<source description>", ...],
  "chunks":  ["<relevant passage>", ...],
  "summary": "<one-paragraph summary of what you found>"
}"""


class RetrieverAgent(BaseAgent):
    """Fetches relevant context; outputs structured JSON for the Analyzer."""

    agent_type = AgentType.RETRIEVER

    async def _run(self, msg: TaskMessage) -> TaskResult:
        query   = msg.payload.get("query", msg.payload.get("original_query", ""))
        context = msg.payload.get("context", "")

        user_prompt = f"Query: {query}"
        if context:
            user_prompt += f"\n\nAdditional context: {context}"

        log.debug("[retriever] query=%s", query[:80])
        raw = await self._llm.complete(_SYSTEM, user_prompt, temperature=0.1)

        # Parse JSON; fall back to wrapping raw text
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"sources": [], "chunks": [raw], "summary": raw}

        return TaskResult(
            task_id        = msg.task_id,
            parent_task_id = msg.parent_task_id,
            root_task_id   = msg.root_task_id,
            agent_type     = AgentType.RETRIEVER,
            status         = TaskStatus.DONE,
            output         = parsed,
            sequence_number= msg.payload.get("sequence", 0),
        )
