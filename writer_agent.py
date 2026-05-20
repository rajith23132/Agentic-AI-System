"""
agents/writer_agent.py
----------------------
Synthesises retrieved facts + analytical insights into a final, flowing prose
answer.

Streaming behaviour
-------------------
In the full pipeline the Writer streams tokens progressively to the StreamBus
so the SSE endpoint can relay them to the client in real time.

In batch mode (called from BatchScheduler.handle_batch) the full text is
buffered and published as a single TaskResult — the StreamBus still receives
a series of TOKEN events before the final EOS.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from models.schemas import (
    AgentType, TaskMessage, TaskResult, TaskStatus,
    StreamEvent, StreamEventType,
)
from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)

_SYSTEM = """You are a Writer agent. You receive analytical insights extracted
from relevant information and must compose a clear, informative, well-structured
response for the end user.

Guidelines
- Write in flowing prose — not bullet lists
- Be concise and accurate; cite specific insights where relevant
- Tailor depth and length to the complexity of the original question
- Do not invent facts not present in the provided insights"""


class WriterAgent(BaseAgent):
    """
    Generates the final user-facing answer.

    If a StreamBus is injected, each token is published as a TOKEN event
    before the TaskResult is returned.
    """

    agent_type = AgentType.WRITER

    def __init__(self, queue: Any, llm: Any, stream_bus: Any | None = None) -> None:
        super().__init__(queue, llm)
        self._bus = stream_bus

    async def _run(self, msg: TaskMessage) -> TaskResult:
        analysis       = msg.payload.get("analysis", {})
        original_query = msg.payload.get("original_query", "")
        style          = msg.payload.get("style", "informative")
        root_task_id   = msg.root_task_id or msg.task_id

        # Normalise analysis payload
        if isinstance(analysis, dict):
            insights  = analysis.get("key_insights", [])
            reasoning = analysis.get("reasoning", "")
            rec       = analysis.get("recommendation", "")
            analysis_text = (
                f"Key insights: {insights}\n"
                f"Reasoning: {reasoning}\n"
                f"Focus: {rec}"
            )
        else:
            analysis_text = str(analysis)

        user_prompt = (
            f"User query: {original_query}\n\n"
            f"Analytical insights:\n{analysis_text}\n\n"
            f"Writing style: {style}"
        )

        log.debug("[writer] composing answer for: %s", original_query[:80])

        full_text   = ""
        token_count = 0
        seq         = 0

        # Stream tokens and publish to bus
        async for token in self._llm.stream(_SYSTEM, user_prompt):
            full_text   += token
            token_count += 1
            seq         += 1

            if self._bus:
                await self._bus.publish(
                    root_task_id,
                    StreamEvent(
                        event_type     = StreamEventType.TOKEN,
                        root_task_id   = root_task_id,
                        sequence_number= seq,
                        data           = {"token": token},
                    ),
                )

        # Publish citation / source event if we have them
        if self._bus:
            sources = msg.payload.get("sources", [])
            if sources:
                await self._bus.publish(
                    root_task_id,
                    StreamEvent(
                        event_type   = StreamEventType.CITATION,
                        root_task_id = root_task_id,
                        data         = {"sources": sources},
                    ),
                )

        return TaskResult(
            task_id        = msg.task_id,
            parent_task_id = msg.parent_task_id,
            root_task_id   = root_task_id,
            agent_type     = AgentType.WRITER,
            status         = TaskStatus.DONE,
            output         = {"text": full_text, "word_count": len(full_text.split())},
            tokens_used    = token_count,
            sequence_number= msg.payload.get("sequence", 2),
        )
