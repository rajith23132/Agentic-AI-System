"""
agents/analyzer_agent.py
------------------------
Reasons over retrieved context and produces structured analytical insights
for the Writer agent.
"""

from __future__ import annotations

import json
import logging

from models.schemas import AgentType, TaskMessage, TaskResult, TaskStatus
from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)

_SYSTEM = """You are an Analyzer agent. You receive retrieved information and
must reason over it to produce structured analytical insights.

Return ONLY valid JSON — no markdown, no preamble:
{
  "key_insights":   ["<insight>", ...],
  "confidence":     <0.0 to 1.0>,
  "reasoning":      "<step-by-step reasoning>",
  "recommendation": "<what the Writer should focus on>"
}"""


class AnalyzerAgent(BaseAgent):
    """Analyzes retrieved context; outputs JSON insights for the Writer."""

    agent_type = AgentType.ANALYZER

    async def _run(self, msg: TaskMessage) -> TaskResult:
        retrieved      = msg.payload.get("retrieved_data", {})
        original_query = msg.payload.get("original_query", "")

        # Normalise retrieved_data: accept dict or raw string
        if isinstance(retrieved, dict):
            retrieved_text = json.dumps(retrieved, indent=2)
        else:
            retrieved_text = str(retrieved)

        user_prompt = (
            f"Original query: {original_query}\n\n"
            f"Retrieved data:\n{retrieved_text}"
        )

        log.debug("[analyzer] original_query=%s", original_query[:80])
        raw = await self._llm.complete(_SYSTEM, user_prompt, temperature=0.2)

        try:
            parsed = json.loads(raw)
            confidence = float(parsed.get("confidence", 0.8))
        except json.JSONDecodeError:
            parsed     = {"key_insights": [raw], "confidence": 0.5,
                          "reasoning": raw, "recommendation": raw}
            confidence = 0.5

        return TaskResult(
            task_id        = msg.task_id,
            parent_task_id = msg.parent_task_id,
            root_task_id   = msg.root_task_id,
            agent_type     = AgentType.ANALYZER,
            status         = TaskStatus.DONE,
            output         = parsed,
            sequence_number= msg.payload.get("sequence", 1),
        )
