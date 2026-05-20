"""
agents/planner_agent.py
-----------------------
Optional agent that uses an LLM to decompose a complex query into a structured
plan instead of using the Orchestrator's hard-coded default decomposition.

Activated by setting ENABLE_PLANNER=true (or enable_planner=True in the
Orchestrator constructor).
"""

from __future__ import annotations

import json
import logging

from models.schemas import AgentType, TaskMessage, TaskResult, TaskStatus
from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)

_SYSTEM = """You are a Planner agent. Your only job is to decompose a user
request into an ordered list of subtasks for specialised AI agents.

Available agent types: retriever, analyzer, writer

Return ONLY a JSON array — no markdown, no preamble:
[
  {
    "agent_type":  "retriever|analyzer|writer",
    "description": "<what this step does>",
    "depends_on":  [<0-based indices of subtasks that must finish first>],
    "payload":     {}
  },
  ...
]

Hard rules
- Always start with one or more retriever tasks (gather facts first)
- Follow with at least one analyzer task
- End with exactly one writer task
- Keep the plan to 3–6 steps
- depends_on references zero-based index positions in this array"""


class PlannerAgent(BaseAgent):
    """Decomposes a user query into a structured plan via LLM."""

    agent_type = AgentType.PLANNER

    async def _run(self, msg: TaskMessage) -> TaskResult:
        query   = msg.payload.get("query", msg.payload.get("original_query", ""))
        context = msg.payload.get("context", "")

        user_prompt = f"User request: {query}"
        if context:
            user_prompt += f"\n\nContext: {context}"

        log.debug("[planner] planning for: %s", query[:80])
        raw = await self._llm.complete(_SYSTEM, user_prompt, temperature=0.1)

        try:
            plan = json.loads(raw)
            if not isinstance(plan, list) or not plan:
                raise ValueError("Plan must be a non-empty list")
            # Validate each step has required keys
            for step in plan:
                assert "agent_type" in step, "Missing agent_type"
                assert step["agent_type"] in ("retriever", "analyzer", "writer", "planner")
        except (json.JSONDecodeError, ValueError, AssertionError) as exc:
            log.warning("[planner] parse failed (%s) — using default plan", exc)
            plan = _default_plan(query)

        # Inject original_query into every step's payload
        for step in plan:
            step.setdefault("payload", {})
            step["payload"]["original_query"] = query

        return TaskResult(
            task_id        = msg.task_id,
            parent_task_id = msg.parent_task_id,
            root_task_id   = msg.root_task_id,
            agent_type     = AgentType.PLANNER,
            status         = TaskStatus.DONE,
            output         = {"plan": plan, "step_count": len(plan)},
            sequence_number= 0,
        )


def _default_plan(query: str) -> list[dict]:
    return [
        {"agent_type": "retriever",
         "description": f"Retrieve relevant information for: {query}",
         "depends_on": [], "payload": {"query": query}},
        {"agent_type": "analyzer",
         "description": "Analyse and structure the retrieved information",
         "depends_on": [0], "payload": {}},
        {"agent_type": "writer",
         "description": "Write the final answer",
         "depends_on": [1], "payload": {}},
    ]
