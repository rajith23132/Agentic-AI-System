"""
agents/base_agent.py
--------------------
Abstract base class shared by all specialised agents.

Provides:
  - handle_batch(batch)    called by BatchScheduler with a list of TaskMessages
  - _execute(msg)          template method — subclasses override _run(msg)
  - Retry via @async_retry on _call_llm
  - Circuit-breaker wrapping of every LLM call
  - Idempotency: tracks processed task_ids in an in-process set
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from models.schemas import AgentType, TaskMessage, TaskResult, TaskStatus
from agents.llm_client import LLMClient
from fault_tolerance.retry import async_retry, get_breaker, CircuitBreakerOpen

log = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    All agents inherit from this class and implement `_run`.

    handle_batch(batch)
      Called by BatchScheduler. Processes each item in the batch and returns
      a list of TaskResults. Failures are isolated per item.
    """

    agent_type: AgentType  # override in subclass

    def __init__(self, queue: Any, llm: Any) -> None:
        """
        queue: QueueAdapter (for publishing results / DLQ moves)
        llm:   raw Anthropic / OpenAI client (or mock) — wrapped in LLMClient
        """
        self._queue   = queue
        self._llm     = LLMClient(client=llm)
        self._breaker = get_breaker(
            f"llm:{self.agent_type.value}",
            failure_threshold=5,
            recovery_timeout=30.0,
        )
        self._processed: set[str] = set()  # in-process idempotency

    # ── Public: called by BatchScheduler ──────────────────────────────────────

    async def handle_batch(self, batch: list[TaskMessage]) -> list[TaskResult]:
        """Process a batch of messages. Each item is isolated — one failure
        does not prevent the rest from completing."""
        results: list[TaskResult] = []
        for msg in batch:
            result = await self._execute(msg)
            results.append(result)
        return results

    # ── Internal execution pipeline ────────────────────────────────────────────

    async def _execute(self, msg: TaskMessage) -> TaskResult:
        task_id = msg.task_id

        # Idempotency check (in-process)
        if task_id in self._processed:
            log.debug("[%s] duplicate skip: %s", self.agent_type.value, task_id)
            return TaskResult(
                task_id=task_id,
                parent_task_id=msg.parent_task_id,
                root_task_id=msg.root_task_id,
                agent_type=self.agent_type,
                status=TaskStatus.DONE,
                output={"cached": True},
            )

        log.info("[%s] processing task %s", self.agent_type.value, task_id)

        try:
            if not self._breaker.allow_request():
                raise CircuitBreakerOpen(
                    f"Circuit {self.agent_type.value} is OPEN"
                )
            result = await self._run_with_retry(msg)
            self._breaker.record_success()
            self._processed.add(task_id)
            log.info("[%s] done task %s", self.agent_type.value, task_id)
            return result

        except CircuitBreakerOpen as exc:
            log.error("[%s] circuit open for %s", self.agent_type.value, task_id)
            return self._error_result(msg, str(exc), TaskStatus.FAILED)

        except Exception as exc:
            self._breaker.record_failure()
            log.error("[%s] failed task %s: %s", self.agent_type.value, task_id, exc)
            # Caller (BatchScheduler) handles retry/DLQ; return FAILED result
            return self._error_result(msg, str(exc), TaskStatus.FAILED)

    @async_retry(max_attempts=3, base_delay=1.0, jitter=0.5)
    async def _run_with_retry(self, msg: TaskMessage) -> TaskResult:
        return await self._run(msg)

    # ── Template method ────────────────────────────────────────────────────────

    @abstractmethod
    async def _run(self, msg: TaskMessage) -> TaskResult:
        """Subclasses implement this to do their actual work."""

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _error_result(
        self,
        msg: TaskMessage,
        error: str,
        status: TaskStatus = TaskStatus.FAILED,
    ) -> TaskResult:
        return TaskResult(
            task_id=msg.task_id,
            parent_task_id=msg.parent_task_id,
            root_task_id=msg.root_task_id,
            agent_type=self.agent_type,
            status=status,
            error=error,
        )
