"""
batching/scheduler.py
---------------------
Manual Batch Scheduler — groups same-type subtasks into a single
LLM inference call on a fixed poll interval.

Design
  - A background asyncio task polls pending queues every `interval_ms`.
  - Messages of the same agent_type are grouped into a batch of up to `max_batch`.
  - The batch is dispatched to the LLM as a single API call using a
    multi-message prompt array.
  - Individual results are extracted by task_id and published back to
    the results stream.
  - If any item in the batch raises, it is retried individually so a
    single bad message doesn't block the whole batch.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine, Dict, List

from models.schemas import AgentType, TaskMessage, TaskResult, TaskStatus
from queue.base import QueueAdapter

log = logging.getLogger(__name__)

# Type alias for the handler each agent registers
BatchHandler = Callable[
    [List[TaskMessage]],
    Coroutine[Any, Any, List[TaskResult]],
]


class BatchScheduler:
    """
    Polls each agent's queue on a fixed interval and dispatches batches.

    Usage:
        scheduler = BatchScheduler(queue, interval_ms=100, max_batch=16)
        scheduler.register(AgentType.RETRIEVER, retriever_agent.handle_batch)
        scheduler.register(AgentType.ANALYZER, analyzer_agent.handle_batch)
        await scheduler.run()   # runs forever until cancelled
    """

    def __init__(
        self,
        queue: QueueAdapter,
        interval_ms: int = 100,
        max_batch: int = 16,
    ):
        self._queue = queue
        self._interval = interval_ms / 1000.0
        self._max_batch = max_batch
        self._handlers: Dict[AgentType, BatchHandler] = {}
        self._running = False

    def register(self, agent_type: AgentType, handler: BatchHandler) -> None:
        """Register a batch handler for an agent type."""
        self._handlers[agent_type] = handler
        log.info("BatchScheduler: registered handler for %s", agent_type.value)

    async def run(self) -> None:
        """Start the polling loop. Runs until cancelled."""
        self._running = True
        log.info(
            "BatchScheduler started (interval=%dms, max_batch=%d)",
            int(self._interval * 1000), self._max_batch,
        )
        tasks = [
            asyncio.create_task(self._poll_loop(agent_type), name=f"batch-{agent_type.value}")
            for agent_type in self._handlers
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            self._running = False
            log.info("BatchScheduler stopped")

    async def _poll_loop(self, agent_type: AgentType) -> None:
        """Per-agent polling loop."""
        handler = self._handlers[agent_type]
        async for batch in self._queue.consume_tasks(
            agent_type=agent_type.value,
            batch_size=self._max_batch,
        ):
            if not batch:
                await asyncio.sleep(self._interval)
                continue

            log.debug(
                "BatchScheduler dispatching %d %s tasks", len(batch), agent_type.value
            )
            asyncio.create_task(
                self._dispatch(batch, handler, agent_type),
                name=f"dispatch-{agent_type.value}",
            )

    async def _dispatch(
        self,
        batch: List[TaskMessage],
        handler: BatchHandler,
        agent_type: AgentType,
    ) -> None:
        """
        Run the batch handler.  On failure, fall back to single-item processing
        so one bad message doesn't block the rest.
        """
        try:
            results = await handler(batch)
            for result in results:
                await self._queue.publish_result(result)
                await self._queue.ack(result.task_id, agent_type.value)
        except Exception as exc:
            log.error(
                "Batch of %d %s tasks failed (%s) — retrying individually",
                len(batch), agent_type.value, exc,
            )
            await self._dispatch_individually(batch, handler, agent_type)

    async def _dispatch_individually(
        self,
        batch: List[TaskMessage],
        handler: BatchHandler,
        agent_type: AgentType,
    ) -> None:
        """Fallback: process each message in the batch one-by-one."""
        for message in batch:
            try:
                results = await handler([message])
                for result in results:
                    await self._queue.publish_result(result)
                await self._queue.ack(message.task_id, agent_type.value)
            except Exception as exc:
                if message.retry_count < message.max_retries:
                    retried = message.increment_retry()
                    await self._queue.publish_task(retried)
                    log.warning(
                        "Re-queued task %s (retry %d/%d)",
                        message.task_id, retried.retry_count, retried.max_retries,
                    )
                else:
                    await self._queue.publish_dlq(message, str(exc))
                    log.error("DLQ: task %s exhausted retries", message.task_id)
                await self._queue.nack(message.task_id, agent_type.value)
