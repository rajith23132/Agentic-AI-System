"""
queue/memory_adapter.py
-----------------------
Pure in-process async queue adapter — no external dependencies.
Perfect for unit tests and local development without Redis.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from typing import AsyncIterator

from models.schemas import TaskMessage, TaskResult
from queue.base import QueueAdapter

log = logging.getLogger(__name__)


class MemoryQueueAdapter(QueueAdapter):
    """
    Async in-memory queue backed by asyncio.Queue objects.
    Not suitable for multi-process deployments but ideal for testing.
    """

    def __init__(self) -> None:
        # agent_type -> asyncio.Queue[TaskMessage]
        self._task_queues: dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        # root_task_id -> asyncio.Queue[TaskResult]
        self._result_queues: dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self._dlq: asyncio.Queue = asyncio.Queue()

    async def connect(self) -> None:
        log.info("MemoryQueueAdapter ready (in-process, no external deps)")

    async def disconnect(self) -> None:
        log.info("MemoryQueueAdapter shut down")

    # ── Publishing ─────────────────────────────────────────────────────────────

    async def publish_task(self, message: TaskMessage) -> None:
        await self._task_queues[message.agent_type.value].put(message)
        log.debug("Published task %s → %s queue", message.task_id, message.agent_type)

    async def publish_result(self, result: TaskResult) -> None:
        await self._result_queues[result.root_task_id].put(result)
        log.debug("Published result for %s", result.task_id)

    async def publish_dlq(self, message: TaskMessage, error: str) -> None:
        await self._dlq.put((message, error))
        log.warning("DLQ: task %s — %s", message.task_id, error)

    # ── Consuming ──────────────────────────────────────────────────────────────

    async def consume_tasks(
        self, agent_type: str, batch_size: int = 1
    ) -> AsyncIterator[list[TaskMessage]]:
        q = self._task_queues[agent_type]
        while True:
            batch: list[TaskMessage] = []
            # Block until we get at least one message
            first = await q.get()
            batch.append(first)
            # Drain up to batch_size - 1 more without blocking
            for _ in range(batch_size - 1):
                try:
                    batch.append(q.get_nowait())
                except asyncio.QueueEmpty:
                    break
            yield batch

    async def consume_results(
        self, root_task_id: str
    ) -> AsyncIterator[TaskResult]:
        q = self._result_queues[root_task_id]
        while True:
            result = await q.get()
            yield result

    # ── Management ─────────────────────────────────────────────────────────────

    async def queue_depth(self, agent_type: str) -> int:
        return self._task_queues[agent_type].qsize()

    async def ack(self, message_id: str, agent_type: str) -> None:
        pass   # asyncio.Queue auto-removes on get()

    async def nack(self, message_id: str, agent_type: str) -> None:
        pass   # caller must re-publish if needed
