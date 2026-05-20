"""
queue/redis_adapter.py
----------------------
Redis Streams implementation of QueueAdapter.

Topics
  TASK:{agent_type}   — subtask delivery per agent type
  RESULTS:{root_id}   — results for a specific root task
  DLQ                 — dead-letter queue

Each stream entry is a JSON-encoded TaskMessage / TaskResult.
Consumer groups allow multiple agent replicas to compete for messages
without duplication.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import redis.asyncio as aioredis

from models.schemas import AgentType, TaskMessage, TaskResult
from queue.base import QueueAdapter

log = logging.getLogger(__name__)

TASK_STREAM = "TASK:{agent_type}"
RESULT_STREAM = "RESULTS:{root_id}"
DLQ_STREAM = "DLQ"
CONSUMER_GROUP = "agents"
CONSUMER_NAME = "worker-{agent_type}"


class RedisQueueAdapter(QueueAdapter):
    """
    Redis Streams-backed message queue.

    Args:
        url:  Redis connection URL, e.g. redis://localhost:6379
        max_len:  MAXLEN per stream (approximate trimming)
    """

    def __init__(self, url: str = "redis://localhost:6379", max_len: int = 10_000):
        self._url = url
        self._max_len = max_len
        self._redis: aioredis.Redis | None = None
        # track message IDs so we can ACK later: {internal_key -> stream_msg_id}
        self._pending: dict[str, str] = {}

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        self._redis = await aioredis.from_url(
            self._url, encoding="utf-8", decode_responses=True
        )
        # Ensure consumer groups exist for every agent type
        for agent_type in AgentType:
            stream = TASK_STREAM.format(agent_type=agent_type.value)
            try:
                await self._redis.xgroup_create(
                    stream, CONSUMER_GROUP, id="0", mkstream=True
                )
            except aioredis.ResponseError as exc:
                if "BUSYGROUP" not in str(exc):
                    raise
        log.info("RedisQueueAdapter connected to %s", self._url)

    async def disconnect(self) -> None:
        if self._redis:
            await self._redis.aclose()
            log.info("RedisQueueAdapter disconnected")

    # ── Publishing ─────────────────────────────────────────────────────────────

    async def publish_task(self, message: TaskMessage) -> None:
        stream = TASK_STREAM.format(agent_type=message.agent_type.value)
        await self._redis.xadd(
            stream,
            {"data": message.model_dump_json()},
            maxlen=self._max_len,
            approximate=True,
        )
        log.debug("Published task %s to %s", message.task_id, stream)

    async def publish_result(self, result: TaskResult) -> None:
        stream = RESULT_STREAM.format(root_id=result.root_task_id)
        await self._redis.xadd(
            stream,
            {"data": result.model_dump_json()},
            maxlen=self._max_len,
            approximate=True,
        )
        # Auto-expire result streams after 1 hour
        await self._redis.expire(stream, 3600)
        log.debug("Published result for task %s", result.task_id)

    async def publish_dlq(self, message: TaskMessage, error: str) -> None:
        await self._redis.xadd(
            DLQ_STREAM,
            {
                "data": message.model_dump_json(),
                "error": error,
            },
            maxlen=self._max_len,
            approximate=True,
        )
        log.warning("DLQ: task %s — %s", message.task_id, error)

    # ── Consuming ──────────────────────────────────────────────────────────────

    async def consume_tasks(
        self, agent_type: str, batch_size: int = 1
    ) -> AsyncIterator[list[TaskMessage]]:
        """
        Blocking read from the agent's consumer group.
        Yields lists of TaskMessages (up to batch_size).
        Caller must call ack() for each message after processing.
        """
        stream = TASK_STREAM.format(agent_type=agent_type)
        consumer = CONSUMER_NAME.format(agent_type=agent_type)

        while True:
            entries = await self._redis.xreadgroup(
                groupname=CONSUMER_GROUP,
                consumername=consumer,
                streams={stream: ">"},
                count=batch_size,
                block=2000,   # 2-second blocking read
            )
            if not entries:
                continue

            batch: list[TaskMessage] = []
            for _stream, messages in entries:
                for msg_id, fields in messages:
                    try:
                        task = TaskMessage.model_validate_json(fields["data"])
                        # store stream msg_id keyed by task_id for ACK
                        self._pending[task.task_id] = (stream, msg_id)
                        batch.append(task)
                    except Exception as exc:
                        log.error("Failed to deserialise task: %s", exc)
                        await self._redis.xack(stream, CONSUMER_GROUP, msg_id)

            if batch:
                yield batch

    async def consume_results(
        self, root_task_id: str
    ) -> AsyncIterator[TaskResult]:
        """Poll the results stream for a given root task."""
        stream = RESULT_STREAM.format(root_id=root_task_id)
        last_id = "0"

        while True:
            entries = await self._redis.xread(
                streams={stream: last_id}, count=50, block=1000
            )
            if not entries:
                continue
            for _stream, messages in entries:
                for msg_id, fields in messages:
                    last_id = msg_id
                    try:
                        yield TaskResult.model_validate_json(fields["data"])
                    except Exception as exc:
                        log.error("Failed to deserialise result: %s", exc)

    # ── Management ─────────────────────────────────────────────────────────────

    async def queue_depth(self, agent_type: str) -> int:
        stream = TASK_STREAM.format(agent_type=agent_type)
        info = await self._redis.xinfo_groups(stream)
        for group in info:
            if group["name"] == CONSUMER_GROUP:
                return group.get("pending", 0)
        return 0

    async def ack(self, message_id: str, agent_type: str) -> None:
        key = (TASK_STREAM.format(agent_type=agent_type), message_id)
        if message_id in self._pending:
            stream, stream_msg_id = self._pending.pop(message_id)
            await self._redis.xack(stream, CONSUMER_GROUP, stream_msg_id)

    async def nack(self, message_id: str, agent_type: str) -> None:
        # For Redis Streams, NACK means we do NOT xack —
        # the message stays in PEL and will be reclaimed by another consumer
        # or by the retry loop via XAUTOCLAIM.
        log.debug("NACK task %s (left in PEL)", message_id)
        self._pending.pop(message_id, None)
