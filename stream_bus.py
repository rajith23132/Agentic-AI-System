"""
streaming/stream_bus.py
-----------------------
In-process pub/sub bus that bridges agents → SSE clients.

Each root_task_id gets an asyncio.Queue.  SSE endpoints subscribe by
task ID and receive StreamEvent objects until the EOS marker arrives.

For multi-process deployments replace the queues with Redis pub/sub:
    SUBSCRIBE results:{root_task_id}
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import AsyncIterator

from models.schemas import StreamEvent, StreamEventType

log = logging.getLogger(__name__)


class StreamBus:
    """
    Fanout pub/sub bus for streaming events.

    Publishers  (agents)   call: await bus.publish(root_task_id, event)
    Subscribers (SSE view) call: async for event in bus.subscribe(root_task_id)
    """

    def __init__(self, max_queue: int = 1000):
        # root_task_id -> list of subscriber queues
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._max_queue = max_queue
        self._lock = asyncio.Lock()

    async def publish(self, root_task_id: str, event: StreamEvent) -> None:
        """Broadcast an event to all subscribers of root_task_id."""
        async with self._lock:
            queues = self._subscribers.get(root_task_id, [])

        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                log.warning("[stream_bus] Queue full for %s, dropping event", root_task_id)

    async def subscribe(self, root_task_id: str) -> AsyncIterator[StreamEvent]:
        """
        Async generator that yields events for root_task_id.
        Automatically unsubscribes after receiving the EOS event.
        """
        q: asyncio.Queue[StreamEvent] = asyncio.Queue(maxsize=self._max_queue)

        async with self._lock:
            self._subscribers[root_task_id].append(q)

        log.debug("[stream_bus] New subscriber for %s", root_task_id)

        try:
            while True:
                event = await asyncio.wait_for(q.get(), timeout=60.0)
                yield event
                if event.event_type == StreamEventType.EOS:
                    break
        except asyncio.TimeoutError:
            log.warning("[stream_bus] Subscriber timed out for %s", root_task_id)
            yield StreamEvent(
                event_type=StreamEventType.ERROR,
                root_task_id=root_task_id,
                data={"error": "Stream timed out"},
            )
        finally:
            async with self._lock:
                subs = self._subscribers.get(root_task_id, [])
                if q in subs:
                    subs.remove(q)
                if not subs:
                    self._subscribers.pop(root_task_id, None)
            log.debug("[stream_bus] Subscriber removed for %s", root_task_id)

    async def emit_status(self, root_task_id: str, message: str) -> None:
        """Helper to push a STATUS event."""
        event = StreamEvent(
            event_type=StreamEventType.STATUS,
            root_task_id=root_task_id,
            data={"message": message},
        )
        await self.publish(root_task_id, event)

    async def emit_error(self, root_task_id: str, error: str) -> None:
        event = StreamEvent(
            event_type=StreamEventType.ERROR,
            root_task_id=root_task_id,
            data={"error": error},
        )
        await self.publish(root_task_id, event)
