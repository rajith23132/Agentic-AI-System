"""
queue/base.py
-------------
Abstract queue adapter interface.  Every backend (Redis, RabbitMQ, Kafka, SQS)
implements this contract so the rest of the system never imports a backend directly.
"""

from __future__ import annotations

import abc
import logging
from typing import AsyncIterator, Optional

from models.schemas import TaskMessage, TaskResult

log = logging.getLogger(__name__)


class QueueAdapter(abc.ABC):
    """Pluggable async message queue interface."""

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    @abc.abstractmethod
    async def connect(self) -> None:
        """Establish connection / declare topics."""

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """Gracefully close connection."""

    # ── Publishing ─────────────────────────────────────────────────────────────

    @abc.abstractmethod
    async def publish_task(self, message: TaskMessage) -> None:
        """Publish a subtask to the appropriate agent topic."""

    @abc.abstractmethod
    async def publish_result(self, result: TaskResult) -> None:
        """Publish an agent result to the results topic."""

    @abc.abstractmethod
    async def publish_dlq(self, message: TaskMessage, error: str) -> None:
        """Route a permanently failed message to the dead-letter queue."""

    # ── Consuming ──────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def consume_tasks(
        self, agent_type: str, batch_size: int = 1
    ) -> AsyncIterator[list[TaskMessage]]:
        """
        Yield batches of TaskMessages for a given agent type.
        Implementations must ACK only after the caller has processed them.
        """

    @abc.abstractmethod
    def consume_results(
        self, root_task_id: str
    ) -> AsyncIterator[TaskResult]:
        """Yield TaskResults belonging to a specific root task."""

    # ── Management ─────────────────────────────────────────────────────────────

    @abc.abstractmethod
    async def queue_depth(self, agent_type: str) -> int:
        """Return approximate number of pending messages for an agent type."""

    @abc.abstractmethod
    async def ack(self, message_id: str, agent_type: str) -> None:
        """Acknowledge that a message has been successfully processed."""

    @abc.abstractmethod
    async def nack(self, message_id: str, agent_type: str) -> None:
        """Negatively acknowledge — return message to queue for retry."""
