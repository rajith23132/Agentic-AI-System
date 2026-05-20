"""
models/schemas.py
-----------------
All Pydantic models used across the system.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────


class AgentType(str, Enum):
    RETRIEVER = "retriever"
    ANALYZER = "analyzer"
    WRITER = "writer"
    PLANNER = "planner"
    ORCHESTRATOR = "orchestrator"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    DEAD_LETTERED = "dead_lettered"


class StreamEventType(str, Enum):
    TOKEN = "token"
    STATUS = "status"
    ERROR = "error"
    EOS = "eos"          # end of stream
    CITATION = "citation"
    PARTIAL = "partial"


# ─── Message Envelope ─────────────────────────────────────────────────────────


class TaskMessage(BaseModel):
    """The canonical envelope published to every queue topic."""

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_task_id: Optional[str] = None
    root_task_id: Optional[str] = None          # top-level user request id
    agent_type: AgentType = AgentType.RETRIEVER
    priority: int = Field(5, ge=1, le=10)       # 10 = highest
    payload: Dict[str, Any] = Field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    timeout_ms: int = 30_000
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def increment_retry(self) -> "TaskMessage":
        return self.model_copy(update={"retry_count": self.retry_count + 1})


class TaskResult(BaseModel):
    """Result produced by any agent and published to the results topic."""

    task_id: str
    parent_task_id: Optional[str] = None
    root_task_id: Optional[str] = None
    agent_type: AgentType
    status: TaskStatus = TaskStatus.DONE
    output: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    sequence_number: int = 0          # for ordering in the aggregator
    completed_at: datetime = Field(default_factory=datetime.utcnow)
    tokens_used: int = 0


# ─── Task Graph Node ──────────────────────────────────────────────────────────


class SubTask(BaseModel):
    """One node in the orchestrator's task DAG."""

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_type: AgentType
    description: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    depends_on: List[str] = Field(default_factory=list)   # list of task_ids
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    result: Optional[TaskResult] = None


class TaskGraph(BaseModel):
    """The full execution plan for one user request."""

    root_task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_query: str
    subtasks: List[SubTask] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed: bool = False


# ─── Streaming ────────────────────────────────────────────────────────────────


class StreamEvent(BaseModel):
    """One SSE event pushed to the client."""

    event_type: StreamEventType
    root_task_id: str
    sequence_number: int = 0
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def to_sse(self) -> str:
        """Format as raw SSE text."""
        payload = self.model_dump_json()
        return f"event: {self.event_type.value}\ndata: {payload}\n\n"


# ─── API ──────────────────────────────────────────────────────────────────────


class UserRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    stream: bool = True
    max_tokens: int = Field(1024, ge=64, le=8192)
    priority: int = Field(5, ge=1, le=10)


class UserResponse(BaseModel):
    root_task_id: str
    message: str = "Task accepted. Connect to /stream/{root_task_id} for results."
    estimated_subtasks: int = 0
