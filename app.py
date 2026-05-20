"""
api/app.py
----------
FastAPI application.

Endpoints:
  POST /tasks          — submit a user query, returns root_task_id
  GET  /stream/{id}    — SSE stream of StreamEvents for a task
  GET  /tasks/{id}     — poll current task graph status
  GET  /health         — liveness probe
  GET  /metrics        — queue depths per agent type
  POST /dlq/replay     — re-submit a dead-lettered task
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from agents import AnalyzerAgent, RetrieverAgent, WriterAgent
from batching.scheduler import BatchScheduler
from models.schemas import AgentType, UserRequest, UserResponse
from orchestrator.orchestrator import Orchestrator
from queue.memory_adapter import MemoryQueueAdapter
from queue.redis_adapter import RedisQueueAdapter
from streaming.stream_bus import StreamBus

log = logging.getLogger(__name__)

# ── Global singletons (initialised in lifespan) ───────────────────────────────

queue = None
stream_bus: StreamBus = None
orchestrator: Orchestrator = None
scheduler: BatchScheduler = None

QUEUE_BACKEND = os.getenv("QUEUE_BACKEND", "memory")   # "memory" | "redis"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
ENABLE_PLANNER = os.getenv("ENABLE_PLANNER", "false").lower() == "true"


def _build_llm_client():
    """
    Return an LLM client.
    Priority: ANTHROPIC_API_KEY > OPENAI_API_KEY > mock stub.
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if anthropic_key:
        import anthropic
        return anthropic.AsyncAnthropic(api_key=anthropic_key)
    if openai_key:
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=openai_key)

    # Mock stub for development without API keys
    class MockLLM:
        class messages:
            @staticmethod
            async def create(**kwargs):
                class Resp:
                    content = [type("B", (), {"text": f"[MOCK] Response to: {kwargs.get('messages', [{}])[-1].get('content', '')[:60]}..."})()]
                return Resp()

    log.warning("No LLM API key found — using mock LLM responses")
    return MockLLM()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    global queue, stream_bus, orchestrator, scheduler

    log.info("Starting Agentic AI System (queue=%s)", QUEUE_BACKEND)

    # Queue
    if QUEUE_BACKEND == "redis":
        queue = RedisQueueAdapter(url=REDIS_URL)
    else:
        queue = MemoryQueueAdapter()
    await queue.connect()

    # Stream bus
    stream_bus = StreamBus()

    # LLM client
    llm = _build_llm_client()

    # Agents
    retriever = RetrieverAgent(queue, llm)
    analyzer = AnalyzerAgent(queue, llm)
    writer = WriterAgent(queue, llm, stream_bus=stream_bus)

    # Orchestrator
    orchestrator = Orchestrator(
        queue=queue,
        stream_bus=stream_bus,
        enable_planner=ENABLE_PLANNER,
        llm_client=llm,
    )

    # Batch scheduler
    scheduler = BatchScheduler(queue, interval_ms=100, max_batch=8)
    scheduler.register(AgentType.RETRIEVER, retriever.handle_batch)
    scheduler.register(AgentType.ANALYZER, analyzer.handle_batch)
    scheduler.register(AgentType.WRITER, writer.handle_batch)

    # Start background tasks
    bg = asyncio.create_task(scheduler.run(), name="batch-scheduler")

    log.info("Agentic AI System ready")
    yield

    # Shutdown
    bg.cancel()
    await queue.disconnect()
    log.info("Agentic AI System shut down")


app = FastAPI(
    title="Agentic AI System",
    description="Multi-agent async task orchestration with streaming",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.post("/tasks", response_model=UserResponse, status_code=202)
async def submit_task(request: UserRequest):
    """
    Submit a complex user query.
    Returns immediately with a root_task_id.
    Connect to GET /stream/{root_task_id} to receive streamed results.
    """
    root_task_id = await orchestrator.handle_request(request)
    # Count planned subtasks from the orchestrator's graph
    graph = orchestrator._graphs.get(root_task_id)
    n = len(graph.subtasks) if graph else 0
    return UserResponse(root_task_id=root_task_id, estimated_subtasks=n)


@app.get("/stream/{root_task_id}")
async def stream_task(root_task_id: str, request: Request):
    """
    Server-Sent Events (SSE) endpoint.
    Streams StreamEvent objects as they are produced by agents.

    Client usage:
        const es = new EventSource('/stream/<root_task_id>');
        es.addEventListener('token', e => console.log(JSON.parse(e.data)));
        es.addEventListener('eos', () => es.close());
    """
    async def event_generator() -> AsyncIterator[str]:
        async for event in stream_bus.subscribe(root_task_id):
            if await request.is_disconnected():
                break
            yield event.to_sse()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/tasks/{root_task_id}")
async def get_task_status(root_task_id: str):
    """Poll the current state of a task graph (non-streaming alternative)."""
    graph = orchestrator._graphs.get(root_task_id)
    if not graph:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "root_task_id": root_task_id,
        "query": graph.user_query,
        "completed": graph.completed,
        "subtasks": [
            {
                "task_id": st.task_id,
                "agent_type": st.agent_type.value,
                "status": st.status.value,
                "description": st.description,
                "output": st.result.output if st.result else None,
            }
            for st in graph.subtasks
        ],
    }


@app.get("/metrics")
async def get_metrics():
    """Return queue depths for monitoring / auto-scaling triggers."""
    depths = {}
    for agent_type in AgentType:
        if agent_type == AgentType.ORCHESTRATOR:
            continue
        depths[agent_type.value] = await queue.queue_depth(agent_type.value)
    return {"queue_depths": depths}


@app.get("/health")
async def health():
    return {"status": "ok", "queue_backend": QUEUE_BACKEND}
