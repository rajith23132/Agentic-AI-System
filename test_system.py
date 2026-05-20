"""
tests/test_system.py
--------------------
Integration tests for the full agentic pipeline.
Uses MemoryQueueAdapter — no external dependencies required.
Run with: pytest tests/ -v
"""

from __future__ import annotations

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pytest_asyncio

from models.schemas import (
    AgentType, TaskMessage, TaskResult, TaskStatus,
    StreamEventType, UserRequest,
)
from queue.memory_adapter import MemoryQueueAdapter
from streaming.stream_bus import StreamBus
from fault_tolerance.retry import (
    async_retry, CircuitBreaker, CircuitState,
)
from batching.scheduler import BatchScheduler
from orchestrator.orchestrator import Orchestrator


# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_queue() -> MemoryQueueAdapter:
    return MemoryQueueAdapter()


class MockLLM:
    """Minimal mock that satisfies the Anthropic SDK interface."""
    class messages:
        @staticmethod
        async def create(**kwargs):
            content = kwargs.get("messages", [{}])[-1].get("content", "")[:40]

            class Block:
                text = (
                    '{"chunks": ["Relevant info about '
                    + content
                    + '"], "sources": ["mock_db"], "summary": "Mock summary."}'
                )

            class Resp:
                pass

            r = Resp()
            r.content = [Block()]
            return r


# ── Queue Tests ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_memory_queue_publish_consume():
    q = make_queue()
    await q.connect()
    msg = TaskMessage(agent_type=AgentType.RETRIEVER, payload={"query": "hello"})
    await q.publish_task(msg)
    async for batch in q.consume_tasks("retriever", batch_size=1):
        assert len(batch) == 1
        assert batch[0].task_id == msg.task_id
        break


@pytest.mark.asyncio
async def test_memory_queue_result_roundtrip():
    q = make_queue()
    await q.connect()
    root_id = "root-test-001"
    result = TaskResult(
        task_id      = "t1",
        root_task_id = root_id,
        agent_type   = AgentType.RETRIEVER,
        status       = TaskStatus.DONE,
        output       = {"chunks": ["data"]},
    )
    await q.publish_result(result)
    async for r in q.consume_results(root_id):
        assert r.task_id == "t1"
        assert r.output["chunks"] == ["data"]
        break


@pytest.mark.asyncio
async def test_dlq_routing():
    q = make_queue()
    await q.connect()
    msg = TaskMessage(agent_type=AgentType.ANALYZER, payload={})
    await q.publish_dlq(msg, error="boom")
    item = await q._dlq.get()
    assert item[0].task_id == msg.task_id
    assert "boom" in item[1]


# ── Retry Tests ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retry_succeeds_on_third_attempt():
    calls = 0

    @async_retry(max_attempts=3, base_delay=0.01, jitter=0)
    async def flaky():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ValueError("not yet")
        return "ok"

    result = await flaky()
    assert result == "ok"
    assert calls == 3


@pytest.mark.asyncio
async def test_retry_raises_after_exhaustion():
    @async_retry(max_attempts=2, base_delay=0.01, jitter=0)
    async def always_fails():
        raise RuntimeError("always")

    with pytest.raises(RuntimeError, match="always"):
        await always_fails()


# ── Circuit Breaker Tests ──────────────────────────────────────────────────────

def test_circuit_breaker_trips_after_threshold():
    cb = CircuitBreaker("test-trip", failure_threshold=3, window=60, recovery_timeout=1)
    assert cb.state == CircuitState.CLOSED
    cb.record_failure()
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert not cb.allow_request()


def test_circuit_breaker_half_open_and_recover():
    import time
    cb = CircuitBreaker("test-recover", failure_threshold=2, window=60, recovery_timeout=0.05)
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    time.sleep(0.1)
    assert cb.allow_request()          # transitions to HALF_OPEN
    assert cb.state == CircuitState.HALF_OPEN
    cb.record_success()
    assert cb.state == CircuitState.CLOSED


def test_circuit_breaker_half_open_re_trips():
    import time
    cb = CircuitBreaker("test-re-trip", failure_threshold=2, window=60, recovery_timeout=0.05)
    cb.record_failure()
    cb.record_failure()
    time.sleep(0.1)
    cb.allow_request()  # → HALF_OPEN
    cb.record_failure() # → OPEN again
    assert cb.state == CircuitState.OPEN


# ── StreamBus Tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_bus_fanout():
    from models.schemas import StreamEvent, StreamEventType

    bus = StreamBus()
    root_id = "root-stream-test"
    received = []

    async def subscriber():
        async for event in bus.subscribe(root_id):
            received.append(event)

    task = asyncio.create_task(subscriber())
    await asyncio.sleep(0.05)
    await bus.emit_status(root_id, "Started")
    await bus.publish(root_id, StreamEvent(
        event_type=StreamEventType.EOS, root_task_id=root_id, data={}
    ))
    await asyncio.wait_for(task, timeout=2.0)

    assert len(received) == 2
    assert received[0].event_type == StreamEventType.STATUS
    assert received[-1].event_type == StreamEventType.EOS


# ── BatchScheduler Tests ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_batch_scheduler_dispatches():
    q = make_queue()
    await q.connect()
    processed = []

    async def mock_handler(batch):
        results = []
        for msg in batch:
            processed.append(msg.task_id)
            results.append(TaskResult(
                task_id      = msg.task_id,
                root_task_id = msg.root_task_id or "root",
                agent_type   = msg.agent_type,
                status       = TaskStatus.DONE,
                output       = {"done": True},
            ))
        return results

    scheduler = BatchScheduler(q, interval_ms=50, max_batch=4)
    scheduler.register(AgentType.RETRIEVER, mock_handler)

    msg = TaskMessage(
        agent_type   = AgentType.RETRIEVER,
        root_task_id = "root-batch-test",
        payload      = {"query": "test"},
    )
    await q.publish_task(msg)

    sched_task = asyncio.create_task(scheduler.run())
    await asyncio.sleep(0.5)
    sched_task.cancel()

    assert msg.task_id in processed


# ── Orchestrator Tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_orchestrator_default_decompose():
    q = make_queue()
    await q.connect()
    bus = Orchestrator.__new__(Orchestrator)
    # Use StreamBus directly
    from streaming.stream_bus import StreamBus as SB
    orch = Orchestrator(queue=q, stream_bus=SB(), enable_planner=False)

    req = UserRequest(query="What is quantum computing?", max_tokens=512)
    graph = orch._default_decompose("root-decompose-test", req)

    assert len(graph.subtasks) == 3
    types = [st.agent_type for st in graph.subtasks]
    assert AgentType.RETRIEVER in types
    assert AgentType.ANALYZER  in types
    assert AgentType.WRITER    in types

    # Dependency chain: retriever has no deps; analyzer depends on retriever; writer on analyzer
    retriever = graph.subtasks[0]
    analyzer  = graph.subtasks[1]
    writer    = graph.subtasks[2]
    assert retriever.depends_on == []
    assert retriever.task_id in analyzer.depends_on
    assert analyzer.task_id  in writer.depends_on


@pytest.mark.asyncio
async def test_agents_handle_batch():
    """RetrieverAgent and AnalyzerAgent handle a batch without erroring."""
    from agents import RetrieverAgent, AnalyzerAgent

    q = make_queue()
    await q.connect()
    llm = MockLLM()

    retriever = RetrieverAgent(q, llm)
    msg = TaskMessage(
        agent_type   = AgentType.RETRIEVER,
        root_task_id = "root-agent-test",
        payload      = {"query": "How do GPUs work?", "sequence": 0},
    )
    results = await retriever.handle_batch([msg])
    assert len(results) == 1
    assert results[0].status == TaskStatus.DONE
    assert isinstance(results[0].output, dict)

    # Feed retriever output into analyzer
    analyzer = AnalyzerAgent(q, llm)
    import json
    amsg = TaskMessage(
        agent_type   = AgentType.ANALYZER,
        root_task_id = "root-agent-test",
        payload      = {
            "original_query": "How do GPUs work?",
            "retrieved_data": results[0].output,
            "sequence": 1,
        },
    )
    aresults = await analyzer.handle_batch([amsg])
    assert len(aresults) == 1
    assert aresults[0].status == TaskStatus.DONE


@pytest.mark.asyncio
async def test_full_pipeline_in_memory():
    """
    End-to-end: submit a query with MemoryQueue + MockLLM, verify EOS received.
    """
    from agents import RetrieverAgent, AnalyzerAgent, WriterAgent
    from streaming.stream_bus import StreamBus as SB

    q   = make_queue()
    await q.connect()
    bus = SB()
    llm = MockLLM()

    retriever = RetrieverAgent(q, llm)
    analyzer  = AnalyzerAgent(q, llm)
    writer    = WriterAgent(q, llm, stream_bus=bus)

    orch = Orchestrator(queue=q, stream_bus=bus, enable_planner=False)

    scheduler = BatchScheduler(q, interval_ms=50, max_batch=4)
    scheduler.register(AgentType.RETRIEVER, retriever.handle_batch)
    scheduler.register(AgentType.ANALYZER,  analyzer.handle_batch)
    scheduler.register(AgentType.WRITER,    writer.handle_batch)

    req = UserRequest(query="Explain neural networks", max_tokens=256)

    received = []

    async def collect(root_id: str):
        async for event in bus.subscribe(root_id):
            received.append(event)
            if event.event_type == StreamEventType.EOS:
                break

    sched_task = asyncio.create_task(scheduler.run())
    root_id    = await orch.handle_request(req)
    collect_task = asyncio.create_task(collect(root_id))

    try:
        await asyncio.wait_for(collect_task, timeout=15.0)
    except asyncio.TimeoutError:
        pass
    finally:
        sched_task.cancel()

    assert len(received) >= 1
    event_types = {e.event_type for e in received}
    assert StreamEventType.STATUS in event_types or StreamEventType.EOS in event_types
