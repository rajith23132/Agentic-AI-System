#!/usr/bin/env python3
"""
scripts/run_worker.py
---------------------
Standalone agent worker process.

Run one worker per agent type for horizontal scaling:

    # Terminal 1 — retriever workers (scale to 3 replicas)
    python scripts/run_worker.py --agent retriever --replicas 3

    # Terminal 2
    python scripts/run_worker.py --agent analyzer

    # Terminal 3
    python scripts/run_worker.py --agent writer

    # Terminal 4 (optional)
    python scripts/run_worker.py --agent planner

In Kubernetes, deploy each agent type as an independent Deployment and
scale replicas based on queue depth (Kafka consumer lag / Redis stream PEL).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

# Make sure project root is on sys.path when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging_config  # noqa: F401 — configures root logger

log = logging.getLogger("worker")


async def run_worker(
    agent_type: str,
    queue_backend: str,
    redis_url: str,
    replicas: int,
) -> None:
    from agents import RetrieverAgent, AnalyzerAgent, WriterAgent, PlannerAgent
    from batching.scheduler import BatchScheduler
    from models.schemas import AgentType
    from streaming.stream_bus import StreamBus

    # Build queue adapter
    if queue_backend == "redis":
        from queue.redis_adapter import RedisQueueAdapter
        queue = RedisQueueAdapter(url=redis_url)
    else:
        from queue.memory_adapter import MemoryQueueAdapter
        queue = MemoryQueueAdapter()

    await queue.connect()
    log.info("Worker connected to %s queue", queue_backend)

    # LLM client (built inside each agent from env vars)
    llm = _build_llm()

    # Stream bus (used only by writer in standalone mode; bus events go nowhere
    # without a connected SSE client, but the writer still generates its answer)
    bus = StreamBus()

    # Instantiate agent(s) for the requested type
    agent_map = {
        "retriever": lambda: RetrieverAgent(queue, llm),
        "analyzer":  lambda: AnalyzerAgent(queue, llm),
        "writer":    lambda: WriterAgent(queue, llm, stream_bus=bus),
        "planner":   lambda: PlannerAgent(queue, llm),
    }

    if agent_type not in agent_map:
        log.error("Unknown agent type %r — choose from %s", agent_type,
                  list(agent_map.keys()))
        return

    enum_type = AgentType(agent_type)
    scheduler = BatchScheduler(
        queue,
        interval_ms=int(os.getenv("BATCH_INTERVAL_MS", "100")),
        max_batch=int(os.getenv("BATCH_MAX_SIZE", "8")),
    )

    for _ in range(replicas):
        agent = agent_map[agent_type]()
        scheduler.register(enum_type, agent.handle_batch)

    log.info("Starting %d %s worker(s)", replicas, agent_type)

    try:
        await scheduler.run()
    except asyncio.CancelledError:
        log.info("Worker cancelled — shutting down")
    finally:
        await queue.disconnect()


def _build_llm():
    """Return raw SDK client; LLMClient wraps it inside each agent."""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    openai_key    = os.getenv("OPENAI_API_KEY", "")
    if anthropic_key:
        import anthropic
        return anthropic.AsyncAnthropic(api_key=anthropic_key)
    if openai_key:
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=openai_key)
    log.warning("No API key — using mock LLM")
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone agent worker for the Agentic AI System"
    )
    parser.add_argument(
        "--agent",
        required=True,
        choices=["retriever", "analyzer", "writer", "planner"],
        help="Which agent type this worker runs",
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=1,
        help="Number of concurrent agent replicas in this process (default: 1)",
    )
    parser.add_argument(
        "--queue",
        default=os.getenv("QUEUE_BACKEND", "memory"),
        choices=["memory", "redis"],
        help="Queue backend (default: env QUEUE_BACKEND or 'memory')",
    )
    parser.add_argument(
        "--redis-url",
        default=os.getenv("REDIS_URL", "redis://localhost:6379"),
        help="Redis URL (default: env REDIS_URL or redis://localhost:6379)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(run_worker(
            agent_type   = args.agent,
            queue_backend= args.queue,
            redis_url    = args.redis_url,
            replicas     = args.replicas,
        ))
    except KeyboardInterrupt:
        log.info("Worker stopped by user")
