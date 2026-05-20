# Agentic AI System — Multi-Step Task Orchestration

A production-ready multi-agent system that decomposes complex user queries into
subtasks, routes them through specialised agents via an async message queue, and
streams progressive output back to the client in real time.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│         Orchestrator            │  Decomposes → task graph (DAG)
│   Planner Agent (optional)      │  Tracks state (Redis optimistic lock)
└────────────┬────────────────────┘
             │ publish subtasks
             ▼
┌─────────────────────────────────┐
│        Message Queue            │  Redis Streams / RabbitMQ / Kafka / SQS
│   ┌─────────┐ ┌─────────────┐  │  + Dead-Letter Queue
│   │ Retry   │ │   Batching  │  │  + Manual batch scheduler (100ms poll)
│   └─────────┘ └─────────────┘  │
└────────────┬────────────────────┘
             │
     ┌───────┼──────────┬──────────┐
     ▼       ▼          ▼          ▼
┌─────────┐ ┌─────────┐ ┌───────┐ ┌────────┐
│Retriever│ │Analyzer │ │Writer │ │Planner │
│  Agent  │ │  Agent  │ │ Agent │ │ Agent  │
└────┬────┘ └────┬────┘ └───┬───┘ └────────┘
     │           │           │  streams tokens
     └───────────┴───────────┘
                 │ results
                 ▼
         ┌────────────────┐
         │  Stream Bus    │  asyncio pub/sub fanout
         └───────┬────────┘
                 │ SSE events
                 ▼
              Client
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install fastapi uvicorn pydantic anthropic openai redis
```

### 2. Configure

```bash
cp .env.example .env
# Add your API key in .env:
# ANTHROPIC_API_KEY=sk-ant-...   (or OPENAI_API_KEY=sk-...)
```

### 3. Run (no external services needed)

```bash
uvicorn main:app --reload
```

The system uses the **in-memory queue** by default — no Redis required.

### 4. Submit a task

```bash
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain how transformers work in machine learning", "stream": true}'
```

Response:
```json
{"root_task_id": "abc-123", "estimated_subtasks": 3}
```

### 5. Stream the results

```bash
curl -N http://localhost:8000/stream/abc-123
```

Or in JavaScript:
```javascript
const es = new EventSource('http://localhost:8000/stream/abc-123');
es.addEventListener('token',  e => process.stdout.write(JSON.parse(e.data).data.token));
es.addEventListener('status', e => console.log('\n[status]', JSON.parse(e.data).data.message));
es.addEventListener('eos',    () => { console.log('\n[done]'); es.close(); });
```

---

## Docker (with Redis)

```bash
# Add your API key to .env first
docker compose up --build
```

This starts:
- `redis` — Redis 7 with AOF persistence
- `api` — the FastAPI server on port 8000

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/tasks` | Submit a query. Returns `root_task_id`. |
| `GET` | `/stream/{id}` | SSE stream of events. |
| `GET` | `/tasks/{id}` | Poll task graph status (non-streaming). |
| `GET` | `/metrics` | Queue depths per agent type. |
| `GET` | `/health` | Liveness probe. |

### SSE Event Types

| Event | Payload | Description |
|-------|---------|-------------|
| `status` | `{message: str}` | Orchestrator status update |
| `citation` | `{sources: [str]}` | Sources found by Retriever |
| `token` | `{token: str}` | One output token from Writer |
| `eos` | `{message: str}` | Stream complete |
| `error` | `{error: str}` | Something went wrong |

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `QUEUE_BACKEND` | `memory` | `memory` or `redis` |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `ANTHROPIC_API_KEY` | — | Anthropic Claude API key |
| `OPENAI_API_KEY` | — | OpenAI API key (alternative) |
| `ENABLE_PLANNER` | `false` | Use LLM-based task decomposition |

---

## Project Structure

```
agentic_ai/
├── main.py                     # Entrypoint (uvicorn main:app)
├── api/
│   └── app.py                  # FastAPI routes, lifespan, SSE endpoint
├── orchestrator/
│   └── orchestrator.py         # Task graph, dispatch, state tracking
├── agents/
│   ├── base_agent.py           # Abstract base: retry, circuit breaker, idempotency
│   ├── retriever_agent.py      # Fetch context (vector DB + web + LLM)
│   ├── analyzer_agent.py       # Reason over retrieved context
│   ├── writer_agent.py         # Synthesise and stream final response
│   └── planner_agent.py        # Dynamic task decomposition (optional)
├── queue/
│   ├── base.py                 # QueueAdapter abstract interface
│   ├── memory_adapter.py       # In-memory (dev/test, no deps)
│   └── redis_adapter.py        # Redis Streams (production)
├── batching/
│   └── scheduler.py            # Manual batch scheduler (100ms poll)
├── streaming/
│   └── stream_bus.py           # Async pub/sub fanout for SSE events
├── fault_tolerance/
│   └── retry.py                # Exponential retry + circuit breaker
├── models/
│   └── schemas.py              # All Pydantic models
├── tests/
│   └── test_system.py          # Pytest integration tests
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

---

## Swapping Queue Backends

The `QueueAdapter` interface is fully pluggable. To use Kafka:

```python
# Create queue/kafka_adapter.py extending QueueAdapter
# Then in api/app.py:
from queue.kafka_adapter import KafkaQueueAdapter
queue = KafkaQueueAdapter(bootstrap_servers="localhost:9092")
```

For RabbitMQ: extend `QueueAdapter` using `aio-pika`.
For AWS SQS: extend using `aiobotocore`.

---

## Extending with a New Agent

1. Create `agents/my_agent.py` extending `BaseAgent`
2. Set `agent_type = AgentType.MY_AGENT`
3. Implement `async def execute(message) -> TaskResult`
4. Register with the `BatchScheduler` in `api/app.py`
5. Add routing in `Orchestrator._default_decompose`

```python
class MyAgent(BaseAgent):
    agent_type = AgentType.MY_AGENT   # add to AgentType enum first

    async def execute(self, message: TaskMessage) -> TaskResult:
        result = await self.call_llm(system="...", user=message.payload["query"])
        return TaskResult(
            task_id=message.task_id,
            root_task_id=message.root_task_id,
            agent_type=self.agent_type,
            status=TaskStatus.DONE,
            output={"result": result},
        )
```

---

## Scaling

- Each agent type is stateless and can run as multiple replicas
- Switch `QUEUE_BACKEND=redis` to enable distributed message passing
- Orchestrator task state uses Redis WATCH/MULTI/EXEC (optimistic locking)
- Auto-scale on `GET /metrics` queue depth (integrate with KEDA or custom HPA)

---

## Design Trade-offs

| Decision | Made | Reason |
|----------|------|--------|
| Framework-free orchestration | ✅ Custom | Per spec; full control over retry/batch |
| In-memory queue for dev | ✅ MemoryAdapter | No external deps; swap to Redis for prod |
| Linear pipeline default | ✅ | Covers 90% of use cases; Planner for complex |
| Manual batching | ✅ BatchScheduler | Per spec; implemented independently |
| SSE over WebSocket | ✅ | Unidirectional; simpler for this use case |
