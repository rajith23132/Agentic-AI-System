"""
orchestrator/orchestrator.py
-----------------------------
Central coordinator.  For each user request it:

  1. Optionally calls PlannerAgent (LLM) to decompose the query, or falls
     back to a hardcoded Retriever → Analyzer → Writer chain.
  2. Builds a TaskGraph (in-memory DAG with dependency tracking).
  3. Walks the graph: dispatches ready subtasks to the message queue, then
     collects results and wires outputs to downstream inputs.
  4. Emits STATUS and EOS events to the StreamBus so SSE clients get
     progressive feedback.

Scaling note
------------
`_graphs` is an in-process dict.  In a multi-replica deployment, replace it
with a Redis-backed TaskGraph (see the scaling section of the design doc).
Optimistic locking (WATCH/MULTI/EXEC) should wrap every state transition.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from models.schemas import (
    AgentType, SubTask, TaskGraph, TaskMessage, TaskResult,
    TaskStatus, UserRequest, StreamEvent, StreamEventType,
)

log = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates the multi-agent pipeline for every user request."""

    def __init__(
        self,
        queue: Any,
        stream_bus: Any,
        enable_planner: bool = False,
        llm_client: Any | None = None,
    ) -> None:
        self._queue          = queue
        self._bus            = stream_bus
        self._enable_planner = enable_planner
        self._llm            = llm_client
        self._graphs:  dict[str, TaskGraph]   = {}   # root_task_id → graph
        self._results: dict[str, TaskResult]  = {}   # task_id → result

    # ── Public API ─────────────────────────────────────────────────────────────

    async def handle_request(self, request: UserRequest) -> str:
        """
        Accept a user request, build a task graph, start execution, and
        return the root_task_id immediately (fire-and-forget style).

        Callers subscribe to the StreamBus to receive progressive output.
        """
        root_task_id = str(uuid.uuid4())
        log.info("[orchestrator] new request %s: %s", root_task_id, request.query[:80])

        # Build plan
        if self._enable_planner and self._llm:
            graph = await self._planner_decompose(root_task_id, request)
        else:
            graph = self._default_decompose(root_task_id, request)

        self._graphs[root_task_id] = graph

        # Start execution in background
        asyncio.create_task(
            self._run_graph(root_task_id, graph),
            name=f"graph-{root_task_id[:8]}",
        )

        await self._bus.emit_status(
            root_task_id,
            f"Plan ready — {len(graph.subtasks)} subtasks queued",
        )
        return root_task_id

    # ── Decomposition ──────────────────────────────────────────────────────────

    def _default_decompose(self, root_task_id: str, request: UserRequest) -> TaskGraph:
        """Hard-coded Retriever → Analyzer → Writer chain (no LLM call)."""
        r_id = str(uuid.uuid4())
        a_id = str(uuid.uuid4())
        w_id = str(uuid.uuid4())

        retriever = SubTask(
            task_id    = r_id,
            agent_type = AgentType.RETRIEVER,
            description= "Retrieve relevant information",
            depends_on = [],
            payload    = {"query": request.query, "sequence": 0},
        )
        analyzer = SubTask(
            task_id    = a_id,
            agent_type = AgentType.ANALYZER,
            description= "Analyse retrieved information",
            depends_on = [r_id],
            payload    = {"original_query": request.query, "sequence": 1},
        )
        writer = SubTask(
            task_id    = w_id,
            agent_type = AgentType.WRITER,
            description= "Write the final answer",
            depends_on = [a_id],
            payload    = {"original_query": request.query, "sequence": 2},
        )

        return TaskGraph(
            root_task_id = root_task_id,
            user_query   = request.query,
            subtasks     = [retriever, analyzer, writer],
        )

    async def _planner_decompose(
        self, root_task_id: str, request: UserRequest
    ) -> TaskGraph:
        """Use PlannerAgent (LLM) to decompose. Falls back to default on failure."""
        from agents.llm_client import LLMClient
        from agents.planner_agent import _default_plan, _SYSTEM

        llm = LLMClient(client=self._llm)
        try:
            raw  = await llm.complete(
                _SYSTEM, f"User request: {request.query}", temperature=0.1
            )
            plan = json.loads(raw)
            assert isinstance(plan, list) and plan
        except Exception as exc:
            log.warning("[orchestrator] planner LLM failed (%s) — using default", exc)
            return self._default_decompose(root_task_id, request)

        # Map plan → SubTask list, assign UUIDs, resolve depends_on by index
        id_map: dict[int, str] = {}
        subtasks: list[SubTask] = []

        for idx, step in enumerate(plan):
            tid = str(uuid.uuid4())
            id_map[idx] = tid
            payload = step.get("payload", {})
            payload["original_query"] = request.query
            payload["sequence"] = idx
            subtasks.append(SubTask(
                task_id    = tid,
                agent_type = AgentType(step["agent_type"]),
                description= step.get("description", ""),
                depends_on = [id_map[i] for i in step.get("depends_on", [])
                              if i in id_map],
                payload    = payload,
            ))

        return TaskGraph(
            root_task_id = root_task_id,
            user_query   = request.query,
            subtasks     = subtasks,
        )

    # ── Graph execution ────────────────────────────────────────────────────────

    async def _run_graph(self, root_task_id: str, graph: TaskGraph) -> None:
        """
        Dispatch subtasks whose dependencies are satisfied, collect results,
        feed outputs into downstream payloads, repeat until all nodes are done.
        """
        log.info("[orchestrator] starting graph %s (%d subtasks)",
                 root_task_id, len(graph.subtasks))

        dispatched: set[str] = set()

        while not graph.completed:
            ready = self._get_ready(graph, dispatched)

            for subtask in ready:
                await self._dispatch(root_task_id, subtask, graph)
                dispatched.add(subtask.task_id)
                subtask.status = TaskStatus.IN_PROGRESS
                await self._bus.emit_status(
                    root_task_id,
                    f"[{subtask.agent_type.value}] started: {subtask.description}",
                )

            # Poll for results on the results queue
            await self._collect_results(root_task_id, graph)

            # Check completion
            if all(
                st.status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.DEAD_LETTERED)
                for st in graph.subtasks
            ):
                graph.completed = True
                break

            if not ready and not dispatched - {
                st.task_id for st in graph.subtasks
                if st.status == TaskStatus.IN_PROGRESS
            }:
                # No progress possible — likely all failed
                log.error("[orchestrator] graph %s stalled", root_task_id)
                break

            await asyncio.sleep(0.05)

        # Emit final answer and EOS
        await self._emit_final(root_task_id, graph)

    def _get_ready(
        self, graph: TaskGraph, dispatched: set[str]
    ) -> list[SubTask]:
        """Return subtasks whose dependencies are all DONE and not yet dispatched."""
        done_ids = {
            st.task_id
            for st in graph.subtasks
            if st.status == TaskStatus.DONE
        }
        return [
            st for st in graph.subtasks
            if st.task_id not in dispatched
            and st.status == TaskStatus.PENDING
            and all(dep in done_ids for dep in st.depends_on)
        ]

    async def _dispatch(
        self,
        root_task_id: str,
        subtask: SubTask,
        graph: TaskGraph,
    ) -> None:
        """
        Enrich the subtask payload with outputs from its dependencies,
        then publish a TaskMessage to the appropriate agent stream.
        """
        payload = dict(subtask.payload)

        for dep_id in subtask.depends_on:
            dep_result = self._results.get(dep_id)
            if not dep_result:
                continue
            dep_subtask = next(
                (st for st in graph.subtasks if st.task_id == dep_id), None
            )
            if not dep_subtask:
                continue
            if dep_subtask.agent_type == AgentType.RETRIEVER:
                payload["retrieved_data"] = dep_result.output
                payload["sources"]        = dep_result.output.get("sources", [])
            elif dep_subtask.agent_type == AgentType.ANALYZER:
                payload["analysis"] = dep_result.output

        msg = TaskMessage(
            task_id        = subtask.task_id,
            parent_task_id = subtask.task_id,
            root_task_id   = root_task_id,
            agent_type     = subtask.agent_type,
            payload        = payload,
            priority       = 5,
            max_retries    = 3,
        )
        await self._queue.publish_task(msg)
        log.debug("[orchestrator] dispatched %s → %s",
                  subtask.task_id[:8], subtask.agent_type.value)

    async def _collect_results(
        self, root_task_id: str, graph: TaskGraph
    ) -> None:
        """
        Read a small batch of results from the queue and update the graph.
        We poll with a short window to avoid stalling on blocking reads.
        """
        try:
            collected = 0
            async for result in self._queue.consume_results(root_task_id):
                self._results[result.task_id] = result

                # Update matching subtask
                for subtask in graph.subtasks:
                    if subtask.task_id == result.task_id:
                        subtask.status = result.status
                        subtask.result = result
                        if result.status == TaskStatus.DONE:
                            await self._bus.emit_status(
                                root_task_id,
                                f"[{result.agent_type.value}] completed",
                            )
                        elif result.status == TaskStatus.FAILED:
                            log.warning(
                                "[orchestrator] subtask %s failed: %s",
                                result.task_id[:8], result.error,
                            )
                        break

                collected += 1
                if collected >= 5:
                    break   # yield control; next loop iteration collects more
        except StopAsyncIteration:
            pass
        except Exception as exc:
            log.error("[orchestrator] collect error: %s", exc)

    async def _emit_final(self, root_task_id: str, graph: TaskGraph) -> None:
        """Push the Writer's output as a STATUS event then send EOS."""
        writer_result = next(
            (
                st.result for st in graph.subtasks
                if st.agent_type == AgentType.WRITER and st.result
            ),
            None,
        )

        if writer_result and writer_result.status == TaskStatus.DONE:
            text = writer_result.output.get("text", "")
            # If StreamBus already received tokens from WriterAgent, this is redundant
            # but harmless for non-streaming (poll) clients.
            await self._bus.emit_status(root_task_id, f"Answer ready ({len(text)} chars)")
        else:
            await self._bus.emit_error(root_task_id, "Pipeline completed with errors")

        # EOS — SSE subscribers will close their connection
        from models.schemas import StreamEvent, StreamEventType
        await self._bus.publish(
            root_task_id,
            StreamEvent(
                event_type   = StreamEventType.EOS,
                root_task_id = root_task_id,
                data         = {"completed": graph.completed},
            ),
        )
        log.info("[orchestrator] graph %s finished", root_task_id)
