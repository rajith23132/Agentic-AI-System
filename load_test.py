#!/usr/bin/env python3
"""
scripts/load_test.py
--------------------
Sends N concurrent queries to the Agentic AI System and reports
latency percentiles, throughput, and error rates.

Usage:
    # Quick smoke test (5 concurrent, 1 round)
    python scripts/load_test.py --concurrency 5 --rounds 1

    # Sustained load (20 concurrent, 3 rounds)
    python scripts/load_test.py --concurrency 20 --rounds 3 --base-url http://localhost:8000

    # Stream mode (SSE) instead of polling
    python scripts/load_test.py --concurrency 10 --mode stream
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from typing import Literal

import httpx

BASE_URL = "http://localhost:8000"

SAMPLE_QUERIES = [
    "Explain how transformer neural networks work",
    "What are the main differences between SQL and NoSQL databases?",
    "How does the TCP/IP handshake work?",
    "Explain the CAP theorem in distributed systems",
    "What is gradient descent and how is it used in machine learning?",
    "How do containerisation and virtual machines differ?",
    "Explain the concept of eventual consistency in distributed databases",
    "What is the difference between supervised and unsupervised learning?",
    "How does public-key cryptography work?",
    "Explain microservices architecture and its trade-offs",
]


async def submit_and_wait_poll(
    client: httpx.AsyncClient,
    query: str,
) -> tuple[float, bool, str]:
    """Submit via POST /tasks, then poll GET /tasks/{id}. Returns (latency_s, ok, error)."""
    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{BASE_URL}/tasks",
            json={"query": query, "stream": False, "max_tokens": 512},
            timeout=10.0,
        )
        resp.raise_for_status()
        root_id = resp.json()["root_task_id"]

        # Poll until completed or timeout
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            status_resp = await client.get(
                f"{BASE_URL}/tasks/{root_id}", timeout=5.0
            )
            data = status_resp.json()
            if data.get("completed"):
                return time.monotonic() - t0, True, ""
            await asyncio.sleep(0.5)

        return time.monotonic() - t0, False, "timeout"

    except Exception as exc:
        return time.monotonic() - t0, False, str(exc)


async def submit_and_stream(
    client: httpx.AsyncClient,
    query: str,
) -> tuple[float, bool, str]:
    """Submit via POST /tasks, then consume SSE stream. Returns (latency_s, ok, error)."""
    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{BASE_URL}/tasks",
            json={"query": query, "stream": True, "max_tokens": 512},
            timeout=10.0,
        )
        resp.raise_for_status()
        root_id = resp.json()["root_task_id"]

        async with client.stream(
            "GET", f"{BASE_URL}/stream/{root_id}", timeout=60.0
        ) as stream:
            async for line in stream.aiter_lines():
                if not line:
                    continue
                if line.startswith("data:"):
                    payload = json.loads(line[5:].strip())
                    et = payload.get("event_type", "")
                    if et == "eos":
                        return time.monotonic() - t0, True, ""
                    if et == "error":
                        return time.monotonic() - t0, False, payload.get("data", {}).get("error", "unknown")

        return time.monotonic() - t0, False, "stream ended without EOS"

    except Exception as exc:
        return time.monotonic() - t0, False, str(exc)


async def run_round(
    concurrency: int,
    mode: Literal["poll", "stream"],
    client: httpx.AsyncClient,
) -> list[tuple[float, bool, str]]:
    queries = [SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)] for i in range(concurrency)]
    fn = submit_and_stream if mode == "stream" else submit_and_wait_poll
    tasks = [fn(client, q) for q in queries]
    return await asyncio.gather(*tasks)


def print_report(all_results: list[tuple[float, bool, str]], concurrency: int, rounds: int) -> None:
    successes = [r for r in all_results if r[1]]
    failures  = [r for r in all_results if not r[1]]
    latencies = [r[0] for r in successes]

    print("\n" + "═" * 56)
    print(f"  LOAD TEST REPORT")
    print("═" * 56)
    print(f"  Concurrency : {concurrency}")
    print(f"  Rounds      : {rounds}")
    print(f"  Total req   : {len(all_results)}")
    print(f"  Successes   : {len(successes)}")
    print(f"  Failures    : {len(failures)}")
    print(f"  Error rate  : {100 * len(failures) / max(len(all_results), 1):.1f}%")

    if latencies:
        print(f"\n  Latency (successful requests)")
        print(f"  ── min    : {min(latencies):.2f}s")
        print(f"  ── median : {statistics.median(latencies):.2f}s")
        print(f"  ── p95    : {sorted(latencies)[int(len(latencies) * 0.95)]:.2f}s")
        print(f"  ── max    : {max(latencies):.2f}s")
        print(f"  ── mean   : {statistics.mean(latencies):.2f}s")

    if failures:
        print(f"\n  Errors (first 5)")
        for _, _, err in failures[:5]:
            print(f"  ✗ {err[:80]}")

    print("═" * 56 + "\n")


async def main(args: argparse.Namespace) -> None:
    print(f"\nLoad test: {args.concurrency} concurrent × {args.rounds} round(s) [{args.mode}]")
    print(f"Target: {args.base_url}\n")

    all_results: list[tuple[float, bool, str]] = []

    async with httpx.AsyncClient(base_url=args.base_url) as client:
        # Health check
        try:
            h = await client.get("/health", timeout=5.0)
            print(f"Health: {h.json()}")
        except Exception as exc:
            print(f"⚠ Health check failed: {exc}")

        for r in range(1, args.rounds + 1):
            print(f"Round {r}/{args.rounds} — sending {args.concurrency} requests...")
            t_round = time.monotonic()
            results = await run_round(args.concurrency, args.mode, client)
            elapsed = time.monotonic() - t_round
            ok = sum(1 for _, s, _ in results if s)
            print(f"  Done in {elapsed:.1f}s — {ok}/{args.concurrency} succeeded")
            all_results.extend(results)
            if r < args.rounds:
                await asyncio.sleep(1.0)

    print_report(all_results, args.concurrency, args.rounds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agentic AI System load tester")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Concurrent requests per round (default: 5)")
    parser.add_argument("--rounds", type=int, default=1,
                        help="Number of rounds to run (default: 1)")
    parser.add_argument("--mode", choices=["poll", "stream"], default="poll",
                        help="poll = GET /tasks/{id}; stream = SSE (default: poll)")
    parser.add_argument("--base-url", default=BASE_URL,
                        help=f"API base URL (default: {BASE_URL})")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
