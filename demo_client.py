#!/usr/bin/env python3
"""
demo_client.py — Stream a query from the command line.
Usage: python scripts/demo_client.py "Explain how neural networks learn"
"""
import asyncio, sys, json
import httpx

BASE_URL = "http://localhost:8000"

async def main(query: str):
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{BASE_URL}/tasks",
            json={"query": query, "stream": True, "max_tokens": 512})
        resp.raise_for_status()
        data = resp.json()
        root_id = data["root_task_id"]
        print(f"\n[submitted] root_task_id={root_id} | subtasks={data['estimated_subtasks']}\n{'─'*60}")
        event_type = "unknown"
        async with client.stream("GET", f"{BASE_URL}/stream/{root_id}") as stream:
            async for line in stream.aiter_lines():
                if not line:
                    continue
                if line.startswith("event:"):
                    event_type = line[7:].strip()
                elif line.startswith("data:"):
                    payload = json.loads(line[5:].strip())
                    et = payload.get("event_type", event_type)
                    if et == "status":
                        print(f"\n[status] {payload['data']['message']}")
                    elif et == "citation":
                        print(f"\n[sources] {', '.join(payload['data'].get('sources', []))}")
                    elif et == "token":
                        print(payload["data"].get("token", ""), end="", flush=True)
                    elif et == "eos":
                        print(f"\n{'─'*60}\n[done]"); return
                    elif et == "error":
                        print(f"\n[error] {payload['data']['error']}"); return

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Explain how transformers work"
    asyncio.run(main(q))
