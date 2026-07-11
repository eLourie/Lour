#!/usr/bin/env python
"""
scripts/demo.py

Scripted end-to-end demo — the entry point behind ``make demo``.

It drives a running instance through the whole stack in one pass, so a reviewer
can see the system work without a UI:

  1. health / readiness probes,
  2. the public skill catalogue (``GET /v1/skills``),
  3. every skill invoked with sample inputs (``/v1/skills/{name}/invoke``),
  4. auto-routing free text to a skill (``/v1/skills/auto``),
  5. a free-form ``/v1/chat`` turn, streamed as SSE, printing the supervisor +
     sub-agent node timeline live.

The gating steps (health, catalogue, the tool-free ``summarize_document`` skill,
and the chat stream) must pass or the process exits non-zero. Skills that lean
on external resources (web search keys, a seeded RAG corpus) are best-effort:
they are reported but do not fail the demo.

Usage:
    make demo
    uv run python scripts/demo.py --base-url http://localhost:8000
    uv run python scripts/demo.py --only chat        # a single step
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_API_KEY = os.getenv("APP_API_KEY", "changeme-user")

_SAMPLE_DOC = (
    "Retrieval-augmented generation grounds a language model in an external "
    "corpus: a retriever fetches relevant passages, which are prepended to the "
    "prompt so the model answers from evidence rather than memory. This cuts "
    "hallucination and lets the knowledge base be updated without retraining."
)
_SAMPLE_CODE = "def add(a, b):\n    return a - b  # bug: should be a + b\n"

# ── pretty printing ──────────────────────────────────────────────────────────


def _hr(title: str) -> None:
    print(f"\n{'=' * 72}\n▶ {title}\n{'=' * 72}")


def _preview(text: str, limit: int = 400) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + " …"


# ── steps ────────────────────────────────────────────────────────────────────


async def step_health(client: httpx.AsyncClient) -> bool:
    _hr("Health & readiness")
    for path in ("/healthz", "/readyz"):
        resp = await client.get(path)
        ok = resp.status_code == 200
        print(f"  GET {path:<10} → {resp.status_code} {'✓' if ok else '✗'}")
        if not ok:
            print(f"    {_preview(resp.text)}")
            return False
    return True


async def step_catalogue(client: httpx.AsyncClient) -> bool:
    _hr("Skill catalogue (GET /v1/skills)")
    resp = await client.get("/v1/skills")
    resp.raise_for_status()
    data = resp.json()
    for skill in data["skills"]:
        desc = _preview(skill["description"], 60)
        print(f"  • {skill['name']:<20} agent={skill['agent']:<11} {desc}")
    ok = data["count"] >= 4
    print(f"  → {data['count']} skills {'✓' if ok else '✗ (expected ≥4)'}")
    return ok


async def _invoke(client: httpx.AsyncClient, name: str, inputs: dict[str, Any]) -> str | None:
    """Invoke one skill; return the result field text, or None on failure."""
    try:
        resp = await client.post(f"/v1/skills/{name}/invoke", json=inputs)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"  ✗ {name}: {exc}")
        return None
    body = resp.json()
    output = body.get("output") or {}
    text = next((str(v) for v in output.values() if v), json.dumps(body))
    print(f"  ✓ {name}: {_preview(text)}")
    return text


async def step_skills(client: httpx.AsyncClient) -> bool:
    _hr("Invoke skills (POST /v1/skills/{name}/invoke)")
    # Gating: the tool-free direct skill must work end to end.
    summary = await _invoke(client, "summarize_document", {"text": _SAMPLE_DOC, "length": "short"})
    # Best-effort: these depend on a seeded corpus / web-search keys.
    await _invoke(client, "answer_from_kb", {"question": "What is retrieval-augmented generation?"})
    await _invoke(client, "review_code", {"code": _SAMPLE_CODE, "language": "python"})
    await _invoke(client, "research_topic", {"topic": "vector databases", "depth": "quick"})
    return summary is not None


async def step_auto(client: httpx.AsyncClient) -> bool:
    _hr("Auto-route free text (POST /v1/skills/auto)")
    # invoke=false → classify only, keeping the demo fast and deterministic.
    resp = await client.post(
        "/v1/skills/auto",
        json={"text": "Please summarize this paragraph for me.", "invoke": False},
    )
    resp.raise_for_status()
    data = resp.json()
    print(f"  → routed to `{data['skill']}` — {_preview(data.get('reasoning', ''), 80)}")
    return bool(data.get("skill"))


async def step_chat(client: httpx.AsyncClient) -> bool:
    _hr("Free-form chat, streamed (POST /v1/chat)")
    message = "Sort the list [5, 3, 8, 1] ascending and show the result."
    print(f"  user: {message}\n  ── stream ──")
    saw_final = False
    async with client.stream(
        "POST", "/v1/chat", json={"message": message}, headers={"Accept": "text/event-stream"}
    ) as resp:
        resp.raise_for_status()
        print(f"  thread: {resp.headers.get('X-Thread-Id', '—')}")
        async for event, data in _async_sse(resp):
            if event == "route_decided":
                print(f"    🧭 route → {data.get('agent')}")
            elif event == "node_started":
                print(f"    ▶️  {data.get('node')}")
            elif event == "tool_called":
                print(f"    🔧 {data.get('name')}({data.get('arguments')})")
            elif event == "tool_result":
                outcome = "ok" if data.get("ok") else data.get("error")
                print(f"    📦 {data.get('name')} → {outcome}")
            elif event == "final":
                print(f"    ✅ final: {_preview(data.get('answer', ''))}")
                saw_final = True
            elif event == "error":
                print(f"    ❌ error: {data.get('message')}")
    return saw_final


async def _async_sse(resp: httpx.Response) -> Any:
    """Async variant of the SSE frame parser over ``resp.aiter_lines()``."""
    event, buf = "message", []
    async for raw in resp.aiter_lines():
        line = raw.rstrip("\n")
        if line == "":
            if buf:
                try:
                    data = json.loads("\n".join(buf))
                except json.JSONDecodeError:
                    data = {}
                yield event, (data if isinstance(data, dict) else {"value": data})
            event, buf = "message", []
        elif line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            buf.append(line[len("data:") :].lstrip(" "))


# gating steps must pass; the rest are informative.
_STEPS = {
    "health": (step_health, True),
    "catalogue": (step_catalogue, True),
    "skills": (step_skills, True),
    "auto": (step_auto, False),
    "chat": (step_chat, True),
}


async def run(base_url: str, api_key: str, timeout: float, only: str | None) -> int:
    headers = {"X-API-Key": api_key}
    names = [only] if only else list(_STEPS)
    outcomes: dict[str, str] = {}
    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=timeout) as client:
        for name in names:
            runner, gating = _STEPS[name]
            try:
                ok = await runner(client)
                outcomes[name] = "pass" if ok else ("fail" if gating else "warn")
            except Exception as exc:  # a step that cannot run at all
                outcomes[name] = "fail" if gating else "warn"
                print(f"  ✗ step {name} errored: {exc}")

    _hr("Demo summary")
    for name, status in outcomes.items():
        mark = {"pass": "✓", "warn": "~", "fail": "✗"}[status]
        print(f"  {mark} {name:<12} {status}")
    return 0 if all(v != "fail" for v in outcomes.values()) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Scripted end-to-end demo.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout (s).")
    parser.add_argument("--only", choices=list(_STEPS), default=None, help="Run one step.")
    args = parser.parse_args()

    print(f"Lour demo → {args.base_url} (timeout {args.timeout:.0f}s)")
    try:
        return asyncio.run(run(args.base_url, args.api_key, args.timeout, args.only))
    except httpx.ConnectError:
        print(f"\n✗ Could not reach {args.base_url}. Is the app running? (make dev)")
        return 2


if __name__ == "__main__":
    sys.exit(main())
