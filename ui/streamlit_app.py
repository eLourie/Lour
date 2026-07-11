"""
ui/streamlit_app.py

Minimal Streamlit demo client for the agent (Phase 9, showcase).

It is a thin front-end over ``POST /v1/chat``: the request streams back
Server-Sent Events (app/gateway/streaming.py), and this app renders the two
levels the taxonomy carries (app/agents/events.py) —

  • node level  — a live timeline of ROUTE_DECIDED / NODE_* / TOOL_* events, so
                  you can watch the supervisor route and the sub-agent work.
  • token level — the answer typed out as TOKEN frames arrive.

It also drives the HITL loop: an APPROVAL_REQUIRED pause surfaces Approve / Deny
buttons that resume the run via ``POST /v1/sessions/{thread_id}/approve``.

Run it (installs Streamlit via the ``ui`` extra):

    uv sync --extra ui
    make ui                 # == uv run --extra ui streamlit run ui/streamlit_app.py

The API base URL and key default to localhost + the dev key; override them in
the sidebar or via ``LOUR_BASE_URL`` / ``APP_API_KEY``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
import streamlit as st

if TYPE_CHECKING:
    from collections.abc import Iterator

DEFAULT_BASE_URL = os.getenv("LOUR_BASE_URL", "http://localhost:8000")
DEFAULT_API_KEY = os.getenv("APP_API_KEY", "changeme-user")

# Icons for the node-level timeline, keyed by SSE event type.
_ICONS = {
    "route_decided": "🧭",
    "node_started": "▶️",
    "node_finished": "✅",
    "tool_called": "🔧",
    "tool_result": "📦",
    "approval_required": "⏸️",
    "error": "❌",
}


# ── SSE parsing ──────────────────────────────────────────────────────────────


@dataclass
class SSEEvent:
    """One parsed Server-Sent Event: its ``event:`` type and JSON ``data:``."""

    event: str
    data: dict[str, Any]


def iter_sse(lines: Iterator[str]) -> Iterator[SSEEvent]:
    """Parse an SSE line stream (``event:`` + ``data:`` frames) into events.

    Pure and side-effect free so it can be reasoned about (and reused) apart
    from Streamlit: give it any iterator of decoded lines and it yields one
    ``SSEEvent`` per blank-line-terminated frame.
    """
    event = "message"
    data_buf: list[str] = []
    for raw in lines:
        line = raw.rstrip("\n")
        if line == "":  # blank line terminates a frame
            if data_buf:
                yield SSEEvent(event=event, data=_decode("\n".join(data_buf)))
            event, data_buf = "message", []
            continue
        if line.startswith(":"):  # comment/heartbeat
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value
        elif field == "data":
            data_buf.append(value)
    if data_buf:  # flush a frame not followed by a trailing blank line
        yield SSEEvent(event=event, data=_decode("\n".join(data_buf)))


def _decode(payload: str) -> dict[str, Any]:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return {"raw": payload}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _timeline_line(ev: SSEEvent) -> str | None:
    """Render a node-level event as one timeline line (or None to skip)."""
    d, icon = ev.data, _ICONS.get(ev.event, "•")
    if ev.event == "route_decided":
        reasoning = f" — {d['reasoning']}" if d.get("reasoning") else ""
        return f"{icon} **route** → `{d.get('agent', '?')}`{reasoning}"
    if ev.event == "node_started":
        return f"{icon} node `{d.get('node', '?')}` started"
    if ev.event == "node_finished":
        return f"{icon} node `{d.get('node', '?')}` finished"
    if ev.event == "tool_called":
        return f"{icon} tool `{d.get('name', '?')}` called `{d.get('arguments', {})}`"
    if ev.event == "tool_result":
        status = "ok" if d.get("ok") else f"error: {d.get('error')}"
        return f"{icon} tool `{d.get('name', '?')}` → {status}"
    if ev.event == "error":
        return f"{icon} **error**: {d.get('message', 'unknown')}"
    return None


# ── HTTP ─────────────────────────────────────────────────────────────────────


def _headers(api_key: str) -> dict[str, str]:
    return {"X-API-Key": api_key, "Accept": "text/event-stream"}


def _render_stream(
    method: str, url: str, payload: dict[str, Any], api_key: str
) -> tuple[str, str | None]:
    """Open an SSE stream and render it live; return (answer, thread_id).

    Node-level events accumulate into an expander timeline; TOKEN frames type
    the answer out; a FINAL frame is the fallback answer when no tokens stream.
    An APPROVAL_REQUIRED frame stashes the pause in session_state and stops.
    """
    timeline: list[str] = []
    answer_parts: list[str] = []
    final_answer = ""
    thread_id: str | None = None

    timeline_box = st.expander("🧠 Agent timeline", expanded=True)
    timeline_ph = timeline_box.empty()
    answer_ph = st.empty()

    with httpx.Client(timeout=None) as client, client.stream(
        method, url, json=payload, headers=_headers(api_key)
    ) as resp:
        resp.raise_for_status()
        thread_id = resp.headers.get("X-Thread-Id")
        for ev in iter_sse(resp.iter_lines()):
            if ev.event == "token":
                answer_parts.append(ev.data.get("text", ""))
                answer_ph.markdown("".join(answer_parts))
            elif ev.event == "final":
                final_answer = ev.data.get("answer", "")
            elif ev.event == "approval_required":
                st.session_state.pending = {"thread_id": thread_id, **ev.data}
                timeline.append(
                    f"⏸️ **approval required** for `{ev.data.get('tool', '?')}`"
                    f" — {ev.data.get('reason', '')}"
                )
                timeline_ph.markdown("\n\n".join(timeline))
                break
            elif ev.event == "done":
                break
            else:
                line = _timeline_line(ev)
                if line:
                    timeline.append(line)
                    timeline_ph.markdown("\n\n".join(timeline))

    answer = "".join(answer_parts) or final_answer
    if answer:
        answer_ph.markdown(answer)
    return answer, thread_id


# ── App ──────────────────────────────────────────────────────────────────────


def _init_state() -> None:
    st.session_state.setdefault("history", [])  # list[{"role", "content"}]
    st.session_state.setdefault("thread_id", None)
    st.session_state.setdefault("pending", None)  # HITL approval, if any


def _sidebar() -> tuple[str, str]:
    st.sidebar.title("⚙️ Connection")
    base_url = st.sidebar.text_input("API base URL", DEFAULT_BASE_URL).rstrip("/")
    api_key = st.sidebar.text_input("API key", DEFAULT_API_KEY, type="password")
    st.sidebar.divider()
    st.sidebar.caption(f"Thread: `{st.session_state.thread_id or '—'}`")
    if st.sidebar.button("🆕 New session", use_container_width=True):
        st.session_state.history = []
        st.session_state.thread_id = None
        st.session_state.pending = None
        st.rerun()
    return base_url, api_key


def _render_history() -> None:
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def _handle_pending(base_url: str, api_key: str) -> None:
    """Render Approve/Deny for a HITL pause and resume the run on click."""
    pending = st.session_state.pending
    st.warning(
        f"⏸️ The agent wants to run **{pending.get('tool', '?')}** — "
        f"{pending.get('reason', 'approval required')}",
        icon="⏸️",
    )
    approve, deny = st.columns(2)
    decision: bool | None = None
    if approve.button("✅ Approve", use_container_width=True):
        decision = True
    if deny.button("🚫 Deny", use_container_width=True):
        decision = False
    if decision is None:
        return

    thread_id = pending["thread_id"]
    st.session_state.pending = None
    with st.chat_message("assistant"):
        answer, _ = _render_stream(
            "POST",
            f"{base_url}/v1/sessions/{thread_id}/approve",
            {"approved": decision},
            api_key,
        )
    st.session_state.history.append({"role": "assistant", "content": answer or "_(no answer)_"})
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="Lour — agent demo", page_icon="🦉", layout="centered")
    st.title("🦉 Lour — multi-agent demo")
    st.caption("Free-form chat over `/v1/chat`, streaming supervisor + sub-agent progress.")

    _init_state()
    base_url, api_key = _sidebar()
    _render_history()

    if st.session_state.pending:
        _handle_pending(base_url, api_key)
        return

    prompt = st.chat_input("Ask anything — e.g. 'research vector databases' or 'sort [3,1,2]'")
    if not prompt:
        return

    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload: dict[str, Any] = {"message": prompt}
    if st.session_state.thread_id:
        payload["thread_id"] = st.session_state.thread_id

    with st.chat_message("assistant"):
        try:
            answer, thread_id = _render_stream(
                "POST", f"{base_url}/v1/chat", payload, api_key
            )
        except httpx.HTTPError as exc:
            st.error(f"Request failed: {exc}")
            return

    if thread_id:
        st.session_state.thread_id = thread_id
    if st.session_state.pending:  # a HITL pause interrupted the turn
        st.rerun()
    st.session_state.history.append({"role": "assistant", "content": answer or "_(no answer)_"})


if __name__ == "__main__":
    main()
