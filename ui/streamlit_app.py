"""
Streamlit UI for Research+FactCheck Chat Agent.

This is the frontend for the Vanna-style backend that provides:
- Chat interface with message history
- Real-time step-by-step progress display
- Collapsible accordions for each workflow step
- Fast mode toggle for reduced verification passes

Usage:
    uv run streamlit run ui/streamlit_app.py --server.port 8501
"""

import json
import time
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st

# Must be the first Streamlit command in the script.
st.set_page_config(page_title="Research+FactCheck Chat (Ollama)", layout="wide")


def _get_backend_base_url() -> str:
    """
    Get the backend base URL from secrets or default.

    Returns:
        str: Backend URL (default: http://localhost:8000).
    """
    from pathlib import Path

    local_secrets = Path(".streamlit") / "secrets.toml"
    user_secrets = Path.home() / ".streamlit" / "secrets.toml"
    if local_secrets.exists() or user_secrets.exists():
        return st.secrets.get("BACKEND_BASE_URL", "http://localhost:8000")
    return "http://localhost:8000"


BACKEND_BASE_URL = _get_backend_base_url()


def _sse_events_post(url: str, payload: Dict[str, Any]):
    """
    Stream SSE events from a POST endpoint.

    This function handles the single-endpoint SSE streaming pattern
    used by the Vanna-style backend.

    Args:
        url: The endpoint URL (e.g., /api/v1/chat/sse).
        payload: JSON payload to send in the POST body.

    Yields:
        tuple[str, str]: (event_type, data_json) pairs.
    """
    with httpx.stream(
        "POST",
        url,
        json=payload,
        timeout=None,
        headers={"Accept": "text/event-stream"},
    ) as r:
        r.raise_for_status()
        event = None
        data_lines: List[str] = []
        for line in r.iter_lines():
            if line is None:
                continue
            if line == "":
                if event and data_lines:
                    data = "\n".join(data_lines)
                    yield event, data
                event = None
                data_lines = []
                continue
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].lstrip())


# =============================================================================
# UI Layout
# =============================================================================

st.title("Research+FactCheck Chat (Ollama qwen3:8b + LangGraph)")

if "messages" not in st.session_state:
    st.session_state.messages = []

left, right = st.columns([2, 1])

STEP_LABELS = {
    "request": "Request received",
    "workflow": "Workflow",
    "router": "Route (easy vs deep)",
    "research": "Research (multi-tool search)",
    "synthesis": "Synthesis (evidence plan)",
    "producer": "Draft answer",
    "factcheck": "Fact-check & revise",
    "final": "Final answer",
    "agent": "Agent",
    "stream": "Stream",
}

with right:
    st.subheader("Live steps")
    status_box = st.empty()
    fast_mode = st.toggle("Fast mode (fewer verification passes)", value=False)
    step_detail_box = st.container()
    research_box = st.expander("Research (queries + tools + sources)", expanded=True)
    synthesis_box = st.expander("Synthesis (plan/outline)", expanded=False)
    verify_box = st.expander("Verification (what changed + why)", expanded=True)

with left:
    # Display message history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Chat input
    prompt = st.chat_input("Ask anything…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build request payload
        payload: Dict[str, Any] = {"message": prompt}
        if fast_mode:
            payload.update({"best_of_n": 1, "verify_max_rounds": 1})

        assistant_placeholder = st.chat_message("assistant").empty()
        final_answer: Optional[str] = None
        research_log: List[str] = []
        verify_log: List[str] = []
        synthesis_text: str = ""
        current_step: str = ""
        verify_round: Optional[int] = None
        step_io: Dict[str, Dict[str, str]] = {}
        step_boxes: Dict[str, Dict[str, Any]] = {}

        status_box.info("Connecting to backend stream…")

        try:
            # Single streaming POST endpoint (Vanna-style)
            for evt_name, data in _sse_events_post(
                f"{BACKEND_BASE_URL}/api/v1/chat/sse", payload
            ):
                if evt_name in ("ping",):
                    continue
                if evt_name in ("done",):
                    break

                # Parse the event data
                try:
                    event_data = json.loads(data)
                except json.JSONDecodeError:
                    continue

                step = event_data.get("step", "")
                typ = event_data.get("type", evt_name)
                dct: Dict[str, Any] = event_data.get("data", {}) or {}

                if step:
                    current_step = step
                    step_label = STEP_LABELS.get(step, step)
                    if step not in step_boxes and step not in ("workflow", "agent", "stream"):
                        exp = step_detail_box.expander(f"{step_label}", expanded=False)
                        step_boxes[step] = {
                            "expander": exp,
                            "input": exp.empty(),
                            "output": exp.empty(),
                            "status": exp.empty(),
                        }

                # Handle error events
                if typ == "error":
                    status_box.error(dct.get("error", "Unknown backend error"))

                # Handle step lifecycle events
                if typ == "step_started":
                    step_label = STEP_LABELS.get(step, step)
                    if step not in ("workflow", "agent"):
                        status_box.info(f"Running: **{step_label}**")
                    step_io.setdefault(step, {})
                    step_io[step]["input"] = dct.get("input", "") or step_io[step].get(
                        "input", ""
                    )
                    if step in step_boxes:
                        step_boxes[step]["status"].markdown("Status: **Running**")

                if typ == "step_finished":
                    step_label = STEP_LABELS.get(step, step)
                    if step not in ("workflow", "agent"):
                        status_box.success(f"Finished: **{step_label}**")
                    step_io.setdefault(step, {})
                    step_io[step]["output"] = dct.get("output", "") or step_io[step].get(
                        "output", ""
                    )
                    if step in step_boxes:
                        step_boxes[step]["status"].markdown("Status: **Done**")

                # Handle tool events
                if typ in ("tool_call", "tool_result"):
                    tool = dct.get("tool", "")
                    q = dct.get("query", "")
                    if typ == "tool_call":
                        research_log.append(f"- **{tool}** searching: `{q}`")
                        step_io.setdefault("research", {})
                        step_io["research"]["input"] = f"Query: {q}"
                    else:
                        note = dct.get("note", "")
                        results = dct.get("results", 0)
                        suffix = f" ({note})" if note else ""
                        research_log.append(
                            f"- **{tool}** found {results} results for `{q}`{suffix}"
                        )
                        step_io.setdefault("research", {})
                        step_io["research"]["output"] = f"{results} results from {tool}"
                    with research_box:
                        st.markdown("\n".join(research_log) or "_No research yet_")

                # Handle research query variations
                if typ == "message" and step == "research":
                    qvs = dct.get("query_variations")
                    if isinstance(qvs, list):
                        with research_box:
                            st.markdown("**Query variations**")
                            st.markdown("\n".join([f"- `{q}`" for q in qvs]))
                        step_io.setdefault("research", {})
                        step_io["research"]["input"] = (
                            "Variations: " + ", ".join(qvs[:4]) + ("…" if len(qvs) > 4 else "")
                        )

                # Handle synthesis output
                if typ == "message" and step == "synthesis":
                    if "synthesis" in dct:
                        synthesis_text = dct["synthesis"] or ""
                        with synthesis_box:
                            st.markdown(synthesis_text or "_No synthesis yet_")
                        step_io.setdefault("synthesis", {})
                        step_io["synthesis"]["output"] = "Synthesis plan generated"

                # Handle verification events
                if typ in ("critic_feedback", "revision_selected"):
                    verify_round = dct.get("round", verify_round)
                    if typ == "critic_feedback":
                        critic = dct.get("critic", "critic")
                        summary = dct.get("summary", "")
                        verify_log.append(
                            f"- Round {verify_round}: **{critic}** → {summary}"
                        )
                        step_io.setdefault("factcheck", {})
                        step_io["factcheck"]["input"] = f"Round {verify_round}: {critic}"
                        step_io["factcheck"]["output"] = summary
                    else:
                        summary = dct.get("summary", "")
                        verify_log.append(
                            f"- Round {verify_round}: **revision selected** → {summary}"
                        )
                        step_io.setdefault("factcheck", {})
                        step_io["factcheck"]["output"] = summary
                    with verify_box:
                        st.markdown("\n".join(verify_log) or "_No verification yet_")

                # Handle final answer
                if typ == "final":
                    final_answer = dct.get("answer", "")
                    assistant_placeholder.markdown(final_answer)
                    step_io.setdefault("final", {})
                    step_io["final"]["output"] = "Final answer ready"

                # Small yield to keep UI responsive
                time.sleep(0.01)

                # Update step accordions with latest IO
                for key, io in step_io.items():
                    if key in ("workflow", "agent", "stream"):
                        continue
                    if key not in step_boxes:
                        label = STEP_LABELS.get(key, key)
                        exp = step_detail_box.expander(f"{label}", expanded=False)
                        step_boxes[key] = {
                            "expander": exp,
                            "input": exp.empty(),
                            "output": exp.empty(),
                            "status": exp.empty(),
                        }
                    step_boxes[key]["input"].markdown(
                        f"Input: {io.get('input', '_pending_')}"
                    )
                    step_boxes[key]["output"].markdown(
                        f"Output: {io.get('output', '_pending_')}"
                    )

        except httpx.HTTPStatusError as e:
            status_box.error(f"Backend error: {e.response.status_code}")
            final_answer = f"_Backend error: HTTP {e.response.status_code}_"
            assistant_placeholder.markdown(final_answer)
        except httpx.ConnectError:
            status_box.error("Cannot connect to backend. Is it running?")
            final_answer = "_Cannot connect to backend. Please start the server._"
            assistant_placeholder.markdown(final_answer)
        except Exception as e:
            status_box.error(f"Error: {str(e)}")
            final_answer = f"_Error: {str(e)}_"
            assistant_placeholder.markdown(final_answer)

        if not final_answer:
            final_answer = "_No final answer received (backend error or interrupted run)._"
            assistant_placeholder.markdown(final_answer)

        st.session_state.messages.append({"role": "assistant", "content": final_answer})
