"""
Shared Pydantic schemas and enums for the API and workflow state.

This module defines the core data models used throughout the application:
- EventType: SSE event types for UI updates
- ChatRequest: Incoming chat request model
- Source: Retrieved source/document representation
- SSEEvent: SSE payload model for streaming
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """
    Normalized event types emitted over SSE to drive UI updates.

    These events are streamed from the backend to the Streamlit UI
    to provide real-time progress updates.

    Attributes:
        step_started: A workflow step has begun.
        step_finished: A workflow step has completed.
        tool_call: A tool is being invoked.
        tool_result: A tool has returned results.
        message: General informational message.
        draft_chunk: Streaming draft content (unused currently).
        critic_feedback: Feedback from a verification critic.
        revision_selected: A revision has been chosen in Best-of-N.
        final: The final answer is ready.
        error: An error occurred.
    """

    step_started = "step_started"
    step_finished = "step_finished"
    tool_call = "tool_call"
    tool_result = "tool_result"
    message = "message"
    draft_chunk = "draft_chunk"
    critic_feedback = "critic_feedback"
    revision_selected = "revision_selected"
    final = "final"
    error = "error"


class ChatRequest(BaseModel):
    """
    Incoming chat request from the UI.

    Attributes:
        message: The user's question or message.
        session_id: Optional session identifier (for future use).
        best_of_n: Override for Best-of-N candidate count.
        verify_max_rounds: Override for max verification rounds.

    Example:
        {
            "message": "What is quantum computing?",
            "best_of_n": 1,
            "verify_max_rounds": 1
        }
    """

    message: str = Field(..., min_length=1, description="The user's question")
    session_id: Optional[str] = Field(None, description="Optional session ID")
    best_of_n: Optional[int] = Field(
        None, ge=1, le=5, description="Best-of-N candidates")
    verify_max_rounds: Optional[int] = Field(
        None, ge=1, le=5, description="Max verification rounds")


class Source(BaseModel):
    """
    Canonical representation of a retrieved source/document.

    Sources are collected from search tools (DuckDuckGo, Wikipedia, arXiv)
    and used to ground the LLM's responses.

    Attributes:
        title: Title of the source.
        url: URL of the source.
        snippet: Short excerpt or summary.
        tool: Name of the tool that retrieved this source.
        score: Relevance score (0-1).
        raw_text: Full page text if fetched.

    Example:
        Source(
            title="Quantum Computing - Wikipedia",
            url="https://en.wikipedia.org/wiki/Quantum_computing",
            snippet="Quantum computing is a type of computation...",
            tool="wikipedia",
            score=0.9,
        )
    """

    title: str
    url: str
    snippet: str = ""
    tool: str
    score: float = 0.0
    raw_text: str = ""


class SSEEvent(BaseModel):
    """
    SSE payload model sent to the UI stream.

    All events streamed to the UI follow this structure.

    Attributes:
        type: The event type (from EventType enum).
        run_id: Unique identifier for this run (legacy, kept for compatibility).
        step: The workflow step that generated this event.
        data: Event-specific payload data.
        ts_ms: Timestamp in milliseconds.

    Example:
        {
            "type": "step_started",
            "run_id": "abc123",
            "step": "research",
            "data": {"input": "What is AI?"},
            "ts_ms": 1705837200000
        }
    """

    type: EventType
    run_id: str = ""
    step: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)
    ts_ms: int = 0
