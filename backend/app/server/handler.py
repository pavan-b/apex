"""
Chat handler for processing chat requests.

The ChatHandler encapsulates the logic for handling chat requests,
separate from the HTTP routing concerns. This follows the Vanna 2.0
pattern of separating handlers from route registration.

Example:
    agent = Agent(llm_service, tool_registry, lifecycle_hooks)
    handler = ChatHandler(agent)

    # In your route
    @app.post("/api/v1/chat/sse")
    async def chat_sse(request: ChatRequest):
        return await handler.chat_sse(request)
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, AsyncIterator

from starlette.responses import StreamingResponse

from backend.app.schemas import ChatRequest

if TYPE_CHECKING:
    from backend.app.core.agent import Agent

logger = logging.getLogger(__name__)


class ChatHandler:
    """
    Handler for chat requests with SSE streaming.

    The ChatHandler:
    - Accepts chat requests with message and optional config
    - Runs the Agent's workflow
    - Streams SSE events to the client in real-time

    This class separates the business logic from HTTP concerns,
    making it easier to test and reuse.

    Attributes:
        agent: The Agent instance that processes requests.

    Example:
        handler = ChatHandler(agent)

        # Process a request (returns StreamingResponse)
        response = await handler.chat_sse(ChatRequest(message="Hello"))
    """

    def __init__(self, agent: Agent) -> None:
        """
        Initialize the chat handler.

        Args:
            agent: The Agent instance to use for processing requests.
        """
        self.agent = agent

    async def chat_sse(self, request: ChatRequest) -> StreamingResponse:
        """
        Handle a chat request with SSE streaming response.

        This method:
        1. Extracts config from the request
        2. Runs the Agent's workflow
        3. Streams SSE events as they're generated

        Args:
            request: The chat request with message and optional config.

        Returns:
            StreamingResponse: SSE stream with events.

        Example:
            response = await handler.chat_sse(ChatRequest(
                message="What is quantum computing?",
                best_of_n=1,
                verify_max_rounds=1,
            ))
        """
        # Build config from request
        config = {}
        if request.best_of_n is not None:
            config["best_of_n"] = request.best_of_n
        if request.verify_max_rounds is not None:
            config["verify_max_rounds"] = request.verify_max_rounds

        logger.info(
            "Chat SSE request: message=%s, config=%s",
            request.message[:50] if request.message else "",
            config,
        )

        async def event_generator() -> AsyncIterator[str]:
            """Generate SSE events from the agent run."""
            try:
                async for event in self.agent.run(request.message, config):
                    # Format as SSE
                    yield self._format_sse_event(event)

            except Exception as e:
                logger.exception("Error in chat SSE stream: %s", e)
                # Yield error event
                yield self._format_sse_event({
                    "type": "error",
                    "step": "stream",
                    "data": {"error": str(e)},
                })
                # Yield done event
                yield self._format_sse_event({
                    "type": "done",
                    "step": "stream",
                    "data": {},
                })

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    def _format_sse_event(self, event: dict) -> str:
        """
        Format an event dictionary as an SSE message.

        SSE format:
            event: <event_type>
            data: <json_payload>

            (blank line to end event)

        Args:
            event: Event dictionary with type, step, and data.

        Returns:
            str: Formatted SSE message.
        """
        event_type = event.get("type", "message")
        data = json.dumps(event)
        return f"event: {event_type}\ndata: {data}\n\n"

    async def health_check(self) -> dict:
        """
        Perform a health check.

        Returns:
            dict: Health status including agent info.
        """
        return {
            "status": "healthy",
            "model": self.agent.llm_service.model,
            "tools": self.agent.tool_registry.list_names(),
        }
