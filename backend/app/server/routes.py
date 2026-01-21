"""
Route registration for the chat API.

This module provides a function to register chat routes on a FastAPI app.
Following the Vanna 2.0 pattern, route registration is separate from
handler implementation.

Example:
    from fastapi import FastAPI
    from backend.app.server import ChatHandler, register_chat_routes

    app = FastAPI()
    handler = ChatHandler(agent)
    register_chat_routes(app, handler)

    # Now app has:
    # - POST /api/v1/chat/sse (SSE streaming chat)
    # - GET /api/v1/health (health check)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import FastAPI

from backend.app.schemas import ChatRequest

if TYPE_CHECKING:
    from backend.app.server.handler import ChatHandler

logger = logging.getLogger(__name__)


def register_chat_routes(app: FastAPI, handler: ChatHandler) -> None:
    """
    Register chat API routes on the FastAPI app.

    Routes registered:
        POST /api/v1/chat/sse - Main chat endpoint with SSE streaming
        GET /api/v1/health - Health check endpoint

    Args:
        app: The FastAPI application instance.
        handler: The ChatHandler instance to handle requests.

    Example:
        app = FastAPI()
        agent = Agent(...)
        handler = ChatHandler(agent)
        register_chat_routes(app, handler)
    """

    @app.post("/api/v1/chat/sse")
    async def chat_sse(request: ChatRequest):
        """
        Process a chat message with SSE streaming response.

        This endpoint:
        1. Accepts a message and optional configuration
        2. Runs the agent workflow (research -> synthesis -> verification)
        3. Streams events in real-time via Server-Sent Events

        Request body:
            {
                "message": "Your question here",
                "best_of_n": 1,  // optional, for fast mode
                "verify_max_rounds": 1  // optional, for fast mode
            }

        Response:
            SSE stream with events:
            - step_started: A workflow step began
            - step_finished: A workflow step completed
            - tool_call: A tool is being called
            - tool_result: A tool returned results
            - critic_feedback: Verification critic feedback
            - revision_selected: A revision was selected
            - final: The final answer
            - error: An error occurred
            - done: Stream complete

        Args:
            request: ChatRequest with message and optional config.

        Returns:
            StreamingResponse with SSE events.
        """
        return await handler.chat_sse(request)

    @app.get("/api/v1/health")
    async def health_check():
        """
        Health check endpoint.

        Returns:
            JSON with health status and agent info.
        """
        return await handler.health_check()

    logger.info("Registered chat routes: POST /api/v1/chat/sse, GET /api/v1/health")
