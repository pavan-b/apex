"""
Server module for FastAPI route handling.

This module separates HTTP concerns from business logic:
- ChatHandler: Handles chat requests and SSE streaming
- register_chat_routes: Registers routes on a FastAPI app

Example:
    from fastapi import FastAPI
    from backend.app.core import Agent
    from backend.app.server import ChatHandler, register_chat_routes

    app = FastAPI()
    agent = Agent(...)
    handler = ChatHandler(agent)
    register_chat_routes(app, handler)
"""

from backend.app.server.handler import ChatHandler
from backend.app.server.routes import register_chat_routes

__all__ = [
    "ChatHandler",
    "register_chat_routes",
]
