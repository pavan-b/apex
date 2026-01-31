"""
FastAPI entrypoint for the chat backend (Vanna-style architecture).

This module wires together all components:
- Agent: Central orchestrator
- LlmService: Ollama integration with middleware
- ToolRegistry: Registered search and fetch tools
- ChatHandler: Request handling
- Routes: POST /api/v1/chat/sse
- Observability: Arize Phoenix tracing

The architecture follows patterns from Vanna 2.0:
- Separation of concerns (handler vs routes vs agent)
- Tool registry for extensibility
- Middleware chain for cross-cutting concerns
- Pluggable observability providers

Example:
    uvicorn backend.app.main:app --reload --port 8000
"""

from __future__ import annotations
from backend.app.tools import (
    ArxivSearchTool,
    DuckDuckGoSearchTool,
    PageFetchTool,
    WikipediaSearchTool,
)
from backend.app.server import ChatHandler, register_chat_routes
from backend.app.core import (
    Agent,
    LifecycleHooks,
    LlmService,
    LoggingMiddleware,
    ToolRegistry,
    setup_logging,
    get_logger,
    get_log_directory,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from backend.app.core.observability import (
    init_observability,
    shutdown_observability,
    tracing_enabled,
    get_provider,
)

import logging
import os
from contextlib import asynccontextmanager

# =============================================================================
# IMPORTANT: Initialize logging FIRST, then observability
# This ensures all logs are captured to files from the start
# =============================================================================
log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
setup_logging(
    log_level=log_level,
    log_to_console=True,
    log_to_file=True,
)
logger = get_logger(__name__)

# Initialize Phoenix observability before LangChain imports

observability = init_observability(provider_type="phoenix")

# Now import components that use LangChain


# =============================================================================
# Create core components
# =============================================================================

# LLM Service with middleware
llm_service = LlmService(
    middlewares=[
        LoggingMiddleware(log_level=logging.DEBUG),
        # Add more middlewares here: CachingMiddleware(), RetryMiddleware(), etc.
    ],
)

# Tool Registry with search tools
tool_registry = ToolRegistry()
tool_registry.register(DuckDuckGoSearchTool())
tool_registry.register(WikipediaSearchTool())
tool_registry.register(ArxivSearchTool())
tool_registry.register(PageFetchTool())

# Lifecycle hooks (use default implementation, can be customized)
lifecycle_hooks = LifecycleHooks()

# Create the Agent
agent = Agent(
    llm_service=llm_service,
    tool_registry=tool_registry,
    lifecycle_hooks=lifecycle_hooks,
)


# =============================================================================
# Lifespan context manager (replaces deprecated on_event)
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.

    This replaces the deprecated @app.on_event("startup") and
    @app.on_event("shutdown") decorators.

    Args:
        app: The FastAPI application instance.

    Yields:
        None: Control returns to FastAPI to handle requests.
    """
    # Startup
    logger.info("=" * 60)
    logger.info("Backend starting up")
    logger.info("  Model: %s", llm_service.model)
    logger.info("  Base URL: %s", llm_service.base_url)
    logger.info("  Tools: %s", tool_registry.list_names())
    logger.info("  Observability: %s (%s)",
                "enabled" if tracing_enabled() else "disabled",
                observability.name)
    logger.info("  Log directory: %s", get_log_directory())
    logger.info("  Routes: POST /api/v1/chat/sse, GET /api/v1/health")
    logger.info("=" * 60)

    yield  # Application runs here

    # Shutdown
    logger.info("Backend shutting down")
    shutdown_observability()


# =============================================================================
# Create FastAPI app
# =============================================================================

app = FastAPI(
    title="LangGraph Research+FactCheck Chat (Ollama qwen3:8b)",
    description="Vanna-style agent with research, synthesis, and fact-checking",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Register routes
# =============================================================================

# Create handler and register routes
handler = ChatHandler(agent)
register_chat_routes(app, handler)


# Legacy health check (also available at /api/v1/health)
@app.get("/health")
async def health():
    """
    Simple health check endpoint.

    Returns:
        dict: Health status with observability info.
    """
    return {
        "ok": True,
        "model": llm_service.model,
        "observability": {
            "provider": observability.name,
            "enabled": tracing_enabled(),
        },
    }
