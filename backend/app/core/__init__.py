"""
Core module for Vanna-style agent architecture.

This module provides the foundational abstractions for building an extensible,
production-ready LLM agent system following patterns from Vanna 2.0.

Components:
    - Tool: Base class for all tools (search, fetch, etc.)
    - ToolContext: Runtime context passed to tool execution
    - ToolResult: Standardized result from tool execution
    - ToolRegistry: Central registry for discovering and managing tools
    - LlmService: Abstraction over LLM providers with middleware support
    - LlmMiddleware: Base class for LLM call interceptors
    - LifecycleHooks: Extension points at key moments in request lifecycle
    - Agent: Central orchestrator tying everything together
    - ObservabilityProvider: Base class for observability backends
    - ArizePhoenixProvider: Arize Phoenix observability implementation
"""

from backend.app.core.tool import Tool, ToolContext, ToolResult
from backend.app.core.registry import ToolRegistry
from backend.app.core.llm_service import LlmService
from backend.app.core.middleware import LlmMiddleware, LoggingMiddleware, CachingMiddleware
from backend.app.core.lifecycle import LifecycleHooks
from backend.app.core.agent import Agent
from backend.app.core.observability import (
    ObservabilityProvider,
    ArizePhoenixProvider,
    init_observability,
    get_provider,
    shutdown_observability,
    tracing_enabled,
    get_tracer,
)

__all__ = [
    # Tool system
    "Tool",
    "ToolContext",
    "ToolResult",
    "ToolRegistry",
    # LLM service
    "LlmService",
    "LlmMiddleware",
    "LoggingMiddleware",
    "CachingMiddleware",
    # Lifecycle
    "LifecycleHooks",
    # Agent
    "Agent",
    # Observability
    "ObservabilityProvider",
    "ArizePhoenixProvider",
    "init_observability",
    "get_provider",
    "get_tracer",
    "shutdown_observability",
    "tracing_enabled",
]
