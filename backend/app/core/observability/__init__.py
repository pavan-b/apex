"""
Observability module for tracing and monitoring.

This module provides a pluggable observability system following the Vanna 2.0
pattern. It includes:
- ObservabilityProvider: Abstract base class for observability backends
- ArizePhoenixProvider: Concrete implementation for Arize Phoenix
- init_observability(): Factory function to initialize the configured provider

Example:
    from backend.app.core.observability import init_observability

    # Initialize at startup (before LangChain imports)
    provider = init_observability()

    # Create custom spans
    with provider.span("my_operation") as span:
        span.set_attribute("key", "value")
        result = do_something()
"""

from backend.app.core.observability.base import ObservabilityProvider, Span, NoOpProvider
from backend.app.core.observability.phoenix_provider import ArizePhoenixProvider
from backend.app.core.observability.factory import (
    init_observability,
    get_provider,
    shutdown_observability,
    tracing_enabled,
    get_tracer,
)

__all__ = [
    "ObservabilityProvider",
    "ArizePhoenixProvider",
    "NoOpProvider",
    "Span",
    "init_observability",
    "get_provider",
    "shutdown_observability",
    "tracing_enabled",
    "get_tracer",
]
