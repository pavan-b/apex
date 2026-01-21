"""
Observability provider factory.

This module provides factory functions to initialize and access the
observability provider. It follows the singleton pattern to ensure
only one provider instance exists.

The factory supports:
- Automatic provider selection based on configuration
- Lazy initialization
- Global access to the provider instance

Example:
    # At application startup (before LangChain imports)
    from backend.app.core.observability import init_observability

    provider = init_observability()

    # Later, anywhere in the code
    from backend.app.core.observability import get_provider

    provider = get_provider()
    with provider.span("my_operation") as span:
        # do work
        pass
"""

from __future__ import annotations

import logging
from typing import Optional

from backend.app.core.observability.base import NoOpProvider, ObservabilityProvider

logger = logging.getLogger(__name__)

# Global singleton instance
_provider: Optional[ObservabilityProvider] = None


def init_observability(
    provider_type: str = "phoenix",
    **kwargs,
) -> ObservabilityProvider:
    """
    Initialize the observability provider.

    This function should be called at application startup, before importing
    any LangChain modules, to ensure auto-instrumentation captures all calls.

    Args:
        provider_type: Provider to use ("phoenix" or "noop").
        **kwargs: Provider-specific configuration options.

    Returns:
        ObservabilityProvider: The initialized provider instance.

    Example:
        # Basic usage
        provider = init_observability()

        # With custom configuration
        provider = init_observability(
            provider_type="phoenix",
            endpoint="http://localhost:6006",
            project_name="my-project",
        )

        # Disable tracing
        provider = init_observability(provider_type="noop")
    """
    global _provider

    if _provider is not None:
        logger.debug("Observability already initialized with %s", _provider.name)
        return _provider

    if provider_type == "noop":
        _provider = NoOpProvider()
        logger.info("Observability disabled (NoOp provider)")
        return _provider

    if provider_type == "phoenix":
        from backend.app.core.observability.phoenix_provider import ArizePhoenixProvider

        _provider = ArizePhoenixProvider(**kwargs)
        if _provider.instrument():
            logger.info("Observability initialized with Arize Phoenix")
        else:
            # Fall back to NoOp if Phoenix fails to connect
            logger.info("Falling back to NoOp provider (Phoenix unavailable)")
            # Keep the Phoenix provider but it will return NoOpSpans
        return _provider

    # Unknown provider type - use NoOp
    logger.warning("Unknown provider type '%s', using NoOp", provider_type)
    _provider = NoOpProvider()
    return _provider


def get_provider() -> ObservabilityProvider:
    """
    Get the current observability provider.

    Returns a NoOpProvider if init_observability() hasn't been called.
    This ensures code can safely use tracing without checking initialization.

    Returns:
        ObservabilityProvider: The current provider instance.

    Example:
        provider = get_provider()
        with provider.span("my_operation") as span:
            span.set_attribute("key", "value")
    """
    global _provider

    if _provider is None:
        # Return a NoOp provider if not initialized
        # This allows code to use tracing without initialization checks
        return NoOpProvider()

    return _provider


def shutdown_observability() -> None:
    """
    Shutdown the observability provider.

    Call this during application shutdown to flush pending spans
    and close connections gracefully.
    """
    global _provider

    if _provider is not None:
        _provider.shutdown()
        _provider = None
        logger.debug("Observability shut down")


def tracing_enabled() -> bool:
    """
    Check if tracing is enabled.

    Returns:
        True if a provider is initialized and active.
    """
    return _provider is not None and _provider.is_enabled


def get_tracer(name: str = "apex"):
    """
    Get a tracer for creating custom spans.

    This is a convenience function that wraps the provider's span method.
    Safe to use even if tracing is disabled - returns a no-op context.

    Args:
        name: Tracer name (for organizing spans).

    Returns:
        A tracer-like object with start_as_current_span method.

    Example:
        tracer = get_tracer("research")
        with tracer.start_as_current_span("search_tool") as span:
            span.set_attribute("query", query)
            results = search(query)
    """
    provider = get_provider()

    class TracerWrapper:
        """Wrapper to provide OTel-compatible tracer interface."""

        def __init__(self, provider, name):
            self._provider = provider
            self._name = name

        def start_as_current_span(self, span_name: str, **kwargs):
            """Create a span as a context manager."""
            return self._provider.span(f"{self._name}.{span_name}", **kwargs)

    return TracerWrapper(provider, name)
