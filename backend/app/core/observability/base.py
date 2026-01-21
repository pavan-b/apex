"""
Base classes for the observability system.

This module defines the abstract interfaces that all observability providers
must implement. Following the Vanna 2.0 pattern, this allows for pluggable
observability backends (Arize Phoenix, LangSmith, custom solutions, etc.).

The key abstractions are:
- Span: Represents a unit of work being traced
- ObservabilityProvider: Backend that creates and manages spans

Example:
    class MyProvider(ObservabilityProvider):
        def span(self, name: str, **kwargs) -> Span:
            return MySpan(name)

        def instrument(self) -> None:
            # Set up auto-instrumentation
            pass
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)


class Span(ABC):
    """
    Abstract base class representing a trace span.

    A span represents a single unit of work within a trace. Spans can be
    nested to represent hierarchical operations. Each span has:
    - A name identifying the operation
    - Optional attributes (key-value pairs)
    - Optional events (timestamped log entries)
    - A status (success/error)

    Implementations should support context manager usage:
        with provider.span("my_op") as span:
            span.set_attribute("key", "value")
            # do work
    """

    @abstractmethod
    def set_attribute(self, key: str, value: Any) -> None:
        """
        Set an attribute on the span.

        Args:
            key: Attribute name.
            value: Attribute value (should be JSON-serializable).
        """
        pass

    @abstractmethod
    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        """
        Set multiple attributes at once.

        Args:
            attributes: Dictionary of attribute key-value pairs.
        """
        pass

    @abstractmethod
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a timestamped event to the span.

        Args:
            name: Event name.
            attributes: Optional event attributes.
        """
        pass

    @abstractmethod
    def set_status(self, status: str, description: Optional[str] = None) -> None:
        """
        Set the span status.

        Args:
            status: Status code ("OK", "ERROR").
            description: Optional status description.
        """
        pass

    @abstractmethod
    def record_exception(self, exception: Exception) -> None:
        """
        Record an exception on the span.

        Args:
            exception: The exception to record.
        """
        pass

    @abstractmethod
    def end(self) -> None:
        """End the span, recording its duration."""
        pass

    def __enter__(self) -> "Span":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - records exception if present."""
        if exc_val is not None:
            self.record_exception(exc_val)
            self.set_status("ERROR", str(exc_val))
        else:
            self.set_status("OK")
        self.end()


class NoOpSpan(Span):
    """
    No-operation span for when tracing is disabled.

    All methods are no-ops that do nothing. This allows code to use
    tracing uniformly without checking if tracing is enabled.
    """

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        pass

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        pass

    def set_status(self, status: str, description: Optional[str] = None) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def end(self) -> None:
        pass


class ObservabilityProvider(ABC):
    """
    Abstract base class for observability backends.

    Implementations handle:
    - Connecting to the observability backend (Phoenix, LangSmith, etc.)
    - Creating and managing spans
    - Auto-instrumenting libraries (LangChain, etc.)
    - Graceful shutdown

    The provider pattern allows swapping observability backends without
    changing application code.

    Example:
        provider = ArizePhoenixProvider(endpoint="http://localhost:6006")
        provider.instrument()

        with provider.span("process_request") as span:
            span.set_attribute("user_id", user_id)
            result = process(request)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name (e.g., 'arize-phoenix')."""
        pass

    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        """Return True if the provider is active and connected."""
        pass

    @abstractmethod
    def instrument(self) -> bool:
        """
        Initialize and instrument the observability backend.

        This should:
        1. Connect to the backend
        2. Set up auto-instrumentation for supported libraries
        3. Return True if successful, False otherwise

        Returns:
            True if instrumentation was successful.
        """
        pass

    @abstractmethod
    @contextmanager
    def span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Generator[Span, None, None]:
        """
        Create a new span as a context manager.

        Args:
            name: Span name (operation being traced).
            attributes: Optional initial attributes.
            **kwargs: Provider-specific options.

        Yields:
            Span: The created span.
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """
        Gracefully shutdown the provider.

        Should flush any pending spans and close connections.
        """
        pass


class NoOpProvider(ObservabilityProvider):
    """
    No-operation provider for when observability is disabled.

    Returns NoOpSpan instances that do nothing. This allows code to
    use tracing uniformly without checking if it's enabled.
    """

    @property
    def name(self) -> str:
        return "noop"

    @property
    def is_enabled(self) -> bool:
        return False

    def instrument(self) -> bool:
        logger.debug("NoOp observability provider - instrumentation skipped")
        return False

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Generator[Span, None, None]:
        yield NoOpSpan()

    def shutdown(self) -> None:
        pass
