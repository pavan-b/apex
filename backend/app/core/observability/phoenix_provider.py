"""
Arize Phoenix observability provider.

This module implements the ObservabilityProvider interface for Arize Phoenix,
providing:
- OTLP trace export to Phoenix
- Auto-instrumentation for LangChain/LangGraph
- Custom span creation for non-instrumented code

Phoenix is an open-source observability tool for LLM applications that provides:
- Trace visualization
- LLM call analysis
- Token usage tracking
- Latency monitoring

Reference: https://docs.arize.com/phoenix/tracing/integrations-tracing/langgraph

Example:
    provider = ArizePhoenixProvider(
        endpoint="http://localhost:6006",
        project_name="my-project",
    )
    if provider.instrument():
        print("Phoenix tracing enabled")
"""

from __future__ import annotations

import logging
import os
import socket
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional
from urllib.parse import urlparse

from backend.app.core.observability.base import (
    NoOpSpan,
    ObservabilityProvider,
    Span,
)

logger = logging.getLogger(__name__)


class PhoenixSpan(Span):
    """
    Span implementation wrapping an OpenTelemetry span.

    This provides a clean interface over the OTel span while maintaining
    compatibility with the ObservabilityProvider pattern.
    """

    def __init__(self, otel_span: Any) -> None:
        """
        Initialize with an OpenTelemetry span.

        Args:
            otel_span: The underlying OTel span object.
        """
        self._span = otel_span

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a single attribute on the span."""
        try:
            self._span.set_attribute(key, value)
        except Exception:
            pass  # Silently ignore attribute errors

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        """Set multiple attributes at once."""
        for key, value in attributes.items():
            self.set_attribute(key, value)

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add a timestamped event to the span."""
        try:
            self._span.add_event(name, attributes=attributes)
        except Exception:
            pass

    def set_status(self, status: str, description: Optional[str] = None) -> None:
        """Set the span status."""
        try:
            from opentelemetry.trace import StatusCode

            if status.upper() == "ERROR":
                self._span.set_status(StatusCode.ERROR, description)
            else:
                self._span.set_status(StatusCode.OK, description)
        except Exception:
            pass

    def record_exception(self, exception: Exception) -> None:
        """Record an exception on the span."""
        try:
            self._span.record_exception(exception)
        except Exception:
            pass

    def end(self) -> None:
        """End the span."""
        try:
            self._span.end()
        except Exception:
            pass


class ArizePhoenixProvider(ObservabilityProvider):
    """
    Arize Phoenix observability provider.

    This provider integrates with Arize Phoenix for LLM observability,
    providing automatic tracing for LangChain/LangGraph applications.

    Features:
    - Automatic LangChain/LangGraph instrumentation
    - OTLP trace export to Phoenix
    - Custom span creation
    - Graceful fallback when Phoenix is unavailable

    Attributes:
        endpoint: Phoenix server endpoint (default: http://localhost:6006).
        project_name: Project name shown in Phoenix UI.
        api_key: Optional API key for Phoenix Cloud.

    Example:
        provider = ArizePhoenixProvider(
            endpoint="http://localhost:6006",
            project_name="apex-research-chat",
        )

        # Initialize (call before LangChain imports)
        provider.instrument()

        # Create custom spans
        with provider.span("search_tool", {"query": "AI"}) as span:
            results = search(query)
            span.set_attribute("result_count", len(results))
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        project_name: Optional[str] = None,
        api_key: Optional[str] = None,
        auto_instrument: bool = True,
    ) -> None:
        """
        Initialize the Phoenix provider.

        Configuration can be provided via:
        1. Constructor arguments
        2. Environment variables (PHOENIX_OTEL_ENDPOINT, etc.)
        3. .env file in the project root

        Args:
            endpoint: Phoenix endpoint URL.
            project_name: Project name for grouping traces.
            api_key: Optional API key for Phoenix Cloud.
            auto_instrument: Whether to auto-instrument LangChain.
        """
        self._endpoint = endpoint
        self._project_name = project_name
        self._api_key = api_key
        self._auto_instrument = auto_instrument
        self._tracer_provider = None
        self._tracer = None
        self._is_enabled = False
        self._langchain_instrumented = False

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "arize-phoenix"

    @property
    def is_enabled(self) -> bool:
        """Return True if Phoenix is connected and tracing is active."""
        return self._is_enabled

    def _load_env_file(self, path: str = ".env") -> Dict[str, str]:
        """Load environment variables from a .env file."""
        env: Dict[str, str] = {}
        if not os.path.exists(path):
            return env
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip().strip('"').strip("'")
        except OSError:
            pass
        return env

    def _get_config(self) -> Dict[str, Optional[str]]:
        """
        Get configuration from multiple sources.

        Priority: constructor args > env vars > .env file > defaults
        """
        file_env = self._load_env_file()

        endpoint = (
            self._endpoint
            or os.getenv("PHOENIX_OTEL_ENDPOINT")
            or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            or file_env.get("PHOENIX_OTEL_ENDPOINT")
            or file_env.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        )

        # Normalize endpoint - strip /v1/traces if present
        if endpoint:
            for suffix in ["/v1/traces", "/v1/logs", "/v1/metrics"]:
                if endpoint.endswith(suffix):
                    endpoint = endpoint[: -len(suffix)]
                    break
            endpoint = endpoint.rstrip("/")

        project_name = (
            self._project_name
            or os.getenv("PHOENIX_SERVICE_NAME")
            or file_env.get("PHOENIX_SERVICE_NAME")
            or "apex-research-chat"
        )

        api_key = (
            self._api_key
            or os.getenv("PHOENIX_API_KEY")
            or file_env.get("PHOENIX_API_KEY")
        )

        return {
            "endpoint": endpoint,
            "project_name": project_name,
            "api_key": api_key,
        }

    def _is_reachable(self, endpoint: str, timeout: float = 2.0) -> bool:
        """Check if Phoenix endpoint is reachable."""
        try:
            parsed = urlparse(endpoint)
            host = parsed.hostname or "localhost"
            port = parsed.port or (6006 if "6006" in endpoint else 80)

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def instrument(self) -> bool:
        """
        Initialize Phoenix tracing and instrument LangChain.

        This method:
        1. Loads configuration from env/files
        2. Checks Phoenix connectivity
        3. Sets up OTLP export
        4. Instruments LangChain/LangGraph

        Returns:
            True if instrumentation was successful.
        """
        if self._is_enabled:
            logger.debug("Phoenix already instrumented")
            return True

        config = self._get_config()
        endpoint = config["endpoint"]
        project_name = config["project_name"]
        api_key = config["api_key"]

        if not endpoint:
            logger.info(
                "Phoenix tracing disabled: No endpoint configured. "
                "Set PHOENIX_OTEL_ENDPOINT environment variable."
            )
            return False

        if not self._is_reachable(endpoint):
            logger.warning(
                "Phoenix tracing disabled: Server not reachable at %s. "
                "Start Phoenix with 'uv run phoenix-serve'.",
                endpoint,
            )
            return False

        try:
            # Import LangChain instrumentor FIRST (before register)
            # This ensures auto_instrument can find it
            try:
                from openinference.instrumentation.langchain import LangChainInstrumentor
                _langchain_instrumentor = LangChainInstrumentor()
            except ImportError:
                _langchain_instrumentor = None
                logger.debug("LangChain instrumentor not available")

            # Use phoenix.otel.register for clean setup
            from phoenix.otel import register

            # Build headers for API key if provided
            headers = {}
            if api_key:
                headers["authorization"] = f"Bearer {api_key}"

            # Build the OTLP HTTP endpoint (Phoenix expects /v1/traces path)
            # Use HTTP transport explicitly for the /v1/traces endpoint
            otlp_endpoint = f"{endpoint}/v1/traces"

            # Register with Phoenix using HTTP transport
            self._tracer_provider = register(
                project_name=project_name,
                endpoint=otlp_endpoint,
                headers=headers or None,
                # Don't use auto_instrument - we'll instrument manually
                auto_instrument=False,
            )

            # Get a tracer for custom spans
            from opentelemetry import trace
            self._tracer = trace.get_tracer("apex.custom")

            # Manually instrument LangChain after tracer provider is set
            if _langchain_instrumentor is not None:
                if not _langchain_instrumentor.is_instrumented_by_opentelemetry:
                    _langchain_instrumentor.instrument()
                    logger.debug("LangChain instrumentation enabled")
                self._langchain_instrumented = True

            self._is_enabled = True
            logger.info(
                "Phoenix tracing enabled: endpoint=%s, project=%s",
                otlp_endpoint,
                project_name,
            )
            return True

        except ImportError as e:
            logger.warning(
                "Phoenix tracing disabled: Missing dependency - %s. "
                "Install with: pip install arize-phoenix openinference-instrumentation-langchain",
                e,
            )
            return False
        except Exception as e:
            logger.warning("Phoenix tracing disabled: %s", e)
            return False

    def _instrument_langchain(self) -> None:
        """Instrument LangChain for automatic tracing."""
        if self._langchain_instrumented:
            return

        try:
            from openinference.instrumentation.langchain import LangChainInstrumentor

            instrumentor = LangChainInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
                logger.debug("LangChain instrumentation enabled")

            self._langchain_instrumented = True

        except ImportError:
            logger.debug(
                "LangChain instrumentor not available. "
                "Install: pip install openinference-instrumentation-langchain"
            )
        except Exception as e:
            logger.debug("LangChain instrumentation failed: %s", e)

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Generator[Span, None, None]:
        """
        Create a new span for tracing custom operations.

        Args:
            name: Span name describing the operation.
            attributes: Optional initial attributes.
            **kwargs: Additional OTel span options.

        Yields:
            Span: The created span (PhoenixSpan or NoOpSpan).

        Example:
            with provider.span("fetch_url", {"url": url}) as span:
                content = fetch(url)
                span.set_attribute("content_length", len(content))
        """
        if not self._is_enabled or self._tracer is None:
            yield NoOpSpan()
            return

        try:
            with self._tracer.start_as_current_span(name, **kwargs) as otel_span:
                phoenix_span = PhoenixSpan(otel_span)
                if attributes:
                    phoenix_span.set_attributes(attributes)
                yield phoenix_span
        except Exception:
            yield NoOpSpan()

    def shutdown(self) -> None:
        """
        Gracefully shutdown the Phoenix provider.

        Flushes pending spans and closes connections.
        """
        if self._tracer_provider is not None:
            try:
                self._tracer_provider.shutdown()
                logger.debug("Phoenix tracer provider shut down")
            except Exception as e:
                logger.debug("Error shutting down Phoenix: %s", e)

        self._is_enabled = False
        self._tracer = None
        self._tracer_provider = None
