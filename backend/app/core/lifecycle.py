"""
Lifecycle hooks for extensibility at key moments in request processing.

Lifecycle hooks allow you to inject custom logic at specific points during
request processing without modifying core code. Use cases include:
- Logging and metrics
- Quota checking
- Content filtering
- Custom validation
- Audit trails

Hooks are async and can be overridden by subclassing LifecycleHooks.

Example:
    class MyHooks(LifecycleHooks):
        async def on_request_start(self, message, config):
            print(f"Starting request: {message[:50]}")
            # Could check quotas, log to database, etc.

        async def on_request_end(self, final_answer):
            print(f"Request completed: {len(final_answer)} chars")

    agent = Agent(lifecycle_hooks=MyHooks())
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class LifecycleHooks:
    """
    Base class for lifecycle hooks.

    All hook methods have default no-op implementations. Subclass and
    override the hooks you need.

    Available hooks:
        - on_request_start: Called when a new request begins
        - on_request_end: Called when a request completes
        - on_tool_call: Called before a tool is executed
        - on_tool_result: Called after a tool completes
        - on_llm_call: Called before an LLM call
        - on_llm_result: Called after an LLM call
        - on_error: Called when an error occurs

    Example:
        class MetricsHooks(LifecycleHooks):
            def __init__(self, metrics_client):
                self.metrics = metrics_client

            async def on_request_start(self, message, config):
                self.metrics.increment("requests_started")

            async def on_request_end(self, final_answer):
                self.metrics.increment("requests_completed")

            async def on_error(self, error, context):
                self.metrics.increment("requests_failed")
    """

    async def on_request_start(
        self,
        message: str,
        config: dict[str, Any],
    ) -> None:
        """
        Called when a new request begins processing.

        This is the first hook called and is ideal for:
        - Quota checking
        - Request logging
        - Setting up request-scoped state

        Args:
            message: The user's input message.
            config: Request configuration (best_of_n, verify_max_rounds, etc.).

        Raises:
            Exception: Raising an exception will abort the request.
        """
        logger.debug("Request started: %s", message[:100] if message else "")

    async def on_request_end(
        self,
        final_answer: str,
    ) -> None:
        """
        Called when a request completes successfully.

        This hook is called after the final answer is generated but
        before it's sent to the client. Ideal for:
        - Completion logging
        - Metrics recording
        - Post-processing

        Args:
            final_answer: The final answer to be returned.
        """
        logger.debug("Request completed: %d chars", len(final_answer) if final_answer else 0)

    async def on_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> None:
        """
        Called before a tool is executed.

        Useful for:
        - Tool usage logging
        - Argument validation
        - Rate limiting specific tools

        Args:
            tool_name: Name of the tool being called.
            args: Arguments being passed to the tool.
        """
        logger.debug("Tool call: %s with %d args", tool_name, len(args))

    async def on_tool_result(
        self,
        tool_name: str,
        result: Any,
        duration_ms: float,
    ) -> None:
        """
        Called after a tool completes execution.

        Useful for:
        - Performance monitoring
        - Result logging
        - Error tracking

        Args:
            tool_name: Name of the tool that was called.
            result: The tool's result.
            duration_ms: Execution time in milliseconds.
        """
        logger.debug("Tool result: %s completed in %.1fms", tool_name, duration_ms)

    async def on_llm_call(
        self,
        messages: list[dict[str, str]],
    ) -> None:
        """
        Called before an LLM call is made.

        Useful for:
        - Prompt logging
        - Token estimation
        - Content filtering

        Args:
            messages: The messages being sent to the LLM.
        """
        logger.debug("LLM call: %d messages", len(messages))

    async def on_llm_result(
        self,
        response: str,
        duration_ms: float,
    ) -> None:
        """
        Called after an LLM call completes.

        Useful for:
        - Response logging
        - Latency tracking
        - Token usage recording

        Args:
            response: The LLM's response.
            duration_ms: Call duration in milliseconds.
        """
        logger.debug("LLM result: %d chars in %.1fms", len(response), duration_ms)

    async def on_error(
        self,
        error: Exception,
        context: dict[str, Any],
    ) -> None:
        """
        Called when an error occurs during processing.

        This hook is called for any unhandled exception. Useful for:
        - Error logging
        - Alerting
        - Error transformation

        Args:
            error: The exception that occurred.
            context: Additional context about where the error occurred.
        """
        logger.error(
            "Error in %s: %s",
            context.get("step", "unknown"),
            str(error),
        )

    async def on_step_start(
        self,
        step_name: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """
        Called when a workflow step begins.

        Useful for:
        - Progress tracking
        - Step timing
        - Debug logging

        Args:
            step_name: Name of the step (e.g., "research", "synthesis").
            data: Optional data associated with the step.
        """
        logger.debug("Step started: %s", step_name)

    async def on_step_end(
        self,
        step_name: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """
        Called when a workflow step completes.

        Args:
            step_name: Name of the step that completed.
            data: Optional data about the step result.
        """
        logger.debug("Step completed: %s", step_name)


class CompositeLifecycleHooks(LifecycleHooks):
    """
    Combines multiple lifecycle hook implementations.

    When you have multiple sets of hooks (e.g., logging + metrics + audit),
    use CompositeLifecycleHooks to call all of them.

    Example:
        hooks = CompositeLifecycleHooks([
            LoggingHooks(),
            MetricsHooks(metrics_client),
            AuditHooks(audit_service),
        ])
        agent = Agent(lifecycle_hooks=hooks)
    """

    def __init__(self, hooks: list[LifecycleHooks]) -> None:
        """
        Initialize with a list of hook implementations.

        Args:
            hooks: List of LifecycleHooks instances to compose.
        """
        self._hooks = hooks

    async def on_request_start(self, message: str, config: dict[str, Any]) -> None:
        """Call on_request_start on all composed hooks."""
        for hook in self._hooks:
            await hook.on_request_start(message, config)

    async def on_request_end(self, final_answer: str) -> None:
        """Call on_request_end on all composed hooks."""
        for hook in self._hooks:
            await hook.on_request_end(final_answer)

    async def on_tool_call(self, tool_name: str, args: dict[str, Any]) -> None:
        """Call on_tool_call on all composed hooks."""
        for hook in self._hooks:
            await hook.on_tool_call(tool_name, args)

    async def on_tool_result(
        self,
        tool_name: str,
        result: Any,
        duration_ms: float,
    ) -> None:
        """Call on_tool_result on all composed hooks."""
        for hook in self._hooks:
            await hook.on_tool_result(tool_name, result, duration_ms)

    async def on_llm_call(self, messages: list[dict[str, str]]) -> None:
        """Call on_llm_call on all composed hooks."""
        for hook in self._hooks:
            await hook.on_llm_call(messages)

    async def on_llm_result(self, response: str, duration_ms: float) -> None:
        """Call on_llm_result on all composed hooks."""
        for hook in self._hooks:
            await hook.on_llm_result(response, duration_ms)

    async def on_error(self, error: Exception, context: dict[str, Any]) -> None:
        """Call on_error on all composed hooks."""
        for hook in self._hooks:
            await hook.on_error(error, context)

    async def on_step_start(
        self,
        step_name: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Call on_step_start on all composed hooks."""
        for hook in self._hooks:
            await hook.on_step_start(step_name, data)

    async def on_step_end(
        self,
        step_name: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Call on_step_end on all composed hooks."""
        for hook in self._hooks:
            await hook.on_step_end(step_name, data)
