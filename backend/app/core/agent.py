"""
Agent class - Central orchestrator for the LLM agent system.

The Agent ties together all components:
- LlmService: For making LLM calls
- ToolRegistry: For discovering and executing tools
- LifecycleHooks: For extensibility at key points
- Workflow: The LangGraph state machine for research + verification

The Agent's run() method is the main entry point that:
1. Fires lifecycle hooks at appropriate points
2. Calls the LangGraph workflow
3. Yields SSE events for real-time UI updates

Example:
    # Create the agent
    agent = Agent(
        llm_service=LlmService(model="qwen3:8b"),
        tool_registry=registry,
        lifecycle_hooks=MyHooks(),
    )

    # Run a query
    async for event in agent.run("What is quantum computing?", config):
        yield f"event: message\\ndata: {json.dumps(event)}\\n\\n"
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, AsyncIterator

from backend.app.core.lifecycle import LifecycleHooks
from backend.app.core.llm_service import LlmService
from backend.app.core.registry import ToolRegistry
from backend.app.core.tool import ToolContext

logger = logging.getLogger(__name__)


class Agent:
    """
    Central orchestrator for the LLM agent system.

    The Agent coordinates:
    - LLM interactions through LlmService
    - Tool execution through ToolRegistry
    - Lifecycle hooks for extensibility
    - The LangGraph workflow for research and verification

    Attributes:
        llm_service: Service for making LLM calls.
        tool_registry: Registry of available tools.
        lifecycle_hooks: Hooks for extensibility.
        config: Default configuration.

    Example:
        agent = Agent(
            llm_service=LlmService(),
            tool_registry=ToolRegistry(),
            lifecycle_hooks=LifecycleHooks(),
        )

        # Process a message
        async for event in agent.run("Explain AI", {}):
            print(event)
    """

    def __init__(
        self,
        llm_service: LlmService | None = None,
        tool_registry: ToolRegistry | None = None,
        lifecycle_hooks: LifecycleHooks | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the Agent.

        Args:
            llm_service: LLM service instance. Created with defaults if None.
            tool_registry: Tool registry. Created empty if None.
            lifecycle_hooks: Lifecycle hooks. Created with defaults if None.
            config: Default configuration dictionary.
        """
        self.llm_service = llm_service or LlmService()
        self.tool_registry = tool_registry or ToolRegistry()
        self.lifecycle_hooks = lifecycle_hooks or LifecycleHooks()
        self.config = config or {}

        # Load default config from environment
        self._load_env_config()

        logger.info(
            "Agent initialized: model=%s, tools=%d",
            self.llm_service.model,
            len(self.tool_registry),
        )

    def _load_env_config(self) -> None:
        """Load configuration from environment variables."""
        if "best_of_n" not in self.config:
            self.config["best_of_n"] = int(os.getenv("BEST_OF_N", "3"))
        if "verify_max_rounds" not in self.config:
            self.config["verify_max_rounds"] = int(os.getenv("VERIFY_MAX_ROUNDS", "3"))
        if "search_max_results" not in self.config:
            self.config["search_max_results"] = int(os.getenv("SEARCH_MAX_RESULTS", "5"))

    def get_tool_context(
        self,
        emit_event: Any,
        config: dict[str, Any],
    ) -> ToolContext:
        """
        Create a ToolContext for tool execution.

        Args:
            emit_event: Callback to emit events to the client.
            config: Request-scoped configuration.

        Returns:
            ToolContext: Context for tool execution.
        """
        return ToolContext(
            agent=self,
            emit_event=emit_event,
            config=config,
        )

    async def run(
        self,
        message: str,
        request_config: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Run the agent on a user message.

        This is the main entry point for processing a user query. It:
        1. Merges request config with defaults
        2. Fires on_request_start hook
        3. Runs the LangGraph workflow
        4. Yields SSE events for real-time updates
        5. Fires on_request_end hook

        Args:
            message: The user's input message.
            request_config: Optional per-request configuration overrides.

        Yields:
            dict: SSE event dictionaries with type, step, and data.

        Example:
            async for event in agent.run("What is AI?", {"best_of_n": 1}):
                # event = {"type": "step_started", "step": "research", "data": {...}}
                yield format_sse(event)
        """
        # Merge configs
        config = {**self.config, **(request_config or {})}

        start_time = time.time()
        final_answer = ""

        try:
            # Fire request start hook
            await self.lifecycle_hooks.on_request_start(message, config)

            # Import workflow here to avoid circular imports
            from backend.app.graph.workflow import run_workflow

            # Run the workflow and yield events
            async for event in run_workflow(
                message=message,
                config=config,
                agent=self,
            ):
                # Fire step hooks if applicable
                if event.get("type") == "step_started":
                    await self.lifecycle_hooks.on_step_start(
                        event.get("step", ""),
                        event.get("data"),
                    )
                elif event.get("type") == "step_finished":
                    await self.lifecycle_hooks.on_step_end(
                        event.get("step", ""),
                        event.get("data"),
                    )
                elif event.get("type") == "tool_call":
                    await self.lifecycle_hooks.on_tool_call(
                        event.get("data", {}).get("tool", ""),
                        event.get("data", {}),
                    )
                elif event.get("type") == "final":
                    final_answer = event.get("data", {}).get("answer", "")

                yield event

            # Fire request end hook
            elapsed_ms = (time.time() - start_time) * 1000
            await self.lifecycle_hooks.on_request_end(final_answer)
            logger.info("Agent run completed in %.1fms", elapsed_ms)

        except Exception as e:
            # Fire error hook
            await self.lifecycle_hooks.on_error(e, {"step": "agent.run"})

            # Yield error event
            yield {
                "type": "error",
                "step": "agent",
                "data": {"error": str(e)},
            }

            # Yield fallback final event
            yield {
                "type": "final",
                "step": "agent",
                "data": {"answer": f"_Error: {e}_"},
            }

        # Always yield done event
        yield {
            "type": "done",
            "step": "agent",
            "data": {},
        }

    async def execute_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        emit_event: Any,
        config: dict[str, Any],
    ) -> Any:
        """
        Execute a tool by name.

        This method:
        1. Looks up the tool in the registry
        2. Validates arguments
        3. Fires lifecycle hooks
        4. Executes the tool
        5. Returns the result

        Args:
            tool_name: Name of the tool to execute.
            args: Arguments for the tool.
            emit_event: Callback to emit events.
            config: Request-scoped configuration.

        Returns:
            The tool's result data.

        Raises:
            ValueError: If the tool is not found.
        """
        tool = self.tool_registry.get(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        # Fire tool call hook
        await self.lifecycle_hooks.on_tool_call(tool_name, args)

        # Create context and execute
        ctx = self.get_tool_context(emit_event, config)
        start_time = time.time()

        # Parse args using the tool's schema
        args_schema = tool.get_args_schema()
        parsed_args = args_schema(**args)

        result = await tool.execute(ctx, parsed_args)

        # Fire tool result hook
        elapsed_ms = (time.time() - start_time) * 1000
        await self.lifecycle_hooks.on_tool_result(tool_name, result, elapsed_ms)

        if result.success:
            return result.data
        else:
            raise RuntimeError(f"Tool {tool_name} failed: {result.error}")
