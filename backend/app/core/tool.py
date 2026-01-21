"""
Tool abstraction for the agent system.

This module defines the base classes for tools that the agent can use:
- ToolContext: Runtime context passed to each tool execution
- ToolResult: Standardized result from tool execution
- Tool: Abstract base class that all tools must extend

Tools are the primary way for the agent to interact with external systems
(search engines, databases, APIs, etc.).

Example:
    class MySearchTool(Tool[SearchArgs]):
        @property
        def name(self) -> str:
            return "my_search"

        @property
        def description(self) -> str:
            return "Search the web for information"

        def get_args_schema(self) -> Type[SearchArgs]:
            return SearchArgs

        async def execute(self, ctx: ToolContext, args: SearchArgs) -> ToolResult:
            results = await do_search(args.query)
            return ToolResult(success=True, data=results)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generic, Type, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from backend.app.core.agent import Agent

# Type variable for tool argument schemas
TArgs = TypeVar("TArgs", bound=BaseModel)


@dataclass
class ToolContext:
    """
    Runtime context provided to tools during execution.

    This context gives tools access to:
    - The parent agent (for accessing LLM, other tools, config)
    - An event emitter for streaming progress to the client
    - Request-scoped configuration

    Attributes:
        agent: The Agent instance orchestrating this execution.
        emit_event: Callback to emit SSE events to the client.
        config: Request-scoped configuration dictionary.
    """

    agent: "Agent"
    emit_event: Callable[[dict[str, Any]], None]
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """
    Standardized result from tool execution.

    All tools return a ToolResult to provide consistent handling of
    success/failure states and data.

    Attributes:
        success: Whether the tool execution succeeded.
        data: The result data (type depends on the tool).
        error: Error message if success is False.
        metadata: Optional metadata about the execution (timing, counts, etc.).

    Example:
        # Success case
        ToolResult(success=True, data={"results": [...]})

        # Failure case
        ToolResult(success=False, error="API rate limit exceeded")
    """

    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Tool(ABC, Generic[TArgs]):
    """
    Abstract base class for all tools in the agent system.

    Tools encapsulate specific capabilities that the agent can use to
    accomplish tasks. Each tool:
    - Has a unique name for identification
    - Has a description for LLM prompting
    - Defines an argument schema (Pydantic model)
    - Implements async execution logic

    Subclasses must implement:
        - name (property): Unique identifier for the tool
        - description (property): Human-readable description
        - get_args_schema(): Returns the Pydantic model for arguments
        - execute(): Performs the tool's action

    Example:
        class WebSearchTool(Tool[WebSearchArgs]):
            @property
            def name(self) -> str:
                return "web_search"

            @property
            def description(self) -> str:
                return "Search the web using DuckDuckGo"

            def get_args_schema(self) -> Type[WebSearchArgs]:
                return WebSearchArgs

            async def execute(self, ctx: ToolContext, args: WebSearchArgs) -> ToolResult:
                hits = await search_ddg(args.query, args.max_results)
                return ToolResult(success=True, data=hits)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this tool.

        Returns:
            str: Tool name (e.g., "web_search", "wikipedia", "arxiv").
        """
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Human-readable description of what this tool does.

        This description is used in LLM prompts to help the model
        understand when and how to use the tool.

        Returns:
            str: Description of the tool's purpose and capabilities.
        """
        ...

    @abstractmethod
    def get_args_schema(self) -> Type[TArgs]:
        """
        Return the Pydantic model class for this tool's arguments.

        The schema is used for:
        - Validating arguments before execution
        - Generating JSON schema for LLM function calling
        - Documentation

        Returns:
            Type[TArgs]: Pydantic model class for arguments.
        """
        ...

    @abstractmethod
    async def execute(self, ctx: ToolContext, args: TArgs) -> ToolResult:
        """
        Execute the tool with the given context and arguments.

        This is where the tool's main logic lives. Tools should:
        - Use ctx.emit_event() to stream progress updates
        - Return ToolResult with success=True and data on success
        - Return ToolResult with success=False and error on failure
        - Handle exceptions gracefully

        Args:
            ctx: Runtime context with agent access and event emitter.
            args: Validated arguments matching get_args_schema().

        Returns:
            ToolResult: Result of the tool execution.
        """
        ...

    def to_function_schema(self) -> dict[str, Any]:
        """
        Generate OpenAI-compatible function schema for this tool.

        This is used when integrating with LLMs that support function calling.

        Returns:
            dict: Function schema with name, description, and parameters.
        """
        schema = self.get_args_schema()
        return {
            "name": self.name,
            "description": self.description,
            "parameters": schema.model_json_schema(),
        }
