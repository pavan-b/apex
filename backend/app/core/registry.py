"""
Tool registry for managing and discovering tools.

The ToolRegistry is a central location where tools are registered and can be
looked up by name. It provides:
- Registration of tool instances
- Lookup by name
- Listing all available tools
- Validation of tool uniqueness

Example:
    registry = ToolRegistry()
    registry.register(WebSearchTool())
    registry.register(WikipediaTool())

    # Later, get a tool by name
    search_tool = registry.get("web_search")
    if search_tool:
        result = await search_tool.execute(ctx, args)

    # List all tools for LLM prompting
    for tool in registry.list_tools():
        print(f"{tool.name}: {tool.description}")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.app.core.tool import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Central registry for tool instances.

    The registry ensures tool names are unique and provides efficient
    lookup and enumeration of registered tools.

    Attributes:
        _tools: Internal dictionary mapping tool names to tool instances.

    Example:
        registry = ToolRegistry()

        # Register tools
        registry.register(SearchTool(max_results=10))
        registry.register(FetchTool(timeout=30))

        # Check if a tool exists
        if registry.has("search"):
            tool = registry.get("search")

        # Get all tools for function calling
        schemas = [t.to_function_schema() for t in registry.list_tools()]
    """

    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """
        Register a tool instance.

        If a tool with the same name already exists, it will be replaced
        and a warning will be logged.

        Args:
            tool: The tool instance to register.

        Raises:
            ValueError: If tool is None or has no name.

        Example:
            registry.register(WebSearchTool(api_key="..."))
        """
        if tool is None:
            raise ValueError("Cannot register None as a tool")

        name = tool.name
        if not name:
            raise ValueError("Tool must have a non-empty name")

        if name in self._tools:
            logger.warning(
                "Tool '%s' is being replaced in registry",
                name,
            )

        self._tools[name] = tool
        logger.debug("Registered tool: %s", name)

    def unregister(self, name: str) -> bool:
        """
        Remove a tool from the registry by name.

        Args:
            name: The name of the tool to remove.

        Returns:
            bool: True if the tool was removed, False if it wasn't found.

        Example:
            if registry.unregister("old_tool"):
                print("Tool removed")
        """
        if name in self._tools:
            del self._tools[name]
            logger.debug("Unregistered tool: %s", name)
            return True
        return False

    def get(self, name: str) -> Tool | None:
        """
        Get a tool by name.

        Args:
            name: The name of the tool to retrieve.

        Returns:
            Tool | None: The tool instance, or None if not found.

        Example:
            tool = registry.get("web_search")
            if tool:
                result = await tool.execute(ctx, args)
        """
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """
        Check if a tool is registered.

        Args:
            name: The name of the tool to check.

        Returns:
            bool: True if the tool is registered.

        Example:
            if registry.has("wikipedia"):
                # Use wikipedia tool
                pass
        """
        return name in self._tools

    def list_tools(self) -> list[Tool]:
        """
        Get all registered tools.

        Returns:
            list[Tool]: List of all registered tool instances.

        Example:
            for tool in registry.list_tools():
                print(f"{tool.name}: {tool.description}")
        """
        return list(self._tools.values())

    def list_names(self) -> list[str]:
        """
        Get names of all registered tools.

        Returns:
            list[str]: List of tool names.

        Example:
            available = registry.list_names()
            print(f"Available tools: {', '.join(available)}")
        """
        return list(self._tools.keys())

    def get_function_schemas(self) -> list[dict]:
        """
        Get OpenAI-compatible function schemas for all tools.

        This is useful when setting up function calling with an LLM.

        Returns:
            list[dict]: List of function schemas.

        Example:
            schemas = registry.get_function_schemas()
            response = llm.chat(messages, functions=schemas)
        """
        return [tool.to_function_schema() for tool in self._tools.values()]

    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered (supports 'in' operator)."""
        return name in self._tools

    def __iter__(self):
        """Iterate over registered tools."""
        return iter(self._tools.values())
