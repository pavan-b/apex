"""
Tools package for the agent system.

This package contains Tool implementations for various capabilities:
- DuckDuckGoSearchTool: Web search via DuckDuckGo
- WikipediaSearchTool: Wikipedia article search
- ArxivSearchTool: arXiv paper search
- PageFetchTool: Web page content extraction

All tools follow the Tool base class pattern from backend.app.core.tool.

Example:
    from backend.app.tools import (
        DuckDuckGoSearchTool,
        WikipediaSearchTool,
        ArxivSearchTool,
        PageFetchTool,
    )
    from backend.app.core import ToolRegistry

    registry = ToolRegistry()
    registry.register(DuckDuckGoSearchTool())
    registry.register(WikipediaSearchTool())
    registry.register(ArxivSearchTool())
    registry.register(PageFetchTool())
"""

from backend.app.tools.search_tool import (
    ArxivSearchTool,
    DuckDuckGoSearchTool,
    SearchHit,
    SearchLog,
    WikipediaSearchTool,
    dedupe_hits,
    generate_query_variations,
)
from backend.app.tools.fetch_tool import (
    PageFetchTool,
    fetch_pages_for_hits,
)

__all__ = [
    # Tool classes
    "DuckDuckGoSearchTool",
    "WikipediaSearchTool",
    "ArxivSearchTool",
    "PageFetchTool",
    # Data classes
    "SearchHit",
    "SearchLog",
    # Utility functions
    "generate_query_variations",
    "dedupe_hits",
    "fetch_pages_for_hits",
]
