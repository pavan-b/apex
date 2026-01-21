"""
Search tools implementing the Tool base class.

This module provides search tools for:
- DuckDuckGo web search
- Wikipedia article search
- arXiv paper search

Each tool follows the Tool abstraction, enabling:
- Registration in ToolRegistry
- Consistent execution interface
- Lifecycle hook integration
- SSE event emission

Example:
    registry = ToolRegistry()
    registry.register(DuckDuckGoSearchTool(max_results=5))
    registry.register(WikipediaSearchTool())
    registry.register(ArxivSearchTool())
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Type

import arxiv
import wikipedia
from duckduckgo_search import DDGS
from pydantic import BaseModel, Field

from backend.app.core.tool import Tool, ToolContext, ToolResult
from backend.app.core.observability import get_tracer

logger = logging.getLogger(__name__)


# ============================================================================
# Shared data structures
# ============================================================================


@dataclass
class SearchHit:
    """
    Normalized search result from any search tool.

    Attributes:
        title: Title of the result.
        url: URL of the result.
        snippet: Short description or excerpt.
        tool: Name of the tool that produced this result.
        score: Relevance score (0-1).
        raw_text: Full page text if fetched.
    """

    title: str
    url: str
    snippet: str
    tool: str
    score: float = 0.0
    raw_text: str = ""


@dataclass
class SearchLog:
    """
    Log entry for a search tool call.

    Used for UI display and debugging.

    Attributes:
        tool: Name of the tool.
        query: Query that was executed.
        results: Number of results returned.
        note: Optional note (e.g., error message).
    """

    tool: str
    query: str
    results: int
    note: str = ""


# ============================================================================
# Argument schemas
# ============================================================================


class WebSearchArgs(BaseModel):
    """Arguments for web search tools."""

    query: str = Field(description="The search query")
    max_results: int = Field(default=5, description="Maximum number of results")


class MultiQuerySearchArgs(BaseModel):
    """Arguments for searching with multiple query variations."""

    queries: List[str] = Field(description="List of query variations to search")
    max_results: int = Field(default=5, description="Maximum results per query")
    fetch_pages: bool = Field(default=True, description="Whether to fetch page content")
    fetch_max: int = Field(default=3, description="Maximum pages to fetch")


# ============================================================================
# DuckDuckGo Search Tool
# ============================================================================


class DuckDuckGoSearchTool(Tool[WebSearchArgs]):
    """
    Search the web using DuckDuckGo.

    DuckDuckGo provides privacy-focused web search without requiring an API key.
    Results include title, URL, and snippet.

    Attributes:
        default_max_results: Default number of results to return.

    Example:
        tool = DuckDuckGoSearchTool()
        result = await tool.execute(ctx, WebSearchArgs(query="python async"))
        for hit in result.data:
            print(f"{hit.title}: {hit.url}")
    """

    def __init__(self, default_max_results: int = 5) -> None:
        """
        Initialize the DuckDuckGo search tool.

        Args:
            default_max_results: Default max results if not specified.
        """
        self.default_max_results = default_max_results

    @property
    def name(self) -> str:
        """Return tool name."""
        return "duckduckgo_search"

    @property
    def description(self) -> str:
        """Return tool description."""
        return "Search the web using DuckDuckGo for general information"

    def get_args_schema(self) -> Type[WebSearchArgs]:
        """Return argument schema."""
        return WebSearchArgs

    async def execute(self, ctx: ToolContext, args: WebSearchArgs) -> ToolResult:
        """
        Execute DuckDuckGo search.

        Args:
            ctx: Tool execution context.
            args: Search arguments.

        Returns:
            ToolResult with list of SearchHit objects.
        """
        tracer = get_tracer("search")
        max_results = args.max_results or self.default_max_results

        # Emit tool call event
        ctx.emit_event({
            "type": "tool_call",
            "step": "research",
            "data": {"tool": self.name, "query": args.query},
        })

        def _run() -> List[SearchHit]:
            hits: List[SearchHit] = []
            with tracer.start_as_current_span("tool.duckduckgo"):
                with DDGS() as ddgs:
                    for r in ddgs.text(args.query, max_results=max_results):
                        hits.append(
                            SearchHit(
                                title=r.get("title") or "",
                                url=r.get("href") or r.get("url") or "",
                                snippet=r.get("body") or "",
                                tool="duckduckgo",
                                score=1.0,
                            )
                        )
            return hits

        try:
            hits = await asyncio.to_thread(_run)

            # Emit result event
            ctx.emit_event({
                "type": "tool_result",
                "step": "research",
                "data": {
                    "tool": self.name,
                    "query": args.query,
                    "results": len(hits),
                },
            })

            return ToolResult(
                success=True,
                data=hits,
                metadata={"query": args.query, "count": len(hits)},
            )

        except Exception as e:
            logger.exception("DuckDuckGo search failed: %s", e)
            return ToolResult(success=False, error=str(e))


# ============================================================================
# Wikipedia Search Tool
# ============================================================================


class WikipediaSearchTool(Tool[WebSearchArgs]):
    """
    Search Wikipedia for encyclopedia articles.

    Wikipedia provides reliable, well-sourced information on a wide range of topics.
    Results include article title, URL, and summary.

    Example:
        tool = WikipediaSearchTool()
        result = await tool.execute(ctx, WebSearchArgs(query="quantum computing"))
        for hit in result.data:
            print(f"{hit.title}: {hit.snippet}")
    """

    def __init__(self, language: str = "en", default_max_results: int = 5) -> None:
        """
        Initialize the Wikipedia search tool.

        Args:
            language: Wikipedia language code.
            default_max_results: Default max results.
        """
        self.language = language
        self.default_max_results = default_max_results

    @property
    def name(self) -> str:
        """Return tool name."""
        return "wikipedia_search"

    @property
    def description(self) -> str:
        """Return tool description."""
        return "Search Wikipedia for authoritative encyclopedia articles"

    def get_args_schema(self) -> Type[WebSearchArgs]:
        """Return argument schema."""
        return WebSearchArgs

    async def execute(self, ctx: ToolContext, args: WebSearchArgs) -> ToolResult:
        """
        Execute Wikipedia search.

        Args:
            ctx: Tool execution context.
            args: Search arguments.

        Returns:
            ToolResult with list of SearchHit objects.
        """
        tracer = get_tracer("search")
        max_results = args.max_results or self.default_max_results

        # Emit tool call event
        ctx.emit_event({
            "type": "tool_call",
            "step": "research",
            "data": {"tool": self.name, "query": args.query},
        })

        def _run() -> List[SearchHit]:
            hits: List[SearchHit] = []
            with tracer.start_as_current_span("tool.wikipedia"):
                wikipedia.set_lang(self.language)
                titles = wikipedia.search(args.query, results=min(5, max_results))
                for t in titles:
                    try:
                        page = wikipedia.page(t, auto_suggest=False)
                        summary = wikipedia.summary(t, sentences=2, auto_suggest=False)
                        hits.append(
                            SearchHit(
                                title=page.title,
                                url=page.url,
                                snippet=summary or "",
                                tool="wikipedia",
                                score=0.9,
                            )
                        )
                    except Exception:
                        continue
            return hits

        try:
            hits = await asyncio.to_thread(_run)

            # Emit result event
            ctx.emit_event({
                "type": "tool_result",
                "step": "research",
                "data": {
                    "tool": self.name,
                    "query": args.query,
                    "results": len(hits),
                },
            })

            return ToolResult(
                success=True,
                data=hits,
                metadata={"query": args.query, "count": len(hits)},
            )

        except Exception as e:
            logger.exception("Wikipedia search failed: %s", e)
            return ToolResult(success=False, error=str(e))


# ============================================================================
# arXiv Search Tool
# ============================================================================


class ArxivSearchTool(Tool[WebSearchArgs]):
    """
    Search arXiv for academic papers.

    arXiv is a preprint repository for scientific papers, particularly useful
    for technical and academic queries.

    Example:
        tool = ArxivSearchTool()
        result = await tool.execute(ctx, WebSearchArgs(query="transformer architecture"))
        for hit in result.data:
            print(f"{hit.title}: {hit.url}")
    """

    def __init__(self, default_max_results: int = 5) -> None:
        """
        Initialize the arXiv search tool.

        Args:
            default_max_results: Default max results.
        """
        self.default_max_results = default_max_results

    @property
    def name(self) -> str:
        """Return tool name."""
        return "arxiv_search"

    @property
    def description(self) -> str:
        """Return tool description."""
        return "Search arXiv for academic papers and preprints"

    def get_args_schema(self) -> Type[WebSearchArgs]:
        """Return argument schema."""
        return WebSearchArgs

    async def execute(self, ctx: ToolContext, args: WebSearchArgs) -> ToolResult:
        """
        Execute arXiv search.

        Args:
            ctx: Tool execution context.
            args: Search arguments.

        Returns:
            ToolResult with list of SearchHit objects.
        """
        tracer = get_tracer("search")
        max_results = args.max_results or self.default_max_results

        # Emit tool call event
        ctx.emit_event({
            "type": "tool_call",
            "step": "research",
            "data": {"tool": self.name, "query": args.query},
        })

        def _run() -> List[SearchHit]:
            hits: List[SearchHit] = []
            with tracer.start_as_current_span("tool.arxiv"):
                search = arxiv.Search(
                    query=args.query,
                    max_results=min(5, max_results),
                    sort_by=arxiv.SortCriterion.Relevance,
                )
                for r in search.results():
                    hits.append(
                        SearchHit(
                            title=r.title or "",
                            url=r.entry_id or "",
                            snippet=(r.summary or "")[:500],
                            tool="arxiv",
                            score=0.95,
                        )
                    )
            return hits

        try:
            hits = await asyncio.to_thread(_run)

            # Emit result event
            ctx.emit_event({
                "type": "tool_result",
                "step": "research",
                "data": {
                    "tool": self.name,
                    "query": args.query,
                    "results": len(hits),
                },
            })

            return ToolResult(
                success=True,
                data=hits,
                metadata={"query": args.query, "count": len(hits)},
            )

        except Exception as e:
            logger.exception("arXiv search failed: %s", e)
            return ToolResult(success=False, error=str(e))


# ============================================================================
# Utility functions (kept for backward compatibility with workflow)
# ============================================================================


async def generate_query_variations(query: str) -> List[str]:
    """
    Produce a small set of query variations for better coverage.

    Args:
        query: Original user query.

    Returns:
        A de-duplicated list of query variations.
    """
    q = query.strip()
    ql = q.lower()

    years = []
    if not re.search(r"\b20\d{2}\b", ql):
        years = [f"{q} 2026", f"{q} 2025"]

    variations = [
        q,
        f"{q} overview",
        f"{q} key concepts",
        f"{q} pros cons",
        f"site:wikipedia.org {q}",
        f"arXiv {q}",
        *years,
    ]
    # dedupe while preserving order
    seen = set()
    out: List[str] = []
    for v in variations:
        vv = re.sub(r"\s+", " ", v).strip()
        if vv and vv.lower() not in seen:
            seen.add(vv.lower())
            out.append(vv)
    return out[:8]


def dedupe_hits(hits: List[SearchHit]) -> List[SearchHit]:
    """
    Deduplicate search results by URL.

    Args:
        hits: Raw search results.

    Returns:
        De-duplicated results.
    """
    seen = set()
    out: List[SearchHit] = []
    for h in hits:
        u = (h.url or "").strip()
        if not u:
            continue
        key = u.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out
