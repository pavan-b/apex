"""
Page fetch tool for extracting content from web pages.

This module provides a tool for fetching web pages and extracting their
visible text content. Useful for enriching search results with full page text.

Example:
    tool = PageFetchTool()
    result = await tool.execute(ctx, FetchArgs(url="https://example.com"))
    print(result.data["text"][:500])
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import List, Type

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from backend.app.core.tool import Tool, ToolContext, ToolResult
from backend.app.core.observability import get_tracer
from backend.app.tools.search_tool import SearchHit, SearchLog

logger = logging.getLogger(__name__)


class FetchArgs(BaseModel):
    """Arguments for the page fetch tool."""

    url: str = Field(description="URL of the page to fetch")
    max_chars: int = Field(default=4000, description="Maximum characters to extract")


class BatchFetchArgs(BaseModel):
    """Arguments for batch page fetching."""

    urls: List[str] = Field(description="List of URLs to fetch")
    max_chars: int = Field(default=4000, description="Maximum characters per page")


class PageFetchTool(Tool[FetchArgs]):
    """
    Fetch and extract text content from web pages.

    This tool:
    - Fetches the HTML content of a URL
    - Removes script, style, and noscript tags
    - Extracts visible text
    - Limits output to max_chars

    Useful for enriching search results with full page content for
    better context in synthesis and fact-checking.

    Attributes:
        timeout: HTTP request timeout in seconds.
        user_agent: User-Agent header for requests.

    Example:
        tool = PageFetchTool(timeout=15)
        result = await tool.execute(ctx, FetchArgs(
            url="https://en.wikipedia.org/wiki/Python",
            max_chars=2000
        ))
        if result.success:
            print(result.data["text"])
    """

    def __init__(
        self,
        timeout: float = 10.0,
        user_agent: str = "apex-research-agent/1.0",
    ) -> None:
        """
        Initialize the page fetch tool.

        Args:
            timeout: HTTP request timeout in seconds.
            user_agent: User-Agent header for requests.
        """
        self.timeout = timeout
        self.user_agent = user_agent

    @property
    def name(self) -> str:
        """Return tool name."""
        return "page_fetch"

    @property
    def description(self) -> str:
        """Return tool description."""
        return "Fetch and extract text content from a web page URL"

    def get_args_schema(self) -> Type[FetchArgs]:
        """Return argument schema."""
        return FetchArgs

    async def execute(self, ctx: ToolContext, args: FetchArgs) -> ToolResult:
        """
        Fetch a web page and extract its text content.

        Args:
            ctx: Tool execution context.
            args: Fetch arguments.

        Returns:
            ToolResult with extracted text in data["text"].
        """
        tracer = get_tracer("search")

        # Emit tool call event
        ctx.emit_event({
            "type": "tool_call",
            "step": "research",
            "data": {"tool": self.name, "query": args.url},
        })

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": self.user_agent},
            ) as client:
                with tracer.start_as_current_span("tool.fetch"):
                    response = await client.get(args.url)

                    if response.status_code >= 400:
                        return ToolResult(
                            success=False,
                            error=f"HTTP {response.status_code}",
                            metadata={"url": args.url, "status": response.status_code},
                        )

                    # Parse and extract text
                    soup = BeautifulSoup(response.text, "lxml")
                    for tag in soup(["script", "style", "noscript"]):
                        tag.extract()
                    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
                    text = text[:args.max_chars]

            # Emit result event
            ctx.emit_event({
                "type": "tool_result",
                "step": "research",
                "data": {
                    "tool": self.name,
                    "query": args.url,
                    "results": 1,
                    "chars": len(text),
                },
            })

            return ToolResult(
                success=True,
                data={"url": args.url, "text": text},
                metadata={"url": args.url, "chars": len(text)},
            )

        except Exception as e:
            logger.exception("Page fetch failed for %s: %s", args.url, e)
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"url": args.url},
            )


async def fetch_pages_for_hits(
    hits: List[SearchHit],
    max_to_fetch: int = 3,
    logs: List[SearchLog] | None = None,
    timeout: float = 10.0,
) -> List[SearchHit]:
    """
    Fetch page content for a list of search hits.

    This is a utility function that enriches SearchHit objects with
    their full page text. Only fetches DuckDuckGo results (Wikipedia
    and arXiv already have good snippets).

    Args:
        hits: List of search hits to enrich.
        max_to_fetch: Maximum number of pages to fetch.
        logs: Optional list to append SearchLog entries.
        timeout: HTTP timeout in seconds.

    Returns:
        The same hits list with raw_text populated where available.
    """
    if logs is None:
        logs = []

    # Filter to fetchable URLs
    to_fetch: List[SearchHit] = []
    for h in hits:
        if len(to_fetch) >= max_to_fetch:
            break
        if h.tool != "duckduckgo":
            continue
        if not h.url.startswith("http"):
            continue
        if h.url.lower().endswith(".pdf"):
            continue
        to_fetch.append(h)

    if not to_fetch:
        return hits

    tracer = get_tracer("search")

    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "apex-research-agent/1.0"},
    ) as client:
        async def _fetch_one(h: SearchHit) -> None:
            try:
                with tracer.start_as_current_span("tool.fetch"):
                    r = await client.get(h.url)
                    if r.status_code >= 400:
                        logs.append(SearchLog(
                            tool="fetch",
                            query=h.url,
                            results=0,
                            note=f"http {r.status_code}",
                        ))
                        return

                    soup = BeautifulSoup(r.text, "lxml")
                    for tag in soup(["script", "style", "noscript"]):
                        tag.extract()
                    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
                    h.raw_text = text[:4000]
                    logs.append(SearchLog(tool="fetch", query=h.url, results=1))

            except Exception as e:
                logs.append(SearchLog(
                    tool="fetch",
                    query=h.url,
                    results=0,
                    note=f"error: {e}",
                ))

        await asyncio.gather(*[_fetch_one(h) for h in to_fetch])

    return hits
