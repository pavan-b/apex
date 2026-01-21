"""
Backward compatibility module for search tools.

This module re-exports from the new tool files to maintain backward
compatibility with existing code that imports from backend.app.tools.search.

For new code, prefer importing directly from:
- backend.app.tools.search_tool
- backend.app.tools.fetch_tool
"""

from __future__ import annotations

import asyncio
from typing import List, Tuple

from backend.app.tools.search_tool import (
    ArxivSearchTool,
    DuckDuckGoSearchTool,
    SearchHit,
    SearchLog,
    WikipediaSearchTool,
    dedupe_hits,
    generate_query_variations,
)
from backend.app.tools.fetch_tool import fetch_pages_for_hits

# Re-export everything for backward compatibility
__all__ = [
    "SearchHit",
    "SearchLog",
    "generate_query_variations",
    "run_parallel_search",
    "dedupe_hits",
]


async def run_parallel_search(
    queries: List[str],
    max_results: int,
    *,
    fetch_pages: bool = True,
    fetch_max: int = 3,
) -> Tuple[List[SearchHit], List[SearchLog]]:
    """
    Run search tools in parallel and return hits + logs.

    This function provides backward compatibility with the old search API.
    It uses the new Tool classes internally.

    Args:
        queries: Query variations to run.
        max_results: Max results per tool/query.
        fetch_pages: Whether to fetch and extract page text.
        fetch_max: Max number of pages to fetch.

    Returns:
        A tuple of (hits, logs).
    """
    from backend.app.core.observability import get_tracer

    logs: List[SearchLog] = []
    tracer = get_tracer("search")

    # 1) DuckDuckGo (parallel per query)
    async def _ddg(q: str) -> List[SearchHit]:
        from duckduckgo_search import DDGS

        def _run() -> List[SearchHit]:
            hits: List[SearchHit] = []
            with tracer.start_as_current_span("tool.duckduckgo"):
                with DDGS() as ddgs:
                    for r in ddgs.text(q, max_results=max_results):
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

        results = await asyncio.to_thread(_run)
        logs.append(SearchLog(tool="duckduckgo", query=q, results=len(results)))
        return results

    ddg_tasks = [_ddg(q) for q in queries]
    ddg_lists = await asyncio.gather(*ddg_tasks, return_exceptions=True)
    ddg_hits: List[SearchHit] = []
    for q, res in zip(queries, ddg_lists):
        if isinstance(res, Exception):
            logs.append(SearchLog(tool="duckduckgo", query=q, results=0, note=f"error: {res}"))
        else:
            ddg_hits.extend(res)

    # 2) Wikipedia (single run on base query)
    async def _wiki(q: str) -> List[SearchHit]:
        import wikipedia

        def _run() -> List[SearchHit]:
            hits: List[SearchHit] = []
            try:
                with tracer.start_as_current_span("tool.wikipedia"):
                    wikipedia.set_lang("en")
                    titles = wikipedia.search(q, results=min(5, max_results))
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
            except Exception:
                return []
            return hits

        results = await asyncio.to_thread(_run)
        logs.append(SearchLog(tool="wikipedia", query=q, results=len(results)))
        return results

    wiki_hits = await _wiki(queries[0] if queries else "")

    # 3) arXiv (single run on base query)
    async def _arxiv(q: str) -> List[SearchHit]:
        import arxiv

        def _run() -> List[SearchHit]:
            hits: List[SearchHit] = []
            try:
                with tracer.start_as_current_span("tool.arxiv"):
                    search = arxiv.Search(
                        query=q,
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
            except Exception:
                return []
            return hits

        results = await asyncio.to_thread(_run)
        logs.append(SearchLog(tool="arxiv", query=q, results=len(results)))
        return results

    arxiv_hits = await _arxiv(queries[0] if queries else "")

    hits = dedupe_hits(ddg_hits + wiki_hits + arxiv_hits)

    if fetch_pages:
        hits = await fetch_pages_for_hits(hits, max_to_fetch=fetch_max, logs=logs)

    return hits, logs
