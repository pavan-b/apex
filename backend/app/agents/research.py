from __future__ import annotations

"""Research agent entrypoint.

Generates query variations, runs parallel search tools, and normalizes the
results into Source objects for downstream synthesis.
"""

from typing import List, Tuple

from backend.app.schemas import Source
from backend.app.tools.search import SearchHit, SearchLog, generate_query_variations, run_parallel_search


async def research(user_message: str, max_results: int) -> Tuple[List[str], List[Source], List[SearchLog]]:
    """Perform deep research for a user query.

    Args:
        user_message: User's input question.
        max_results: Max results per tool/query.

    Returns:
        Tuple of (query variations, normalized sources, search logs).
    """
    variations = await generate_query_variations(user_message)
    hits, logs = await run_parallel_search(variations, max_results=max_results, fetch_pages=True)
    sources = [
        Source(title=h.title, url=h.url, snippet=h.snippet, tool=h.tool, score=h.score, raw_text="")
        for h in hits
    ]
    # attach raw text when available
    for s, h in zip(sources, hits):
        s.raw_text = h.raw_text or ""
    return variations, sources, logs

