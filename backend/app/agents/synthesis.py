from __future__ import annotations

"""Synthesis agent.

Builds a concise evidence-backed plan/outline from retrieved sources.
"""

from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from backend.app.schemas import Source


def _dedupe_sources(sources: List[Source]) -> List[Source]:
    """Deduplicate sources by URL.

    Args:
        sources: Retrieved sources.

    Returns:
        De-duplicated list of sources.
    """
    seen = set()
    out: List[Source] = []
    for s in sources:
        u = (s.url or "").strip().lower()
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(s)
    return out


def _evidence_table(sources: List[Source], max_items: int = 8) -> str:
    """Format a short evidence table suitable for prompting the LLM.

    Args:
        sources: Sources to include in the table.
        max_items: Maximum number of sources to include.

    Returns:
        A formatted evidence table string.
    """
    lines = []
    for s in sources[:max_items]:
        snippet = (s.snippet or "").replace("\n", " ").strip()
        if len(snippet) > 220:
            snippet = snippet[:217] + "..."
        lines.append(f"- {s.title} ({s.tool}) — {s.url}\n  - {snippet}")
    return "\n".join(lines)


async def synthesize(user_message: str, sources: List[Source], llm: BaseChatModel) -> str:
    """Produce a synthesis plan for the final answer.

    Args:
        user_message: User's input question.
        sources: Retrieved sources to ground the plan.
        llm: LLM instance used for synthesis.

    Returns:
        A synthesis plan/outline string.
    """
    sources = sorted(_dedupe_sources(sources), key=lambda s: (s.score, len(s.raw_text or "")), reverse=True)
    evidence = _evidence_table(sources, max_items=10)

    system = SystemMessage(
        content=(
            "You are a research synthesizer. Use only the provided evidence list.\n"
            "Your job is to plan the best answer: key points, what to cite, and what is uncertain.\n"
            "Do not invent sources. When referencing a source, include its URL exactly."
        )
    )
    prompt = (
        f"User question:\n{user_message}\n\n"
        f"Evidence list:\n{evidence}\n\n"
        "Produce:\n"
        "1) A short answer strategy (bullets)\n"
        "2) A 5-10 bullet outline with citations (URLs)\n"
        "3) Uncertainties / conflicts to be careful about\n"
    )
    resp = await llm.ainvoke([system, {"role": "user", "content": prompt}])
    return getattr(resp, "content", str(resp))

