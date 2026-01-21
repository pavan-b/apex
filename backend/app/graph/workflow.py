"""
LangGraph workflow definition for the research+factcheck agent.

This module defines:
  - A lightweight router (easy vs deep research)
  - Research, synthesis, producer, and fact-check nodes
  - A LangGraph StateGraph tying all nodes together
  - run_workflow() as an async generator yielding SSE events

Architecture:
    The workflow uses the Vanna-style Agent architecture where run_workflow()
    yields events directly as an async generator, enabling real-time SSE
    streaming to the UI.

Flow:
    router → (easy: producer | deep: research → synthesis → producer) → factcheck → final

Example:
    async for event in run_workflow("What is AI?", config, agent):
        yield f"event: {event['type']}\\ndata: {json.dumps(event)}\\n\\n"
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, List, Literal, TypedDict

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from backend.app.agents.factcheck import verify_and_revise
from backend.app.agents.research import research as do_research
from backend.app.agents.synthesis import synthesize
from backend.app.core.observability import get_tracer

if TYPE_CHECKING:
    from backend.app.core.agent import Agent


class LGState(TypedDict, total=False):
    """
    Internal LangGraph state used by the workflow nodes.

    Attributes:
        user_message: The original user query.
        route: Routing decision ('easy' or 'deep').
        query_variations: Generated query variations for research.
        sources: List of source dictionaries from research.
        synthesis: Synthesized summary of sources.
        draft: Initial draft answer from producer.
        verified_answer: Final verified answer after fact-checking.
        critic_notes: Notes from the critic during verification.
        verify_max_rounds: Maximum verification rounds.
        best_of_n: Number of candidates for best-of-N selection.
    """

    user_message: str
    route: Literal["easy", "deep"]
    query_variations: List[str]
    sources: List[dict]
    synthesis: str
    draft: str
    verified_answer: str
    critic_notes: List[dict]
    verify_max_rounds: int
    best_of_n: int


# =============================================================================
# Event queue for async generator pattern
# =============================================================================


class EventQueue:
    """
    Simple async queue for collecting events during workflow execution.

    Used to bridge the LangGraph node execution (which uses callbacks)
    with the async generator pattern expected by the Agent.
    """

    def __init__(self) -> None:
        """Initialize empty event queue."""
        self._events: List[dict] = []
        self._queue: asyncio.Queue[dict] = asyncio.Queue()

    async def put(self, event: dict) -> None:
        """Add an event to the queue."""
        await self._queue.put(event)

    async def get(self) -> dict:
        """Get the next event from the queue."""
        return await self._queue.get()

    def put_nowait(self, event: dict) -> None:
        """Add an event without waiting."""
        self._queue.put_nowait(event)


# =============================================================================
# Utility functions
# =============================================================================


def _is_deep_query(q: str) -> bool:
    """
    Heuristic to decide whether a query needs deep research.

    Args:
        q: User question.

    Returns:
        True if the query should go through deep research.
    """
    ql = q.lower()
    if any(
        k in ql
        for k in [
            "latest",
            "2025",
            "2026",
            "news",
            "compare",
            "vs ",
            "benchmark",
            "paper",
            "sources",
            "citations",
        ]
    ):
        return True
    if len(q.split()) >= 18:
        return True
    if "?" in q and any(
        k in ql for k in ["why", "how", "should", "tradeoff", "pros", "cons"]
    ):
        return True
    return False


def _get_llm(config: Dict[str, Any]) -> ChatOllama:
    """
    Create the Ollama chat model with runtime configuration.

    Args:
        config: Runtime configuration dictionary.

    Returns:
        Configured ChatOllama instance.
    """
    return ChatOllama(
        model=config.get("ollama_model", os.getenv(
            "OLLAMA_MODEL", "qwen3:8b")),
        base_url=config.get("ollama_base_url", os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434")),
        temperature=float(config.get("ollama_temperature",
                          os.getenv("OLLAMA_TEMPERATURE", "0.2"))),
    )


# =============================================================================
# LangGraph nodes
# =============================================================================


async def _router_node(state: LGState, config: RunnableConfig) -> LGState:
    """
    Choose easy vs deep routing and emit UI events.

    Args:
        state: Current LangGraph state.
        config: Runtime LangGraph config with context.

    Returns:
        Partial state updates containing the chosen route.
    """
    emit = config["configurable"]["emit"]
    q = state["user_message"]
    route: Literal["easy", "deep"] = "deep" if _is_deep_query(q) else "easy"

    tracer = get_tracer("router")
    with tracer.start_as_current_span("route_decision"):
        await emit({
            "type": "step_started",
            "step": "router",
            "data": {"input": q[:200], "decision": route},
        })
        await emit({
            "type": "step_finished",
            "step": "router",
            "data": {"decision": route, "output": f"Route: {route}"},
        })

    return {"route": route}


async def _research_node(state: LGState, config: RunnableConfig) -> LGState:
    """
    Run parallel research tools and normalize sources.

    Args:
        state: Current LangGraph state.
        config: Runtime LangGraph config with context.

    Returns:
        Partial state updates with query variations and sources.
    """
    emit = config["configurable"]["emit"]
    cfg: Dict[str, Any] = config["configurable"]["cfg"]

    await emit({
        "type": "step_started",
        "step": "research",
        "data": {"input": state["user_message"][:200]},
    })

    tracer = get_tracer("research")
    with tracer.start_as_current_span("research.run"):
        max_results = int(cfg.get("search_max_results", 5))
        variations, sources, logs = await do_research(
            state["user_message"], max_results=max_results
        )

    # Emit tool calls
    for lg in logs:
        await emit({
            "type": "tool_call",
            "step": "research",
            "data": {"tool": lg.tool, "query": lg.query},
        })
        await emit({
            "type": "tool_result",
            "step": "research",
            "data": {
                "tool": lg.tool,
                "query": lg.query,
                "results": lg.results,
                "note": lg.note,
            },
        })

    await emit({
        "type": "step_finished",
        "step": "research",
        "data": {
            "sources_count": len(sources),
            "query_variations": variations,
            "output": f"Found {len(sources)} sources from {len(logs)} tool calls",
        },
    })

    return {
        "query_variations": variations,
        "sources": [s.model_dump() for s in sources],
    }


async def _synthesis_node(state: LGState, config: RunnableConfig) -> LGState:
    """
    Summarize sources into an evidence-backed plan/outline.

    Args:
        state: Current LangGraph state.
        config: Runtime LangGraph config with context.

    Returns:
        Partial state updates with synthesis text.
    """
    emit = config["configurable"]["emit"]
    llm: ChatOllama = config["configurable"]["llm"]

    await emit({
        "type": "step_started",
        "step": "synthesis",
        "data": {"input": f"{len(state.get('sources', []))} sources"},
    })

    from backend.app.schemas import Source

    sources = [Source(**d) for d in state.get("sources", [])]

    tracer = get_tracer("synthesis")
    with tracer.start_as_current_span("synthesis.run"):
        synthesis = await synthesize(state["user_message"], sources=sources, llm=llm)

    await emit({
        "type": "step_finished",
        "step": "synthesis",
        "data": {
            "output": synthesis[:500] + ("..." if len(synthesis) > 500 else ""),
            "len": len(synthesis),
        },
    })

    return {"synthesis": synthesis}


async def _producer_node(state: LGState, config: RunnableConfig) -> LGState:
    """
    Generate the initial draft response from the LLM.

    Args:
        state: Current LangGraph state.
        config: Runtime LangGraph config with context.

    Returns:
        Partial state updates with the draft answer.
    """
    emit = config["configurable"]["emit"]
    llm: ChatOllama = config["configurable"]["llm"]

    await emit({
        "type": "step_started",
        "step": "producer",
        "data": {"input": f"Question + {len(state.get('synthesis', ''))} chars synthesis"},
    })

    tracer = get_tracer("producer")

    system = SystemMessage(
        content=(
            "You are a helpful research assistant. If you use any factual claims, keep them conservative.\n"
            "When sources are provided, include inline citations as URLs.\n"
            "If sources are missing, clearly label statements as general knowledge or uncertainty."
        )
    )

    prompt = (
        f"User question:\n{state['user_message']}\n\n"
        f"Synthesis notes (may be empty):\n{state.get('synthesis', '')}\n\n"
        "Write the best possible answer. If you make multiple claims, structure them with headings.\n"
    )

    with tracer.start_as_current_span("producer.generate"):
        resp = await llm.ainvoke([system, {"role": "user", "content": prompt}])
    draft = getattr(resp, "content", str(resp))

    await emit({
        "type": "step_finished",
        "step": "producer",
        "data": {
            "output": draft[:500] + ("..." if len(draft) > 500 else ""),
            "len": len(draft),
        },
    })

    return {"draft": draft}


async def _factcheck_node(state: LGState, config: RunnableConfig) -> LGState:
    """
    Run the multi-critic verification loop and produce a final answer.

    Args:
        state: Current LangGraph state.
        config: Runtime LangGraph config with context.

    Returns:
        Partial state updates with verified answer and notes.
    """
    emit = config["configurable"]["emit"]
    cfg: Dict[str, Any] = config["configurable"]["cfg"]
    llm: ChatOllama = config["configurable"]["llm"]

    max_rounds = int(cfg.get("verify_max_rounds", 3))
    best_of_n = int(cfg.get("best_of_n", 3))

    await emit({
        "type": "step_started",
        "step": "factcheck",
        "data": {
            "input": f"Draft: {len(state.get('draft', ''))} chars",
            "max_rounds": max_rounds,
            "best_of_n": best_of_n,
        },
    })

    from backend.app.schemas import Source

    sources = [Source(**d) for d in state.get("sources", [])]
    tracer = get_tracer("factcheck")

    async def _emit_inner(kind: str, data: Dict[str, Any]) -> None:
        """Emit inner verification events."""
        await emit({
            "type": kind,
            "step": "factcheck",
            "data": data,
        })

    with tracer.start_as_current_span("factcheck.run"):
        verified, notes = await verify_and_revise(
            user_message=state["user_message"],
            draft=state.get("draft", ""),
            sources=sources,
            max_rounds=max_rounds,
            best_of_n=best_of_n,
            llm=llm,
            emit=_emit_inner,
        )

    await emit({
        "type": "step_finished",
        "step": "factcheck",
        "data": {
            "output": f"Verified answer: {len(verified)} chars, {len(notes)} critic notes",
            "notes": len(notes),
        },
    })

    return {"verified_answer": verified, "critic_notes": notes}


async def _final_node(state: LGState, config: RunnableConfig) -> LGState:
    """
    Emit the final answer to the UI stream.

    Args:
        state: Current LangGraph state.
        config: Runtime LangGraph config with context.

    Returns:
        Empty state update.
    """
    emit = config["configurable"]["emit"]
    answer = state.get("verified_answer") or state.get("draft") or ""

    await emit({
        "type": "final",
        "step": "final",
        "data": {"answer": answer},
    })

    return {}


# =============================================================================
# Graph builder
# =============================================================================


def _build_graph() -> Any:
    """
    Compile and return the LangGraph StateGraph.

    Returns:
        A compiled StateGraph ready for invocation.
    """
    g = StateGraph(LGState)
    g.add_node("router", _router_node)
    g.add_node("research", _research_node)
    g.add_node("synthesis_node", _synthesis_node)
    g.add_node("producer", _producer_node)
    g.add_node("factcheck", _factcheck_node)
    g.add_node("final", _final_node)

    g.set_entry_point("router")
    g.add_conditional_edges(
        "router", lambda s: s["route"], {
            "easy": "producer", "deep": "research"}
    )
    g.add_edge("research", "synthesis_node")
    g.add_edge("synthesis_node", "producer")
    g.add_edge("producer", "factcheck")
    g.add_edge("factcheck", "final")
    g.add_edge("final", END)

    return g.compile()


# =============================================================================
# Main entry points
# =============================================================================


async def run_workflow(
    message: str,
    config: Dict[str, Any],
    agent: "Agent | None" = None,
) -> AsyncIterator[dict]:
    """
    Orchestrate the full LangGraph workflow as an async generator.

    This function runs the complete research + verification pipeline:
    1. Router decides if query needs deep research or direct answer
    2. Research phase (if deep): parallel search across DDG, Wikipedia, arXiv
    3. Synthesis: builds evidence-backed outline
    4. Producer: generates draft answer
    5. FactCheck: multi-critic verification loop (up to max_rounds)
    6. Final: emits the verified answer

    Args:
        message: User's input message/question.
        config: Runtime configuration dict with keys:
            - verify_max_rounds: Max verification iterations (default: 3)
            - best_of_n: Candidates for Best-of-N selection (default: 3)
            - search_max_results: Max results per search tool (default: 5)
        agent: Optional Agent instance (provides LlmService access).

    Yields:
        dict: SSE event dictionaries with keys:
            - type: Event type (step_started, tool_call, final, etc.)
            - step: Workflow step name
            - data: Event-specific payload

    Example:
        async for event in run_workflow("What is AI?", config, agent):
            print(f"[{event['step']}] {event['type']}")
    """
    # Event queue for collecting events
    event_queue: asyncio.Queue[dict | None] = asyncio.Queue()

    async def emit(event: dict) -> None:
        """Put event in queue."""
        await event_queue.put(event)

    # Get LLM - prefer agent's llm_service if available
    if agent and agent.llm_service:
        llm = agent.llm_service.get_client()
    else:
        llm = _get_llm(config)

    # Build merged config
    merged_config = {
        "ollama_model": os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "ollama_temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.2")),
        "verify_max_rounds": config.get("verify_max_rounds", int(os.getenv("VERIFY_MAX_ROUNDS", "3"))),
        "best_of_n": config.get("best_of_n", int(os.getenv("BEST_OF_N", "3"))),
        "search_max_results": config.get("search_max_results", int(os.getenv("SEARCH_MAX_RESULTS", "5"))),
        **config,
    }

    graph = _build_graph()
    init: LGState = {
        "user_message": message,
        "route": "deep",
        "query_variations": [],
        "sources": [],
        "synthesis": "",
        "draft": "",
        "verified_answer": "",
        "critic_notes": [],
        "verify_max_rounds": int(merged_config["verify_max_rounds"]),
        "best_of_n": int(merged_config["best_of_n"]),
    }

    # Emit workflow start
    await emit({
        "type": "step_started",
        "step": "workflow",
        "data": {"input": message[:200]},
    })

    # Run graph in background task
    async def _run_graph() -> None:
        try:
            tracer = get_tracer("workflow")
            with tracer.start_as_current_span("workflow.run"):
                await graph.ainvoke(
                    init,
                    config={
                        "configurable": {
                            "emit": emit,
                            "cfg": merged_config,
                            "llm": llm,
                        }
                    },
                )
        except Exception as e:
            await emit({
                "type": "error",
                "step": "workflow",
                "data": {"error": str(e)},
            })
        finally:
            await emit({
                "type": "step_finished",
                "step": "workflow",
                "data": {},
            })
            await event_queue.put(None)  # Signal end

    # Start graph execution
    task = asyncio.create_task(_run_graph())

    # Yield events as they arrive
    while True:
        event = await event_queue.get()
        if event is None:
            break
        yield event

    # Ensure task is done
    await task
