# Apex


**Apex** is a production-ready AI research assistant powered by **Ollama (qwen3:8b)**, **LangGraph**, and **FastAPI**. Features intelligent query routing, parallel multi-source research, evidence synthesis, and robust hallucination detection using a multi-critic verification system.

> *"Apex"* — reaching the peak of AI-assisted research through verified, factual answers.

## Features

- **Intelligent Routing** — Simple queries answered directly; complex ones trigger deep research
- **Parallel Research** — Concurrent searches across DuckDuckGo, Wikipedia, and arXiv with query variations
- **Evidence Synthesis** — Deduplicates sources and builds structured evidence tables
- **Multi-Critic Verification** — Producer-Critic pattern with Chain-of-Verification (CoVe) and Best-of-N selection
- **Real-time UI** — Streamlit frontend with SSE streaming, step accordions, and progress tracking
- **LLM Observability** — Optional Arize Phoenix integration for tracing and debugging
- **Extensible Architecture** — Vanna 2.0-inspired patterns for tools, middleware, and lifecycle hooks

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Request Lifecycle](#request-lifecycle)
3. [LangGraph Workflow](#langgraph-workflow)
4. [Component Reference](#component-reference)
5. [API Reference](#api-reference)
6. [Getting Started](#getting-started)
7. [Configuration](#configuration)
8. [Observability](#observability)
9. [Extending the System](#extending-the-system)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              STREAMLIT UI                                    │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │ Chat Input  │  │ Step Display │  │  Accordions │  │ Fast Mode Toggle│   │
│  └──────┬──────┘  └──────────────┘  └─────────────┘  └─────────────────┘   │
└─────────┼───────────────────────────────────────────────────────────────────┘
          │ POST /api/v1/chat/sse (SSE Stream)
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             FASTAPI BACKEND                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                           HTTP Layer                                  │  │
│  │  routes.py ──► ChatHandler ──► Agent.run() ──► SSE EventSourceResponse│  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                          Core Layer                                   │  │
│  │  ┌─────────┐  ┌────────────┐  ┌────────────┐  ┌─────────────────┐   │  │
│  │  │  Agent  │  │ LlmService │  │ToolRegistry│  │ LifecycleHooks  │   │  │
│  │  │ (orch.) │  │(+middleware)│  │            │  │                 │   │  │
│  │  └────┬────┘  └─────┬──────┘  └──────┬─────┘  └────────┬────────┘   │  │
│  └───────┼─────────────┼────────────────┼─────────────────┼────────────┘  │
│          │             │                │                 │                │
│  ┌───────▼─────────────▼────────────────▼─────────────────▼────────────┐  │
│  │                        LangGraph Workflow                            │  │
│  │  router ──► research ──► synthesis ──► producer ──► factcheck ──► final│
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                          Tools Layer                                  │  │
│  │  ┌────────────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────┐ │  │
│  │  │DuckDuckGoSearch│  │WikipediaSearch│  │ ArxivSearch│  │PageFetch │ │  │
│  │  └────────────────┘  └──────────────┘  └────────────┘  └──────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────┐     ┌─────────────────────┐
│    Ollama Server    │     │   Arize Phoenix     │
│   (qwen3:8b LLM)    │     │   (Observability)   │
└─────────────────────┘     └─────────────────────┘
```

### Directory Structure

```
backend/app/
├── main.py                 # FastAPI app entry point, wiring
├── schemas.py              # Pydantic models (ChatRequest, SSEEvent, etc.)
│
├── core/                   # Core abstractions (Vanna 2.0 pattern)
│   ├── agent.py            # Agent: central orchestrator
│   ├── llm_service.py      # LlmService: LLM abstraction + middleware chain
│   ├── middleware.py       # LlmMiddleware base + LoggingMiddleware
│   ├── lifecycle.py        # LifecycleHooks: request lifecycle extension points
│   ├── registry.py         # ToolRegistry: tool discovery and management
│   ├── tool.py             # Tool base class, ToolContext, ToolResult
│   └── observability/      # Observability provider pattern
│       ├── base.py         # ObservabilityProvider ABC
│       ├── phoenix_provider.py  # Arize Phoenix implementation
│       └── factory.py      # Provider factory and global access
│
├── server/                 # HTTP layer
│   ├── routes.py           # Route registration (POST /api/v1/chat/sse)
│   └── handler.py          # ChatHandler: request → SSE stream
│
├── graph/                  # LangGraph workflow
│   └── workflow.py         # StateGraph definition and run_workflow()
│
├── agents/                 # Domain logic (research, synthesis, verification)
│   ├── research.py         # Query variations + parallel search
│   ├── synthesis.py        # Source deduplication + evidence table
│   └── factcheck.py        # Multi-critic CoVe verification loop
│
└── tools/                  # Tool implementations
    ├── search_tool.py      # DuckDuckGoSearchTool, WikipediaSearchTool, ArxivSearchTool
    ├── fetch_tool.py       # PageFetchTool (URL content extraction)
    └── search.py           # Backward-compatible exports

ui/
└── streamlit_app.py        # Streamlit frontend with SSE consumption
```

---

## Request Lifecycle

Every chat request flows through these stages:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         REQUEST LIFECYCLE                                 │
└──────────────────────────────────────────────────────────────────────────┘

1. HTTP REQUEST
   └─► POST /api/v1/chat/sse { message, best_of_n?, verify_max_rounds? }

2. ROUTE HANDLER (routes.py)
   └─► Validates request via Pydantic
   └─► Calls ChatHandler.handle_streaming()

3. CHAT HANDLER (handler.py)
   └─► Creates RequestContext with config
   └─► Invokes LifecycleHooks.on_request_start()
   └─► Calls Agent.run(message, config)
   └─► Streams SSE events to client

4. AGENT ORCHESTRATION (agent.py)
   └─► Builds LangGraph initial state
   └─► Invokes run_workflow() generator
   └─► Yields SSEEvent objects for each step

5. LANGGRAPH WORKFLOW (workflow.py)
   └─► Executes nodes: router → research → synthesis → producer → factcheck → final
   └─► Each node yields events (step_started, step_finished, tool_call, etc.)

6. LIFECYCLE HOOKS
   └─► on_request_end() called with final answer
   └─► on_error() called if exception occurs

7. SSE RESPONSE
   └─► Events streamed as `data: {...}\n\n`
   └─► Final `done` event signals completion
```

### Lifecycle Hooks

Extension points for custom logic at key moments:

| Hook | When Called | Use Cases |
|------|-------------|-----------|
| `on_request_start(message, config)` | Before workflow starts | Logging, quota checks, request enrichment |
| `on_request_end(final_answer)` | After successful completion | Metrics, caching, notifications |
| `on_error(error, context)` | On any exception | Error tracking, alerting |
| `on_tool_start(tool_name, args)` | Before tool execution | Tool-level logging |
| `on_tool_end(tool_name, result)` | After tool execution | Result caching |

---

## LangGraph Workflow

The core workflow is a LangGraph `StateGraph` that orchestrates the research and verification pipeline:

```
                              ┌─────────────┐
                              │   START     │
                              └──────┬──────┘
                                     │
                                     ▼
                              ┌─────────────┐
                              │   ROUTER    │
                              │ (classify)  │
                              └──────┬──────┘
                                     │
                     ┌───────────────┴───────────────┐
                     │                               │
              is_deep=True                    is_deep=False
                     │                               │
                     ▼                               │
              ┌─────────────┐                        │
              │  RESEARCH   │                        │
              │ (parallel   │                        │
              │  searches)  │                        │
              └──────┬──────┘                        │
                     │                               │
                     ▼                               │
              ┌─────────────┐                        │
              │  SYNTHESIS  │                        │
              │ (dedupe +   │                        │
              │  evidence)  │                        │
              └──────┬──────┘                        │
                     │                               │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                              ┌─────────────┐
                              │  PRODUCER   │
                              │ (draft      │
                              │  answer)    │
                              └──────┬──────┘
                                     │
                                     ▼
                              ┌─────────────┐
                              │  FACTCHECK  │◄────────┐
                              │ (verify +   │         │
                              │  revise)    │    retry (max 3)
                              └──────┬──────┘         │
                                     │                │
                              passed? ────No──────────┘
                                     │
                                    Yes
                                     │
                                     ▼
                              ┌─────────────┐
                              │    FINAL    │
                              │ (emit       │
                              │  answer)    │
                              └──────┬──────┘
                                     │
                                     ▼
                              ┌─────────────┐
                              │     END     │
                              └─────────────┘
```

### Node Details

| Node | Description | Key Logic |
|------|-------------|-----------|
| **router** | Classifies query complexity | LLM prompt → `is_deep: true/false` |
| **research** | Parallel multi-source search | Generates query variations, searches DDG/Wiki/arXiv concurrently |
| **synthesis** | Evidence consolidation | Deduplicates sources, builds evidence table with URL + snippet |
| **producer** | Initial answer draft | LLM generates answer from evidence (or direct answer if simple) |
| **factcheck** | Multi-critic verification | CoVe questions → parallel critics → Best-of-N selection → revision loop |
| **final** | Output emission | Emits `final` SSE event with verified answer |

### Factcheck Verification Flow

The factcheck node implements a sophisticated verification system:

```
┌────────────────────────────────────────────────────────────────────┐
│                    FACTCHECK VERIFICATION                          │
└────────────────────────────────────────────────────────────────────┘

For each round (up to VERIFY_MAX_ROUNDS):

1. DECOMPOSITION
   └─► Break answer into atomic claims

2. COVE QUESTION GENERATION (parallel)
   └─► Generate verification questions for each claim
   └─► Questions designed to catch hallucinations

3. MULTI-CRITIC EVALUATION (parallel)
   └─► N critics independently evaluate (BEST_OF_N)
   └─► Each critic scores: factual accuracy, source support, completeness

4. CONSENSUS CHECK
   └─► If majority approve → PASS
   └─► If issues found → collect feedback

5. REVISION (if needed)
   └─► LLM revises answer based on critic feedback
   └─► Loop back to step 1

6. BEST-OF-N SELECTION
   └─► If multiple revisions, select highest-scoring version
```

---

## Component Reference

### Agent (`core/agent.py`)

The central orchestrator that wires together all components:

```python
class Agent:
    """
    Central orchestrator for the research chat system.
    
    Responsibilities:
    - Manages LlmService, ToolRegistry, and LifecycleHooks
    - Provides unified run() method for workflow execution
    - Handles configuration merging (request + defaults)
    """
    
    def __init__(
        self,
        llm_service: LlmService,
        tool_registry: ToolRegistry,
        lifecycle_hooks: LifecycleHooks | None = None,
    ): ...
    
    async def run(
        self,
        message: str,
        config: dict | None = None,
    ) -> AsyncGenerator[SSEEvent, None]:
        """Execute workflow and yield SSE events."""
```

### LlmService (`core/llm_service.py`)

Abstraction over LLM providers with middleware support:

```python
class LlmService:
    """
    LLM abstraction with middleware chain support.
    
    Middleware is executed in order:
    - Pre-processing: logging, validation, caching lookup
    - LLM call: actual Ollama invocation
    - Post-processing: response logging, cache storage, metrics
    """
    
    async def invoke(
        self,
        messages: list[dict],
        **kwargs,
    ) -> str:
        """Invoke LLM with middleware chain."""
    
    async def invoke_structured(
        self,
        messages: list[dict],
        schema: type[BaseModel],
        **kwargs,
    ) -> BaseModel:
        """Invoke LLM with structured output parsing."""
```

### ToolRegistry (`core/registry.py`)

Manages tool discovery and lookup:

```python
class ToolRegistry:
    """
    Registry for tool discovery and management.
    
    Supports:
    - Registration by name
    - Lookup by name
    - Iteration over all tools
    - Schema generation for LLM function calling
    """
    
    def register(self, tool: Tool) -> None: ...
    def get(self, name: str) -> Tool | None: ...
    def list_tools(self) -> list[Tool]: ...
    def get_schemas(self) -> list[dict]: ...
```

### Tool Base Class (`core/tool.py`)

Base class for all tools:

```python
class Tool(ABC, Generic[TArgs]):
    """
    Abstract base class for tools.
    
    Subclasses must implement:
    - name: Unique identifier
    - description: LLM-facing description
    - get_args_schema(): Pydantic model for arguments
    - execute(): Async execution logic
    """
    
    @abstractmethod
    async def execute(
        self,
        ctx: ToolContext,
        args: TArgs,
    ) -> ToolResult: ...
```

---

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/chat/sse` | POST | Main chat endpoint with SSE streaming |
| `/api/v1/health` | GET | Health check with model info |
| `/health` | GET | Simple health check |

### POST /api/v1/chat/sse

**Request:**

```json
{
  "message": "What are the latest advances in quantum computing?",
  "best_of_n": 3,
  "verify_max_rounds": 3
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `message` | string | required | User query |
| `best_of_n` | int | 3 | Number of critic evaluations per round |
| `verify_max_rounds` | int | 3 | Maximum verification iterations |

**Response:** Server-Sent Events stream

```
data: {"event": "step_started", "data": {"step": "router", "input": "..."}}

data: {"event": "step_finished", "data": {"step": "router", "output": {"is_deep": true}}}

data: {"event": "tool_call", "data": {"tool": "duckduckgo_search", "args": {"query": "..."}}}

data: {"event": "tool_result", "data": {"tool": "duckduckgo_search", "result": [...]}}

data: {"event": "critic_feedback", "data": {"round": 1, "critic": 1, "passed": false, "feedback": "..."}}

data: {"event": "revision_selected", "data": {"round": 1, "score": 0.85}}

data: {"event": "final", "data": {"answer": "...", "sources": [...]}}

data: {"event": "done", "data": {}}
```

### SSE Event Types

| Event | Description | Data Fields |
|-------|-------------|-------------|
| `step_started` | Workflow step begins | `step`, `input` |
| `step_finished` | Workflow step ends | `step`, `output` |
| `tool_call` | Tool invocation | `tool`, `args` |
| `tool_result` | Tool response | `tool`, `result` |
| `critic_feedback` | Critic evaluation | `round`, `critic`, `passed`, `feedback` |
| `revision_selected` | Best revision chosen | `round`, `score` |
| `final` | Final answer | `answer`, `sources` |
| `error` | Error occurred | `message`, `step` |
| `done` | Stream complete | — |

---

## Getting Started

### Prerequisites

1. **Python 3.11 or 3.12** (3.14 may have build issues with some dependencies)
2. **Ollama** installed and running
3. **qwen3:8b model** pulled

```bash
# Install Ollama (see https://ollama.ai)
# Then pull the model:
ollama pull qwen3:8b
```

### Installation

**Using uv (recommended):**

```bash
# Install uv
python -m pip install -U uv

# Create venv and sync dependencies
uv venv --python 3.12
uv sync
```

**Using pip (fallback):**

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Running

**Quick start with shorthand commands:**

```bash
# Show available commands
uv run dev

# Terminal 1: Start Phoenix (optional, for tracing)
uv run phoenix-serve

# Terminal 2: Start backend
uv run backend

# Terminal 3: Start UI
uv run ui
```

**Manual commands:**

```bash
# Phoenix (optional)
uv run phoenix serve

# Backend
uv run uvicorn backend.app.main:app --reload --port 8000

# UI
uv run streamlit run ui/streamlit_app.py --server.port 8501
```

**Access:**
- Streamlit UI: http://localhost:8501
- FastAPI docs: http://localhost:8000/docs
- Phoenix UI: http://localhost:6006 (if running)

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# LLM Configuration
OLLAMA_MODEL=qwen3:8b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TEMPERATURE=0.2

# Workflow Configuration
BEST_OF_N=3                 # Number of critics per verification round
VERIFY_MAX_ROUNDS=3         # Maximum verification iterations
SEARCH_MAX_RESULTS=5        # Results per search tool

# Phoenix Observability (optional)
PHOENIX_OTEL_ENDPOINT=http://localhost:6006
PHOENIX_SERVICE_NAME=apex
PHOENIX_API_KEY=            # Optional, for Phoenix Cloud
```

### Fast Mode (UI)

The Streamlit UI includes a "Fast Mode" toggle that reduces:
- `BEST_OF_N`: 3 → 1
- `VERIFY_MAX_ROUNDS`: 3 → 1

This significantly speeds up responses at the cost of less thorough verification.

---

## Observability

### Arize Phoenix Integration

Phoenix provides LLM observability with trace visualization, helping debug and optimize the workflow.

**Setup:**

1. Start Phoenix before the backend:
   ```bash
   uv run phoenix-serve
   ```

2. Configure endpoint in `.env`:
   ```bash
   PHOENIX_OTEL_ENDPOINT=http://localhost:6006
   ```

3. Start backend (tracing auto-enables if Phoenix is reachable):
   ```bash
   uv run backend
   ```

4. View traces at http://localhost:6006

**What's traced:**
- All LLM calls (inputs, outputs, latency, tokens)
- Tool executions
- LangGraph node transitions
- Custom spans for research/synthesis/factcheck

**Architecture:**

```
┌─────────────────────────────────────────────────────────┐
│                 Observability Layer                     │
├─────────────────────────────────────────────────────────┤
│  ObservabilityProvider (ABC)                            │
│    └─► ArizePhoenixProvider                             │
│          └─► phoenix.otel.register()                    │
│          └─► LangChainInstrumentor                      │
│          └─► OpenTelemetry TracerProvider               │
├─────────────────────────────────────────────────────────┤
│  Usage:                                                 │
│    from backend.app.core.observability import get_tracer│
│    tracer = get_tracer()                                │
│    with tracer.start_as_current_span("my-span"):        │
│        ...                                              │
└─────────────────────────────────────────────────────────┘
```

---

## Extending the System

### Adding a New Tool

```python
# backend/app/tools/my_tool.py
from backend.app.core import Tool, ToolContext, ToolResult
from pydantic import BaseModel, Field

class MyToolArgs(BaseModel):
    """Arguments for MyTool."""
    query: str = Field(..., description="Search query")
    limit: int = Field(5, description="Max results")

class MyTool(Tool[MyToolArgs]):
    """Custom tool implementation."""
    
    @property
    def name(self) -> str:
        return "my_tool"
    
    @property
    def description(self) -> str:
        return "Searches my custom data source for relevant information."
    
    def get_args_schema(self) -> type[MyToolArgs]:
        return MyToolArgs
    
    async def execute(self, ctx: ToolContext, args: MyToolArgs) -> ToolResult:
        # Implement your logic
        results = await my_search_function(args.query, args.limit)
        return ToolResult(
            success=True,
            data={"results": results},
        )

# Register in backend/app/main.py
tool_registry.register(MyTool())
```

### Adding Middleware

```python
# backend/app/core/middleware.py
from backend.app.core import LlmMiddleware

class CostTrackingMiddleware(LlmMiddleware):
    """Track LLM call costs."""
    
    def __init__(self):
        self.total_tokens = 0
    
    async def process(self, messages, call_next):
        result = await call_next(messages)
        # Track tokens (implementation depends on LLM response format)
        self.total_tokens += result.get("usage", {}).get("total_tokens", 0)
        return result

class CachingMiddleware(LlmMiddleware):
    """Cache LLM responses."""
    
    def __init__(self, cache: dict):
        self.cache = cache
    
    async def process(self, messages, call_next):
        key = self._compute_key(messages)
        if key in self.cache:
            return self.cache[key]
        result = await call_next(messages)
        self.cache[key] = result
        return result

# Add to LlmService in main.py
llm_service = LlmService(
    middlewares=[
        LoggingMiddleware(),
        CostTrackingMiddleware(),
        CachingMiddleware(cache={}),
    ]
)
```

### Custom Lifecycle Hooks

```python
# backend/app/core/lifecycle.py
from backend.app.core import LifecycleHooks

class MetricsHooks(LifecycleHooks):
    """Custom hooks for metrics collection."""
    
    async def on_request_start(self, message: str, config: dict) -> None:
        # Start timer, log request
        self.start_time = time.time()
        logger.info(f"Request started: {message[:50]}...")
    
    async def on_request_end(self, final_answer: str) -> None:
        # Record latency, log completion
        latency = time.time() - self.start_time
        metrics.record("request_latency", latency)
        logger.info(f"Request completed in {latency:.2f}s")
    
    async def on_error(self, error: Exception, context: dict) -> None:
        # Alert on errors
        alerting.send(f"Error in {context.get('step')}: {error}")

# Use in main.py
agent = Agent(
    llm_service=llm_service,
    tool_registry=tool_registry,
    lifecycle_hooks=MetricsHooks(),
)
```

### Adding a New Workflow Node

```python
# backend/app/graph/workflow.py

async def _my_custom_node(state: LGState, agent: Agent) -> AsyncGenerator:
    """Custom workflow node."""
    yield SSEEvent(event=EventType.STEP_STARTED, data={"step": "my_node"})
    
    # Your logic here
    result = await some_processing(state)
    
    yield SSEEvent(event=EventType.STEP_FINISHED, data={"step": "my_node", "output": result})
    
    # Return state updates
    return {"my_field": result}

# Add to graph builder
builder.add_node("my_node", lambda s: _my_custom_node(s, agent))
builder.add_edge("previous_node", "my_node")
builder.add_edge("my_node", "next_node")
```

---

## License

MIT

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

For major changes, please open an issue first to discuss what you would like to change.
