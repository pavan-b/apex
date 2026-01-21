# Ollama (qwen3:8b) LangGraph Research + FactCheck Chat

This repo contains:
- **FastAPI backend** with Vanna-style architecture (Agent, Tools, Middlewares)
- **LangGraph workflow** (research → synthesis → producer → 3-pass verification)
- **Streamlit UI** with real-time SSE updates and step accordions

## Architecture (Vanna 2.0 Pattern)

```
backend/app/
├── core/           # Core abstractions
│   ├── agent.py        # Central orchestrator
│   ├── tool.py         # Tool base class
│   ├── registry.py     # ToolRegistry
│   ├── llm_service.py  # LLM abstraction with middleware
│   ├── middleware.py   # Logging, Caching middlewares
│   └── lifecycle.py    # Request lifecycle hooks
├── tools/          # Tool implementations
│   ├── search_tool.py  # DuckDuckGo, Wikipedia, arXiv
│   └── fetch_tool.py   # Page content extraction
├── server/         # HTTP layer
│   ├── handler.py      # ChatHandler
│   └── routes.py       # Route registration
├── agents/         # Business logic
│   ├── research.py     # Research orchestration
│   ├── synthesis.py    # Evidence synthesis
│   └── factcheck.py    # Verification loop
├── graph/          # LangGraph workflow
│   └── workflow.py
└── main.py         # App entry point
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/chat/sse` | POST | Main chat endpoint (SSE streaming) |
| `/api/v1/health` | GET | Health check with model info |
| `/health` | GET | Simple health check |

### POST /api/v1/chat/sse

Request:
```json
{
  "message": "Your question here",
  "best_of_n": 1,
  "verify_max_rounds": 1
}
```

Response: Server-Sent Events stream with events:
- `step_started` / `step_finished`: Workflow step lifecycle
- `tool_call` / `tool_result`: Tool execution
- `critic_feedback` / `revision_selected`: Verification loop
- `final`: Final answer
- `error`: Error occurred
- `done`: Stream complete

## Prereqs

1. Install and run Ollama.
2. Pull the model:

```bash
ollama pull qwen3:8b
```

## Setup

### Python version note (important)
This project is tested with **Python 3.11 / 3.12**.

If you are on **Python 3.14**, some dependencies (notably `pydantic-core`) may require a local **Rust + Cargo** toolchain to build, which often fails if Cargo isn't on `PATH`.

Recommended: install Python **3.12.x**, then create a fresh venv.

### Install with `uv` (recommended)

Install uv:

```bash
python -m pip install -U uv
```

Create venv + sync deps:

```bash
uv venv --python 3.12
uv sync
```

### Install with `pip` (fallback)
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

### Quick Start (shorthand commands)

```bash
# Show all commands
uv run dev

# Start Phoenix (optional, for tracing)
uv run phoenix-serve

# Start Backend (in a new terminal)
uv run backend

# Start UI (in a new terminal)
uv run ui
```

### Manual Commands (if shorthands don't work)

```bash
# Phoenix
uv run phoenix serve

# Backend
uv run uvicorn backend.app.main:app --reload --port 8000

# UI
uv run streamlit run ui/streamlit_app.py --server.port 8501
```

Open Streamlit and chat. The UI will display:
- routing decision (easy vs deep research)
- parallel research queries and tool hits
- synthesis
- producer answer draft
- up to 3 rounds of multi-critic verification (CoVe + aspect checks) and revisions

## Config (optional)

Environment variables (backend):
- `OLLAMA_MODEL` (default: `qwen3:8b`)
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_TEMPERATURE` (default: `0.2`)
- `BEST_OF_N` (default: `3`)
- `VERIFY_MAX_ROUNDS` (default: `3`)
- `SEARCH_MAX_RESULTS` (default: `5`)

### Arize Phoenix (LLM observability)

Tracing is **optional** and enabled only if an OTLP endpoint is set. Phoenix accepts OTLP traces and renders them in the Phoenix UI.

Docs:
- https://arize.com/docs/phoenix
- https://arize-ai.github.io/openinference/
- https://arize.com/docs/phoenix/integrations/python/langchain/langchain-tracing

Start a local Phoenix instance **before** the backend:

```bash
uv run phoenix serve
```

Environment variables:
- `PHOENIX_OTEL_ENDPOINT` (preferred) or `OTEL_EXPORTER_OTLP_ENDPOINT` (default: `http://localhost:6006/`)
- `PHOENIX_SERVICE_NAME` (default: `apex-research-chat`)
- `PHOENIX_API_KEY` (optional; sent as `Authorization: Bearer ...`)
- `PHOENIX_OTEL_HEADERS` (optional, comma-separated `k=v` pairs)

## Extending

### Adding a new tool

```python
from backend.app.core import Tool, ToolContext, ToolResult
from pydantic import BaseModel

class MyToolArgs(BaseModel):
    query: str

class MyTool(Tool[MyToolArgs]):
    @property
    def name(self) -> str:
        return "my_tool"

    @property
    def description(self) -> str:
        return "Description for LLM"

    def get_args_schema(self):
        return MyToolArgs

    async def execute(self, ctx: ToolContext, args: MyToolArgs) -> ToolResult:
        # Your logic here
        return ToolResult(success=True, data={"result": "..."})

# Register in main.py
tool_registry.register(MyTool())
```

### Adding middleware

```python
from backend.app.core import LlmMiddleware

class CostTrackingMiddleware(LlmMiddleware):
    async def process(self, messages, call_next):
        # Pre-processing
        result = await call_next(messages)
        # Post-processing (track costs, etc.)
        return result

# Add to LlmService
llm_service = LlmService(middlewares=[CostTrackingMiddleware()])
```

### Custom lifecycle hooks

```python
from backend.app.core import LifecycleHooks

class MyHooks(LifecycleHooks):
    async def on_request_start(self, message, config):
        # Log, check quotas, etc.
        pass

    async def on_request_end(self, final_answer):
        # Record metrics, etc.
        pass

# Use in Agent
agent = Agent(lifecycle_hooks=MyHooks())
```
