"""
Apex – AI Research Assistant UI.

Single chat window with a clean workflow panel on the right.
Background-threaded SSE so the UI stays responsive.

Usage:
    uv run streamlit run ui/streamlit_app.py --server.port 8501
"""

import json
import random
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st

st.set_page_config(page_title="Apex", page_icon="🔬", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = r"""
<style>
:root{
  --bg-primary:#0a0e17;--bg-secondary:#111827;
  --bg-card:rgba(17,24,39,.7);--bg-glass:rgba(255,255,255,.03);
  --border-subtle:rgba(255,255,255,.06);--border-hover:rgba(255,255,255,.12);
  --text-primary:#f1f5f9;--text-secondary:#94a3b8;--text-muted:#64748b;
  --accent-purple:#8b5cf6;--accent-blue:#3b82f6;--accent-teal:#14b8a6;
  --accent-green:#22c55e;--accent-amber:#f59e0b;--accent-red:#ef4444;
  --gradient-primary:linear-gradient(135deg,#8b5cf6,#3b82f6,#14b8a6);
  --gradient-glow:linear-gradient(135deg,rgba(139,92,246,.15),rgba(59,130,246,.15),rgba(20,184,166,.15));
  --shadow-lg:0 8px 32px rgba(0,0,0,.3);--shadow-glow:0 0 20px rgba(139,92,246,.1);
  --radius-sm:8px;--radius-md:12px;--radius-lg:16px;
}

/* hide chrome */
header[data-testid="stHeader"]{background:transparent!important;height:0!important;min-height:0!important;pointer-events:none}
#MainMenu,footer,header .stDeployButton{display:none!important}

/* global */
.block-container{padding-top:.8rem!important;padding-bottom:5.5rem!important;max-width:1440px!important}
.stApp{background:var(--bg-primary)!important}
h1,h2,h3{font-weight:700!important;letter-spacing:-.02em}
::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:rgba(255,255,255,.08);border-radius:3px}

/* pinned chat input */
div[data-testid="stChatInput"]{
  position:fixed!important;bottom:0;left:0;right:0;z-index:999;
  padding:12px 16px 14px;border-top:1px solid var(--border-subtle);
  background:rgba(10,14,23,.92)!important;
  backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
}

/* header */
.apex-header{display:flex;align-items:center;gap:14px;padding:6px 0 2px;margin-bottom:4px}
.apex-logo{width:38px;height:38px;border-radius:var(--radius-md);background:var(--gradient-primary);display:flex;align-items:center;justify-content:center;font-size:1.25rem;box-shadow:var(--shadow-glow)}
.apex-title-group{display:flex;flex-direction:column}
.apex-title{font-size:1.4rem;font-weight:800;background:var(--gradient-primary);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.2}
.apex-tagline{font-size:.7rem;color:var(--text-muted);font-weight:400;margin-top:1px}

/* empty state */
.empty-state{text-align:center;padding:3rem 1rem 1rem}
.empty-icon-wrap{width:72px;height:72px;margin:0 auto 14px;border-radius:50%;background:var(--gradient-glow);border:2px solid transparent;display:flex;align-items:center;justify-content:center;font-size:2rem;position:relative;animation:icon-float 3s ease-in-out infinite}
.empty-icon-wrap::before{content:'';position:absolute;inset:-2px;border-radius:50%;background:var(--gradient-primary);z-index:-1;opacity:.45;animation:ring-spin 4s linear infinite}
@keyframes icon-float{0%,100%{transform:translateY(0)}50%{transform:translateY(-5px)}}
@keyframes ring-spin{to{transform:rotate(360deg)}}
.empty-title{font-size:1.15rem;font-weight:700;color:var(--text-primary);margin-bottom:6px}
.empty-desc{font-size:.82rem;color:var(--text-muted);max-width:420px;margin:0 auto 16px;line-height:1.5}
.feature-row{display:flex;justify-content:center;flex-wrap:wrap;gap:7px;margin-bottom:18px}
.feature-pill{display:inline-flex;align-items:center;gap:5px;font-size:.7rem;font-weight:500;padding:4px 11px;border-radius:20px;background:rgba(255,255,255,.03);border:1px solid var(--border-subtle);color:var(--text-secondary)}
.feature-pill .fp-dot{width:5px;height:5px;border-radius:50%}
.fp-purple{background:var(--accent-purple)}.fp-blue{background:var(--accent-blue)}.fp-teal{background:var(--accent-teal)}.fp-green{background:var(--accent-green)}
.suggestion-label{font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--text-muted);margin-bottom:8px;text-align:center}

/* suggestion buttons */
.stButton>button{
  width:100%!important;text-align:left!important;justify-content:flex-start!important;
  background:rgba(255,255,255,.02)!important;border:1px solid var(--border-subtle)!important;
  border-radius:var(--radius-md)!important;padding:12px 14px!important;
  color:var(--text-secondary)!important;font-size:.8rem!important;font-weight:500!important;
  transition:all .25s ease!important;cursor:pointer!important;line-height:1.4!important;
  height:auto!important;min-height:48px!important;
}
.stButton>button:hover{border-color:var(--accent-purple)!important;background:rgba(139,92,246,.06)!important;color:var(--text-primary)!important;transform:translateY(-2px)!important;box-shadow:0 4px 16px rgba(139,92,246,.12)!important}

/* workflow panel */
.panel-card{background:var(--bg-card);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border:1px solid var(--border-subtle);border-radius:var(--radius-lg);padding:16px;margin-bottom:8px;box-shadow:var(--shadow-lg)}
.panel-title{display:flex;align-items:center;gap:10px;margin-bottom:12px}
.panel-title-icon{width:28px;height:28px;border-radius:var(--radius-sm);background:var(--gradient-glow);border:1px solid var(--border-subtle);display:flex;align-items:center;justify-content:center;font-size:.9rem}
.panel-title-text{font-size:.9rem;font-weight:700;color:var(--text-primary)}

/* progress */
.progress-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}
.progress-label{font-size:.7rem;color:var(--text-muted);font-weight:500}
.progress-pct{font-size:.7rem;font-weight:700;background:var(--gradient-primary);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.progress-track{height:3px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden;margin-bottom:14px}
.progress-fill{height:100%;background:var(--gradient-primary);border-radius:2px;transition:width .5s cubic-bezier(.4,0,.2,1)}

/* timeline */
.tl-container{padding:0}
.tl-step{display:flex;align-items:flex-start;gap:12px;padding:5px 0;position:relative}
.tl-step:not(:last-child)::before{content:'';position:absolute;left:13px;top:28px;bottom:-5px;width:2px;background:rgba(255,255,255,.06);border-radius:1px}
.tl-step.done:not(:last-child)::before{background:linear-gradient(180deg,var(--accent-green),rgba(255,255,255,.06))}
.tl-step.running:not(:last-child)::before{background:linear-gradient(180deg,var(--accent-blue),rgba(255,255,255,.06))}
.tl-dot{width:26px;height:26px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;flex-shrink:0;z-index:1;border:2px solid transparent;transition:all .3s ease}
.tl-dot.pending{background:rgba(255,255,255,.04);border-color:rgba(255,255,255,.08);color:var(--text-muted)}
.tl-dot.running{background:rgba(59,130,246,.15);border-color:var(--accent-blue);color:var(--accent-blue);animation:dot-pulse 2s infinite}
.tl-dot.done{background:rgba(34,197,94,.15);border-color:var(--accent-green);color:var(--accent-green)}
.tl-dot.error{background:rgba(239,68,68,.15);border-color:var(--accent-red);color:var(--accent-red)}
@keyframes dot-pulse{0%,100%{box-shadow:0 0 0 0 rgba(59,130,246,.4)}50%{box-shadow:0 0 0 6px rgba(59,130,246,0)}}
.tl-body{flex:1;min-width:0;padding-top:3px}
.tl-label{font-size:.76rem;font-weight:600;line-height:1.3;transition:color .3s ease}
.tl-label.pending{color:var(--text-muted)}.tl-label.running{color:var(--accent-blue)}.tl-label.done{color:var(--accent-green)}.tl-label.error{color:var(--accent-red)}
.tl-note{font-size:.66rem;color:var(--text-muted);margin-top:1px;line-height:1.3}
.tl-note .note-badge{display:inline-block;padding:1px 6px;border-radius:4px;font-size:.63rem;font-weight:600;background:rgba(255,255,255,.05);border:1px solid var(--border-subtle)}

/* source cards */
.detail-section{margin-top:4px;padding-top:4px}
.detail-label{font-size:.66rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--text-muted);margin-bottom:6px;display:flex;align-items:center;gap:5px}
.detail-label .label-dot{width:5px;height:5px;border-radius:50%;background:var(--accent-purple)}
.src-card{background:rgba(255,255,255,.02);border:1px solid var(--border-subtle);border-radius:var(--radius-sm);padding:7px 9px;margin-bottom:5px;transition:all .2s ease}
.src-card:hover{border-color:var(--border-hover);background:rgba(255,255,255,.04)}
.src-row{display:flex;align-items:center;gap:6px;margin-bottom:2px}
.src-badge{display:inline-flex;align-items:center;gap:3px;font-size:.63rem;font-weight:600;padding:2px 6px;border-radius:5px;background:rgba(139,92,246,.12);color:var(--accent-purple);border:1px solid rgba(139,92,246,.2);white-space:nowrap}
.src-query{font-size:.72rem;color:var(--text-secondary);line-height:1.4}
.src-result{font-size:.68rem;color:var(--accent-green);font-weight:500;display:flex;align-items:center;gap:4px;margin-top:2px}
.src-result .result-dot{width:4px;height:4px;border-radius:50%;background:var(--accent-green)}

/* query pills */
.q-pills{display:flex;flex-wrap:wrap;gap:4px;margin:5px 0}
.q-pill{display:inline-block;font-size:.66rem;padding:2px 8px;border-radius:16px;color:var(--text-secondary);background:rgba(59,130,246,.08);border:1px solid rgba(59,130,246,.18)}

/* verify cards */
.vf-card{background:rgba(255,255,255,.02);border:1px solid var(--border-subtle);border-radius:var(--radius-sm);padding:7px 9px;margin-bottom:5px;border-left:3px solid var(--border-subtle)}
.vf-card.feedback{border-left-color:var(--accent-amber)}.vf-card.accepted{border-left-color:var(--accent-teal)}
.vf-header{display:flex;align-items:center;gap:6px;margin-bottom:2px}
.vf-round{font-size:.61rem;font-weight:700;padding:1px 5px;border-radius:4px;background:rgba(255,255,255,.06);color:var(--text-muted);border:1px solid var(--border-subtle)}
.vf-critic{font-size:.72rem;font-weight:600;color:var(--text-primary)}
.vf-summary{font-size:.7rem;color:var(--text-muted);line-height:1.4}

/* synthesis */
.synth-card{background:rgba(139,92,246,.04);border:1px solid rgba(139,92,246,.12);border-radius:var(--radius-sm);padding:10px 12px;font-size:.74rem;color:var(--text-secondary);line-height:1.5}

/* thinking */
.think-wrap{display:flex;align-items:center;gap:10px;padding:8px 0}
.think-orb{width:30px;height:30px;border-radius:50%;background:var(--gradient-glow);border:1px solid var(--border-subtle);display:flex;align-items:center;justify-content:center;animation:orb-breathe 2s infinite ease-in-out}
@keyframes orb-breathe{0%,100%{transform:scale(1);opacity:.7}50%{transform:scale(1.08);opacity:1}}
.think-orb-inner{width:9px;height:9px;border-radius:50%;background:var(--gradient-primary);animation:orb-spin 3s infinite linear}
@keyframes orb-spin{0%{transform:rotate(0) scale(1)}50%{transform:rotate(180deg) scale(.7)}100%{transform:rotate(360deg) scale(1)}}
.think-text{font-size:.78rem;color:var(--text-muted);font-weight:500}
.think-dots{display:inline-flex;gap:3px;margin-left:2px}
.think-dots span{width:4px;height:4px;border-radius:50%;background:var(--accent-purple);animation:dot-bounce 1.4s infinite ease-in-out both}
.think-dots span:nth-child(1){animation-delay:-.32s}.think-dots span:nth-child(2){animation-delay:-.16s}
@keyframes dot-bounce{0%,80%,100%{transform:scale(0);opacity:.3}40%{transform:scale(1);opacity:1}}

/* idle */
.idle-panel{text-align:center;padding:1.2rem .5rem}
.idle-icon{font-size:1.6rem;margin-bottom:5px;opacity:.35}
.idle-text{font-size:.76rem;color:var(--text-muted);line-height:1.45}

/* workflow radio */
.stRadio>div{gap:0!important;flex-wrap:wrap}
.stRadio label{font-size:.72rem!important;padding:4px 10px!important}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Backend
# ─────────────────────────────────────────────────────────────────────────────

def _get_backend_base_url() -> str:
    from pathlib import Path
    for p in [Path(".streamlit") / "secrets.toml", Path.home() / ".streamlit" / "secrets.toml"]:
        if p.exists():
            return st.secrets.get("BACKEND_BASE_URL", "http://localhost:8000")
    return "http://localhost:8000"


BACKEND_BASE_URL = _get_backend_base_url()

# ─────────────────────────────────────────────────────────────────────────────
# Step Config
# ─────────────────────────────────────────────────────────────────────────────

STEP_ORDER = ["router", "research", "synthesis", "producer", "factcheck", "final"]
STEP_CONFIG = {
    "router":    {"label": "Routing",       "icon": "🧭"},
    "research":  {"label": "Researching",   "icon": "🔍"},
    "synthesis": {"label": "Synthesizing",  "icon": "📝"},
    "producer":  {"label": "Drafting",      "icon": "✍️"},
    "factcheck": {"label": "Fact-Checking", "icon": "🛡️"},
    "final":     {"label": "Delivering",    "icon": "✨"},
}
TOOL_ICONS = {"duckduckgo": "🌐", "wikipedia": "📚", "arxiv": "📄", "page_fetch": "🔗"}
SKIP_STEPS = {"workflow", "agent", "stream", "request"}

_SUGGESTION_POOL = [
    {"icon": "🧬", "text": "Latest advances in CRISPR gene therapy"},
    {"icon": "🌍", "text": "Impact of AI on climate change research"},
    {"icon": "🚀", "text": "Current state of quantum computing"},
    {"icon": "📊", "text": "Compare transformer vs. diffusion models"},
    {"icon": "🧠", "text": "How do large language models reason?"},
    {"icon": "⚛️", "text": "Breakthroughs in nuclear fusion energy"},
    {"icon": "🦠", "text": "mRNA vaccine technology beyond COVID-19"},
    {"icon": "🔭", "text": "James Webb Space Telescope latest discoveries"},
    {"icon": "🏥", "text": "AI-assisted drug discovery pipeline explained"},
    {"icon": "🌊", "text": "Deep-sea mining environmental risks and benefits"},
    {"icon": "🤖", "text": "Current progress in humanoid robotics"},
    {"icon": "🛰️", "text": "Starlink and the future of satellite internet"},
    {"icon": "🧪", "text": "Advances in solid-state battery technology"},
    {"icon": "📡", "text": "6G wireless research and expected capabilities"},
    {"icon": "🌱", "text": "Vertical farming scalability and economics"},
    {"icon": "💊", "text": "GLP-1 receptor agonists: mechanisms and impact"},
    {"icon": "🏗️", "text": "3D-printed housing: feasibility and adoption"},
    {"icon": "🧊", "text": "Arctic permafrost thaw and methane feedback loops"},
    {"icon": "🎭", "text": "Deepfake detection methods and their limitations"},
    {"icon": "🔐", "text": "Post-quantum cryptography standardization status"},
]


def _get_suggestions() -> List[Dict[str, str]]:
    if "_suggestion_seed" not in st.session_state:
        st.session_state._suggestion_seed = random.randint(0, 2**31)
    rng = random.Random(st.session_state._suggestion_seed)
    return rng.sample(_SUGGESTION_POOL, min(4, len(_SUGGESTION_POOL)))


# ─────────────────────────────────────────────────────────────────────────────
# SSE helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sse_events_post(url: str, payload: Dict[str, Any]):
    with httpx.stream("POST", url, json=payload, timeout=None,
                      headers={"Accept": "text/event-stream"}) as r:
        r.raise_for_status()
        event = None
        data_lines: List[str] = []
        for line in r.iter_lines():
            if line is None:
                continue
            if line == "":
                if event and data_lines:
                    yield event, "\n".join(data_lines)
                event = None
                data_lines = []
                continue
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].lstrip())


# ─────────────────────────────────────────────────────────────────────────────
# Background SSE worker
# ─────────────────────────────────────────────────────────────────────────────

def _new_running_state() -> Dict[str, Any]:
    return {
        "done": False, "error": None,
        "step_states": {}, "step_notes": {},
        "research_entries": [], "query_variations": [],
        "verify_entries": [], "synthesis_text": "",
        "draft_text": "", "final_answer": None,
        "current_label": "Connecting",
    }


def _run_sse_worker(url: str, payload: Dict[str, Any], state: Dict[str, Any]):
    current_step = ""
    verify_round: Optional[int] = None
    try:
        for evt_name, data in _sse_events_post(url, payload):
            if evt_name == "ping":
                continue
            if evt_name == "done":
                break
            try:
                event_data = json.loads(data)
            except json.JSONDecodeError:
                continue
            step = event_data.get("step", "")
            typ = event_data.get("type", evt_name)
            dct: Dict[str, Any] = event_data.get("data", {}) or {}
            if step:
                current_step = step
            if step and step not in SKIP_STEPS:
                cfg = STEP_CONFIG.get(step, {})
                if cfg and typ == "step_started":
                    state["current_label"] = cfg.get("label", step)
            if typ == "error":
                state["step_states"][current_step] = "error"
                state["step_notes"][current_step] = dct.get("error", "Error")[:60]
            if typ == "step_started" and step not in SKIP_STEPS:
                state["step_states"][step] = "running"
                state["step_notes"][step] = "In progress…"
            if typ == "step_finished" and step not in SKIP_STEPS:
                state["step_states"][step] = "done"
                state["step_notes"][step] = "Done"
                if step == "router":
                    output = dct.get("output", "")
                    if isinstance(output, str) and output:
                        route = "Deep research" if "deep" in output.lower() else "Quick answer"
                        state["step_notes"][step] = f'<span class="note-badge">{route}</span>'
                elif step == "research":
                    n = sum(1 for e in state["research_entries"] if e.get("type") == "tool_result")
                    state["step_notes"][step] = f"{n} sources"
                elif step == "factcheck":
                    state["step_notes"][step] = f"{len(state['verify_entries'])} rounds"
            if typ in ("tool_call", "tool_result"):
                tool, query = dct.get("tool", ""), dct.get("query", "")
                if typ == "tool_call":
                    state["research_entries"].append({"type": "tool_call", "tool": tool, "query": query})
                else:
                    state["research_entries"].append({
                        "type": "tool_result", "tool": tool, "query": query,
                        "results": dct.get("results", 0), "note": dct.get("note", ""),
                    })
            if typ == "message" and step == "research":
                qvs = dct.get("query_variations")
                if isinstance(qvs, list) and qvs:
                    state["query_variations"] = qvs
            if typ == "message" and step == "synthesis":
                syn = dct.get("synthesis")
                if syn:
                    state["synthesis_text"] = syn
            if typ == "step_finished" and step == "synthesis":
                out = dct.get("output", "")
                if out and not state["synthesis_text"]:
                    state["synthesis_text"] = out if isinstance(out, str) else str(out)
            if typ == "draft_chunk":
                chunk = dct.get("chunk", "")
                if chunk:
                    state["draft_text"] += chunk
            if typ in ("critic_feedback", "revision_selected"):
                verify_round = dct.get("round", verify_round)
                entry: Dict[str, Any] = {"type": typ, "round": verify_round}
                if typ == "critic_feedback":
                    entry["critic"] = dct.get("critic", "Critic")
                    entry["summary"] = dct.get("summary", "")
                else:
                    entry["summary"] = dct.get("summary", "")
                state["verify_entries"].append(entry)
            if typ == "final":
                state["final_answer"] = dct.get("answer", "")
                state["step_states"]["final"] = "done"
                state["step_notes"]["final"] = "✨ Delivered"
    except httpx.HTTPStatusError as e:
        state["error"] = f"_Backend error: HTTP {e.response.status_code}_"
        state["step_states"][current_step or "router"] = "error"
        state["step_notes"][current_step or "router"] = f"HTTP {e.response.status_code}"
    except httpx.ConnectError:
        state["error"] = "_Cannot connect to backend. Please start the server._"
    except Exception as e:
        state["error"] = f"_Error: {str(e)}_"
    finally:
        state["done"] = True


# ─────────────────────────────────────────────────────────────────────────────
# HTML renderers
# ─────────────────────────────────────────────────────────────────────────────

def _render_timeline(ss: Dict[str, str], sn: Dict[str, str], done: int) -> str:
    total = len(STEP_ORDER)
    pct = int((done / max(total, 1)) * 100)
    h = [f'<div class="progress-row"><span class="progress-label">Progress</span>'
         f'<span class="progress-pct">{pct}%</span></div>'
         f'<div class="progress-track"><div class="progress-fill" style="width:{pct}%"></div></div>'
         '<div class="tl-container">']
    icons = {"pending": "○", "running": "◉", "done": "✓", "error": "✗"}
    for key in STEP_ORDER:
        cfg = STEP_CONFIG[key]
        state = ss.get(key, "pending")
        note = sn.get(key, "")
        dot = icons.get(state, "○")
        nh = f'<div class="tl-note">{note}</div>' if note else ""
        h.append(f'<div class="tl-step {state}"><div class="tl-dot {state}">{dot}</div>'
                 f'<div class="tl-body"><div class="tl-label {state}">{cfg["icon"]} {cfg["label"]}</div>'
                 f'{nh}</div></div>')
    h.append('</div>')
    return "".join(h)


def _render_sources(entries: List[Dict[str, Any]]) -> str:
    if not entries:
        return ""
    h = ['<div class="detail-section">',
         '<div class="detail-label"><span class="label-dot"></span>Research Activity</div>']
    for e in entries:
        tool, query = e.get("tool", ""), e.get("query", "")
        icon = TOOL_ICONS.get(tool.lower().replace("_search", "").replace("search", "duckduckgo"), "🔎")
        nice = tool.replace("_", " ").title()
        if e["type"] == "tool_call":
            h.append(f'<div class="src-card"><div class="src-row"><span class="src-badge">{icon} {nice}</span></div>'
                     f'<div class="src-query">⟶ <em>{query}</em></div></div>')
        else:
            cnt, note = e.get("results", 0), e.get("note", "")
            extra = f" — {note}" if note else ""
            h.append(f'<div class="src-card"><div class="src-row"><span class="src-badge">{icon} {nice}</span></div>'
                     f'<div class="src-query">{query}</div>'
                     f'<div class="src-result"><span class="result-dot"></span>{cnt} results{extra}</div></div>')
    h.append("</div>")
    return "".join(h)


def _render_pills(variations: List[str]) -> str:
    pills = "".join(f'<span class="q-pill">{q}</span>' for q in variations)
    return (f'<div class="detail-section"><div class="detail-label">'
            f'<span class="label-dot"></span>Query Variations</div>'
            f'<div class="q-pills">{pills}</div></div>')


def _render_verify(entries: List[Dict[str, Any]]) -> str:
    if not entries:
        return ""
    h = ['<div class="detail-section">',
         '<div class="detail-label"><span class="label-dot"></span>Verification</div>']
    for e in entries:
        rnd, typ = e.get("round", "?"), e.get("type", "")
        cls = "accepted" if typ == "revision_selected" else "feedback"
        if typ == "critic_feedback":
            critic, summary = e.get("critic", "Critic"), e.get("summary", "")
            h.append(f'<div class="vf-card {cls}"><div class="vf-header"><span class="vf-round">R{rnd}</span>'
                     f'<span class="vf-critic">💬 {critic}</span></div>'
                     f'<div class="vf-summary">{summary}</div></div>')
        else:
            summary = e.get("summary", "")
            h.append(f'<div class="vf-card {cls}"><div class="vf-header"><span class="vf-round">R{rnd}</span>'
                     f'<span class="vf-critic">✅ Accepted</span></div>'
                     f'<div class="vf-summary">{summary}</div></div>')
    h.append("</div>")
    return "".join(h)


def _thinking_html(label: str = "Thinking") -> str:
    return (f'<div class="think-wrap"><div class="think-orb"><div class="think-orb-inner"></div></div>'
            f'<div class="think-text">{label}'
            f'<span class="think-dots"><span></span><span></span><span></span></span></div></div>')


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

def _init():
    if "messages" not in st.session_state:
        st.session_state.messages = []            # [{role, content, workflow?}]
    if "running" not in st.session_state:
        st.session_state.running = None            # background state dict
    if "running_prompt" not in st.session_state:
        st.session_state.running_prompt = None
    if "selected_suggestion" not in st.session_state:
        st.session_state.selected_suggestion = None
    if "viewing_wf_idx" not in st.session_state:
        st.session_state.viewing_wf_idx = None
    if "_last_processed_prompt" not in st.session_state:
        st.session_state._last_processed_prompt = None


_init()


# ─────────────────────────────────────────────────────────────────────────────
# Finalize completed background query
# ─────────────────────────────────────────────────────────────────────────────

def _finalize():
    rs = st.session_state.running
    if rs is None or not rs["done"]:
        return
    answer = rs.get("final_answer") or rs.get("error") or rs.get("draft_text")
    if not answer:
        answer = "_No answer received — backend may have errored._"
    wf = {
        "step_states": dict(rs.get("step_states", {})),
        "step_notes": dict(rs.get("step_notes", {})),
        "research_entries": list(rs.get("research_entries", [])),
        "query_variations": list(rs.get("query_variations", [])),
        "verify_entries": list(rs.get("verify_entries", [])),
        "synthesis_text": rs.get("synthesis_text", ""),
    }
    st.session_state.messages.append({"role": "assistant", "content": answer, "workflow": wf})
    st.session_state.viewing_wf_idx = len(st.session_state.messages) - 1
    st.session_state.running = None
    st.session_state.running_prompt = None


_finalize()

is_running = st.session_state.running is not None and not st.session_state.running.get("done", True)
msgs = st.session_state.messages


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="apex-header">'
    '<div class="apex-logo">🔬</div>'
    '<div class="apex-title-group">'
    '<span class="apex-title">Apex</span>'
    '<span class="apex-tagline">AI Research Assistant · LangGraph</span>'
    '</div></div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Chat input (pinned to bottom via CSS)
# ─────────────────────────────────────────────────────────────────────────────

prompt = st.chat_input("Ask anything — Apex will research it…", disabled=is_running)
if prompt is None and not st.session_state.selected_suggestion:
    # Widget consumed the value — safe to allow next submission
    st.session_state._last_processed_prompt = None
if st.session_state.selected_suggestion:
    prompt = st.session_state.selected_suggestion
    st.session_state.selected_suggestion = None

# ─────────────────────────────────────────────────────────────────────────────
# Fast-mode + New chat toggle
# ─────────────────────────────────────────────────────────────────────────────

_toolbar = st.columns([6, 1])
with _toolbar[1]:
    fast_mode = st.toggle("⚡", value=False, help="Fast mode: fewer verification passes")

# ─────────────────────────────────────────────────────────────────────────────
# Two-column layout: Chat | Workflow
# ─────────────────────────────────────────────────────────────────────────────

col_chat, col_wf = st.columns([3, 2])

# ═══════════════════════════════════════════════════════════════════════════
# WORKFLOW PANEL
# ═══════════════════════════════════════════════════════════════════════════

with col_wf:
    # Determine active workflow data
    active_wf: Optional[Dict[str, Any]] = None
    _wf_topic = ""

    if is_running and st.session_state.running:
        active_wf = st.session_state.running
        rp = st.session_state.running_prompt or ""
        if rp:
            _wf_topic = rp[:60] + ("…" if len(rp) > 60 else "")
    else:
        v_idx = st.session_state.viewing_wf_idx
        if v_idx is not None and 0 <= v_idx < len(msgs):
            m = msgs[v_idx]
            if m["role"] == "assistant" and m.get("workflow"):
                active_wf = m["workflow"]
                if v_idx > 0 and msgs[v_idx - 1]["role"] == "user":
                    _wf_topic = msgs[v_idx - 1]["content"][:60]
                    if len(msgs[v_idx - 1]["content"]) > 60:
                        _wf_topic += "…"

    _sub = (f'<div style="font-size:.68rem;color:var(--text-muted);margin-top:2px;'
            f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">'
            f'📌 {_wf_topic}</div>') if _wf_topic else ""

    st.markdown(
        '<div class="panel-card">'
        '<div class="panel-title">'
        '<div class="panel-title-icon">⚡</div>'
        '<span class="panel-title-text">Workflow</span>'
        '</div>'
        f'{_sub}'
        '</div>',
        unsafe_allow_html=True,
    )

    _VIEWS = ["⏱ Timeline", "🔍 Research", "📝 Synthesis", "🛡️ Verification"]
    view = st.radio("View", _VIEWS, horizontal=True, label_visibility="collapsed")

    wf_ph = st.empty()

    if view == _VIEWS[0]:
        if active_wf:
            ss = active_wf.get("step_states", {})
            sn = active_wf.get("step_notes", {})
            done_n = sum(1 for v in ss.values() if v == "done")
            wf_ph.html(CUSTOM_CSS + _render_timeline(ss, sn, done_n))
        else:
            wf_ph.html(CUSTOM_CSS + '<div class="idle-panel"><div class="idle-icon">🔬</div>'
                       '<div class="idle-text">Workflow steps appear here when a query runs.</div></div>')

    elif view == _VIEWS[1]:
        if active_wf:
            parts: List[str] = []
            if active_wf.get("research_entries"):
                parts.append(_render_sources(active_wf["research_entries"]))
            if active_wf.get("query_variations"):
                parts.append(_render_pills(active_wf["query_variations"]))
            if parts:
                wf_ph.html(CUSTOM_CSS + "<div>" + "".join(parts) + "</div>")
            else:
                wf_ph.html(CUSTOM_CSS + '<div class="idle-panel"><div class="idle-icon">🔍</div>'
                           '<div class="idle-text">No research data yet.</div></div>')
        else:
            wf_ph.html(CUSTOM_CSS + '<div class="idle-panel"><div class="idle-icon">🔍</div>'
                       '<div class="idle-text">Research activity will appear here.</div></div>')

    elif view == _VIEWS[2]:
        if active_wf and active_wf.get("synthesis_text"):
            wf_ph.html(CUSTOM_CSS + f'<div class="synth-card">{active_wf["synthesis_text"]}</div>')
        else:
            wf_ph.html(CUSTOM_CSS + '<div class="idle-panel"><div class="idle-icon">📝</div>'
                       '<div class="idle-text">Synthesis output will appear here.</div></div>')

    elif view == _VIEWS[3]:
        if active_wf and active_wf.get("verify_entries"):
            wf_ph.html(CUSTOM_CSS + "<div>" + _render_verify(active_wf["verify_entries"]) + "</div>")
        else:
            wf_ph.html(CUSTOM_CSS + '<div class="idle-panel"><div class="idle-icon">🛡️</div>'
                       '<div class="idle-text">Verification rounds will appear here.</div></div>')

# ═══════════════════════════════════════════════════════════════════════════
# CHAT PANEL
# ═══════════════════════════════════════════════════════════════════════════

with col_chat:
    # ---- Empty state ----
    if not msgs and not is_running:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-icon-wrap">💡</div>'
            '<div class="empty-title">What would you like to research?</div>'
            '<div class="empty-desc">Apex searches multiple sources, synthesizes evidence, '
            'and fact-checks to deliver verified answers.</div>'
            '<div class="feature-row">'
            '<span class="feature-pill"><span class="fp-dot fp-purple"></span>Multi-source</span>'
            '<span class="feature-pill"><span class="fp-dot fp-blue"></span>Synthesis</span>'
            '<span class="feature-pill"><span class="fp-dot fp-teal"></span>Fact-check</span>'
            '<span class="feature-pill"><span class="fp-dot fp-green"></span>Verified</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="suggestion-label">Try one of these</div>', unsafe_allow_html=True)
        _sugs = _get_suggestions()
        sc1, sc2 = st.columns(2)
        for si, sug in enumerate(_sugs):
            col = sc1 if si % 2 == 0 else sc2
            with col:
                if st.button(f"{sug['icon']}  {sug['text']}", key=f"sug_{si}", use_container_width=True):
                    st.session_state.selected_suggestion = sug["text"]
                    st.rerun()

    # ---- Saved messages ----
    def _set_wf_idx(idx: int):
        st.session_state.viewing_wf_idx = idx

    for mi, m in enumerate(msgs):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant" and m.get("workflow"):
                wf = m["workflow"]
                ss = wf.get("step_states", {})
                done_n = sum(1 for v in ss.values() if v == "done")
                pct = int((done_n / max(len(STEP_ORDER), 1)) * 100)
                dots = ""
                for key in STEP_ORDER:
                    s = ss.get(key, "pending")
                    cmap = {"done": "var(--accent-green)", "running": "var(--accent-blue)",
                            "error": "var(--accent-red)", "pending": "rgba(255,255,255,.12)"}
                    c = cmap.get(s, "rgba(255,255,255,.12)")
                    dots += f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{c};margin-right:3px"></span>'
                is_active = st.session_state.viewing_wf_idx == mi
                label = "✅ Viewing workflow" if is_active else "🔍 View workflow"
                st.markdown(
                    f'<div class="msg-workflow-bar">{dots}'
                    f'<span style="font-size:.68rem;color:var(--text-muted);margin-left:4px">{pct}%</span></div>',
                    unsafe_allow_html=True,
                )
                st.button(label, key=f"wf_{mi}", on_click=_set_wf_idx, args=(mi,))

    # ---- In-progress query ----
    if is_running and st.session_state.running:
        rs = st.session_state.running
        rp = st.session_state.running_prompt or ""
        if rp:
            with st.chat_message("user"):
                st.markdown(rp)
        with st.chat_message("assistant"):
            draft = rs.get("draft_text", "")
            if draft:
                st.markdown(draft + " ▌")
            else:
                st.html(CUSTOM_CSS + _thinking_html(rs.get("current_label", "Thinking")))

    # ---- Launch query ----
    if prompt and not is_running and prompt != st.session_state._last_processed_prompt:
        st.session_state._last_processed_prompt = prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        payload: Dict[str, Any] = {"message": prompt}
        if fast_mode:
            payload.update({"best_of_n": 1, "verify_max_rounds": 1})
        rs = _new_running_state()
        st.session_state.running = rs
        st.session_state.running_prompt = prompt
        t = threading.Thread(
            target=_run_sse_worker,
            args=(f"{BACKEND_BASE_URL}/api/v1/chat/sse", payload, rs),
            daemon=True,
        )
        t.start()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Auto-rerun poller
# ─────────────────────────────────────────────────────────────────────────────

if is_running:
    time.sleep(1.0)
    st.rerun()
