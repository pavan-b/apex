"""
CLI entry points for running the application.

These functions are registered as console scripts in pyproject.toml,
allowing you to run:
    uv run backend    # Start the FastAPI backend
    uv run ui         # Start the Streamlit UI
    uv run dev        # Start both in development mode
"""

import os
import subprocess
import sys


def run_backend():
    """
    Start the FastAPI backend server.

    Usage:
        uv run backend
        uv run backend --port 8080
    """
    import uvicorn

    # Get port from args or default
    port = 8000
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--port" and i + 2 < len(sys.argv):
            port = int(sys.argv[i + 2])
        elif arg.startswith("--port="):
            port = int(arg.split("=")[1])

    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )


def run_ui():
    """
    Start the Streamlit UI.

    Usage:
        uv run ui
        uv run ui --port 8501
    """
    # Get port from args or default
    port = "8501"
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--port" and i + 2 < len(sys.argv):
            port = sys.argv[i + 2]
        elif arg.startswith("--port="):
            port = arg.split("=")[1]

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "ui/streamlit_app.py",
        "--server.port",
        port,
    ]
    subprocess.run(cmd)


def run_phoenix():
    """
    Start the Phoenix observability server.

    Usage:
        uv run phoenix-serve
    """
    cmd = [sys.executable, "-m", "phoenix.server.main", "serve"]
    subprocess.run(cmd)


def run_dev():
    """
    Print instructions for development mode.

    Since we can't easily run multiple processes from one script,
    this prints the commands to run in separate terminals.
    """
    print("=" * 60)
    print("Development Mode - Run these in separate terminals:")
    print("=" * 60)
    print()
    print("1. Start Phoenix (optional, for tracing):")
    print("   uv run phoenix-serve")
    print()
    print("2. Start Backend:")
    print("   uv run backend")
    print()
    print("3. Start UI:")
    print("   uv run ui")
    print()
    print("=" * 60)


if __name__ == "__main__":
    # Quick test
    print("Available commands: backend, ui, phoenix-serve, dev")
