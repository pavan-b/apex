"""
LLM service abstraction with middleware support.

This module provides a clean interface for interacting with LLMs
(OpenAI or Ollama via LangChain) while supporting middleware chains
for cross-cutting concerns like logging, caching, and cost tracking.

Provider selection is automatic:
  - If OPENAI_API_KEY is set → uses OpenAI (default model: gpt-5.1)
  - Otherwise → falls back to Ollama (default model: qwen3:8b)

The LlmService wraps the underlying LLM client and applies middlewares in order,
allowing each middleware to intercept, modify, or short-circuit LLM calls.

Example:
    # Create service with middlewares (provider auto-detected from env)
    service = LlmService(
        middlewares=[LoggingMiddleware(), CachingMiddleware()],
    )

    # Make LLM calls
    response = await service.invoke("What is the capital of France?")
    response = await service.chat([
        {"role": "user", "content": "Hello!"}
    ])
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from backend.app.core.middleware import LlmMiddleware

logger = logging.getLogger(__name__)


def _to_langchain_messages(messages: list[dict[str, str]]) -> list[BaseMessage]:
    """
    Convert dict-based messages to LangChain message objects.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.

    Returns:
        list[BaseMessage]: LangChain message objects.
    """
    lc_messages: list[BaseMessage] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))
    return lc_messages


class LlmService:
    """
    Abstraction over LLM providers with middleware support.

    LlmService provides a unified interface for making LLM calls while
    supporting a middleware chain pattern. Middlewares can:
    - Log requests and responses
    - Cache responses
    - Track costs and token usage
    - Modify prompts
    - Short-circuit calls (e.g., return cached response)

    Provider is auto-detected from environment variables:
    - If OPENAI_API_KEY is present → OpenAI (model defaults to gpt-5.1)
    - Otherwise → Ollama (model defaults to qwen3:8b)

    Attributes:
        provider: The active provider name ("openai" or "ollama").
        model: The model name (e.g., "gpt-5.1" or "qwen3:8b").
        base_url: The Ollama server URL (only used with Ollama provider).
        temperature: Sampling temperature.
        _client: The underlying LangChain BaseChatModel client.
        _middlewares: List of middleware instances.

    Example:
        service = LlmService(
            middlewares=[LoggingMiddleware()],
        )

        # Simple invocation
        response = await service.invoke("Explain quantum computing")

        # Chat with message history
        response = await service.chat([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ])
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        middlewares: Sequence[LlmMiddleware] | None = None,
    ) -> None:
        """
        Initialize the LLM service.

        Provider is auto-detected: if OPENAI_API_KEY env var is set, OpenAI
        is used; otherwise Ollama is used.

        Args:
            model: Model name override. Defaults to env var or provider default.
            base_url: Ollama server URL (ignored for OpenAI). Defaults to OLLAMA_BASE_URL.
            temperature: Sampling temperature. Defaults to OLLAMA_TEMPERATURE or 0.2.
            middlewares: List of middleware instances to apply to all calls.
        """
        self.temperature = temperature if temperature is not None else float(
            os.getenv("OLLAMA_TEMPERATURE", "0.2")
        )
        self._middlewares: list[LlmMiddleware] = list(middlewares) if middlewares else []

        # Auto-detect provider from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.provider = "openai"
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.1")
            self.base_url = ""  # Not used for OpenAI
            self._client: BaseChatModel = ChatOpenAI(
                model=self.model,
                api_key=openai_api_key,
                temperature=self.temperature,
            )
        else:
            self.provider = "ollama"
            self.model = model or os.getenv("OLLAMA_MODEL", "qwen3:8b")
            self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self._client = ChatOllama(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
            )

        logger.info(
            "LlmService initialized: provider=%s, model=%s, temperature=%s, middlewares=%d",
            self.provider,
            self.model,
            self.temperature,
            len(self._middlewares),
        )

    def add_middleware(self, middleware: LlmMiddleware) -> None:
        """
        Add a middleware to the chain.

        Middlewares are applied in order (first added = outermost).

        Args:
            middleware: The middleware instance to add.
        """
        self._middlewares.append(middleware)

    def get_client(self) -> BaseChatModel:
        """
        Get the underlying LangChain client.

        This is useful when you need direct access to the client for
        LangChain-specific operations (e.g., building chains).

        Returns:
            BaseChatModel: The underlying client instance.
        """
        return self._client

    async def invoke(self, prompt: str) -> str:
        """
        Send a single prompt to the LLM and get a response.

        This is a convenience method that wraps the prompt in a user message.

        Args:
            prompt: The prompt text.

        Returns:
            str: The LLM's response text.

        Example:
            response = await service.invoke("What is the capital of France?")
            print(response)  # "The capital of France is Paris."
        """
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages)

    async def chat(self, messages: list[dict[str, str]]) -> str:
        """
        Send a chat conversation to the LLM and get a response.

        This method applies all middlewares in order before making the
        actual LLM call.

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Returns:
            str: The LLM's response text.

        Example:
            response = await service.chat([
                {"role": "system", "content": "You are a pirate."},
                {"role": "user", "content": "Hello!"},
            ])
            print(response)  # "Ahoy, matey!"
        """
        # Build the middleware chain
        async def core_call(msgs: list[dict[str, str]]) -> str:
            lc_messages = _to_langchain_messages(msgs)
            result = await self._client.ainvoke(lc_messages)
            return str(result.content)

        # Wrap with middlewares (innermost first, so reverse order)
        call_fn: Callable[[list[dict[str, str]]], Coroutine[Any, Any, str]] = core_call
        for middleware in reversed(self._middlewares):
            call_fn = self._wrap_middleware(middleware, call_fn)

        return await call_fn(messages)

    def _wrap_middleware(
        self,
        middleware: LlmMiddleware,
        next_fn: Callable[[list[dict[str, str]]], Coroutine[Any, Any, str]],
    ) -> Callable[[list[dict[str, str]]], Coroutine[Any, Any, str]]:
        """
        Wrap a middleware around the next function in the chain.

        Args:
            middleware: The middleware to wrap.
            next_fn: The next function to call.

        Returns:
            A new async function that applies the middleware.
        """

        async def wrapped(messages: list[dict[str, str]]) -> str:
            return await middleware.process(messages, next_fn)

        return wrapped

    async def chat_stream(self, messages: list[dict[str, str]]):
        """
        Stream chat responses token by token.

        Note: Middlewares are not applied to streaming calls currently.
        This is a direct passthrough to the underlying client.

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Yields:
            str: Chunks of the response as they arrive.

        Example:
            async for chunk in service.chat_stream(messages):
                print(chunk, end="", flush=True)
        """
        lc_messages = _to_langchain_messages(messages)
        async for chunk in self._client.astream(lc_messages):
            if chunk.content:
                yield str(chunk.content)
