"""
LLM middleware system for cross-cutting concerns.

Middlewares intercept LLM calls and can:
- Log requests and responses
- Cache responses
- Track costs and token usage
- Modify prompts
- Short-circuit calls (e.g., return cached response)

The middleware pattern follows a "chain of responsibility" where each
middleware wraps the next, allowing pre-processing and post-processing
of LLM calls.

Example:
    class MyMiddleware(LlmMiddleware):
        async def process(self, messages, call_next):
            # Pre-processing
            print(f"Calling LLM with {len(messages)} messages")

            # Call the next middleware (or the actual LLM)
            response = await call_next(messages)

            # Post-processing
            print(f"Got response: {response[:50]}...")

            return response
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class LlmMiddleware(ABC):
    """
    Abstract base class for LLM middlewares.

    Middlewares wrap LLM calls to provide cross-cutting functionality.
    Each middleware receives the messages and a function to call the next
    middleware in the chain (or the actual LLM call if this is the last).

    Subclasses must implement the `process` method.

    Example:
        class TimingMiddleware(LlmMiddleware):
            async def process(self, messages, call_next):
                start = time.time()
                result = await call_next(messages)
                elapsed = time.time() - start
                print(f"LLM call took {elapsed:.2f}s")
                return result
    """

    @abstractmethod
    async def process(
        self,
        messages: list[dict[str, str]],
        call_next: Callable[[list[dict[str, str]]], Coroutine[Any, Any, str]],
    ) -> str:
        """
        Process an LLM call.

        Args:
            messages: The messages being sent to the LLM.
            call_next: Function to call the next middleware or LLM.

        Returns:
            str: The LLM response (possibly modified).
        """
        ...


class LoggingMiddleware(LlmMiddleware):
    """
    Middleware that logs LLM calls and responses.

    Logs include:
    - Number of messages and total characters
    - Execution time
    - Response length
    - Errors

    Attributes:
        log_level: Logging level (default: DEBUG).
        log_content: Whether to log message content (default: False for privacy).

    Example:
        service = LlmService(middlewares=[LoggingMiddleware(log_content=True)])
    """

    def __init__(
        self,
        log_level: int = logging.DEBUG,
        log_content: bool = False,
    ) -> None:
        """
        Initialize the logging middleware.

        Args:
            log_level: Logging level to use.
            log_content: Whether to log message content.
        """
        self.log_level = log_level
        self.log_content = log_content

    async def process(
        self,
        messages: list[dict[str, str]],
        call_next: Callable[[list[dict[str, str]]], Coroutine[Any, Any, str]],
    ) -> str:
        """
        Log the LLM call and delegate to the next middleware.

        Args:
            messages: The messages being sent to the LLM.
            call_next: Function to call the next middleware or LLM.

        Returns:
            str: The LLM response.
        """
        msg_count = len(messages)
        total_chars = sum(len(m.get("content", "")) for m in messages)

        logger.log(
            self.log_level,
            "LLM call starting: %d messages, %d chars",
            msg_count,
            total_chars,
        )

        if self.log_content:
            for i, msg in enumerate(messages):
                logger.log(
                    self.log_level,
                    "  [%d] %s: %s",
                    i,
                    msg.get("role", "?"),
                    msg.get("content", "")[:200],
                )

        start_time = time.time()
        try:
            response = await call_next(messages)
            elapsed = time.time() - start_time

            logger.log(
                self.log_level,
                "LLM call completed: %.2fs, %d chars response",
                elapsed,
                len(response),
            )

            if self.log_content:
                logger.log(
                    self.log_level,
                    "  Response: %s",
                    response[:200],
                )

            return response

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                "LLM call failed after %.2fs: %s",
                elapsed,
                str(e),
            )
            raise


class CachingMiddleware(LlmMiddleware):
    """
    Middleware that caches LLM responses.

    This is an in-memory cache that stores responses keyed by a hash
    of the input messages. Useful for:
    - Reducing API costs during development
    - Speeding up repeated queries
    - Testing

    Note: This is a simple in-memory cache. For production, consider
    using Redis or another distributed cache.

    Attributes:
        _cache: In-memory cache dictionary.
        max_size: Maximum number of entries to cache.
        ttl_seconds: Time-to-live for cache entries (0 = no expiry).

    Example:
        cache_middleware = CachingMiddleware(max_size=1000)
        service = LlmService(middlewares=[cache_middleware])

        # First call hits the LLM
        response1 = await service.invoke("What is 2+2?")

        # Second call returns cached response
        response2 = await service.invoke("What is 2+2?")
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 0,
    ) -> None:
        """
        Initialize the caching middleware.

        Args:
            max_size: Maximum cache entries (LRU eviction).
            ttl_seconds: Time-to-live in seconds (0 = no expiry).
        """
        self._cache: dict[str, tuple[str, float]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

    def _cache_key(self, messages: list[dict[str, str]]) -> str:
        """Generate a cache key from messages."""
        content = json.dumps(messages, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def _is_expired(self, timestamp: float) -> bool:
        """Check if a cache entry is expired."""
        if self.ttl_seconds <= 0:
            return False
        return time.time() - timestamp > self.ttl_seconds

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache is full."""
        if len(self._cache) >= self.max_size:
            # Remove oldest 10% of entries
            to_remove = max(1, self.max_size // 10)
            sorted_keys = sorted(
                self._cache.keys(),
                key=lambda k: self._cache[k][1],
            )
            for key in sorted_keys[:to_remove]:
                del self._cache[key]

    async def process(
        self,
        messages: list[dict[str, str]],
        call_next: Callable[[list[dict[str, str]]], Coroutine[Any, Any, str]],
    ) -> str:
        """
        Check cache before calling LLM, cache the response.

        Args:
            messages: The messages being sent to the LLM.
            call_next: Function to call the next middleware or LLM.

        Returns:
            str: The LLM response (cached or fresh).
        """
        key = self._cache_key(messages)

        # Check cache
        if key in self._cache:
            cached_response, timestamp = self._cache[key]
            if not self._is_expired(timestamp):
                logger.debug("Cache hit for LLM call")
                return cached_response
            else:
                del self._cache[key]

        # Cache miss - call LLM
        logger.debug("Cache miss for LLM call")
        response = await call_next(messages)

        # Store in cache
        self._evict_if_needed()
        self._cache[key] = (response, time.time())

        return response

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
        }


class RetryMiddleware(LlmMiddleware):
    """
    Middleware that retries failed LLM calls.

    Implements exponential backoff for transient failures.

    Attributes:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.

    Example:
        service = LlmService(middlewares=[
            RetryMiddleware(max_retries=3, base_delay=1.0)
        ])
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ) -> None:
        """
        Initialize the retry middleware.

        Args:
            max_retries: Maximum retry attempts.
            base_delay: Initial delay in seconds.
            max_delay: Maximum delay in seconds.
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    async def process(
        self,
        messages: list[dict[str, str]],
        call_next: Callable[[list[dict[str, str]]], Coroutine[Any, Any, str]],
    ) -> str:
        """
        Retry LLM calls on failure with exponential backoff.

        Args:
            messages: The messages being sent to the LLM.
            call_next: Function to call the next middleware or LLM.

        Returns:
            str: The LLM response.

        Raises:
            Exception: The last exception if all retries fail.
        """
        import asyncio

        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return await call_next(messages)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = min(
                        self.base_delay * (2**attempt),
                        self.max_delay,
                    )
                    logger.warning(
                        "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                        str(e),
                    )
                    await asyncio.sleep(delay)

        logger.error("LLM call failed after %d attempts", self.max_retries + 1)
        raise last_exception  # type: ignore
