"""
fault_tolerance/retry.py
------------------------
Provides:
  - async_retry  — decorator for async functions with exponential backoff + jitter
  - CircuitBreaker — three-state circuit breaker (CLOSED / OPEN / HALF_OPEN)
  - CircuitState   — enum of breaker states
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from enum import Enum
from typing import Any, Callable, Coroutine, Iterable, Type

log = logging.getLogger(__name__)


# ── Retry decorator ────────────────────────────────────────────────────────────


def async_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: float = 0.5,
    exceptions: Iterable[Type[Exception]] = (Exception,),
) -> Callable:
    """
    Decorator: retry an async function up to `max_attempts` times using
    exponential backoff with uniform jitter.

    wait = min(base_delay * 2^attempt, max_delay) + uniform(0, jitter)

    Usage:
        @async_retry(max_attempts=3, base_delay=0.5)
        async def call_llm():
            ...
    """
    exc_tuple = tuple(exceptions)

    def decorator(fn: Callable[..., Coroutine]) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return await fn(*args, **kwargs)
                except exc_tuple as exc:
                    last_exc = exc
                    if attempt == max_attempts - 1:
                        break
                    wait = min(base_delay * (2 ** attempt), max_delay)
                    wait += random.uniform(0, jitter)
                    log.warning(
                        "[retry] %s attempt %d/%d failed (%s) — retrying in %.2fs",
                        fn.__name__, attempt + 1, max_attempts, exc, wait,
                    )
                    await asyncio.sleep(wait)
            log.error("[retry] %s exhausted %d attempts", fn.__name__, max_attempts)
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator


# ── Circuit Breaker ────────────────────────────────────────────────────────────


class CircuitState(str, Enum):
    CLOSED    = "closed"     # normal — all requests pass through
    OPEN      = "open"       # tripped — requests fail fast
    HALF_OPEN = "half_open"  # probing — one request let through


class CircuitBreakerOpen(Exception):
    """Raised when a call is blocked by an open circuit."""


class CircuitBreaker:
    """
    Thread-safe circuit breaker.

    State transitions
    -----------------
    CLOSED  → OPEN      after `failure_threshold` failures in `window` seconds
    OPEN    → HALF_OPEN after `recovery_timeout` seconds
    HALF_OPEN→ CLOSED   on success
    HALF_OPEN→ OPEN     on failure

    Synchronous API (used by tests and callers without async context):
        cb.allow_request() -> bool
        cb.record_success()
        cb.record_failure()

    Async convenience API:
        await cb.call(async_func, *args, **kwargs)
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        window: float = 60.0,
        recovery_timeout: float = 30.0,
    ) -> None:
        self.name               = name
        self.failure_threshold  = failure_threshold
        self.window             = window
        self.recovery_timeout   = recovery_timeout

        self._state:           CircuitState = CircuitState.CLOSED
        self._failure_count:   int          = 0
        self._window_start:    float        = time.monotonic()
        self._opened_at:       float        = 0.0

    # ── State ──────────────────────────────────────────────────────────────────

    @property
    def state(self) -> CircuitState:
        self._maybe_transition()
        return self._state

    def _maybe_transition(self) -> None:
        now = time.monotonic()
        if self._state == CircuitState.OPEN:
            if now - self._opened_at >= self.recovery_timeout:
                log.info("[circuit_breaker:%s] → HALF_OPEN", self.name)
                self._state = CircuitState.HALF_OPEN
        elif self._state == CircuitState.CLOSED:
            if now - self._window_start > self.window:
                self._failure_count = 0
                self._window_start  = now

    # ── Sync API ───────────────────────────────────────────────────────────────

    def allow_request(self) -> bool:
        """Return True if a request should be allowed through."""
        self._maybe_transition()
        return self._state != CircuitState.OPEN

    def record_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            log.info("[circuit_breaker:%s] probe succeeded → CLOSED", self.name)
        self._state         = CircuitState.CLOSED
        self._failure_count = 0
        self._window_start  = time.monotonic()

    def record_failure(self) -> None:
        self._failure_count += 1
        if self._state == CircuitState.HALF_OPEN:
            self._trip()
        elif (self._state == CircuitState.CLOSED and
              self._failure_count >= self.failure_threshold):
            self._trip()

    def _trip(self) -> None:
        self._state     = CircuitState.OPEN
        self._opened_at = time.monotonic()
        log.warning(
            "[circuit_breaker:%s] OPEN — %d failures in %.0fs window",
            self.name, self._failure_count, self.window,
        )

    # ── Async convenience ──────────────────────────────────────────────────────

    async def call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        if not self.allow_request():
            raise CircuitBreakerOpen(f"Circuit '{self.name}' is OPEN — failing fast")
        try:
            result = await fn(*args, **kwargs)
            self.record_success()
            return result
        except Exception as exc:
            self.record_failure()
            raise exc

    # ── Status ─────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "name":            self.name,
            "state":           self.state.value,
            "failure_count":   self._failure_count,
            "failure_threshold": self.failure_threshold,
        }


# ── Registry of all breakers for /health endpoint ─────────────────────────────

_registry: dict[str, CircuitBreaker] = {}


def get_breaker(name: str, **kwargs: Any) -> CircuitBreaker:
    if name not in _registry:
        _registry[name] = CircuitBreaker(name, **kwargs)
    return _registry[name]


def all_statuses() -> list[dict]:
    return [cb.status() for cb in _registry.values()]
