from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-haiku-20241022"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.3

    # ── Queue ─────────────────────────────────────────────────────────────────
    queue_backend: Literal["redis"] = "redis"
    redis_url: str = "redis://localhost:6379"
    queue_consumer_group: str = "agentic_ai_group"
    queue_consumer_name: str = "consumer_1"
    queue_block_ms: int = 2000          # how long XREAD blocks before polling again
    queue_batch_size: int = 10          # max messages per XREAD call

    # ── Batching ──────────────────────────────────────────────────────────────
    batch_max_size: int = 8             # max items per LLM batch call
    batch_window_ms: int = 150          # max wait before flushing a partial batch

    # ── Retry / Circuit-breaker ───────────────────────────────────────────────
    max_retries: int = 3
    retry_base_delay_s: float = 1.0
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout_s: float = 30.0

    # ── Streaming ─────────────────────────────────────────────────────────────
    sse_ping_interval_s: float = 15.0

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
