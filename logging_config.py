"""
logging_config.py
-----------------
Configures Python's standard logging for structured, levelled output.

Call `setup_logging()` once at application startup (done automatically
by api/app.py via import).

Format (development):
    2024-01-15 12:00:00 | INFO     | orchestrator | [orchestrator] new request ...

Format (production / JSON — set LOG_JSON=true):
    {"timestamp": "...", "level": "INFO", "logger": "orchestrator", "message": "..."}
"""

from __future__ import annotations

import logging
import os
import sys


def setup_logging(level: str | None = None) -> None:
    """
    Configure root logger.

    Parameters
    ----------
    level : str, optional
        Override log level (DEBUG, INFO, WARNING, ERROR).
        Defaults to LOG_LEVEL env var or INFO.
    """
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    use_json  = os.getenv("LOG_JSON", "false").lower() == "true"

    numeric_level = getattr(logging, log_level, logging.INFO)

    if use_json:
        formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt   = "%(asctime)s | %(levelname)-8s | %(name)-24s | %(message)s",
            datefmt= "%Y-%m-%d %H:%M:%S",
        )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(numeric_level)
    # Remove existing handlers to avoid duplicate output
    root.handlers.clear()
    root.addHandler(handler)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "anthropic", "redis"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).debug("Logging configured (level=%s)", log_level)


class _JsonFormatter(logging.Formatter):
    """Minimal JSON log formatter (no extra dependencies)."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        from datetime import datetime, timezone

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


# Auto-setup when imported (allows `import logging_config` pattern)
setup_logging()
