"""
main.py
-------
Application entrypoint.

Run locally:
    uvicorn main:app --reload --port 8000

Run via Docker:
    docker compose up

The full application logic lives in api/app.py.
"""

from api.app import app  # noqa: F401  re-exported for uvicorn

__all__ = ["app"]
