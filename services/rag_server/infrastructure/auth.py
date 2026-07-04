"""Bearer-token auth for the server deployment tier (Task 2.2).

Local tier (default): RAG_SERVER_AUTH_TOKEN_FILE unset, dependency is a no-op.
Server tier: mount a Docker secret and set RAG_SERVER_AUTH_TOKEN_FILE to its
path; every route except /health then requires `Authorization: Bearer <token>`.
"""
import os
import secrets
from pathlib import Path

from fastapi import Header, HTTPException, status

_token_cache: str | None = None
_token_loaded = False


def _load_auth_token() -> str | None:
    global _token_cache, _token_loaded
    if _token_loaded:
        return _token_cache

    token_file = os.environ.get("RAG_SERVER_AUTH_TOKEN_FILE")
    if token_file:
        path = Path(token_file)
        if path.exists():
            _token_cache = path.read_text().strip().replace("\x00", "")

    _token_loaded = True
    return _token_cache


def reset_auth_token_cache() -> None:
    """Reset cached token. Useful for testing."""
    global _token_cache, _token_loaded
    _token_cache = None
    _token_loaded = False


async def require_bearer_token(authorization: str | None = Header(default=None)) -> None:
    """FastAPI dependency enforcing bearer-token auth when configured."""
    expected = _load_auth_token()
    if expected is None:
        return

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    provided = authorization[len("Bearer "):]
    if not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token")
