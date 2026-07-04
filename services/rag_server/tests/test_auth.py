from pathlib import Path
import sys

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.auth import require_bearer_token, reset_auth_token_cache


@pytest.fixture
def app_client():
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(require_bearer_token)])
    def protected():
        return {"ok": True}

    with TestClient(app) as client:
        yield client

    reset_auth_token_cache()


def test_no_token_file_configured_allows_request(app_client, monkeypatch):
    """Local tier: env var unset means the dependency is a no-op."""
    monkeypatch.delenv("RAG_SERVER_AUTH_TOKEN_FILE", raising=False)
    reset_auth_token_cache()

    response = app_client.get("/protected")
    assert response.status_code == 200


def test_missing_token_rejected(app_client, monkeypatch, tmp_path):
    token_file = tmp_path / "auth_token"
    token_file.write_text("secret-token")
    monkeypatch.setenv("RAG_SERVER_AUTH_TOKEN_FILE", str(token_file))
    reset_auth_token_cache()

    response = app_client.get("/protected")
    assert response.status_code == 401


def test_wrong_token_rejected(app_client, monkeypatch, tmp_path):
    token_file = tmp_path / "auth_token"
    token_file.write_text("secret-token")
    monkeypatch.setenv("RAG_SERVER_AUTH_TOKEN_FILE", str(token_file))
    reset_auth_token_cache()

    response = app_client.get("/protected", headers={"Authorization": "Bearer wrong-token"})
    assert response.status_code == 401


def test_correct_token_accepted(app_client, monkeypatch, tmp_path):
    token_file = tmp_path / "auth_token"
    token_file.write_text("secret-token")
    monkeypatch.setenv("RAG_SERVER_AUTH_TOKEN_FILE", str(token_file))
    reset_auth_token_cache()

    response = app_client.get("/protected", headers={"Authorization": "Bearer secret-token"})
    assert response.status_code == 200
