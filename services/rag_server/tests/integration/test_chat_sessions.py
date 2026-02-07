"""
Chat session lifecycle tests via HTTP API.

Run with: pytest tests/integration/test_chat_sessions.py -v --run-integration --run-slow
"""
import uuid
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.mark.integration
class TestSessionCreationViaQuery:
    """Querying creates sessions and builds history."""

    def test_query_creates_session_and_history(
        self, api_client, test_document, session_cleanup
    ):
        session_id = str(uuid.uuid4())
        session_cleanup.append(session_id)

        resp = api_client.post(
            "/query",
            json={
                "query": "What does the test document contain?",
                "session_id": session_id,
            },
        )
        assert resp.status_code == 200

        # Check history has user + assistant messages
        hist_resp = api_client.get(f"/chat/history/{session_id}")
        assert hist_resp.status_code == 200
        messages = hist_resp.json().get("messages", [])
        roles = [m["role"] for m in messages]
        assert "user" in roles, f"History should contain user message. Roles: {roles}"
        assert "assistant" in roles, f"History should contain assistant message. Roles: {roles}"

    def test_second_query_grows_history(
        self, api_client, test_document, session_cleanup
    ):
        session_id = str(uuid.uuid4())
        session_cleanup.append(session_id)

        # First query
        api_client.post(
            "/query",
            json={"query": "First question about the document.", "session_id": session_id},
        )
        # Second query on same session
        api_client.post(
            "/query",
            json={"query": "Second follow-up question.", "session_id": session_id},
        )

        hist_resp = api_client.get(f"/chat/history/{session_id}")
        assert hist_resp.status_code == 200
        messages = hist_resp.json().get("messages", [])
        assert len(messages) >= 4, (
            f"Two queries should produce >= 4 messages (2 user + 2 assistant), "
            f"got {len(messages)}"
        )


@pytest.mark.integration
class TestSessionHistory:
    """Session history edge cases."""

    def test_nonexistent_session_returns_empty(self, api_client):
        fake_id = str(uuid.uuid4())
        resp = api_client.get(f"/chat/history/{fake_id}")
        assert resp.status_code == 200, (
            f"Nonexistent session history should return 200, got {resp.status_code}"
        )
        messages = resp.json().get("messages", [])
        assert len(messages) == 0, (
            f"Nonexistent session should have empty messages, got {len(messages)}"
        )


@pytest.mark.integration
class TestSessionClear:
    """Clearing session history."""

    def test_clear_empties_history(self, api_client, test_document, session_cleanup):
        session_id = str(uuid.uuid4())
        session_cleanup.append(session_id)

        # Query to create history
        api_client.post(
            "/query",
            json={"query": "Question to create history.", "session_id": session_id},
        )

        # Clear session
        clear_resp = api_client.post(
            "/chat/clear",
            json={"session_id": session_id},
        )
        assert clear_resp.status_code == 200

        # Verify history is empty
        hist_resp = api_client.get(f"/chat/history/{session_id}")
        assert hist_resp.status_code == 200
        messages = hist_resp.json().get("messages", [])
        assert len(messages) == 0, (
            f"History should be empty after clear, got {len(messages)} messages"
        )


@pytest.mark.integration
class TestTemporarySession:
    """Temporary sessions should not appear in session list."""

    def test_temporary_not_in_session_list(
        self, api_client, test_document, session_cleanup
    ):
        session_id = str(uuid.uuid4())
        # No need to add to session_cleanup — temporary sessions aren't persisted

        resp = api_client.post(
            "/query",
            json={
                "query": "Temporary session query.",
                "session_id": session_id,
                "is_temporary": True,
            },
        )
        assert resp.status_code == 200

        # Verify session does NOT appear in session list
        sessions_resp = api_client.get("/chat/sessions")
        assert sessions_resp.status_code == 200
        session_ids = [
            s["session_id"] for s in sessions_resp.json().get("sessions", [])
        ]
        assert session_id not in session_ids, (
            f"Temporary session {session_id} should not appear in /chat/sessions"
        )
