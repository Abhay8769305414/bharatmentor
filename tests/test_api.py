"""
BharatMentor — Test Suite: FastAPI Endpoints
Uses httpx async client with TestClient (synchronous mode).
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app
from app.db import get_db


# Override DB dependency to use test DB
def override_get_db(seeded_db):
    def _override():
        try:
            yield seeded_db
        finally:
            pass
    return _override


class TestHealthEndpoint:
    def test_health_ok(self, test_client):
        response = test_client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestProgressEndpoint:
    def test_existing_student(self, test_client):
        response = test_client.get("/api/v1/progress/1")
        assert response.status_code == 200
        data = response.json()
        assert data["student_id"] == 1
        assert isinstance(data["progress"], list)
        assert len(data["progress"]) > 0

    def test_nonexistent_student(self, test_client):
        response = test_client.get("/api/v1/progress/9999")
        assert response.status_code == 200
        data = response.json()
        assert data["student_id"] == 9999
        assert data["progress"] == []


class TestChatEndpoint:
    def test_calculator_via_chat(self, test_client, mock_llm_calculator):
        """Test that the chat endpoint handles a calculator question."""
        with patch("app.agent.get_llm_provider", return_value=mock_llm_calculator):
            response = test_client.post(
                "/api/v1/chat",
                json={
                    "message": "What is 10 + 5?",
                    "student_id": 1,
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "tool_calls_made" in data

    def test_empty_message_rejected(self, test_client):
        response = test_client.post(
            "/api/v1/chat",
            json={"message": "", "student_id": 1},
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_chat_returns_error_field(self, test_client, mock_llm_calculator):
        with patch("app.agent.get_llm_provider", return_value=mock_llm_calculator):
            response = test_client.post(
                "/api/v1/chat",
                json={"message": "Hello", "student_id": 1},
            )
        assert response.status_code == 200
        # error field should be None or a string
        data = response.json()
        assert data.get("error") is None or isinstance(data.get("error"), str)
