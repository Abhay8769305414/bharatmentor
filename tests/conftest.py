"""
BharatMentor — pytest Fixtures
Shared fixtures: in-memory DB, seeded data, mock LLM providers, test client.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base, Student, StudentProgress, get_db
from app.main import app


# ── In-memory SQLite DB ───────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def engine():
    e = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(e)
    return e


@pytest.fixture(scope="session")
def SessionFactory(engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session")
def seeded_db(SessionFactory):
    db = SessionFactory()
    # Students
    students = [
        Student(id=1, name="Arjun Sharma", email="arjun@example.com"),
        Student(id=2, name="Priya Patel", email="priya@example.com"),
        Student(id=3, name="Rahul Gupta", email="rahul@example.com"),
    ]
    db.add_all(students)
    db.flush()

    # Progress
    progress_data = [
        (1, "Mechanics", 20, 16),
        (1, "Thermodynamics", 15, 10),
        (1, "Optics", 8, 5),
        (2, "Mechanics", 25, 22),
        (2, "Electricity", 18, 14),
        (3, "Mechanics", 10, 6),
        (3, "Optics", 12, 9),
        (3, "Nuclear Physics", 5, 3),
    ]
    for sid, topic, attempted, correct in progress_data:
        db.add(StudentProgress(
            student_id=sid,
            topic=topic,
            questions_attempted=attempted,
            correct_answers=correct,
            accuracy=correct / attempted,
        ))
    db.commit()
    yield db
    db.close()


# ── FastAPI test client ───────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_client(seeded_db):
    def override_get_db():
        try:
            yield seeded_db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


# ── Mock LLM Providers ────────────────────────────────────────────────────────

def _make_tool_call_response(tool_name: str, tool_args: dict, call_id: str = "call_1"):
    """Build a mock Groq-style response that requests a tool call."""
    tool_call = MagicMock()
    tool_call.id = call_id
    tool_call.function.name = tool_name
    tool_call.function.arguments = json.dumps(tool_args)

    message = MagicMock()
    message.content = None
    message.tool_calls = [tool_call]

    choice = MagicMock()
    choice.finish_reason = "tool_calls"
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


def _make_final_response(text: str):
    """Build a mock Groq-style response with a final text answer."""
    message = MagicMock()
    message.content = text
    message.tool_calls = None

    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.fixture
def mock_llm_calculator():
    """LLM that calls calculator("6 * 7") then gives a final answer."""
    llm = MagicMock()
    llm.chat.side_effect = [
        _make_tool_call_response("calculator", {"expression": "6 * 7"}),
        _make_final_response("6 times 7 is 42."),
    ]
    return llm


@pytest.fixture
def mock_llm_progress():
    """LLM that calls get_student_progress(student_id=1) then answers."""
    llm = MagicMock()
    llm.chat.side_effect = [
        _make_tool_call_response("get_student_progress", {"student_id": 1}),
        _make_final_response("You have attempted 20 Mechanics questions with 80% accuracy."),
    ]
    return llm


@pytest.fixture
def mock_llm_search():
    """LLM that calls search_knowledge then answers."""
    llm = MagicMock()
    llm.chat.side_effect = [
        _make_tool_call_response("search_knowledge", {"query": "Newton's second law"}),
        _make_final_response("Newton's second law states F = ma."),
    ]
    return llm


@pytest.fixture
def mock_llm_infinite_tools():
    """LLM that always requests a tool call — triggers MAX_TOOL_ROUNDS guard."""
    llm = MagicMock()
    llm.chat.return_value = _make_tool_call_response("calculator", {"expression": "1 + 1"})
    return llm
