"""
BharatMentor — Test Suite: Agent Loop
Tests the agent's tool selection, dispatch, and response assembly
using a mock LLM provider to avoid real API calls.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from app.agent import run_agent, AgentResponse
from app.tools import calculator


class TestAgentDirectCalculator:
    """Test agent with a mocked LLM that immediately calls the calculator tool."""

    def test_calculator_tool_dispatched(self, seeded_db, mock_llm_calculator):
        result = run_agent(
            user_message="What is 6 times 7?",
            student_id=1,
            db=seeded_db,
            llm=mock_llm_calculator,
        )
        assert isinstance(result, AgentResponse)
        assert result.answer  # should have some text
        # Calculator should appear in tool calls
        tool_names = [tc["tool"] for tc in result.tool_calls_made]
        assert "calculator" in tool_names

    def test_calculator_result_correct(self, seeded_db, mock_llm_calculator):
        # Direct tool test (not agent)
        r = calculator("6 * 7")
        assert r["status"] == "ok"
        assert r["result"] == 42


class TestAgentProgressTool:
    def test_progress_tool_dispatched(self, seeded_db, mock_llm_progress):
        result = run_agent(
            user_message="What is my progress?",
            student_id=1,
            db=seeded_db,
            llm=mock_llm_progress,
        )
        assert isinstance(result, AgentResponse)
        tool_names = [tc["tool"] for tc in result.tool_calls_made]
        assert "get_student_progress" in tool_names


class TestAgentSearchTool:
    def test_search_tool_dispatched(self, seeded_db, mock_llm_search):
        result = run_agent(
            user_message="Explain Newton's second law.",
            student_id=1,
            db=seeded_db,
            llm=mock_llm_search,
        )
        assert isinstance(result, AgentResponse)
        tool_names = [tc["tool"] for tc in result.tool_calls_made]
        assert "search_knowledge" in tool_names


class TestAgentMaxRounds:
    def test_max_rounds_guard(self, seeded_db, mock_llm_infinite_tools):
        """Agent should stop and return after MAX_TOOL_ROUNDS iterations."""
        result = run_agent(
            user_message="Keep calling tools forever.",
            student_id=1,
            db=seeded_db,
            llm=mock_llm_infinite_tools,
        )
        assert isinstance(result, AgentResponse)
        assert result.error == "max_tool_rounds_exceeded"


class TestSafeEval:
    """Directly test the safe AST evaluator without going through the agent."""

    def _eval(self, expr: str) -> float:
        from app.tools import _safe_eval
        import ast
        tree = ast.parse(expr, mode="eval")
        return _safe_eval(tree)

    def test_basic(self):
        assert self._eval("2 + 2") == 4.0

    def test_nested(self):
        assert abs(self._eval("(3**2 + 4**2)**0.5") - 5.0) < 1e-6
