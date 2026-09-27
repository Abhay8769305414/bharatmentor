"""
BharatMentor — Test Suite: Tools
All four tools tested without LLM or network calls where possible.
"""

import pytest
from app.tools import calculator, search_knowledge, get_student_progress, generate_quiz


# ── Calculator ────────────────────────────────────────────────────────────────

class TestCalculator:
    def test_basic_addition(self):
        r = calculator("2 + 3")
        assert r["status"] == "ok"
        assert r["result"] == 5

    def test_multiplication(self):
        r = calculator("25 * 17")
        assert r["status"] == "ok"
        assert r["result"] == 425

    def test_division(self):
        r = calculator("10 / 4")
        assert r["status"] == "ok"
        assert abs(r["result"] - 2.5) < 1e-9

    def test_floor_division(self):
        r = calculator("10 // 3")
        assert r["status"] == "ok"
        assert r["result"] == 3

    def test_power(self):
        r = calculator("2 ** 10")
        assert r["status"] == "ok"
        assert r["result"] == 1024

    def test_complex_expression(self):
        r = calculator("(3**2 + 4**2)**0.5")
        assert r["status"] == "ok"
        assert abs(r["result"] - 5.0) < 1e-6

    def test_division_by_zero(self):
        r = calculator("1 / 0")
        assert r["status"] == "error"
        assert "zero" in r["error"].lower()

    def test_no_eval_injection(self):
        """Ensure arbitrary code cannot be injected."""
        r = calculator("__import__('os').system('echo pwned')")
        assert r["status"] == "error"

    def test_string_rejected(self):
        r = calculator("'hello'")
        assert r["status"] == "error"

    def test_negative_number(self):
        r = calculator("-5 * 3")
        assert r["status"] == "ok"
        assert r["result"] == -15

    def test_modulo(self):
        r = calculator("17 % 5")
        assert r["status"] == "ok"
        assert r["result"] == 2


# ── search_knowledge ──────────────────────────────────────────────────────────

class TestSearchKnowledge:
    def test_returns_dict_with_status(self):
        """search_knowledge should never raise — always returns a dict."""
        result = search_knowledge("Newton's law")
        assert isinstance(result, dict)
        assert "status" in result
        assert "results" in result

    def test_results_is_list(self):
        result = search_knowledge("gravity")
        assert isinstance(result["results"], list)

    def test_query_preserved(self):
        result = search_knowledge("projectile motion")
        assert result["query"] == "projectile motion"


# ── get_student_progress ──────────────────────────────────────────────────────

class TestGetStudentProgress:
    def test_existing_student(self, seeded_db):
        result = get_student_progress(1, seeded_db)
        assert result["status"] == "ok"
        assert result["student_id"] == 1
        assert len(result["progress"]) > 0
        # Each progress record should have required fields
        for p in result["progress"]:
            assert "topic" in p
            assert "accuracy" in p
            assert "questions_attempted" in p

    def test_nonexistent_student(self, seeded_db):
        result = get_student_progress(9999, seeded_db)
        assert result["status"] == "not_found"
        assert result["progress"] == []

    def test_accuracy_between_0_and_1(self, seeded_db):
        result = get_student_progress(1, seeded_db)
        for p in result["progress"]:
            assert 0.0 <= p["accuracy"] <= 1.0
