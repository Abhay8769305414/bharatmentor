"""
BharatMentor — Four Agent Tools
=================================
1. search_knowledge  — RAG retrieval over course material
2. get_student_progress — real DB lookup
3. generate_quiz     — structured quiz generation
4. calculator        — safe AST-based expression evaluator (no eval())

Each tool is a plain Python function.
The TOOL_DEFINITIONS list is the OpenAI-compatible tool schema consumed by the agent.
"""

from __future__ import annotations

import ast
import logging
import operator
from typing import Any

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ── Tool 1: search_knowledge ──────────────────────────────────────────────────

def search_knowledge(query: str, top_k: int = 5) -> dict[str, Any]:
    """
    Search the RAG knowledge base.
    Falls back to TF-IDF if sentence-transformer index is not yet built.
    Returns list of chunks with source metadata.
    """
    try:
        from rag.retrieval import retrieve
        results = retrieve(query, top_k=top_k)
        return {
            "status": "ok",
            "query": query,
            "results": results,
        }
    except Exception as exc:
        logger.error("search_knowledge failed: %s", exc, exc_info=True)
        return {
            "status": "error",
            "query": query,
            "results": [],
            "error": str(exc),
        }


# ── Tool 2: get_student_progress ──────────────────────────────────────────────

def get_student_progress(student_id: int, db: Session) -> dict[str, Any]:
    """
    Retrieve topic-level progress for a given student from the database.
    """
    try:
        from app.db import StudentProgress
        rows = (
            db.query(StudentProgress)
            .filter(StudentProgress.student_id == student_id)
            .all()
        )
        if not rows:
            return {
                "status": "not_found",
                "student_id": student_id,
                "progress": [],
            }
        progress = [
            {
                "topic": r.topic,
                "questions_attempted": r.questions_attempted,
                "correct_answers": r.correct_answers,
                "accuracy": round(r.accuracy, 2),
            }
            for r in rows
        ]
        return {
            "status": "ok",
            "student_id": student_id,
            "progress": progress,
        }
    except Exception as exc:
        logger.error("get_student_progress failed: %s", exc, exc_info=True)
        return {
            "status": "error",
            "student_id": student_id,
            "progress": [],
            "error": str(exc),
        }


# ── Tool 3: generate_quiz ─────────────────────────────────────────────────────

def generate_quiz(
    topic: str,
    number_of_questions: int = 3,
    difficulty: str = "medium",
) -> dict[str, Any]:
    """
    Generate a structured quiz.
    Uses the LLM directly to produce JSON questions.
    """
    try:
        from app.llm import get_llm_provider

        llm = get_llm_provider()
        prompt = (
            f"Generate exactly {number_of_questions} multiple-choice questions "
            f"on the topic '{topic}' at difficulty '{difficulty}'. "
            "Return ONLY a JSON array. Each element must have: "
            "question (string), options (array of 4 strings), answer (string), "
            "explanation (string). No extra text."
        )
        messages = [
            {"role": "system", "content": "You are an educational quiz generator. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ]
        response = llm.chat(messages, temperature=0.5)

        # Parse text from provider response
        raw_text = _extract_text(response)

        # Strip markdown code fences if present
        raw_text = raw_text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        import json
        questions = json.loads(raw_text.strip())

        return {
            "status": "ok",
            "topic": topic,
            "difficulty": difficulty,
            "questions": questions,
        }
    except Exception as exc:
        logger.error("generate_quiz failed: %s", exc, exc_info=True)
        return {
            "status": "error",
            "topic": topic,
            "questions": [],
            "error": str(exc),
        }


def _extract_text(response: Any) -> str:
    """Extract text content from a provider response object."""
    # Groq / OpenAI
    if hasattr(response, "choices"):
        return response.choices[0].message.content or ""
    # Gemini
    if hasattr(response, "text"):
        return response.text or ""
    return str(response)


# ── Tool 4: calculator ────────────────────────────────────────────────────────

# Allowed AST node types — nothing that can execute arbitrary code
_ALLOWED_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Num,        # Python < 3.8
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
}

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    if type(node) not in _ALLOWED_NODES:
        raise ValueError(f"Disallowed expression node: {type(node).__name__}")
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Non-numeric constant: {node.value!r}")
        return float(node.value)
    if isinstance(node, ast.Num):  # Python < 3.8 compatibility
        return float(node.n)
    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        op_fn = _OPERATORS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_fn(left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        op_fn = _OPERATORS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_fn(operand)
    raise ValueError(f"Unexpected node: {type(node).__name__}")


def calculator(expression: str) -> dict[str, Any]:
    """
    Safe arithmetic evaluator.
    Uses AST parsing — never calls eval() directly.
    Supports: + - * / // % **
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        # Return int if result is a whole number
        display = int(result) if result == int(result) else result
        return {
            "status": "ok",
            "expression": expression,
            "result": display,
        }
    except ZeroDivisionError:
        return {"status": "error", "expression": expression, "error": "Division by zero"}
    except Exception as exc:
        logger.warning("calculator error for '%s': %s", expression, exc)
        return {"status": "error", "expression": expression, "error": str(exc)}


# ── OpenAI-compatible Tool Schemas ────────────────────────────────────────────

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "Search the educational knowledge base (course material, textbooks). "
                "Use this when the student asks a subject-matter question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query in any supported language.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_student_progress",
            "description": (
                "Retrieve the student's learning progress — topics, accuracy, "
                "questions attempted. Use when the student asks about their performance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {
                        "type": "integer",
                        "description": "The student's numeric ID.",
                    },
                },
                "required": ["student_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_quiz",
            "description": (
                "Generate multiple-choice quiz questions on a topic. "
                "Use when the student asks for practice questions or a quiz."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The subject or topic for the quiz.",
                    },
                    "number_of_questions": {
                        "type": "integer",
                        "description": "How many questions to generate (default 3).",
                        "default": 3,
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": ["easy", "medium", "hard"],
                        "description": "Difficulty level.",
                        "default": "medium",
                    },
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate a safe arithmetic expression. "
                "Use for mathematical calculations the student asks for."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "An arithmetic expression, e.g. '25 * 17' or '(3**2 + 4**2)**0.5'.",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]
