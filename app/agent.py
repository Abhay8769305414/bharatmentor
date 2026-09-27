"""
BharatMentor — Agent Loop
==========================
Agentic loop: model selects tool → tool executes → result injected → model responds.
Max tool rounds configurable to prevent infinite loops.

Design:
- Stateless: caller passes full conversation history
- Provider-agnostic: uses LLMProvider abstraction
- DB session injected for tools that need it (get_student_progress)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy.orm import Session

from app.llm import LLMProvider, get_llm_provider
from app.tools import (
    TOOL_DEFINITIONS,
    calculator,
    generate_quiz,
    get_student_progress,
    search_knowledge,
)

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 5

SYSTEM_PROMPT = """You are BharatMentor, a voice-first AI learning mentor for Indian students.
You support English, Hindi, and Hinglish conversations.

Guidelines:
- Always respond in the same language/style the student used.
- For subject questions, use search_knowledge to ground your answer in course material.
- Cite sources when answering from retrieved knowledge: (Source: <document>, Page <N>)
- For progress questions, use get_student_progress with the student's ID.
- For math, use the calculator tool.
- For quiz requests, use generate_quiz.
- Be concise, encouraging, and pedagogically sound.
- If you cannot answer confidently, say so honestly.
"""


class AgentResponse:
    """Structured result returned to the API layer."""

    def __init__(
        self,
        answer: str,
        tool_calls_made: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        error: str | None = None,
    ):
        self.answer = answer
        self.tool_calls_made = tool_calls_made
        self.sources = sources
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "tool_calls_made": self.tool_calls_made,
            "sources": self.sources,
            "error": self.error,
        }


def run_agent(
    user_message: str,
    conversation_history: list[dict[str, str]] | None = None,
    student_id: int = 1,
    db: Session | None = None,
    llm: LLMProvider | None = None,
) -> AgentResponse:
    """
    Execute the agent loop for one user turn.

    Args:
        user_message: The student's input (text, already transcribed from speech if voice).
        conversation_history: Prior messages [{"role": ..., "content": ...}].
        student_id: Used by get_student_progress.
        db: SQLAlchemy session (injected by FastAPI or caller).
        llm: LLMProvider instance (defaults to get_llm_provider()).

    Returns:
        AgentResponse with answer, tool call log, and source citations.
    """
    if llm is None:
        llm = get_llm_provider()

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": user_message})

    tool_calls_log: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []

    for round_num in range(MAX_TOOL_ROUNDS):
        logger.debug("Agent round %d — sending %d messages", round_num + 1, len(messages))

        response = llm.chat(messages, tools=TOOL_DEFINITIONS, tool_choice="auto")

        # ── Parse response ────────────────────────────────────────────────────
        finish_reason = _get_finish_reason(response)
        message = _get_message(response)

        if finish_reason == "tool_calls" or _has_tool_calls(message):
            tool_calls = _extract_tool_calls(message)
            # Append the assistant's tool-call message
            messages.append(_message_to_dict(message))

            for tc in tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_call_id = tc["id"]

                logger.info("Executing tool: %s(%s)", tool_name, tool_args)
                result = _dispatch_tool(tool_name, tool_args, student_id=student_id, db=db)

                tool_calls_log.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result_status": result.get("status", "unknown"),
                })

                # Collect sources from knowledge search
                if tool_name == "search_knowledge" and result.get("results"):
                    sources.extend(result["results"])

                # Inject tool result back into conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(result),
                })

        else:
            # Model gave a final answer — extract and return
            final_text = _extract_content_text(message)
            logger.info("Agent finished in %d round(s)", round_num + 1)
            return AgentResponse(
                answer=final_text,
                tool_calls_made=tool_calls_log,
                sources=_deduplicate_sources(sources),
            )

    # Exceeded max rounds — return best available answer
    logger.warning("Agent hit MAX_TOOL_ROUNDS=%d — returning partial response", MAX_TOOL_ROUNDS)
    last_content = _extract_content_text(_get_message(response))
    return AgentResponse(
        answer=last_content or "I reached the maximum number of reasoning steps. Please rephrase your question.",
        tool_calls_made=tool_calls_log,
        sources=_deduplicate_sources(sources),
        error="max_tool_rounds_exceeded",
    )


# ── Tool Dispatcher ───────────────────────────────────────────────────────────

def _dispatch_tool(
    name: str,
    args: dict[str, Any],
    student_id: int,
    db: Session | None,
) -> dict[str, Any]:
    try:
        if name == "search_knowledge":
            return search_knowledge(**args)
        if name == "get_student_progress":
            # Use student_id from args if provided, else fall back to context
            sid = args.get("student_id", student_id)
            if db is None:
                return {"status": "error", "error": "Database unavailable"}
            return get_student_progress(sid, db)
        if name == "generate_quiz":
            return generate_quiz(**args)
        if name == "calculator":
            return calculator(**args)
        return {"status": "error", "error": f"Unknown tool: {name}"}
    except Exception as exc:
        logger.error("Tool dispatch error for %s: %s", name, exc, exc_info=True)
        return {"status": "error", "error": str(exc)}


# ── Response Parsing Helpers (Groq/OpenAI format) ─────────────────────────────

def _get_finish_reason(response: Any) -> str:
    try:
        return response.choices[0].finish_reason or ""
    except Exception:
        return ""


def _get_message(response: Any) -> Any:
    try:
        return response.choices[0].message
    except Exception:
        return response


def _has_tool_calls(message: Any) -> bool:
    return bool(getattr(message, "tool_calls", None))


def _extract_tool_calls(message: Any) -> list[dict[str, Any]]:
    calls = []
    for tc in (message.tool_calls or []):
        try:
            args = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, AttributeError):
            args = {}
        calls.append({
            "id": tc.id,
            "name": tc.function.name,
            "args": args,
        })
    return calls


def _message_to_dict(message: Any) -> dict[str, Any]:
    """Convert a provider message object to a dict for the messages list."""
    d: dict[str, Any] = {"role": "assistant"}
    content = getattr(message, "content", None)
    if content:
        d["content"] = content
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
    return d


def _extract_content_text(message: Any) -> str:
    """Extract text content from a message object."""
    if hasattr(message, "content") and message.content:
        return message.content
    if hasattr(message, "text"):
        return message.text
    return str(message)


def _deduplicate_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for s in sources:
        key = (s.get("document"), s.get("page"), s.get("chunk_id"))
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique
