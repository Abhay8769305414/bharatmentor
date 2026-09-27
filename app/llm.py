"""
BharatMentor — LLM Provider Abstraction
Supports Groq, Gemini, and local MockLLMProvider (deterministic rule-based tool-caller for offline/local testing).
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Response Data structures for Mock Provider ───────────────────────────────

@dataclass
class _Function:
    name: str
    arguments: str

@dataclass
class _ToolCall:
    id: str
    function: _Function

@dataclass
class _Message:
    content: str | None
    tool_calls: list[_ToolCall] | None = None

@dataclass
class _Choice:
    finish_reason: str
    message: _Message

@dataclass
class _ChatCompletion:
    choices: list[_Choice]


# ── Base ──────────────────────────────────────────────────────────────────────

class LLMProvider(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.3,
    ) -> Any:
        """
        Send a chat completion request.
        Returns the raw provider response object.
        The agent layer is responsible for parsing it.
        """


# ── Groq ──────────────────────────────────────────────────────────────────────

class GroqProvider(LLMProvider):
    def __init__(self) -> None:
        try:
            from groq import Groq
            if not settings.groq_api_key or "your_" in settings.groq_api_key:
                raise ValueError("GROQ_API_KEY is not configured with a valid key.")
            self._client = Groq(api_key=settings.groq_api_key)
            self._model = settings.groq_model
            logger.info("GroqProvider initialised with model=%s", self._model)
        except ImportError as exc:
            raise RuntimeError("groq package not installed — pip install groq") from exc

    def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.3,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        try:
            return self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            logger.warning("Groq chat API call failed (%s) — falling back to MockLLMProvider", exc)
            return MockLLMProvider().chat(messages, tools, tool_choice, temperature)


# ── Gemini ────────────────────────────────────────────────────────────────────

class GeminiProvider(LLMProvider):
    """
    Thin wrapper around google-generativeai.
    """

    def __init__(self) -> None:
        try:
            import google.generativeai as genai
            if not settings.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is not set.")
            genai.configure(api_key=settings.gemini_api_key)
            self._genai = genai
            self._model_name = settings.gemini_model
            logger.info("GeminiProvider initialised with model=%s", self._model_name)
        except ImportError as exc:
            raise RuntimeError(
                "google-generativeai package not installed — pip install google-generativeai"
            ) from exc

    def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.3,
    ) -> Any:
        from google.generativeai.types import HarmCategory, HarmBlockThreshold

        history = []
        system_instruction = None

        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "system":
                system_instruction = content
            elif role == "user":
                history.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                history.append({"role": "model", "parts": [content]})

        model = self._genai.GenerativeModel(
            self._model_name,
            system_instruction=system_instruction,
            generation_config={"temperature": temperature},
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            },
        )

        chat_session = model.start_chat(history=history[:-1] if history else [])
        last_user = history[-1]["parts"][0] if history else ""
        return chat_session.send_message(last_user)


# ── Mock / Local Rule-based LLM Provider ──────────────────────────────────────

class MockLLMProvider(LLMProvider):
    """
    Offline/local rule-based agent provider for reliable local testing.
    Routes queries to the appropriate tools and synthesizes coherent responses.
    """

    def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.3,
    ) -> Any:
        # Check if the last message is a tool response
        last_msg = messages[-1] if messages else {}
        if last_msg.get("role") == "tool":
            try:
                tool_data = json.loads(last_msg.get("content", "{}"))
            except Exception:
                tool_data = {}

            # Generate final response based on tool output
            return self._synthesize_final_answer(messages, tool_data)

        # Look at the most recent user message
        user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_text = m.get("content", "")
                break

        low_text = user_text.lower()

        # 1. Check for Math / Calculation
        math_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:multiplied by|\*|times|divided by|/|plus|\+|minus|-)\s*(\d+(?:\.\d+)?)", low_text)
        if math_match or any(w in low_text for w in ["calculate", "multiplied by", "divided by", "plus", "minus", "power of", "sqrt"]):
            expr = user_text
            # Normalize common speech phrasing to math expression
            expr = re.sub(r"[^\d\+\-\*\/\.\(\)\^eE\s]", "", expr)
            if "multiplied by" in low_text:
                parts = re.findall(r"\d+(?:\.\d+)?", low_text)
                if len(parts) >= 2:
                    expr = f"{parts[0]} * {parts[1]}"
            elif "divided by" in low_text:
                parts = re.findall(r"\d+(?:\.\d+)?", low_text)
                if len(parts) >= 2:
                    expr = f"{parts[0]} / {parts[1]}"
            elif "plus" in low_text:
                parts = re.findall(r"\d+(?:\.\d+)?", low_text)
                if len(parts) >= 2:
                    expr = f"{parts[0]} + {parts[1]}"
            elif "minus" in low_text:
                parts = re.findall(r"\d+(?:\.\d+)?", low_text)
                if len(parts) >= 2:
                    expr = f"{parts[0]} - {parts[1]}"

            if not expr.strip():
                expr = "25 * 17"

            return _ChatCompletion(choices=[
                _Choice(
                    finish_reason="tool_calls",
                    message=_Message(
                        content=None,
                        tool_calls=[
                            _ToolCall(
                                id="call_calc",
                                function=_Function(
                                    name="calculator",
                                    arguments=json.dumps({"expression": expr.strip()}),
                                )
                            )
                        ]
                    )
                )
            ])

        # 2. Check for Student Progress / Score
        if any(w in low_text for w in ["score", "progress", "accuracy", "performance", "physics score"]):
            return _ChatCompletion(choices=[
                _Choice(
                    finish_reason="tool_calls",
                    message=_Message(
                        content=None,
                        tool_calls=[
                            _ToolCall(
                                id="call_prog",
                                function=_Function(
                                    name="get_student_progress",
                                    arguments=json.dumps({"student_id": 1}),
                                )
                            )
                        ]
                    )
                )
            ])

        # 3. Check for Physics / Knowledge Search
        if any(w in low_text for w in ["newton", "law", "force", "gravity", "energy", "friction", "kinematics", "motion", "explain", "what is", "which is"]):
            query = user_text
            if "newton" in low_text:
                query = "Newton's second law" if "second" in low_text else ("Newton's first law" if "first" in low_text else ("Newton's third law" if "third" in low_text else "Newton's laws of motion"))
            return _ChatCompletion(choices=[
                _Choice(
                    finish_reason="tool_calls",
                    message=_Message(
                        content=None,
                        tool_calls=[
                            _ToolCall(
                                id="call_search",
                                function=_Function(
                                    name="search_knowledge",
                                    arguments=json.dumps({"query": query}),
                                )
                            )
                        ]
                    )
                )
            ])

        # Default greeting / conversational response
        return _ChatCompletion(choices=[
            _Choice(
                finish_reason="stop",
                message=_Message(
                    content=f"Hello! I am BharatMentor, your AI physics and science mentor. You said: '{user_text}'. How can I help you today?",
                    tool_calls=None,
                )
            )
        ])

    def _synthesize_final_answer(self, messages: list[dict[str, str]], tool_data: dict[str, Any]) -> Any:
        # Check what kind of tool data we got
        if "result" in tool_data and isinstance(tool_data["result"], (int, float)):
            val = tool_data["result"]
            expr = tool_data.get("expression", "")
            ans = f"The result of {expr} is {val}." if expr else f"The calculated result is {val}."
            return _ChatCompletion(choices=[_Choice(finish_reason="stop", message=_Message(content=ans))])

        if "progress" in tool_data:
            progs = tool_data.get("progress", [])
            if progs:
                lines = [f"- {p['topic']}: {p['correct_answers']}/{p['questions_attempted']} correct ({int(p['accuracy']*100)}% accuracy)" for p in progs]
                ans = f"Here is your progress report:\n" + "\n".join(lines)
            else:
                ans = "No progress records found for this student."
            return _ChatCompletion(choices=[_Choice(finish_reason="stop", message=_Message(content=ans))])

        if "results" in tool_data:
            results = tool_data.get("results", [])
            if results:
                top = results[0]
                text = top.get("text", "")
                sec = top.get("section", "")
                ans = f"According to {sec}: {text[:300]}"
            else:
                ans = "I searched the knowledge base but could not find matching information."
            return _ChatCompletion(choices=[_Choice(finish_reason="stop", message=_Message(content=ans))])

        ans = "I have processed your request."
        return _ChatCompletion(choices=[_Choice(finish_reason="stop", message=_Message(content=ans))])


# ── Factory ───────────────────────────────────────────────────────────────────

def get_llm_provider() -> LLMProvider:
    provider = settings.llm_provider.lower()
    if provider == "groq":
        if settings.groq_api_key:
            try:
                return GroqProvider()
            except Exception as exc:
                logger.warning("GroqProvider initialisation failed: %s — falling back to MockLLMProvider", exc)
                return MockLLMProvider()
        else:
            logger.info("GROQ_API_KEY not provided — using MockLLMProvider for offline/local execution.")
            return MockLLMProvider()

    if provider == "gemini":
        if settings.gemini_api_key:
            try:
                return GeminiProvider()
            except Exception as exc:
                logger.warning("GeminiProvider initialisation failed: %s — falling back to MockLLMProvider", exc)
                return MockLLMProvider()
        else:
            logger.info("GEMINI_API_KEY not provided — using MockLLMProvider for offline/local execution.")
            return MockLLMProvider()

    if provider == "mock":
        return MockLLMProvider()

    raise ValueError(
        f"Unknown LLM_PROVIDER='{provider}'. Supported: 'groq', 'gemini', 'mock'."
    )
