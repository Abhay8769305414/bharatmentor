"""
BharatMentor — Voice Pipeline
================================
Orchestrates the full voice interaction loop:
  Audio → STT → language detection → Agent → TTS → response

Latency is measured at every stage and included in the response.
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.agent import run_agent
from voice.stt import get_stt_provider
from voice.tts import get_tts_provider

logger = logging.getLogger(__name__)


def detect_language_style(
    text: str,
    whisper_language: str,
) -> str:
    """
    Heuristically classify the language style of transcribed text.

    Returns: "en" | "hi" | "hinglish"

    Hinglish detection: text contains a mix of Devanagari and Latin characters,
    or uses common Hinglish patterns.
    """
    devanagari_chars = sum(1 for c in text if "\u0900" <= c <= "\u097f")
    latin_chars = sum(1 for c in text if c.isalpha() and c.isascii())
    total_alpha = devanagari_chars + latin_chars

    if total_alpha == 0:
        return whisper_language if whisper_language in ("en", "hi") else "en"

    devanagari_ratio = devanagari_chars / total_alpha

    if devanagari_ratio > 0.7:
        return "hi"
    if devanagari_ratio < 0.1:
        return "en"
    # Mixed — classify as Hinglish
    return "hinglish"


def voice_pipeline(
    audio_path: str | Path,
    student_id: int = 1,
    language_hint: str | None = None,
    db: Session | None = None,
) -> dict[str, Any]:
    """
    Full voice interaction pipeline.

    Args:
        audio_path: Path to audio file.
        student_id: Student identifier for progress tracking.
        language_hint: Optional language override ("en", "hi").
        db: SQLAlchemy session.

    Returns:
        Dict with transcript, response, language, audio_b64, and latency breakdown.
    """
    pipeline_start = time.perf_counter()
    latency: dict[str, float] = {}

    # ── 1. STT ────────────────────────────────────────────────────────────────
    try:
        stt = get_stt_provider()
        t0 = time.perf_counter()
        stt_result = stt.transcribe(audio_path, language_hint=language_hint)
        latency["stt_seconds"] = round(time.perf_counter() - t0, 3)

        transcript = stt_result.text
        if not transcript:
            return _error_response("I couldn't hear anything. Please speak clearly and try again.", latency)

    except Exception as exc:
        logger.error("STT failed: %s", exc, exc_info=True)
        return _error_response("I couldn't process the audio. Please try again.", latency)

    # ── 2. Language detection ─────────────────────────────────────────────────
    detected_lang = detect_language_style(transcript, stt_result.language)
    response_lang = detected_lang if detected_lang in ("en", "hi") else "en"

    # ── 3. Agent ──────────────────────────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        agent_result = run_agent(
            user_message=transcript,
            student_id=student_id,
            db=db,
        )
        latency["agent_seconds"] = round(time.perf_counter() - t0, 3)
        answer_text = agent_result.answer
        sources = agent_result.sources
        tool_calls = agent_result.tool_calls_made

    except Exception as exc:
        logger.error("Agent failed: %s", exc, exc_info=True)
        return _error_response(
            "I encountered a problem generating a response. Please try again.",
            latency,
            transcript=transcript,
            language=detected_lang,
        )

    # ── 4. TTS ────────────────────────────────────────────────────────────────
    audio_b64: str | None = None
    try:
        tts = get_tts_provider()
        t0 = time.perf_counter()
        audio_bytes = tts.synthesize(answer_text, language=response_lang)
        latency["tts_seconds"] = round(time.perf_counter() - t0, 3)
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    except Exception as exc:
        logger.error("TTS failed: %s", exc, exc_info=True)
        # TTS failure is non-fatal — return text response without audio
        latency["tts_error"] = str(exc)

    latency["total_seconds"] = round(time.perf_counter() - pipeline_start, 3)

    return {
        "status": "ok",
        "transcript": transcript,
        "language": detected_lang,
        "whisper_language": stt_result.language,
        "whisper_language_probability": stt_result.language_probability,
        "answer": answer_text,
        "sources": sources,
        "tool_calls_made": tool_calls,
        "audio_b64": audio_b64,   # base64-encoded MP3 or None if TTS failed
        "latency": latency,
    }


def _error_response(
    message: str,
    latency: dict[str, float],
    transcript: str = "",
    language: str = "en",
) -> dict[str, Any]:
    return {
        "status": "error",
        "transcript": transcript,
        "language": language,
        "whisper_language": None,
        "whisper_language_probability": None,
        "answer": message,
        "sources": [],
        "tool_calls_made": [],
        "audio_b64": None,
        "latency": latency,
    }
