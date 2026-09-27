"""
BharatMentor — FastAPI Application
====================================
Endpoints:
  POST /api/v1/chat             — text chat
  POST /api/v1/voice/chat       — audio upload → STT → agent → TTS response
  GET  /api/v1/progress/{id}   — student progress
  GET  /api/v1/health           — health check
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import create_tables, get_db

settings = get_settings()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("BharatMentor API starting up…")
    create_tables()
    yield
    logger.info("BharatMentor API shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="BharatMentor API",
    description="Voice-first multilingual AI learning mentor",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    student_id: int = Field(default=1, ge=1)
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    language: str = Field(default="en")


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    tool_calls_made: list[dict[str, Any]]
    language: str
    error: str | None = None


class ProgressResponse(BaseModel):
    student_id: int
    progress: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    version: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["agent"])
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
) -> ChatResponse:
    """
    Text-based chat endpoint.
    Runs the full agent loop and returns the response.
    """
    from app.agent import run_agent

    try:
        result = run_agent(
            user_message=request.message,
            conversation_history=request.conversation_history,
            student_id=request.student_id,
            db=db,
        )
        return ChatResponse(
            answer=result.answer,
            sources=result.sources,
            tool_calls_made=result.tool_calls_made,
            language=request.language,
            error=result.error,
        )
    except Exception as exc:
        logger.error("Chat endpoint error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="I couldn't process your request. Please try again.",
        )


@app.post("/api/v1/voice/transcribe", tags=["voice"])
async def voice_transcribe(
    audio: UploadFile = File(..., description="Audio file (wav/mp3/ogg/webm)"),
    language_hint: str = Form(default="en"),
) -> dict[str, Any]:
    """
    Speech-to-Text endpoint.
    Accepts audio → transcribes using STT provider → returns transcript and language metadata.
    """
    from voice.stt import get_stt_provider

    content = await audio.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid audio input: audio file is empty.",
        )

    suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        stt = get_stt_provider()
        res = stt.transcribe(tmp_path, language_hint=language_hint)
        return {
            "status": "ok",
            "transcript": res.text,
            "language": res.language,
            "language_probability": res.language_probability,
            "latency_seconds": res.latency_seconds,
            "segments": res.segments,
        }
    except Exception as exc:
        logger.error("STT endpoint error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Speech recognition temporarily unavailable.",
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@app.post("/api/v1/voice/respond", tags=["voice"])
@app.post("/api/v1/voice/chat", tags=["voice"])
async def voice_chat(
    audio: UploadFile = File(..., description="Audio file (wav/mp3/ogg/webm)"),
    student_id: int = Form(default=1),
    language_hint: str = Form(default="en"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Full Voice Loop endpoint (STT → Agent → TTS).
    Accepts audio → transcribes → runs agent → synthesizes audio → returns text + audio_b64.
    """
    from voice.pipeline import voice_pipeline

    content = await audio.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid audio input: audio file is empty.",
        )

    suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        result = voice_pipeline(
            audio_path=tmp_path,
            student_id=student_id,
            language_hint=language_hint,
            db=db,
        )
        return result

    except Exception as exc:
        logger.error("Voice chat error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="I couldn't process the audio. Please try again.",
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@app.get(
    "/api/v1/progress/{student_id}",
    response_model=ProgressResponse,
    tags=["progress"],
)
async def get_progress(
    student_id: int,
    db: Session = Depends(get_db),
) -> ProgressResponse:
    """Return topic-level progress for a student."""
    from app.db import StudentProgress

    rows = (
        db.query(StudentProgress)
        .filter(StudentProgress.student_id == student_id)
        .all()
    )
    progress = [
        {
            "topic": r.topic,
            "questions_attempted": r.questions_attempted,
            "correct_answers": r.correct_answers,
            "accuracy": round(r.accuracy, 2),
        }
        for r in rows
    ]
    return ProgressResponse(student_id=student_id, progress=progress)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
