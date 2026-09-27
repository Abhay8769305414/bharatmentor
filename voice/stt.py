"""
BharatMentor — Speech-to-Text (STT) Provider
==============================================
Primary: faster-whisper (CPU-friendly, base/small model)
Interface: STTProvider.transcribe(audio_path) → TranscriptionResult

Future providers (e.g. IndicConformer) can be added by subclassing STTProvider
and switching STT_PROVIDER env var — no other code changes needed.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    latency_seconds: float
    segments: list[dict] = field(default_factory=list)


# ── Base ──────────────────────────────────────────────────────────────────────

class STTProvider(ABC):
    @abstractmethod
    def transcribe(
        self,
        audio_path: str | Path,
        language_hint: str | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file (wav/mp3/ogg/webm/m4a).
            language_hint: ISO-639-1 code ("en", "hi") or None for auto-detect.

        Returns:
            TranscriptionResult with text, detected language, and latency.
        """


# ── faster-whisper ────────────────────────────────────────────────────────────

class WhisperSTTProvider(STTProvider):
    """
    Uses faster-whisper with CPU-friendly int8 quantisation.
    Model is loaded once and cached as a class attribute.
    """

    _model = None  # Module-level cache

    def __init__(self) -> None:
        if WhisperSTTProvider._model is None:
            self._load_model()

    @classmethod
    def _load_model(cls) -> None:
        try:
            from faster_whisper import WhisperModel
            from app.config import get_settings

            settings = get_settings()
            logger.info(
                "Loading faster-whisper model=%s device=%s compute_type=%s",
                settings.whisper_model_size,
                settings.whisper_device,
                settings.whisper_compute_type,
            )
            cls._model = WhisperModel(
                settings.whisper_model_size,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
            )
            logger.info("faster-whisper model loaded.")
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper not installed — pip install faster-whisper"
            ) from exc

    def transcribe(
        self,
        audio_path: str | Path,
        language_hint: str | None = None,
    ) -> TranscriptionResult:
        audio_path = str(audio_path)
        start = time.perf_counter()

        try:
            kwargs = {}
            if language_hint and language_hint != "auto":
                kwargs["language"] = language_hint

            segments_iter, info = WhisperSTTProvider._model.transcribe(
                audio_path,
                beam_size=5,
                vad_filter=True,          # skip silence
                vad_parameters={"min_silence_duration_ms": 500},
                **kwargs,
            )

            segments = []
            full_text_parts = []
            for seg in segments_iter:
                segments.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text.strip(),
                })
                full_text_parts.append(seg.text.strip())

            full_text = " ".join(full_text_parts).strip()
            latency = time.perf_counter() - start

            logger.info(
                "STT: language=%s (%.2f%%), latency=%.2fs, text='%s'",
                info.language,
                info.language_probability * 100,
                latency,
                full_text[:80],
            )

            return TranscriptionResult(
                text=full_text,
                language=info.language,
                language_probability=info.language_probability,
                latency_seconds=round(latency, 3),
                segments=segments,
            )

        except Exception as exc:
            latency = time.perf_counter() - start
            logger.error("STT transcription error: %s", exc, exc_info=True)
            raise RuntimeError(f"STT failed: {exc}") from exc


# ── Factory ───────────────────────────────────────────────────────────────────

_stt_instance: STTProvider | None = None


def get_stt_provider() -> STTProvider:
    global _stt_instance
    if _stt_instance is None:
        from app.config import get_settings
        settings = get_settings()
        provider = settings.stt_provider.lower()
        if provider == "whisper":
            _stt_instance = WhisperSTTProvider()
        else:
            raise ValueError(f"Unknown STT_PROVIDER='{provider}'. Supported: 'whisper'.")
    return _stt_instance
