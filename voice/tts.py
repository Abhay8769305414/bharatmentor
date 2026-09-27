"""
BharatMentor — Text-to-Speech (TTS) Provider
==============================================
Primary: gTTS (Google TTS) — reliable, multilingual, free.
Interface: TTSProvider.synthesize(text, language) → bytes (MP3)

Supported language codes:
  "en"  — English
  "hi"  — Hindi

Future providers (IndicF5, etc.) can be added without changing call sites.
"""

from __future__ import annotations

import io
import logging
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ── Base ──────────────────────────────────────────────────────────────────────

class TTSProvider(ABC):
    @abstractmethod
    def synthesize(self, text: str, language: str = "en") -> bytes:
        """
        Convert text to speech audio (MP3 bytes).

        Args:
            text: Text to synthesize.
            language: ISO-639-1 language code ("en", "hi").

        Returns:
            MP3 audio as bytes.
        """


# ── gTTS Provider ─────────────────────────────────────────────────────────────

# Map from whisper/internal language codes to gTTS language codes
_LANG_MAP = {
    "en": "en",
    "hi": "hi",
    "hinglish": "hi",  # gTTS uses Hindi for Hinglish
    "mr": "mr",        # Marathi
    "ta": "ta",        # Tamil
    "te": "te",        # Telugu
    "bn": "bn",        # Bengali
}


class GTTSProvider(TTSProvider):
    def synthesize(self, text: str, language: str = "en") -> bytes:
        try:
            from gtts import gTTS

            gtts_lang = _LANG_MAP.get(language.lower(), "en")
            start = time.perf_counter()

            tts = gTTS(text=text, lang=gtts_lang, slow=False)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            audio_bytes = buf.read()

            latency = time.perf_counter() - start
            logger.info(
                "TTS: lang=%s, chars=%d, latency=%.2fs, size=%d bytes",
                gtts_lang,
                len(text),
                latency,
                len(audio_bytes),
            )
            return audio_bytes

        except ImportError as exc:
            raise RuntimeError("gTTS not installed — pip install gTTS") from exc
        except Exception as exc:
            logger.error("TTS synthesis error: %s", exc, exc_info=True)
            raise RuntimeError(f"TTS failed: {exc}") from exc


# ── Factory ───────────────────────────────────────────────────────────────────

_tts_instance: TTSProvider | None = None


def get_tts_provider() -> TTSProvider:
    global _tts_instance
    if _tts_instance is None:
        from app.config import get_settings
        settings = get_settings()
        provider = settings.tts_provider.lower()
        if provider == "gtts":
            _tts_instance = GTTSProvider()
        else:
            raise ValueError(f"Unknown TTS_PROVIDER='{provider}'. Supported: 'gtts'.")
    return _tts_instance
