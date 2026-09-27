"""
BharatMentor — Unit & Integration Tests for Voice Pipeline
===========================================================
Tests:
  - STT provider initialization and mocked transcription
  - TTS provider synthesis and failure handling
  - Voice pipeline orchestration (STT -> Agent -> TTS)
  - Error handling for invalid/empty audio and service failures
  - API endpoints (/api/v1/voice/transcribe, /api/v1/voice/respond, /api/v1/voice/chat)
"""

import io
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from voice.stt import STTProvider, TranscriptionResult, WhisperSTTProvider, get_stt_provider
from voice.tts import TTSProvider, GTTSProvider, get_tts_provider
from voice.pipeline import voice_pipeline, detect_language_style, _error_response
from app.agent import AgentResponse


# ── Mock STT & TTS Providers for Fast Unit Tests ─────────────────────────────

class DummySTT(STTProvider):
    def __init__(self, text: str = "Explain Newton's second law", lang: str = "en"):
        self.text = text
        self.lang = lang

    def transcribe(self, audio_path, language_hint=None) -> TranscriptionResult:
        return TranscriptionResult(
            text=self.text,
            language=self.lang,
            language_probability=0.99,
            latency_seconds=0.1,
            segments=[{"start": 0.0, "end": 2.0, "text": self.text}],
        )


class FailingSTT(STTProvider):
    def transcribe(self, audio_path, language_hint=None) -> TranscriptionResult:
        raise RuntimeError("STT hardware device error")


class DummyTTS(TTSProvider):
    def synthesize(self, text: str, language: str = "en") -> bytes:
        return b"fake_mp3_audio_bytes_12345"


class FailingTTS(TTSProvider):
    def synthesize(self, text: str, language: str = "en") -> bytes:
        raise RuntimeError("TTS connection error")


# ── Unit Tests: Language Detection ───────────────────────────────────────────

class TestLanguageDetection:
    def test_english_text(self):
        assert detect_language_style("What is Newton's second law?", "en") == "en"

    def test_hindi_devanagari(self):
        assert detect_language_style("न्यूटन का दूसरा नियम क्या है?", "hi") == "hi"

    def test_hinglish_mixed(self):
        # Mixture of Hindi Devanagari and English letters
        assert detect_language_style("Newton का second law क्या है explain karo", "hi") == "hinglish"


# ── Unit Tests: STT & TTS Abstractions ───────────────────────────────────────

class TestSTTProvider:
    @patch.object(WhisperSTTProvider, "_load_model")
    def test_whisper_provider_instantiation(self, mock_load):
        provider = WhisperSTTProvider()
        assert isinstance(provider, STTProvider)
        assert mock_load.called

    def test_mock_transcription(self, tmp_path):
        dummy_audio = tmp_path / "test.wav"
        dummy_audio.write_bytes(b"RIFF....WAVEfmt ")
        stt = DummySTT(text="What is force?")
        res = stt.transcribe(dummy_audio)
        assert res.text == "What is force?"
        assert res.language == "en"
        assert res.latency_seconds == 0.1


class TestTTSProvider:
    def test_gtts_provider_instantiation(self):
        tts = get_tts_provider()
        assert isinstance(tts, TTSProvider)

    def test_dummy_tts_synthesis(self):
        tts = DummyTTS()
        audio = tts.synthesize("Hello world", language="en")
        assert audio == b"fake_mp3_audio_bytes_12345"


# ── Unit Tests: Voice Pipeline ───────────────────────────────────────────────

class TestVoicePipeline:
    @patch("voice.pipeline.get_stt_provider")
    @patch("voice.pipeline.get_tts_provider")
    @patch("voice.pipeline.run_agent")
    def test_full_pipeline_success(self, mock_agent, mock_tts_prov, mock_stt_prov, tmp_path):
        mock_stt_prov.return_value = DummySTT("What is Newton's second law?")
        mock_tts_prov.return_value = DummyTTS()
        mock_agent.return_value = AgentResponse(
            answer="Force equals mass times acceleration (F=ma).",
            tool_calls_made=[{"tool": "search_knowledge", "args": {"query": "Newton's second law"}}],
            sources=[{"section": "Newton's Second Law"}],
        )

        dummy_audio = tmp_path / "audio.mp3"
        dummy_audio.write_bytes(b"dummy")

        result = voice_pipeline(dummy_audio, student_id=1)
        assert result["status"] == "ok"
        assert result["transcript"] == "What is Newton's second law?"
        assert "Force equals mass times acceleration" in result["answer"]
        assert result["audio_b64"] is not None
        assert result["tool_calls_made"][0]["tool"] == "search_knowledge"
        assert "stt_seconds" in result["latency"]
        assert "agent_seconds" in result["latency"]
        assert "tts_seconds" in result["latency"]
        assert "total_seconds" in result["latency"]

    @patch("voice.pipeline.get_stt_provider")
    def test_pipeline_stt_failure(self, mock_stt_prov, tmp_path):
        mock_stt_prov.return_value = FailingSTT()
        dummy_audio = tmp_path / "audio.mp3"
        dummy_audio.write_bytes(b"dummy")

        result = voice_pipeline(dummy_audio, student_id=1)
        assert result["status"] == "error"
        assert result["audio_b64"] is None
        assert "couldn't process the audio" in result["answer"]

    @patch("voice.pipeline.get_stt_provider")
    def test_pipeline_empty_speech(self, mock_stt_prov, tmp_path):
        mock_stt_prov.return_value = DummySTT(text="")
        dummy_audio = tmp_path / "audio.mp3"
        dummy_audio.write_bytes(b"dummy")

        result = voice_pipeline(dummy_audio, student_id=1)
        assert result["status"] == "error"
        assert "couldn't hear anything" in result["answer"]

    @patch("voice.pipeline.get_stt_provider")
    @patch("voice.pipeline.get_tts_provider")
    @patch("voice.pipeline.run_agent")
    def test_pipeline_tts_failure_is_non_fatal(self, mock_agent, mock_tts_prov, mock_stt_prov, tmp_path):
        mock_stt_prov.return_value = DummySTT("What is 10 + 10?")
        mock_tts_prov.return_value = FailingTTS()
        mock_agent.return_value = AgentResponse(answer="10 + 10 is 20.", tool_calls_made=[], sources=[])

        dummy_audio = tmp_path / "audio.mp3"
        dummy_audio.write_bytes(b"dummy")

        result = voice_pipeline(dummy_audio, student_id=1)
        # Should still succeed with text answer even if TTS failed
        assert result["status"] == "ok"
        assert result["answer"] == "10 + 10 is 20."
        assert result["audio_b64"] is None
        assert "tts_error" in result["latency"]


# ── Integration Tests: FastAPI Voice Endpoints ───────────────────────────────

class TestVoiceEndpoints:
    def test_transcribe_empty_file_rejected(self, test_client):
        response = test_client.post(
            "/api/v1/voice/transcribe",
            files={"audio": ("empty.wav", b"", "audio/wav")},
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_respond_empty_file_rejected(self, test_client):
        response = test_client.post(
            "/api/v1/voice/respond",
            files={"audio": ("empty.wav", b"", "audio/wav")},
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    @patch("voice.stt.get_stt_provider")
    def test_transcribe_success(self, mock_stt, test_client):
        mock_stt.return_value = DummySTT("Calculate 5 times 5")
        response = test_client.post(
            "/api/v1/voice/transcribe",
            files={"audio": ("test.wav", b"dummy_audio_bytes", "audio/wav")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["transcript"] == "Calculate 5 times 5"
        assert data["language"] == "en"

    @patch("voice.pipeline.get_stt_provider")
    @patch("voice.pipeline.get_tts_provider")
    def test_respond_endpoint_with_calculator(self, mock_tts, mock_stt, test_client):
        mock_stt.return_value = DummySTT("What is 25 multiplied by 17?")
        mock_tts.return_value = DummyTTS()

        response = test_client.post(
            "/api/v1/voice/respond",
            files={"audio": ("calc.wav", b"dummy_audio_bytes", "audio/wav")},
            data={"student_id": 1},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["transcript"] == "What is 25 multiplied by 17?"
        assert "425" in data["answer"]
        assert data["audio_b64"] is not None
