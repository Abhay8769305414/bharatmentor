# BharatMentor — Architecture

## Overview

```
                    ┌─────────────────────────────┐
                    │      Streamlit UI            │
                    │   (frontend/streamlit_app.py) │
                    └──────────────┬───────────────┘
                                   │ HTTP
                                   ▼
                    ┌─────────────────────────────┐
                    │       FastAPI API             │
                    │       (app/main.py)           │
                    │                              │
                    │  POST /api/v1/chat           │
                    │  POST /api/v1/voice/chat     │
                    │  GET  /api/v1/progress/{id} │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │       Agent Loop             │
                    │       (app/agent.py)         │
                    │                              │
                    │  MAX_TOOL_ROUNDS = 5         │
                    └───┬───────┬────┬─────────────┘
                        │       │    │
           ┌────────────┘       │    └────────────┐
           ▼                    ▼                 ▼
  search_knowledge()   get_student_progress()  calculator()
  (rag/retrieval.py)   (app/db.py)             (AST eval)
           │                    │
           ▼                    ▼
  Embeddings Index         SQLite / PostgreSQL
  (data/index/)            (bharatmentor.db)
           │
           ▼
    LLM Provider
    (app/llm.py)
    Groq | Gemini
           │
           ▼
      Final Answer
           │
           ▼
    TTS Provider
    (voice/tts.py)
    gTTS | IndicF5 (future)
           ▲
           │
    STT Provider
    (voice/stt.py)
    faster-whisper
           ▲
           │
    Audio Input
```

## Voice Pipeline

```
Student speaks
      │
      ▼
Audio file (WAV/MP3/WebM)
      │
      ▼
faster-whisper STT
  - base or small model
  - VAD filter (skip silence)
  - Auto language detection
  - Latency measured
      │
      ▼
Language classifier
  - Devanagari/Latin ratio
  - "en" | "hi" | "hinglish"
      │
      ▼
Agent (text)
  - Tool selection
  - Tool execution
  - Source collection
      │
      ▼
gTTS / TTS Provider
  - Language-matched synthesis
  - Returns MP3 bytes
      │
      ▼
Student hears answer
```

## RAG Pipeline

```
PDF documents
      │ (PyMuPDF)
      ▼
Per-page text extraction
      │
      ▼
Text cleaning
      │
      ▼
Overlapping chunking
  - ~400 char chunks
  - 80 char overlap
  - Preserves: document, page, section
      │
      ▼
sentence-transformers embeddings
  - all-MiniLM-L6-v2 (384-dim)
      │
      ▼
numpy array saved to disk
  - data/index/chunks.json
  - data/index/embeddings.npy
      │
      ▼ (at query time)
Cosine similarity retrieval
  - Top-K results
  - Score attached to each chunk
      │
      ▼
Retrieved chunks → LLM context
      │
      ▼
Answer with source citations
```

## Provider Abstraction

All external services use interfaces:

```python
class LLMProvider:   # app/llm.py
class STTProvider:   # voice/stt.py
class TTSProvider:   # voice/tts.py
```

Switch providers by changing `.env` only.

## Database Schema

```
students
  id | name | email | created_at

student_progress
  id | student_id | topic | questions_attempted | correct_answers | accuracy | last_updated

conversation_logs
  id | student_id | role | content | language | created_at
```
