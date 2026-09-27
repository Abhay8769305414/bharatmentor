# 🎓 BharatMentor

**Voice-first Multilingual AI Learning Mentor — English, Hindi & Hinglish**

> Built as proof-of-work for an AI/ML Applied Engineer / Voice-first AI internship,
> specifically to demonstrate voice AI, multilingual interaction, RAG quality,
> and reproducible evaluation.

---

## Why BharatMentor?

Most AI tutoring systems are text-first English-only chatbots.
Indian students often think and speak in Hindi, Hinglish, or their mother tongue — not formal English.

BharatMentor focuses on:
1. **Voice-first interaction** — speak your question, hear the answer
2. **Multilingual** — English, Hindi, Hinglish supported
3. **Grounded answers** — RAG over course material, source-cited
4. **Measured quality** — WER, Recall@5, MRR, conversation evaluation

---

## Architecture

```
Streamlit UI
     │
     ▼
FastAPI (app/main.py)
     │
     ▼
Agent Loop (app/agent.py)
  ├── search_knowledge()   → RAG retrieval
  ├── get_student_progress() → SQLite/PostgreSQL
  ├── generate_quiz()      → LLM-generated JSON quiz
  └── calculator()         → Safe AST evaluator (no eval())
     │
     ▼
LLM Provider (Groq | Gemini) — configurable via .env

Voice Pipeline:
  Audio → faster-whisper STT → Agent → gTTS TTS → Audio
```

See [`docs/architecture.md`](docs/architecture.md) for detailed diagrams.

---

## Agent Workflow

```
Student speaks
      ↓ faster-whisper (base model, CPU, int8)
Transcript + language detection
      ↓
Agent (max 5 tool rounds)
      ↓ tool selected
Tool executes (RAG / DB / calculator / quiz)
      ↓
LLM synthesises answer with source citations
      ↓ gTTS
Audio response
```

---

## RAG Methodology

- **Corpus:** 2–3 educational PDFs (Physics — Mechanics)
- **Chunking:** ~400 char overlapping chunks, preserving document/page/section
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Retrieval:** vectorised cosine similarity
- **Fallback:** TF-IDF when index not built

Each answer includes source citations:
```
Sources:
  mechanics — Page 14
  mechanics — Page 16
```

---

## Evaluation

### RAG Evaluation

| Metric | Value | Target |
|--------|-------|--------|
| Recall@1 | **86.67%** | > 80% |
| Recall@3 | **100.0%** | > 90% |
| Recall@5 | **100.0%** | > 95% |
| MRR | **92.22%** | > 85% |

Dataset: 15 in-domain + 2 out-of-domain verified questions on Mechanics.
See [`experiments/rag/`](experiments/rag/).

### ASR Evaluation (English Baseline)

| Metric | Measured Value |
|--------|----------------|
| STT Engine | faster-whisper (`base`, CPU, `int8`) |
| Average RTF (Real-Time Factor) | **0.54** (2.3s audio in ~1.2s) |
| English Transcription Accuracy | **100%** on target queries |
| TTS Engine | gTTS (`GTTSProvider`, MP3) |

See [`experiments/asr/`](experiments/asr/).

### Conversation Evaluation

10 representative conversations evaluated on:
- Relevance, Correctness, Groundedness, Conciseness (LLM-judge, 1–5)
- Language consistency (deterministic check)

> ⚠️ LLM-judge scores are approximate. See [`docs/limitations.md`](docs/limitations.md).

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/yourhandle/bharatmentor
cd bharatmentor
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env: set GROQ_API_KEY (or GEMINI_API_KEY)
```

### 3. Seed database

```bash
python seed.py
```

### 4. Run tests

```bash
pytest tests/ -v
```

### 5. Start API

```bash
uvicorn app.main:app --reload
# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 6. Start Streamlit UI

```bash
streamlit run frontend/streamlit_app.py
```

---

## Build RAG Index

Place 1–2 Physics PDFs in `data/knowledge/`:
```bash
python -m rag.ingestion
```

Run evaluation:
```bash
python -m rag.evaluation
```

---

## Run Voice Pipeline Tests

```bash
# Requires audio samples in experiments/asr/dataset/en/ and /hi/
python experiments/asr/evaluate.py
```

---

## Project Structure

```
bharatmentor/
├── app/
│   ├── agent.py          # Agent loop
│   ├── tools.py          # Four tools
│   ├── db.py             # SQLAlchemy models
│   ├── llm.py            # LLM provider abstraction
│   ├── config.py         # pydantic-settings config
│   └── main.py           # FastAPI app
├── voice/
│   ├── stt.py            # faster-whisper STT
│   ├── tts.py            # gTTS TTS
│   └── pipeline.py       # Voice pipeline
├── rag/
│   ├── ingestion.py      # PDF → chunks → embeddings
│   ├── retrieval.py      # Cosine similarity retrieval
│   └── evaluation.py     # Recall@5, MRR
├── evaluation/
│   └── conversation_eval.py
├── frontend/
│   └── streamlit_app.py
├── experiments/
│   ├── rag/              # dataset.json, results.json, report.md
│   ├── asr/              # evaluate.py, dataset/, results.json
│   └── conversation/     # cases.json, results.json
├── tests/
│   ├── conftest.py       # Fixtures + mock LLMs
│   ├── test_tools.py
│   ├── test_db.py
│   ├── test_api.py
│   └── test_agent.py
├── docs/
│   ├── architecture.md
│   ├── limitations.md
│   └── progress.md
├── data/
│   ├── knowledge/        # Place PDFs here
│   └── index/            # Generated: chunks.json + embeddings.npy
├── seed.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## LLM Provider

Configurable via `.env`:

```env
LLM_PROVIDER=groq   # or gemini
GROQ_API_KEY=...
```

Groq free tier: https://console.groq.com
Gemini free tier: https://aistudio.google.com

---

## Limitations

See [`docs/limitations.md`](docs/limitations.md) for honest assessment of:
- gTTS internet requirement
- faster-whisper Hindi accuracy on CPU
- Hinglish ASR challenges
- LLM-as-judge reliability

---

## Related Work

**AgencyOS** (separate project) demonstrates:
- Multi-agent orchestration
- Redis + Docker infrastructure
- Complex API workflows
- Production auth

BharatMentor focuses specifically on voice AI and measurable evaluation quality.

---

## License

MIT
