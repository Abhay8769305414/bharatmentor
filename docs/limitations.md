# BharatMentor — Known Limitations

## Voice & STT

### gTTS (Text-to-Speech)
- Requires internet access — calls Google TTS API
- Hinglish code-switching may produce unnatural output (Hindi TTS for Hinglish text)
- For production: replace with a local model (IndicF5 or Kokoro TTS)

### faster-whisper (Speech-to-Text)
- `base` model: faster but lower accuracy, especially for Hindi
- `small` model: better accuracy, ~500MB RAM, 2–5s per utterance on CPU
- Real-time performance NOT claimed — latency varies by hardware
- Hindi transcription quality depends on audio clarity and accent

### Hinglish
- No single ASR model is optimised for Hinglish
- faster-whisper auto-detect may classify Hinglish as English or Hindi
- Language detection uses character-ratio heuristic — not perfect

---

## RAG

### TF-IDF fallback
- Used when the embedding index has not been built
- No semantic understanding — keyword matching only

### Embedding index
- Built from a small corpus (2–3 PDFs)
- Not suitable for a large knowledge base without chunking optimisation
- Recall degrades if documents are long and poorly structured

### LLM hallucination
- If retrieved chunks don't contain the answer, the LLM may still fabricate one
- Groundedness is evaluated but not enforced

---

## LLM

### Groq free tier
- Subject to rate limits and model availability
- Free tier models may change — fallback to Gemini documented in `.env.example`

### Gemini
- Tool calling via function declarations has slightly different formatting
- Gemini wrapper in `app/llm.py` is a best-effort implementation
- Recommended: use Groq for consistent tool-calling support

---

## Evaluation

### LLM-as-judge
- The same LLM that generates answers also evaluates them
- Scores are approximate quality indicators, not ground truth
- Human evaluation would be more reliable for final reporting

### ASR WER
- WER penalises all token differences equally
- Hindi/Devanagari romanisation inconsistencies inflate WER
- CER (character error rate) may be more informative for Hindi

### RAG evaluation dataset
- 12 questions hand-curated, focused on Mechanics
- Generalisation to other topics not measured

---

## Deployment

### SQLite
- Default for local development
- Not suitable for concurrent writes in production
- Switch to PostgreSQL by changing `DATABASE_URL` in `.env`

### No authentication
- The demo assumes a trusted local environment
- Student ID is passed as a parameter — not authenticated
- For production: add JWT or session-based auth

---

## Phase 11 (Indic models) — Not Yet Implemented

IndicConformer (ASR) and IndicF5 (TTS) require:
- AI4Bharat model weights (~2GB+)
- GPU recommended for real-time performance
- Additional dependencies (torch, transformers)

The provider architecture (`STTProvider`, `TTSProvider`) is designed to accept
these models without rewriting the application.

See [AI4Bharat](https://ai4bharat.iitm.ac.in/) for model details.
