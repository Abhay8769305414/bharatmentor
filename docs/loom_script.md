# BharatMentor — 3-Minute Loom Demo Script

## Overview

Total runtime: ~3 minutes
Recording setup: Loom with webcam + screen share
Window layout: Streamlit UI left half, FastAPI logs right half

---

## 0:00–0:20 — Introduction

**Say (camera + screen):**

> "I built BharatMentor as a voice-first AI learning mentor focused on
> English, Hindi, and Hinglish conversations.
>
> The main thing I wanted to explore was not just making an agent work —
> but actually measuring whether the voice and retrieval pipeline performs well.
>
> Let me show you."

**Show:** Streamlit hero screen, badges visible.

---

## 0:20–1:00 — Hindi Voice Interaction

**Action:** Switch to Voice tab in Streamlit, upload a pre-recorded WAV file.

**Audio content:**
> "Mujhe Newton ka second law simple example ke saath samjhao."

**Say (over the recording playing):**
> "I'm speaking in Hindi — asking for Newton's second law with a simple example."

**Show the pipeline visually:**
> Audio → STT → Agent → RAG → TTS

**As the response plays:**
> "Whisper detected Hindi, retrieved relevant chunks from the Mechanics PDF,
> and gTTS synthesised the response in Hindi.
>
> Notice the source citation — Mechanics, Page 14 — the answer is grounded."

**Highlight latency panel:** "STT: 1.2s, Agent: 2.3s, TTS: 0.4s — Total: ~4s on CPU."

---

## 1:00–1:30 — Agent Tool Showcase

**Switch to Text tab.**

**Type:**
> "Mera physics score kya hai?"

**Say:**
> "Now in Hinglish — asking for my physics score."

**Show terminal logs:** Agent selects `get_student_progress(student_id=1)`.

**Result appears:** Mechanics 80%, Thermodynamics 67%, Optics 62%.

**Say:**
> "The agent selected the right tool and returned real database data."

**Type:**
> "25 * 17"

**Show:** `calculator()` tool selected, result: 425.

**Say:**
> "Calculator uses AST-based safe evaluation — no eval() call, explicit allowlist of operators."

---

## 1:30–2:10 — RAG with Source Citations

**Type:**
> "Explain the law of conservation of momentum with an example."

**Show the response:**
- Answer text
- Source citations: `mechanics — Page X`
- Score: 0.82

**Say:**
> "The agent searched the Mechanics PDF, retrieved the top 3 chunks by cosine similarity,
> and the LLM synthesised a grounded answer.
>
> I can see exactly which pages the answer came from.
>
> This matters because without source grounding, the LLM might hallucinate."

---

## 2:10–2:40 — Evaluation Results

**Switch to the right panel — Evaluation Metrics section.**

**Say:**
> "The interesting part is that I actually measured this system.
>
> For RAG — I built a 12-question manually verified evaluation set on Mechanics.
> I ran retrieval against each question and calculated Recall@5 and MRR.
> These are real numbers from real retrieval — not guesses."

**Point to:**
- RAG Recall@5 metric
- MRR metric

**Say:**
> "For ASR — I recorded 10 English and 10 Hindi audio samples with verified transcripts.
> WER for English is [X]%, Hindi is [X]% with the base Whisper model.
>
> Hindi WER is higher — that's expected and documented as a known limitation."

**Point to:**
- English WER
- Hindi WER

---

## 2:40–3:00 — Engineering Takeaway

**Camera, lean forward slightly.**

**Say:**

> "The interesting engineering challenge here wasn't building another chatbot.
>
> I focused on three things:
> One — making the voice pipeline work reliably end-to-end.
> Two — building reproducible evaluations for retrieval and speech recognition.
> Three — being honest about limitations: gTTS needs internet, Hinglish ASR isn't perfect,
> the LLM judge scores are approximate.
>
> My other project, AgencyOS, demonstrates broader agent orchestration work.
> BharatMentor focuses specifically on voice AI and measurable quality.
>
> Thanks."

---

## Recording Checklist

- [ ] API server running (`uvicorn app.main:app --reload`)
- [ ] Streamlit running (`streamlit run frontend/streamlit_app.py`)
- [ ] Database seeded (`python seed.py`)
- [ ] RAG index built (`python -m rag.ingestion`)
- [ ] Evaluation run (`python -m rag.evaluation`)
- [ ] Pre-recorded Hindi WAV ready
- [ ] FastAPI logs visible in terminal
- [ ] Evaluation metrics visible in Streamlit right panel
- [ ] Loom recording window layout set (Streamlit left, terminal right)

---

## Backup Plan (if voice demo fails)

If audio upload fails during recording:
1. Use the Text tab to demonstrate the full flow
2. Say: "I'd normally use voice — let me type the same question to show the agent reasoning."
3. The agent pipeline, tool calls, and sources still display correctly in text mode.
