## Phase 0 — Audit (2026-09-27)

**Status:** ✅ COMPLETE — built from scratch (workspace was empty)

Files created: 25+ across app/, voice/, rag/, evaluation/, frontend/, tests/, docs/

---

## Phase 1 — Stabilize Agent + Database

**Status:** ✅ VERIFIED

### Test Results (2026-09-27)

```
============================= 35 passed in 1.63s ==============================

tests/test_agent.py::TestAgentDirectCalculator::test_calculator_tool_dispatched  PASSED
tests/test_agent.py::TestAgentDirectCalculator::test_calculator_result_correct   PASSED
tests/test_agent.py::TestAgentProgressTool::test_progress_tool_dispatched       PASSED
tests/test_agent.py::TestAgentSearchTool::test_search_tool_dispatched            PASSED
tests/test_agent.py::TestAgentMaxRounds::test_max_rounds_guard                  PASSED
tests/test_agent.py::TestSafeEval::test_basic                                   PASSED
tests/test_agent.py::TestSafeEval::test_nested                                  PASSED
tests/test_api.py::TestHealthEndpoint::test_health_ok                           PASSED
tests/test_api.py::TestProgressEndpoint::test_existing_student                  PASSED
tests/test_api.py::TestProgressEndpoint::test_nonexistent_student               PASSED
tests/test_api.py::TestChatEndpoint::test_calculator_via_chat                   PASSED
tests/test_api.py::TestChatEndpoint::test_empty_message_rejected                PASSED
tests/test_api.py::TestChatEndpoint::test_chat_returns_error_field              PASSED
tests/test_db.py::TestDatabase::test_student_created                            PASSED
tests/test_db.py::TestDatabase::test_progress_created                           PASSED
tests/test_db.py::TestDatabase::test_student_unique_email                       PASSED
tests/test_db.py::TestDatabase::test_progress_accuracy_calculated               PASSED
tests/test_db.py::TestDatabase::test_filter_by_student                          PASSED
tests/test_tools.py::TestCalculator::test_basic_addition                        PASSED
tests/test_tools.py::TestCalculator::test_multiplication                        PASSED
tests/test_tools.py::TestCalculator::test_division                              PASSED
tests/test_tools.py::TestCalculator::test_floor_division                        PASSED
tests/test_tools.py::TestCalculator::test_power                                 PASSED
tests/test_tools.py::TestCalculator::test_complex_expression                    PASSED
tests/test_tools.py::TestCalculator::test_division_by_zero                      PASSED
tests/test_tools.py::TestCalculator::test_no_eval_injection                     PASSED
tests/test_tools.py::TestCalculator::test_string_rejected                       PASSED
tests/test_tools.py::TestCalculator::test_negative_number                       PASSED
tests/test_tools.py::TestCalculator::test_modulo                                PASSED
tests/test_tools.py::TestSearchKnowledge::test_returns_dict_with_status        PASSED
tests/test_tools.py::TestSearchKnowledge::test_results_is_list                 PASSED
tests/test_tools.py::TestSearchKnowledge::test_query_preserved                 PASSED
tests/test_tools.py::TestGetStudentProgress::test_existing_student             PASSED
tests/test_tools.py::TestGetStudentProgress::test_nonexistent_student          PASSED
tests/test_tools.py::TestGetStudentProgress::test_accuracy_between_0_and_1    PASSED
```

### Database Seed (VERIFIED)
```
Students: 3
Progress records: 8
```

### API (VERIFIED — live server)
```
GET /api/v1/health → {"status": "ok", "version": "1.0.0"}
GET /api/v1/progress/1 → {
  "student_id": 1,
  "progress": [
    {"topic": "Mechanics", "questions_attempted": 20, "correct_answers": 16, "accuracy": 0.8},
    {"topic": "Thermodynamics", "questions_attempted": 15, "correct_answers": 10, "accuracy": 0.67},
    {"topic": "Optics", "questions_attempted": 8, "correct_answers": 5, "accuracy": 0.62}
  ]
}
```

### Calculator safety (VERIFIED)
- `__import__('os').system('echo pwned')` → `error: Disallowed expression node: Call` ✅
- `'hello'` → `error: Non-numeric constant` ✅
- `(3**2 + 4**2)**0.5` → `5.0` ✅

### Agent MAX_TOOL_ROUNDS guard (VERIFIED)
- Infinite-tool mock triggered 5 rounds then returned `error: max_tool_rounds_exceeded` ✅

---


---

## Phase 2 — RAG Build & Ingestion

**Status:** ✅ COMPLETE & VERIFIED

- Controlled knowledge base: `data/knowledge/mechanics.md` (12 core mechanics sections)
- Ingestion engine: Section-aware chunking preserving section hierarchy and heading metadata
- Indexed chunks: 22 chunks across 12 sections (`data/index/chunks.json`)
- Retrieval baseline: TF-IDF (sublinear TF, bigrams, stopword filtering) with out-of-domain detection threshold

---

## Phase 3 — RAG Evaluation

**Status:** ✅ COMPLETE & VERIFIED

Evaluation script: `experiments/rag/evaluate.py`
Dataset: `experiments/rag/dataset.json` (15 in-domain across direct/conceptual/application/paraphrased + 2 out-of-domain)

### Benchmark Results
- **Recall@1**: 0.8667 (86.7%)
- **Recall@3**: 1.0000 (100%)
- **Recall@5**: 1.0000 (100%)
- **MRR (Mean Reciprocal Rank)**: 0.9222 (92.2%)
- **OOD Detection**: Out-of-domain questions correctly flagged via relevance thresholding

Artifacts generated:
- `experiments/rag/results.json`
- `experiments/rag/report.md`
- `tests/test_rag.py` (8 new test cases, 43/43 total pytest suite passing)

---

## Phase 4 — Voice Pipeline (English First)

**Status:** ✅ COMPLETE & VERIFIED

### Real Audio & Hardware Benchmarks
- **STT Model**: `faster-whisper` (`base`, CPU, `int8` quantisation)
- **TTS Engine**: `gTTS` (`GTTSProvider`, MP3 output)
- **Average STT RTF**: `0.54` (transcription takes ~54% of actual audio duration)
- **Transcriptions Verified**:
  - Sample 1: "Explain Newton's second law in simple terms." -> `RTF=0.66`
  - Sample 2: "What is Newton's second law?" -> `RTF=0.62`
  - Sample 3: "What is 25 multiplied by 17?" -> `RTF=0.35`
  - Sample 4: "What is my physics score?" -> `RTF=0.54`

### End-to-End Voice Loops Verified
1. **RAG Voice Request**:
   - Audio -> STT ("Which is Newton's second law?") -> Agent -> `search_knowledge` -> RAG context retrieved -> TTS -> Audio generated (`SUCCESS`).
2. **Calculator Voice Request (Non-RAG)**:
   - Audio -> STT ("What is 25 multiplied by 17?") -> Agent -> `calculator` (25 * 17) -> Answer 425 -> TTS -> Audio generated (`SUCCESS`). No RAG invoked.
3. **Database Progress Request**:
   - Audio -> STT ("Which is my physics score?") -> Agent -> `get_student_progress` (student_id=1) -> Progress breakdown -> TTS -> Audio generated (`SUCCESS`).

### API Endpoints
- `POST /api/v1/voice/transcribe`: Audio upload -> STT -> transcript & language metadata.
- `POST /api/v1/voice/respond` & `POST /api/v1/voice/chat`: Full Voice Loop (STT -> Agent -> TTS).
- Total passing tests: **58/58** (15 voice unit/integration tests).

---

## Phase 6 — Hindi + Hinglish

**Status:** PENDING

Test: Upload Hindi audio or send Hinglish text via /api/v1/chat

---

## Phase 7 — ASR Evaluation

**Status:** PENDING — requires audio samples in `experiments/asr/dataset/`

Run: `python experiments/asr/evaluate.py`

---

## Phase 8 — Conversation Evaluation

**Status:** PENDING — requires live LLM API key

Run: `python evaluation/conversation_eval.py`

---

## Known Issues / Limitations

- gTTS requires internet access (uses Google TTS API)
- faster-whisper model download required on first run (~150MB for base)
- Hindi TTS via gTTS may not perfectly handle Hinglish code-switching
- LLM-as-judge scores are approximate (documented in report)
