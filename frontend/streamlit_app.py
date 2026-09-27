"""
BharatMentor — Streamlit Demo UI
==================================
Voice-first multilingual AI learning mentor.

Run:
    streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BharatMentor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_BASE = "http://localhost:8000/api/v1"

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Sans+Devanagari:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hero section */
.hero-container {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
    border-radius: 20px;
    padding: 40px 50px;
    margin-bottom: 30px;
    border: 1px solid rgba(255,255,255,0.08);
    position: relative;
    overflow: hidden;
}

.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.1;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.65);
    margin-top: 10px;
    font-weight: 300;
}

.hero-badges {
    margin-top: 20px;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

.badge {
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.4);
    color: #a5b4fc;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
}

/* Cards */
.section-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
}

.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #818cf8;
    margin-bottom: 10px;
}

.transcript-box {
    background: rgba(99,102,241,0.08);
    border-left: 3px solid #818cf8;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 1rem;
    color: #e2e8f0;
    font-style: italic;
}

.answer-box {
    background: rgba(16,185,129,0.06);
    border-left: 3px solid #10b981;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 1rem;
    color: #e2e8f0;
    line-height: 1.7;
}

.source-item {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.88rem;
    color: #fcd34d;
}

.lang-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}

.lang-en { background: rgba(59,130,246,0.2); color: #93c5fd; border: 1px solid rgba(59,130,246,0.3); }
.lang-hi { background: rgba(239,68,68,0.2); color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); }
.lang-hinglish { background: rgba(16,185,129,0.2); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.3); }

/* Latency bar */
.latency-item {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    color: #94a3b8;
    padding: 4px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}

.latency-value {
    color: #e2e8f0;
    font-weight: 500;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(168,85,247,0.1));
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}

.metric-number {
    font-size: 2rem;
    font-weight: 700;
    color: #a5b4fc;
}

.metric-label {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 4px;
}

/* Tool call pill */
.tool-pill {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    color: #c4b5fd;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-family: monospace;
    margin-right: 6px;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def send_text_message(message: str, student_id: int, history: list) -> dict:
    try:
        r = requests.post(
            f"{API_BASE}/chat",
            json={
                "message": message,
                "student_id": student_id,
                "conversation_history": history,
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to BharatMentor API. Is the server running?"}
    except Exception as e:
        return {"error": str(e)}


def send_audio(audio_bytes: bytes, student_id: int, lang_hint: str = "en") -> dict:
    try:
        r = requests.post(
            f"{API_BASE}/voice/chat",
            files={"audio": ("recording.wav", audio_bytes, "audio/wav")},
            data={"student_id": str(student_id), "language_hint": lang_hint},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to BharatMentor API. Is the server running?"}
    except Exception as e:
        return {"error": str(e)}


def lang_badge_html(lang: str) -> str:
    label_map = {"en": "🇬🇧 English", "hi": "🇮🇳 Hindi", "hinglish": "🌐 Hinglish"}
    css_map = {"en": "lang-en", "hi": "lang-hi", "hinglish": "lang-hinglish"}
    label = label_map.get(lang, lang.upper())
    css = css_map.get(lang, "lang-en")
    return f'<span class="lang-badge {css}">{label}</span>'


def autoplay_audio_html(audio_b64: str) -> str:
    return f"""
    <audio autoplay controls style="width:100%; border-radius:8px; margin-top:10px;">
        <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
    </audio>
    """


def load_eval_results() -> dict:
    """Load pre-computed evaluation results if they exist."""
    results = {}
    rag_file = Path("experiments/rag/results.json")
    asr_file = Path("experiments/asr/results.json")
    conv_file = Path("experiments/conversation/results.json")

    if rag_file.exists():
        with open(rag_file) as f:
            results["rag"] = json.load(f)
    if asr_file.exists():
        with open(asr_file) as f:
            results["asr"] = json.load(f)
    if conv_file.exists():
        with open(conv_file) as f:
            results["conversation"] = json.load(f)
    return results


# ── Session state ─────────────────────────────────────────────────────────────
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "student_id" not in st.session_state:
    st.session_state.student_id = 1


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <p class="hero-title">🎓 BharatMentor</p>
    <p class="hero-subtitle">Voice-first Multilingual AI Learning Mentor</p>
    <div class="hero-badges">
        <span class="badge">🎤 Voice AI</span>
        <span class="badge">🇮🇳 Hindi + Hinglish</span>
        <span class="badge">📚 RAG</span>
        <span class="badge">📊 Evaluated</span>
        <span class="badge">⚡ faster-whisper</span>
    </div>
</div>
""", unsafe_allow_html=True)

# API status
api_ok = check_api_health()
if api_ok:
    st.success("✅ API Connected — BharatMentor is ready")
else:
    st.error("❌ API not reachable — run: `uvicorn app.main:app --reload`")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.session_state.student_id = st.selectbox(
        "Student Profile",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Arjun (Physics)", 2: "Priya (Advanced)", 3: "Rahul (Beginner)"}[x],
    )
    lang_hint = st.selectbox("Language Hint", ["auto", "en", "hi"], index=0)
    st.markdown("---")
    st.markdown("### 💡 Try Asking")
    examples = [
        "What is Newton's second law?",
        "Mujhe momentum samjhao.",
        "What is my physics score?",
        "Calculate 25 × 17",
        "Give me 3 quiz questions on Mechanics",
        "Newton ka second law real example ke saath explain karo",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
            st.session_state["prefill_text"] = ex

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.conversation_history = []
        st.session_state.last_result = None
        st.rerun()


# ── Main columns ──────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    # ── Input section ─────────────────────────────────────────────────────────
    st.markdown("### 💬 Ask BharatMentor")

    tab_text, tab_voice = st.tabs(["⌨️ Text", "🎤 Voice"])

    with tab_text:
        prefill = st.session_state.pop("prefill_text", "")
        user_input = st.text_area(
            "Type your question (English, Hindi, or Hinglish)",
            value=prefill,
            height=100,
            placeholder="e.g. What is Newton's second law? / Mujhe gravity samjhao.",
            key="text_input",
            label_visibility="collapsed",
        )
        send_btn = st.button("🚀 Ask", type="primary", use_container_width=True)

        if send_btn and user_input.strip() and api_ok:
            with st.spinner("Thinking…"):
                result = send_text_message(
                    user_input.strip(),
                    st.session_state.student_id,
                    st.session_state.conversation_history,
                )
                if "error" not in result:
                    st.session_state.conversation_history.append(
                        {"role": "user", "content": user_input.strip()}
                    )
                    st.session_state.conversation_history.append(
                        {"role": "assistant", "content": result.get("answer", "")}
                    )
                    # Keep last 10 turns
                    st.session_state.conversation_history = st.session_state.conversation_history[-20:]
                st.session_state.last_result = result
                result["input_text"] = user_input.strip()
                result["is_voice"] = False

    with tab_voice:
        st.info("📤 Upload a WAV/MP3 audio file of your question.")
        uploaded = st.file_uploader(
            "Upload audio",
            type=["wav", "mp3", "ogg", "webm", "m4a"],
            label_visibility="collapsed",
        )
        voice_send = st.button("🎤 Transcribe & Ask", type="primary", use_container_width=True)

        if voice_send and uploaded and api_ok:
            with st.spinner("Processing audio…"):
                audio_bytes = uploaded.read()
                result = send_audio(
                    audio_bytes,
                    st.session_state.student_id,
                    lang_hint if lang_hint != "auto" else "en",
                )
                st.session_state.last_result = result
                result["is_voice"] = True

    # ── Response display ──────────────────────────────────────────────────────
    result = st.session_state.last_result
    if result:
        if "error" in result and result.get("status") == "error" or "error" in str(result.get("error", "")):
            err_msg = result.get("error") or result.get("answer", "Unknown error")
            st.error(f"⚠️ {err_msg}")
        else:
            st.markdown("---")

            # Transcript (voice only)
            if result.get("is_voice") and result.get("transcript"):
                st.markdown('<div class="section-label">🎤 Transcript</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="transcript-box">"{result["transcript"]}"</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("")

            # Language
            lang = result.get("language", "en")
            st.markdown(
                f'<div class="section-label">🌐 Language &nbsp; {lang_badge_html(lang)}</div>',
                unsafe_allow_html=True,
            )

            # Tool calls made
            if result.get("tool_calls_made"):
                tools_html = "".join(
                    f'<span class="tool-pill">🔧 {tc["tool"]}</span>'
                    for tc in result["tool_calls_made"]
                )
                st.markdown(
                    f'<div class="section-label">🛠️ Tools Used &nbsp; {tools_html}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("")

            # Answer
            st.markdown('<div class="section-label">💡 Answer</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="answer-box">{result.get("answer", "")}</div>',
                unsafe_allow_html=True,
            )

            # Audio playback
            if result.get("audio_b64"):
                st.markdown("**🔊 Listen:**")
                st.markdown(autoplay_audio_html(result["audio_b64"]), unsafe_allow_html=True)

            # Sources
            sources = result.get("sources", [])
            if sources:
                st.markdown("")
                st.markdown('<div class="section-label">📖 Sources</div>', unsafe_allow_html=True)
                for s in sources[:3]:
                    score = s.get("score", 0)
                    doc = s.get("document", "unknown")
                    page = s.get("page", "?")
                    section = s.get("section", "")
                    st.markdown(
                        f'<div class="source-item">📄 <strong>{doc}</strong> — Page {page} &nbsp; '
                        f'<span style="opacity:0.6">{section}</span> &nbsp; '
                        f'<span style="opacity:0.5">score: {score:.3f}</span></div>',
                        unsafe_allow_html=True,
                    )

            # Latency
            if result.get("latency"):
                lat = result["latency"]
                st.markdown("")
                st.markdown('<div class="section-label">⏱️ Latency</div>', unsafe_allow_html=True)
                lat_rows = [
                    ("STT", lat.get("stt_seconds")),
                    ("Agent", lat.get("agent_seconds")),
                    ("TTS", lat.get("tts_seconds")),
                    ("Total", lat.get("total_seconds")),
                ]
                for label, val in lat_rows:
                    if val is not None:
                        st.markdown(
                            f'<div class="latency-item"><span>{label}</span>'
                            f'<span class="latency-value">{val:.2f}s</span></div>',
                            unsafe_allow_html=True,
                        )


# ── Right column: Progress + Evaluation ───────────────────────────────────────
with col_right:

    # Student progress
    st.markdown("### 📊 Student Progress")
    try:
        pr = requests.get(
            f"{API_BASE}/progress/{st.session_state.student_id}",
            timeout=5,
        )
        if pr.status_code == 200:
            progress_data = pr.json().get("progress", [])
            if progress_data:
                for p in progress_data:
                    accuracy = p["accuracy"]
                    color = "#10b981" if accuracy >= 0.7 else "#f59e0b" if accuracy >= 0.5 else "#ef4444"
                    st.markdown(
                        f"""<div class="section-card">
                        <div style="display:flex; justify-content:space-between; align-items:center">
                            <span style="font-weight:500; color:#e2e8f0">{p['topic']}</span>
                            <span style="color:{color}; font-weight:600">{accuracy:.0%}</span>
                        </div>
                        <div style="color:#64748b; font-size:0.82rem; margin-top:4px">
                            {p['correct_answers']}/{p['questions_attempted']} correct
                        </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No progress data yet.")
    except Exception:
        if not api_ok:
            st.warning("API not running.")

    # Evaluation metrics
    st.markdown("### 🧪 Evaluation Metrics")
    eval_results = load_eval_results()

    if eval_results:
        rag = eval_results.get("rag", {})
        asr = eval_results.get("asr", {})
        conv = eval_results.get("conversation", {})

        # RAG metrics
        recall_key = next((k for k in rag if k.startswith("recall_at_")), None)
        if recall_key:
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-number">{rag[recall_key]:.0%}</div>'
                    f'<div class="metric-label">RAG Recall@5</div></div>',
                    unsafe_allow_html=True,
                )
            with col_r2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-number">{rag.get("mrr", 0):.2f}</div>'
                    f'<div class="metric-label">RAG MRR</div></div>',
                    unsafe_allow_html=True,
                )

        # ASR WER
        if asr.get("languages"):
            wer_cols = st.columns(len(asr["languages"]))
            for i, lang_data in enumerate(asr["languages"]):
                wer = lang_data.get("wer")
                with wer_cols[i]:
                    wer_str = f"{wer:.0%}" if wer is not None else "N/A"
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-number">{wer_str}</div>'
                        f'<div class="metric-label">{lang_data["language"].upper()} WER</div></div>',
                        unsafe_allow_html=True,
                    )

        # Conversation
        if conv.get("avg_relevance"):
            st.markdown(
                f'<div class="metric-card"><div class="metric-number">{conv["avg_relevance"]:.1f}/5</div>'
                f'<div class="metric-label">Conversation Quality (LLM-judge)</div></div>',
                unsafe_allow_html=True,
            )
            st.caption("⚠️ LLM-judge scores are approximate — see experiments/ for methodology.")
    else:
        st.info(
            "No evaluation results yet. Run:\n\n"
            "```\npython -m rag.evaluation\npython experiments/asr/evaluate.py\n```"
        )

    # Conversation history
    st.markdown("### 🕒 Conversation History")
    history = st.session_state.conversation_history
    if history:
        for msg in reversed(history[-6:]):
            role = msg["role"]
            icon = "🧑" if role == "user" else "🤖"
            bg = "rgba(99,102,241,0.08)" if role == "user" else "rgba(16,185,129,0.06)"
            content_preview = msg["content"][:150] + "…" if len(msg["content"]) > 150 else msg["content"]
            st.markdown(
                f'<div style="background:{bg}; border-radius:8px; padding:8px 12px; '
                f'margin-bottom:6px; font-size:0.85rem; color:#e2e8f0">'
                f'<strong>{icon} {role.capitalize()}</strong><br>{content_preview}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No messages yet. Ask something above!")
