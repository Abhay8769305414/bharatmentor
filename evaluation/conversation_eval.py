"""
BharatMentor — Conversation Quality Evaluation (Phase 8)
=========================================================
Evaluates ~10 representative conversations on:
  - relevance
  - correctness
  - groundedness
  - language_consistency
  - conciseness

Approach: Deterministic checks + LLM-as-judge.
LLM judge scores are CLEARLY LABELLED as approximate and subjective.

Usage:
    python evaluation/conversation_eval.py

Output:
    experiments/conversation/results.json
    experiments/conversation/report.md
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
CASES_FILE = ROOT / "experiments" / "conversation" / "cases.json"
RESULTS_FILE = ROOT / "experiments" / "conversation" / "results.json"
REPORT_FILE = ROOT / "experiments" / "conversation" / "report.md"


# ── Deterministic checks ──────────────────────────────────────────────────────

def check_language_consistency(question: str, answer: str, expected_lang: str) -> bool:
    """
    Rough check: if expected_lang is 'hi', answer should contain some Devanagari.
    If 'en', answer should be mostly ASCII.
    """
    if expected_lang == "hi":
        devanagari = sum(1 for c in answer if "\u0900" <= c <= "\u097f")
        return devanagari > 5
    if expected_lang == "en":
        ascii_alpha = sum(1 for c in answer if c.isalpha() and c.isascii())
        total_alpha = sum(1 for c in answer if c.isalpha())
        return (ascii_alpha / total_alpha > 0.8) if total_alpha > 0 else True
    return True  # hinglish — accept anything


def check_source_cited(answer: str, sources: list[dict]) -> bool:
    """Check if the answer references a source or if sources were retrieved."""
    return len(sources) > 0 or "source" in answer.lower() or "page" in answer.lower()


# ── LLM-as-judge ──────────────────────────────────────────────────────────────

def llm_judge(question: str, answer: str, context: str) -> dict[str, Any]:
    """
    Ask the LLM to score the answer on relevance, correctness, groundedness, conciseness.
    Returns raw scores (1–5 scale) with a clear disclaimer.

    LIMITATION: LLM self-evaluation is not objective. These scores indicate rough quality
    but should not be treated as ground truth. Human evaluation is preferred for high-stakes
    assessments.
    """
    from app.llm import get_llm_provider
    from app.agent import _extract_text  # reuse text extractor

    llm = get_llm_provider()
    prompt = f"""You are evaluating an AI tutoring system's response.

QUESTION: {question}

CONTEXT (retrieved knowledge):
{context or "(none)"}

ANSWER: {answer}

Rate the answer on each dimension (1=very poor, 5=excellent):
1. Relevance: Does the answer address the question?
2. Correctness: Is the information factually accurate?
3. Groundedness: Is the answer supported by the context (if context is provided)?
4. Conciseness: Is the answer appropriately concise?

Respond with ONLY a JSON object like:
{{"relevance": 4, "correctness": 4, "groundedness": 3, "conciseness": 4}}
"""
    try:
        response = llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        raw = _extract_text(response).strip()
        # Strip code fences
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        scores = json.loads(raw)
        return {k: int(v) for k, v in scores.items()}
    except Exception as exc:
        logger.error("LLM judge failed: %s", exc)
        return {"relevance": None, "correctness": None, "groundedness": None, "conciseness": None}


def _extract_text(response: Any) -> str:
    if hasattr(response, "choices"):
        return response.choices[0].message.content or ""
    if hasattr(response, "text"):
        return response.text or ""
    return str(response)


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_conversation_eval() -> dict[str, Any]:
    if not CASES_FILE.exists():
        raise FileNotFoundError(
            f"Conversation cases not found: {CASES_FILE}\n"
            "Generate them by running the agent against test questions first."
        )

    with open(CASES_FILE, encoding="utf-8") as f:
        cases = json.load(f)

    from app.agent import run_agent
    from app.db import SessionLocal

    db = SessionLocal()
    case_results = []

    try:
        for i, case in enumerate(cases, start=1):
            logger.info("Evaluating case %d/%d: %s", i, len(cases), case["question"][:50])

            # Run agent
            agent_result = run_agent(
                user_message=case["question"],
                student_id=case.get("student_id", 1),
                db=db,
            )

            answer = agent_result.answer
            sources = agent_result.sources
            expected_lang = case.get("expected_language", "en")

            # Deterministic checks
            lang_ok = check_language_consistency(case["question"], answer, expected_lang)
            source_cited = check_source_cited(answer, sources)

            # LLM judge
            context_text = "\n".join(s.get("text", "")[:200] for s in sources[:3])
            llm_scores = llm_judge(case["question"], answer, context_text)

            case_results.append({
                "case_id": i,
                "question": case["question"],
                "expected_language": expected_lang,
                "answer": answer[:500],  # truncate for storage
                "sources_count": len(sources),
                "language_consistent": lang_ok,
                "source_cited": source_cited,
                "llm_scores": llm_scores,
                "tool_calls": [tc["tool"] for tc in agent_result.tool_calls_made],
            })

    finally:
        db.close()

    # Aggregate
    scored_cases = [c for c in case_results if c["llm_scores"].get("relevance") is not None]
    avg = lambda key: (
        round(sum(c["llm_scores"][key] for c in scored_cases) / len(scored_cases), 2)
        if scored_cases else None
    )

    lang_ok_rate = sum(1 for c in case_results if c["language_consistent"]) / len(case_results)

    return {
        "total_cases": len(case_results),
        "scored_by_llm": len(scored_cases),
        "language_consistency_rate": round(lang_ok_rate, 3),
        "avg_relevance": avg("relevance"),
        "avg_correctness": avg("correctness"),
        "avg_groundedness": avg("groundedness"),
        "avg_conciseness": avg("conciseness"),
        "cases": case_results,
        "_disclaimer": (
            "LLM-as-judge scores are approximate. The same LLM both generates "
            "and evaluates responses — treat these as rough indicators, not objective metrics."
        ),
    }


def write_report(results: dict[str, Any]) -> None:
    lines = [
        "# Conversation Quality Evaluation",
        "",
        f"**Cases:** {results['total_cases']}",
        f"**LLM-scored:** {results['scored_by_llm']}",
        "",
        "> ⚠️ **Disclaimer:** Scores from LLM-as-judge are approximate.",
        "> The same LLM both produces and evaluates responses.",
        "> Human evaluation would be more reliable for final reporting.",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Score (1–5) |",
        "|--------|-------------|",
        f"| Relevance | {results['avg_relevance'] or 'N/A'} |",
        f"| Correctness | {results['avg_correctness'] or 'N/A'} |",
        f"| Groundedness | {results['avg_groundedness'] or 'N/A'} |",
        f"| Conciseness | {results['avg_conciseness'] or 'N/A'} |",
        "",
        "## Deterministic Checks",
        "",
        f"| Check | Result |",
        f"|-------|--------|",
        f"| Language consistency | {results['language_consistency_rate']:.1%} |",
    ]

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {REPORT_FILE}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
    results = run_conversation_eval()
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    write_report(results)
    print(f"\nConversation evaluation complete. Cases: {results['total_cases']}")


if __name__ == "__main__":
    main()
