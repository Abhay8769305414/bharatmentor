"""
BharatMentor — ASR WER Evaluation (Phase 7)
=============================================
Calculates Word Error Rate for English and Hindi audio samples.

Dataset structure:
    experiments/asr/dataset/
        en/
            sample_01.wav
            sample_01.txt   ← ground truth transcript
        hi/
            sample_01.wav
            sample_01.txt

Usage:
    python experiments/asr/evaluate.py

Output:
    experiments/asr/results.json
    experiments/asr/report.md
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
DATASET_DIR = ROOT / "experiments" / "asr" / "dataset"
RESULTS_FILE = ROOT / "experiments" / "asr" / "results.json"
REPORT_FILE = ROOT / "experiments" / "asr" / "report.md"


def evaluate_language(lang_dir: Path, lang_code: str) -> dict[str, Any]:
    """Evaluate WER for all samples in a language directory."""
    from jiwer import wer as compute_wer
    from voice.stt import get_stt_provider

    stt = get_stt_provider()
    audio_files = sorted(lang_dir.glob("*.wav")) + sorted(lang_dir.glob("*.mp3"))

    if not audio_files:
        return {
            "language": lang_code,
            "samples": 0,
            "wer": None,
            "note": f"No audio files found in {lang_dir}",
        }

    references = []
    hypotheses = []
    per_sample = []

    for audio_path in audio_files:
        ref_path = audio_path.with_suffix(".txt")
        if not ref_path.exists():
            logger.warning("No ground-truth transcript for %s — skipping", audio_path.name)
            continue

        reference = ref_path.read_text(encoding="utf-8").strip()
        lang_hint = lang_code if lang_code != "hinglish" else None

        try:
            result = stt.transcribe(audio_path, language_hint=lang_hint)
            hypothesis = result.text.strip()
        except Exception as exc:
            logger.error("STT failed for %s: %s", audio_path, exc)
            hypothesis = ""

        sample_wer = float(compute_wer(reference, hypothesis)) if reference else None
        per_sample.append({
            "file": audio_path.name,
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": round(sample_wer, 4) if sample_wer is not None else None,
        })
        references.append(reference)
        hypotheses.append(hypothesis)

    if not references:
        return {
            "language": lang_code,
            "samples": 0,
            "wer": None,
            "note": "No valid samples processed",
        }

    overall_wer = float(compute_wer(references, hypotheses))
    logger.info("Language=%s, samples=%d, WER=%.4f", lang_code, len(references), overall_wer)

    return {
        "language": lang_code,
        "samples": len(references),
        "wer": round(overall_wer, 4),
        "per_sample": per_sample,
    }


def write_report(results: dict[str, Any]) -> None:
    lines = [
        "# ASR Evaluation Report — Word Error Rate",
        "",
        "| Language | Samples | WER |",
        "|----------|---------|-----|",
    ]
    for lang_result in results["languages"]:
        wer_str = f"{lang_result['wer']:.4f}" if lang_result["wer"] is not None else "N/A"
        lines.append(
            f"| {lang_result['language'].upper()} | {lang_result['samples']} | {wer_str} |"
        )

    if results.get("overall_wer") is not None:
        lines += ["", f"**Overall WER:** {results['overall_wer']:.4f}"]

    lines += [
        "",
        "## Notes",
        "",
        "- WER computed using `jiwer` library.",
        "- Audio samples are manually recorded with verified ground-truth transcripts.",
        "- faster-whisper with `base` model used for transcription.",
        "- Hindi WER may be higher due to romanization inconsistencies in ground truth.",
        "",
        "> **Limitation:** WER penalises all token differences equally. "
        "For Hindi, character-level metrics (CER) may be more informative.",
    ]

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {REPORT_FILE}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    lang_dirs = {
        "en": DATASET_DIR / "en",
        "hi": DATASET_DIR / "hi",
    }

    lang_results = []
    all_refs = []
    all_hyps = []

    for lang_code, lang_dir in lang_dirs.items():
        if lang_dir.exists():
            result = evaluate_language(lang_dir, lang_code)
        else:
            result = {
                "language": lang_code,
                "samples": 0,
                "wer": None,
                "note": f"Directory not found: {lang_dir}",
            }
            logger.warning("ASR dataset dir not found: %s", lang_dir)
        lang_results.append(result)

        # Collect for overall WER
        for sample in result.get("per_sample", []):
            if sample.get("reference") and sample.get("hypothesis") is not None:
                all_refs.append(sample["reference"])
                all_hyps.append(sample["hypothesis"])

    overall_wer = None
    if all_refs:
        from jiwer import wer as compute_wer
        overall_wer = round(float(compute_wer(all_refs, all_hyps)), 4)

    results = {
        "languages": lang_results,
        "overall_wer": overall_wer,
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    write_report(results)

    print("\n" + "=" * 40)
    for r in lang_results:
        wer_str = f"{r['wer']:.4f}" if r["wer"] is not None else "N/A (no samples)"
        print(f"{r['language'].upper()} WER: {wer_str}  ({r['samples']} samples)")
    if overall_wer is not None:
        print(f"Overall WER: {overall_wer:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
