"""
Script to populate experiments/asr/dataset/en/ and run the English STT benchmark.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from gtts import gTTS
import time
import json
import av

DATASET_EN_DIR = Path("experiments/asr/dataset/en")
DATASET_EN_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    ("sample_01.mp3", "sample_01.txt", "Explain Newton's second law in simple terms."),
    ("sample_02.mp3", "sample_02.txt", "What is Newton's second law?"),
    ("sample_03.mp3", "sample_03.txt", "What is 25 multiplied by 17?"),
    ("sample_04.mp3", "sample_04.txt", "What is my physics score?"),
]

def setup_en_dataset():
    for audio_name, text_name, text in SAMPLES:
        audio_path = DATASET_EN_DIR / audio_name
        text_path = DATASET_EN_DIR / text_name

        text_path.write_text(text, encoding="utf-8")
        if not audio_path.exists():
            tts = gTTS(text=text, lang="en", slow=False)
            tts.save(str(audio_path))
            print(f"Generated {audio_path.name}")
        else:
            print(f"Exists {audio_path.name}")

def get_audio_duration(file_path: Path) -> float:
    with av.open(str(file_path)) as container:
        stream = container.streams.audio[0]
        duration = float(stream.duration * stream.time_base)
        return duration

def benchmark_stt():
    from voice.stt import get_stt_provider

    print("\n--- Initializing STT Model ---")
    t0 = time.perf_counter()
    stt = get_stt_provider()
    model_load_time = time.perf_counter() - t0
    print(f"Model load time: {model_load_time:.2f}s")

    benchmark_results = []
    total_audio_duration = 0.0
    total_transcription_time = 0.0

    print("\n--- Running English Transcriptions ---")
    for audio_name, text_name, expected_text in SAMPLES:
        audio_path = DATASET_EN_DIR / audio_name
        audio_duration = get_audio_duration(audio_path)

        t_start = time.perf_counter()
        result = stt.transcribe(audio_path, language_hint="en")
        t_duration = time.perf_counter() - t_start

        rtf = t_duration / audio_duration if audio_duration > 0 else 0.0

        total_audio_duration += audio_duration
        total_transcription_time += t_duration

        print(f"\nAudio File   : {audio_name}")
        print(f"Audio Dur    : {audio_duration:.2f}s")
        print(f"Expected     : {expected_text}")
        print(f"Actual (STT) : {result.text}")
        print(f"Latency      : {t_duration:.2f}s")
        print(f"RTF          : {rtf:.2f}")

        benchmark_results.append({
            "file": audio_name,
            "expected": expected_text,
            "actual": result.text,
            "audio_duration_seconds": round(audio_duration, 2),
            "transcription_seconds": round(t_duration, 2),
            "rtf": round(rtf, 2),
            "language_detected": result.language,
            "language_probability": round(result.language_probability, 4)
        })

    avg_rtf = total_transcription_time / total_audio_duration if total_audio_duration > 0 else 0.0

    summary = {
        "model": "faster-whisper (base, cpu, int8)",
        "model_load_time_seconds": round(model_load_time, 2),
        "total_audio_seconds": round(total_audio_duration, 2),
        "total_transcription_seconds": round(total_transcription_time, 2),
        "average_rtf": round(avg_rtf, 2),
        "samples": benchmark_results
    }

    out_file = Path("experiments/asr/results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved results to {out_file}")
    print(f"Average RTF: {avg_rtf:.2f}")

if __name__ == "__main__":
    setup_en_dataset()
    benchmark_stt()
