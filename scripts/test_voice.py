"""
BharatMentor — Manual Voice Integration Test
============================================
Runs the full local STT -> Agent -> Tool -> TTS pipeline on an input audio file.

Usage:
    python scripts/test_voice.py data/audio/sample_01.mp3
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import av
from app.db import get_db, SessionLocal
from voice.pipeline import voice_pipeline


def get_audio_duration(file_path: Path) -> float:
    with av.open(str(file_path)) as container:
        stream = container.streams.audio[0]
        return float(stream.duration * stream.time_base)


def run_voice_test(audio_path_str: str, student_id: int = 1):
    audio_path = Path(audio_path_str)
    if not audio_path.exists():
        print(f"Error: audio file '{audio_path}' does not exist.")
        sys.exit(1)

    audio_duration = get_audio_duration(audio_path)
    db = SessionLocal()

    print("\n" + "=" * 48)
    print("      BHARATMENTOR COMPLETE VOICE TEST")
    print("=" * 48)
    print(f"Audio File     : {audio_path.name}")
    print(f"Audio Duration : {audio_duration:.2f} sec")
    print("-" * 48)

    t0 = time.perf_counter()
    result = voice_pipeline(
        audio_path=audio_path,
        student_id=student_id,
        db=db,
    )
    total_latency = time.perf_counter() - t0

    stt_latency = result.get("latency", {}).get("stt_seconds", 0.0)
    rtf = stt_latency / audio_duration if audio_duration > 0 else 0.0

    print(f"Transcript     : {result.get('transcript')}")
    print(f"Language       : {result.get('language')}")
    print(f"STT Latency    : {stt_latency:.2f} sec")
    print(f"STF / RTF      : {rtf:.2f}")
    print("-" * 48)
    print(f"Tool(s) Called : {result.get('tool_calls_made')}")
    print(f"Sources Found  : {len(result.get('sources', []))} chunks")
    print("-" * 48)
    print(f"Agent Response : {result.get('answer')}")
    print("-" * 48)
    
    has_audio = result.get("audio_b64") is not None
    tts_latency = result.get("latency", {}).get("tts_seconds", 0.0)
    print(f"TTS Output     : {'SUCCESS (' + str(len(result['audio_b64'])) + ' chars b64)' if has_audio else 'FAILED'}")
    print(f"TTS Latency    : {tts_latency:.2f} sec")
    print("-" * 48)
    print(f"Total Latency  : {total_latency:.2f} sec")
    print("=" * 48 + "\n")

    db.close()
    return result


if __name__ == "__main__":
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "data/audio/test_rag.mp3"
    run_voice_test(audio_file)
