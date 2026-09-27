"""
Test TTS generation and validity.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from voice.tts import get_tts_provider
import av

def test_tts():
    text = "Newton's second law states that force equals mass multiplied by acceleration."
    tts = get_tts_provider()
    
    print(f"Synthesizing: '{text}'")
    audio_bytes = tts.synthesize(text, language="en")
    
    assert audio_bytes is not None, "Audio bytes must not be None"
    assert len(audio_bytes) > 0, "Audio bytes must not be empty"
    print(f"Generated {len(audio_bytes)} bytes of audio.")

    out_path = ROOT / "data" / "audio" / "test_tts_output.mp3"
    out_path.write_bytes(audio_bytes)
    print(f"Saved to {out_path}")

    # Verify audio container validity with PyAV
    with av.open(str(out_path)) as container:
        stream = container.streams.audio[0]
        duration = float(stream.duration * stream.time_base)
        print(f"Verified valid MP3 container. Duration: {duration:.2f}s, Sample rate: {stream.sample_rate}Hz")

    print("[OK] TTS verification complete.")

if __name__ == "__main__":
    test_tts()
