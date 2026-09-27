"""
Generate test audio files for voice pipeline evaluation using gTTS.
"""

from pathlib import Path
from gtts import gTTS

AUDIO_DIR = Path(__file__).parent.parent / "data" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = {
    "test_english.mp3": "Explain Newton's second law in simple terms.",
    "test_rag.mp3": "What is Newton's second law?",
    "test_calculator.mp3": "What is 25 multiplied by 17?",
    "test_progress.mp3": "What is my physics score?",
}

def generate():
    for filename, text in SAMPLES.items():
        out_path = AUDIO_DIR / filename
        print(f"Generating {out_path.name} -> '{text}'")
        tts = gTTS(text=text, lang="en", slow=False)
        tts.save(str(out_path))
    print("\n[OK] All test audio generated successfully.")

if __name__ == "__main__":
    generate()
