# ASR Evaluation Dataset
#
# Place audio samples here in language-specific subdirectories:
#
# dataset/
#   en/
#     sample_01.wav   ← recorded audio
#     sample_01.txt   ← exact ground truth transcript (one line)
#     sample_02.wav
#     sample_02.txt
#     ...
#   hi/
#     sample_01.wav
#     sample_01.txt   ← Hindi ground truth in Devanagari script
#     ...
#
# Recording guidelines:
# - Clear audio, minimal background noise
# - Natural speaking pace
# - WAV format, 16kHz mono recommended for best Whisper compatibility
# - English: 5-15 second utterances about Physics topics
# - Hindi: Same topics but in Hindi
#
# Example English ground truths:
# "Newton's second law states that force equals mass times acceleration."
# "The conservation of momentum means the total momentum stays constant."
#
# Example Hindi ground truths:
# "न्यूटन का दूसरा नियम कहता है कि बल बराबर द्रव्यमान गुणा त्वरण है।"
# "संवेग के संरक्षण का नियम कहता है कि कुल संवेग स्थिर रहता है।"
