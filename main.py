import whisper
import time
import os

# Put the audio file here
audiofile = "audio.m4a"

# Full OpenAI whisper installation instructions (PIP, ffmpeg) here:
# https://github.com/openai/whisper

# ==================================================

os.system("clear")
tic = time.perf_counter()

model = whisper.load_model("base")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio(audiofile)
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"\n\nDetected language: {max(probs, key=probs.get)}\n\n")

# decode the audio
result = model.transcribe(audiofile)

tx = result["text"]

file1 = open("OutputText.txt", "w")
file1.write(tx+"\n")
file1.close()   

toc = time.perf_counter()
print(f"\nDone in {(toc - tic)/60:0.1f} mins")
