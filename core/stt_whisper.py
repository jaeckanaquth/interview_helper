import os
import yaml
from faster_whisper import WhisperModel

# Avoid OpenMP duplicate-lib issues
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

cfg = yaml.safe_load(open("config/settings.yaml"))["stt"]

model = WhisperModel(cfg["model"], compute_type=cfg.get("compute_type", "int8"))

def transcribe_file(file_path: str):
    """Run full transcription on a WAV file."""
    print(f"[INFO] Transcribing file: {file_path}")
    segments, info = model.transcribe(
        file_path,
        beam_size=cfg.get("beam_size", 1),
        temperature=cfg.get("temperature", 0.0),
        vad_filter=False,
        language="en"
    )

    text_path = file_path.replace(".wav", ".txt")
    with open(text_path, "w", encoding="utf-8") as f:
        for s in segments:
            line = f"[{s.start:.2f} s → {s.end:.2f} s] {s.text.strip()}"
            print(line)
            f.write(line + "\n")

    print(f"[INFO] Transcript saved: {text_path}")
    return text_path
