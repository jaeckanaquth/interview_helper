# main.py (live mode with question finder)

qa_log = []  # list of dicts: {"q": ..., "bullets": [...]}

qa_log_file = "qa_log.json"
import yaml
import threading
import numpy as np
from queue import Queue
# from core.answer_suggester import AnswerSuggester
from core.answer_llm import AnswerEngine

from core.audio_capture import capture_stream
from core.stt_whisper_stream import transcribe_window
from core.question_finder import QuestionFinder

# suggester = AnswerSuggester()

answer_engine = AnswerEngine(
    role="MLOps Engineer",
    resume_path="data/resume.md",
    jd_path="data/current_jd.md"
)


cfg = yaml.safe_load(open("config/settings.yaml"))
audio_cfg = cfg["audio"]
stream_cfg = cfg["streaming"]

RATE = int(audio_cfg["rate"])
CHANNELS = int(audio_cfg["channels"])
WINDOW_S = int(stream_cfg["window_s"])
STEP_S = int(stream_cfg["step_s"])
VAD_RMS_THR = float(stream_cfg["vad_rms_thresh"])
MIN_SPEECH_MS = int(stream_cfg["min_speech_ms"])

WINDOW_SAMPLES = RATE * WINDOW_S
STEP_SAMPLES = RATE * STEP_S

def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    f = x.astype(np.float32)
    return float(np.sqrt(np.mean(np.square(f))) / 32768.0)

def has_enough_voiced(x: np.ndarray) -> bool:
    if x.size == 0:
        return False
    f = x.astype(np.float32)
    thr = 0.08 * 32768.0
    voiced = np.sum(np.abs(f) >= thr)
    voiced_ms = (voiced / RATE) * 1000.0
    return voiced_ms >= MIN_SPEECH_MS

def longest_common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

def main():
    q = Queue(maxsize=20)
    stop_flag = threading.Event()
    cap_thread = threading.Thread(target=capture_stream, args=(q, stop_flag), daemon=True)
    cap_thread.start()

    buffer = np.array([], dtype=np.int16)
    printed_text_tail = ""
    qfinder = QuestionFinder()

    print("[INFO] Starting live transcription + question finder...")
    try:
        while not stop_flag.is_set():
            # drain queue quickly
            while not q.empty():
                buffer = np.concatenate((buffer, q.get()))

            if buffer.size >= WINDOW_SAMPLES:
                window = buffer[:WINDOW_SAMPLES]

                level = rms(window)
                if level >= VAD_RMS_THR and has_enough_voiced(window):
                    text = transcribe_window(window, input_rate=RATE, channels_hint=CHANNELS)
                    if text:
                        last_tail = printed_text_tail[-2000:]
                        lcp = longest_common_prefix(last_tail + text, last_tail)
                        new_part = (last_tail + text)[lcp:]
                        new_part = new_part.strip()

                        if new_part:
                            # 1) Print transcript snippets (optional)
                            for line in new_part.split(". "):
                                line = line.strip()
                                if line:
                                    print("🗣️", line)

                            printed_text_tail += new_part
                            if len(printed_text_tail) > 8000:
                                printed_text_tail = printed_text_tail[-8000:]

                            # 2) Feed into question finder
                            new_questions = qfinder.process(new_part)
                            # for q_text in new_questions:
                            #     print("❓ Q:", q_text)
                            #     answers = suggester.suggest(q_text)
                            #     for b in answers:
                            #         print("➡", b)
                            for q_text in new_questions:
                                print("❓ Q:", q_text)
                                bullets = answer_engine.generate_answer(q_text)
                                for b in bullets:
                                    print("➡", b)
                                print("--------------------------------")

                                qa_log.append({"q": q_text, "bullets": bullets})




                # slide window
                buffer = buffer[STEP_SAMPLES:]
    except KeyboardInterrupt:
        pass
    finally:
        stop_flag.set()
        cap_thread.join()
        # write QA log
        from datetime import datetime
        import os, textwrap

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = f"data/sessions/{ts}"
        os.makedirs(session_dir, exist_ok=True)
        out_path = os.path.join(session_dir, "qa_log.md")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# Q&A Transcript\n\n")
            for i, item in enumerate(qa_log, 1):
                f.write(f"## Q{i}: {item['q']}\n\n")
                for b in item["bullets"]:
                    f.write(f"- {b}\n")
                f.write("\n---\n\n")

        print(f"[INFO] Q&A log saved to {out_path}")
        print("[INFO] Exiting live mode.")


if __name__ == "__main__":
    main()
