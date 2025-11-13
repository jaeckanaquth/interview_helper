import pyaudio
import numpy as np
import yaml
from queue import Queue

cfg_audio = yaml.safe_load(open("config/settings.yaml"))["audio"]

def capture_stream(q: Queue, stop_flag):
    """
    Continuous capture from WASAPI loopback (desktop audio) and push int16 numpy arrays to q.
    """
    FORMAT = pyaudio.paInt16
    RATE = int(cfg_audio["rate"])
    CHANNELS = int(cfg_audio["channels"])
    CHUNK_MS = int(cfg_audio["chunk_ms"])
    FRAMES_PER_BUFFER = int(RATE * CHUNK_MS / 1000)
    DEVICE_INDEX = int(cfg_audio["input_device"])

    pa = pyaudio.PyAudio()
    print(f"[INFO] Opening WASAPI loopback device {DEVICE_INDEX} at {RATE} Hz...")
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=DEVICE_INDEX,
        frames_per_buffer=FRAMES_PER_BUFFER,
    )

    print("[INFO] Capturing system audio. Press Ctrl+C to stop.")
    try:
        while not stop_flag.is_set():
            data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
            np_data = np.frombuffer(data, dtype=np.int16)
            try:
                q.put_nowait(np_data)
            except:
                # If the queue is full, drop the oldest by getting once then put.
                try:
                    q.get_nowait()
                    q.put_nowait(np_data)
                except:
                    pass
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        stop_flag.set()
        print("[INFO] Audio stream closed.")
