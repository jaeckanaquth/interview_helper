# tools/record_test.py
import wave
import pyaudio
import time
import numpy as np
import sys

# configure here or pass as args
RATE = 48000
CHANNELS = 2
DURATION = 3        # seconds
DEVICE_INDEX = 16   # try 16, then 36 if 16 doesn't work
FRAMES_PER_BUFFER = 1024

def rms_from_bytes(b):
    arr = np.frombuffer(b, dtype=np.int16)
    # if stereo interleaved, mix to mono for RMS
    if arr.size % 2 == 0 and CHANNELS == 2:
        arr = arr.reshape(-1, 2).mean(axis=1).astype(np.int16)
    return np.sqrt(np.mean(arr.astype(np.float32) ** 2)) / 32768.0

def main(device_index):
    pa = pyaudio.PyAudio()
    print("Available device count:", pa.get_device_count())
    try:
        info = pa.get_device_info_by_index(device_index)
        print("Using device:", device_index, info.get("name"))
    except Exception as e:
        print("Failed to query device index", device_index, e)
        pa.terminate()
        return

    try:
        stream = pa.open(format=pyaudio.paInt16,
                         channels=CHANNELS,
                         rate=RATE,
                         input=True,
                         input_device_index=device_index,
                         frames_per_buffer=FRAMES_PER_BUFFER)
    except Exception as e:
        print("Failed to open stream:", e)
        pa.terminate()
        return

    frames = []
    print(f"Recording {DURATION}s ... (watch RMS below)")
    num_iters = int(RATE / FRAMES_PER_BUFFER * DURATION)
    rms_vals = []
    for i in range(num_iters):
        data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
        frames.append(data)
        r = rms_from_bytes(data)
        rms_vals.append(r)
        # print simple progress and RMS
        if i % max(1, num_iters // 10) == 0:
            print(f"iter {i}/{num_iters}, rms={r:.6f}")

    print("Recording done. Closing stream.")
    stream.stop_stream()
    stream.close()
    pa.terminate()

    wav_path = "capture_test.wav"
    wf = wave.open(wav_path, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

    avg_rms = float(np.mean(rms_vals)) if rms_vals else 0.0
    peak_rms = float(np.max(rms_vals)) if rms_vals else 0.0
    print(f"Wrote {wav_path} — avg_rms={avg_rms:.6f}, peak_rms={peak_rms:.6f}")

    if avg_rms < 0.0005:
        print("**Low energy captured (avg_rms < 0.0005).** Likely wrong device or muted output.")
    else:
        print("Captured sound looks present. Play capture_test.wav to verify.")

if __name__ == '__main__':
    idx = DEVICE_INDEX
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
    main(idx)
