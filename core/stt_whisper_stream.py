import os
import yaml
import numpy as np
from scipy.signal import resample_poly
from faster_whisper import WhisperModel

# Avoid OpenMP runtime clashes
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

_cfg = yaml.safe_load(open("config/settings.yaml"))
cfg_stt = _cfg["stt"]

# Load model once
print(f"[INFO] Loading Whisper model '{cfg_stt['model']}' (compute={cfg_stt.get('compute_type','int8')})...")
model = WhisperModel(
    cfg_stt["model"],
    compute_type=cfg_stt.get("compute_type", "int8")
)

def _to_mono_int16(x: np.ndarray, channels_hint: int = 2) -> np.ndarray:
    """
    Ensure mono int16. Input is 1-D interleaved int16 (PyAudio gives 1-D).
    If channels_hint == 2, de-interleave and average L/R.
    """
    if channels_hint == 2:
        # Interleaved stereo -> N x 2 then mean
        if len(x) % 2 == 0:
            stereo = x.reshape(-1, 2)
            mono = stereo.mean(axis=1).astype(np.int16)
            return mono
        else:
            # Fallback if odd length: drop last sample
            stereo = x[:-1].reshape(-1, 2)
            mono = stereo.mean(axis=1).astype(np.int16)
            return mono
    return x  # already mono

def _normalize_to_float32(mono_int16: np.ndarray) -> np.ndarray:
    f = mono_int16.astype(np.float32)
    peak = np.max(np.abs(f))
    if peak > 0:
        f /= peak  # peak normalize to [-1, 1]
    return f

def _resample_48k_to_16k(x_float: np.ndarray) -> np.ndarray:
    # 48k -> 16k using polyphase (down by 3)
    return resample_poly(x_float, 1, 3)

def transcribe_window(raw_int16: np.ndarray, input_rate: int = 48000, channels_hint: int = 2) -> str:
    """
    Transcribe a window of raw PCM int16 captured at input_rate (48k), returns text.
    """
    # Downmix
    mono_i16 = _to_mono_int16(raw_int16, channels_hint=channels_hint)
    # Normalize
    mono_f32 = _normalize_to_float32(mono_i16)
    if mono_f32.size == 0:
        return ""
    # Resample to 16k
    if input_rate == 48000:
        audio_16k = _resample_48k_to_16k(mono_f32)
    else:
        # Generic path if you ever change device rate
        from scipy.signal import resample
        target_len = int(len(mono_f32) * 16000 / float(input_rate))
        audio_16k = resample(mono_f32, target_len)

    segments, _ = model.transcribe(
        audio_16k,
        beam_size=cfg_stt.get("beam_size", 1),
        temperature=cfg_stt.get("temperature", 0.0),
        vad_filter=False,
        language="en",
    )
    text = "".join(s.text for s in segments).strip()
    return text
