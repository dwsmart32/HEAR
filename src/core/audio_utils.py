"""Shared audio utilities for loading/resampling and CVA concatenation.

Used by both vLLM handlers (src/backends/vllm_handlers/base.py) and the API
backend (src/backends/api_backend.py) so ref-audio handling stays consistent.
"""

import io
import os
import wave
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf


# Directory holding the spoken option letters A.wav .. E.wav that are announced
# before each voice sample in a Content-to-Voice Attribution prompt.
# Override with HEAR_PROMPT_AUDIO_DIR if you keep them somewhere else.
PROMPT_AUDIO_DIR = os.environ.get(
    "HEAR_PROMPT_AUDIO_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "hear", "assets", "prompt_audio"),
)
LETTERS = ["A", "B", "C", "D", "E"]


def load_ref_audio(ref_info: Dict, sr: int = 16000) -> Optional[Tuple[np.ndarray, int]]:
    """Load a reference audio segment [start, end] from `source_wav`.

    Returns (audio_np_float32, sr) or None on failure.
    """
    source_wav = ref_info.get("source_wav", "")
    start = float(ref_info.get("start", 0))
    end = float(ref_info.get("end", 0))
    if not source_wav or not os.path.exists(source_wav):
        return None
    try:
        info = sf.info(source_wav)
        s_sample = int(start * info.samplerate)
        n_frames = int((end - start) * info.samplerate) if end > start else -1
        if n_frames <= 0:
            audio, orig_sr = sf.read(source_wav, dtype="float32")
        else:
            audio, orig_sr = sf.read(source_wav, start=s_sample, frames=n_frames, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if orig_sr != sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
        return (audio.astype(np.float32), sr)
    except Exception as e:
        print(f"[audio_utils] load_ref_audio failed: {source_wav} [{start:.1f}-{end:.1f}]: {e}")
        return None


def build_cva_concat_audio(ref_audios: List[Dict], sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Concatenate letter announcements + ref voices into a single audio.

    Layout: A.wav + 0.5s silence + ref1 + 0.5s silence + B.wav + ...
    """
    silence = np.zeros(int(sr * 0.5), dtype=np.float32)
    parts: List[np.ndarray] = []
    for i, ref_info in enumerate(ref_audios[:5]):
        letter_path = os.path.join(PROMPT_AUDIO_DIR, f"{LETTERS[i]}.wav")
        if os.path.exists(letter_path):
            letter_audio, _ = librosa.load(letter_path, sr=sr, mono=True)
            parts.append(letter_audio.astype(np.float32))
        parts.append(silence)

        ref_data = load_ref_audio(ref_info, sr=sr)
        if ref_data is not None:
            parts.append(ref_data[0])
        else:
            parts.append(np.zeros(int(sr * 0.5), dtype=np.float32))
        parts.append(silence)
    return (np.concatenate(parts), sr)


def audio_np_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """Encode a mono float32 numpy array to 16-bit PCM WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def load_ref_wav_bytes(ref_info: Dict, sr: int = 16000) -> Optional[bytes]:
    """Load a ref segment and return as WAV bytes (for API requests)."""
    data = load_ref_audio(ref_info, sr=sr)
    if data is None:
        return None
    audio, out_sr = data
    return audio_np_to_wav_bytes(audio, out_sr)


def build_cva_concat_wav_bytes(ref_audios: List[Dict], sr: int = 16000) -> bytes:
    """CVA concat result as WAV bytes (for API requests)."""
    audio, out_sr = build_cva_concat_audio(ref_audios, sr=sr)
    return audio_np_to_wav_bytes(audio, out_sr)
