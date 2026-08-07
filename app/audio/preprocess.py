"""Audio cleanup applied before speech-to-text.

The goal is to hand the STT engines the cleanest possible signal regardless
of the caller's environment: phone-line hum, HVAC background noise, wind,
clipping from cheap microphones, very quiet speakers, etc.

Pipeline:
  1. DC-offset removal
  2. High-pass filter at 80 Hz  (kills rumble/hum below the speech band)
  3. Low-pass filter at 7.6 kHz (kills hiss above telephone-band speech)
  4. Spectral-gating noise reduction (learns the noise profile from the
     clip itself, so it adapts to whatever environment the caller is in)
  5. Soft de-clipping via tanh companding when clipping is detected
  6. Loudness normalization to a consistent RMS target
"""
from __future__ import annotations

import numpy as np
import noisereduce as nr
from scipy.signal import butter, sosfilt

from app.config import SAMPLE_RATE

_TARGET_RMS = 0.08
_HP_SOS = butter(4, 80, btype="highpass", fs=SAMPLE_RATE, output="sos")
_LP_SOS = butter(6, 7600, btype="lowpass", fs=SAMPLE_RATE, output="sos")


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def float_to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def clean(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Run the full enhancement chain on a float32 mono signal."""
    if audio.size == 0:
        return audio

    audio = audio - float(np.mean(audio))          # DC offset
    audio = sosfilt(_HP_SOS, audio)                # rumble / hum
    audio = sosfilt(_LP_SOS, audio)                # high-frequency hiss

    # Adaptive spectral-gating noise reduction. `stationary=False` lets it
    # track changing background noise (traffic, machinery, crowd).
    try:
        audio = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            stationary=False,
            prop_decrease=0.9,
        )
    except Exception:
        # Extremely short/degenerate clips can make the estimator fail;
        # the filtered signal is still usable as-is.
        pass

    # Soft de-clip: if a meaningful share of samples sits at the rails,
    # compand the waveform so STT sees rounded peaks instead of squares.
    if np.mean(np.abs(audio) > 0.985) > 0.001:
        audio = np.tanh(1.5 * audio) / np.tanh(1.5)

    # RMS loudness normalization so quiet callers aren't lost.
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms > 1e-6:
        audio = audio * min(_TARGET_RMS / rms, 12.0)
        audio = np.clip(audio, -1.0, 1.0)

    return audio.astype(np.float32)


def clean_pcm(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Convenience wrapper: raw PCM16 bytes in, enhanced float32 out."""
    return clean(pcm16_to_float(pcm), sample_rate)
