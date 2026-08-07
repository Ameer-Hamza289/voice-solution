"""Audio-robustness test: synthesizes a spoken caller phrase with TTS,
degrades it (background noise, hum, clipping, quiet volume), then runs it
through the real enhancement + dual-engine STT pipeline and reports what
came back.

Run:  python -m scripts.test_audio_pipeline
"""
from __future__ import annotations

import asyncio
import io

import numpy as np
import soundfile as sf

from app.audio.preprocess import clean
from app.audio.stt import stt
from app.audio.tts import synthesize
from app.config import SAMPLE_RATE

PHRASE = (
    "Hi, my air conditioner stopped working at 1204 Maple Grove Lane, "
    "ZIP code 78745. My phone number is 512-555-0184."
)


async def tts_to_pcm(text: str) -> np.ndarray:
    """Generate speech with edge-tts and decode to 16 kHz float mono."""
    mp3 = await synthesize(text)
    # soundfile can decode mp3 (libsndfile >= 1.1)
    audio, sr = sf.read(io.BytesIO(mp3), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        # naive linear resample is fine for a test signal
        target_len = int(len(audio) * SAMPLE_RATE / sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target_len),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
    return audio


def degrade(audio: np.ndarray, kind: str) -> np.ndarray:
    rng = np.random.default_rng(7)
    if kind == "clean":
        return audio
    if kind == "noisy":  # strong broadband background noise (SNR ~8 dB)
        noise = rng.normal(0, 1, len(audio)).astype(np.float32)
        sig_rms = np.sqrt(np.mean(audio**2))
        noise = noise / np.sqrt(np.mean(noise**2)) * sig_rms / (10 ** (8 / 20))
        return audio + noise
    if kind == "hum_and_quiet":  # 60 Hz mains hum over a very quiet voice
        t = np.arange(len(audio)) / SAMPLE_RATE
        hum = 0.05 * np.sin(2 * np.pi * 60 * t).astype(np.float32)
        return audio * 0.08 + hum
    if kind == "clipped":  # cheap-microphone hard clipping
        return np.clip(audio * 6.0, -1.0, 1.0)
    raise ValueError(kind)


async def run() -> None:
    print("Synthesizing test phrase...")
    base = await tts_to_pcm(PHRASE)
    print(f'Reference: "{PHRASE}"\n')

    for kind in ("clean", "noisy", "hum_and_quiet", "clipped"):
        degraded = degrade(base, kind)
        enhanced = clean(degraded)
        text = await stt.transcribe(enhanced)
        print(f"[{kind:>14}] -> {text or '(no transcript)'}")


if __name__ == "__main__":
    asyncio.run(run())
