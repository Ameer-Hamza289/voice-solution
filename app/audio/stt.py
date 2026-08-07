"""Dual-engine speech-to-text.

Primary engine:  Gemini native audio understanding — state-of-the-art
                 robustness to accents, dialects and background noise.
Fallback engine: faster-whisper running locally (Whisper weights, CTranslate2
                 runtime) — fully offline, also trained on 680k hours of
                 highly diverse multilingual audio, so it remains accurate
                 across accents even with no network at all.

Every utterance has already been VAD-segmented and enhanced by
`preprocess.clean` before it reaches this module, so both engines receive
a denoised, normalized, de-clipped 16 kHz mono signal.

If the primary engine fails or returns an empty/garbage result, the
fallback runs automatically. The caller never sees an engine error — worst
case they get an empty transcript, which the dialog layer treats as
"I didn't catch that".
"""
from __future__ import annotations

import asyncio
import io
import logging
import re

import numpy as np
import soundfile as sf

from app.config import (
    GEMINI_STT_MODEL,
    GOOGLE_API_KEY,
    SAMPLE_RATE,
    STT_PRIMARY,
    WHISPER_MODEL_SIZE,
)

logger = logging.getLogger(__name__)

_TRANSCRIBE_PROMPT = (
    "Transcribe this phone-call audio verbatim. The speaker may have any "
    "accent and there may be background noise. Return ONLY the spoken words "
    "as plain text with normal punctuation. If numbers, addresses, ZIP codes "
    "or phone numbers are spoken, transcribe them as digits. If there is no "
    "intelligible speech, return an empty string."
)


def _to_wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _looks_valid(text: str) -> bool:
    """Reject empty results and obvious non-transcript responses."""
    text = text.strip()
    if not text:
        return False
    if re.fullmatch(r"[\W_]+", text):
        return False
    lowered = text.lower()
    refusals = ("i cannot", "i'm sorry", "no intelligible speech", "no speech")
    return not any(lowered.startswith(r) for r in refusals)


class SpeechToText:
    def __init__(self) -> None:
        self._gemini_client = None
        self._whisper_model = None
        self._whisper_lock = asyncio.Lock()

    # -- Gemini ----------------------------------------------------------

    def _gemini(self):
        if self._gemini_client is None:
            from google import genai

            self._gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
        return self._gemini_client

    async def _transcribe_gemini(self, audio: np.ndarray) -> str:
        from google.genai import types

        wav = _to_wav_bytes(audio)
        client = self._gemini()
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=GEMINI_STT_MODEL,
                contents=[
                    _TRANSCRIBE_PROMPT,
                    types.Part.from_bytes(data=wav, mime_type="audio/wav"),
                ],
                config=types.GenerateContentConfig(temperature=0.0),
            ),
            timeout=20,
        )
        return (response.text or "").strip()

    # -- faster-whisper ----------------------------------------------------

    def _whisper(self):
        if self._whisper_model is None:
            from faster_whisper import WhisperModel

            logger.info("Loading faster-whisper model '%s'...", WHISPER_MODEL_SIZE)
            self._whisper_model = WhisperModel(
                WHISPER_MODEL_SIZE, device="cpu", compute_type="int8"
            )
        return self._whisper_model

    def _transcribe_whisper_sync(self, audio: np.ndarray) -> str:
        model = self._whisper()
        segments, _info = model.transcribe(
            audio,
            language="en",
            beam_size=5,                      # beam search: better accent accuracy
            vad_filter=True,                  # second-stage VAD inside whisper
            condition_on_previous_text=False, # avoids hallucination loops on noise
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    async def _transcribe_whisper(self, audio: np.ndarray) -> str:
        async with self._whisper_lock:
            return await asyncio.to_thread(self._transcribe_whisper_sync, audio)

    # -- public ------------------------------------------------------------

    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe enhanced float32 mono audio; never raises."""
        if STT_PRIMARY == "whisper":
            engines = [("whisper", self._transcribe_whisper),
                       ("gemini", self._transcribe_gemini)]
        else:
            engines = [("gemini", self._transcribe_gemini),
                       ("whisper", self._transcribe_whisper)]

        for name, engine in engines:
            try:
                text = await engine(audio)
            except Exception as exc:
                logger.warning("STT engine '%s' failed: %s", name, exc)
                continue
            if _looks_valid(text):
                return text.strip()
            logger.info("STT engine '%s' returned no usable speech.", name)

        return ""


stt = SpeechToText()
