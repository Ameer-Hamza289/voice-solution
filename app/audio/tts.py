"""Text-to-speech using Microsoft Edge neural voices (free, no key needed).

Returns MP3 bytes, which every browser can decode natively.
"""
from __future__ import annotations

import logging
import re

import edge_tts

from app.config import TTS_RATE, TTS_VOICE

logger = logging.getLogger(__name__)


def _strip_markup(text: str) -> str:
    """Remove markdown artifacts the LLM might emit so they aren't spoken."""
    text = re.sub(r"[*_#`]+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # [label](url) -> label
    return re.sub(r"\s+", " ", text).strip()


async def synthesize(text: str) -> bytes:
    text = _strip_markup(text)
    if not text:
        return b""
    communicate = edge_tts.Communicate(text, voice=TTS_VOICE, rate=TTS_RATE)
    chunks: list[bytes] = []
    async for message in communicate.stream():
        if message["type"] == "audio":
            chunks.append(message["data"])
    return b"".join(chunks)
