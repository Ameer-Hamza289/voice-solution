"""Utterance segmentation using WebRTC VAD.

The server receives a continuous stream of 16 kHz PCM16 frames from the
browser. This class decides where an utterance starts and ends so that the
STT engines get one clean, complete phrase at a time instead of arbitrary
chunks cut mid-word.
"""
from __future__ import annotations

import webrtcvad

from app.config import (
    MAX_UTTERANCE_S,
    MIN_UTTERANCE_MS,
    SAMPLE_RATE,
    SILENCE_END_MS,
    VAD_AGGRESSIVENESS,
    VAD_FRAME_MS,
)

_FRAME_BYTES = SAMPLE_RATE * VAD_FRAME_MS // 1000 * 2  # 16-bit mono


class UtteranceDetector:
    """Feed PCM bytes in, get complete utterances out."""

    def __init__(self) -> None:
        self._vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self._pending = b""          # unframed remainder of the stream
        self._utterance = bytearray()
        self._in_speech = False
        self._silence_ms = 0
        # Keep a short rolling pre-buffer so the first syllable (spoken
        # before the VAD flips to "speech") isn't chopped off.
        self._prebuffer: list[bytes] = []
        self._prebuffer_ms = 240

    def feed(self, pcm: bytes) -> list[bytes]:
        """Append stream bytes; return any utterances completed by them."""
        self._pending += pcm
        completed: list[bytes] = []

        while len(self._pending) >= _FRAME_BYTES:
            frame = self._pending[:_FRAME_BYTES]
            self._pending = self._pending[_FRAME_BYTES:]
            result = self._process_frame(frame)
            if result is not None:
                completed.append(result)

        return completed

    def flush(self) -> bytes | None:
        """Call at end of stream to emit any in-progress utterance."""
        if self._in_speech and self._long_enough():
            data = bytes(self._utterance)
            self._reset()
            return data
        self._reset()
        return None

    # ------------------------------------------------------------------

    def _process_frame(self, frame: bytes) -> bytes | None:
        try:
            is_speech = self._vad.is_speech(frame, SAMPLE_RATE)
        except Exception:
            is_speech = False

        if not self._in_speech:
            self._prebuffer.append(frame)
            max_frames = self._prebuffer_ms // VAD_FRAME_MS
            if len(self._prebuffer) > max_frames:
                self._prebuffer.pop(0)
            if is_speech:
                self._in_speech = True
                self._silence_ms = 0
                for buffered in self._prebuffer:
                    self._utterance.extend(buffered)
                self._prebuffer.clear()
            return None

        self._utterance.extend(frame)

        if is_speech:
            self._silence_ms = 0
        else:
            self._silence_ms += VAD_FRAME_MS

        utterance_s = len(self._utterance) / (SAMPLE_RATE * 2)
        if self._silence_ms >= SILENCE_END_MS or utterance_s >= MAX_UTTERANCE_S:
            if self._long_enough():
                data = bytes(self._utterance)
                self._reset()
                return data
            self._reset()

        return None

    def _long_enough(self) -> bool:
        total_ms = len(self._utterance) / (SAMPLE_RATE * 2) * 1000
        speech_ms = total_ms - self._silence_ms  # exclude trailing silence
        return speech_ms >= MIN_UTTERANCE_MS

    def _reset(self) -> None:
        self._utterance = bytearray()
        self._in_speech = False
        self._silence_ms = 0
        self._prebuffer.clear()
