"""FastAPI server: serves the call UI and runs the realtime voice loop.

WebSocket protocol (/ws/call):
  client -> server:  binary frames  = 16 kHz mono PCM16 microphone audio
                     text frames    = JSON control: {"type": "end_call"}
  server -> client:  text frames    = JSON events:
                       {"type": "ready", "greeting": ...}
                       {"type": "state", "value": "listening"|"thinking"|"speaking"}
                       {"type": "transcript", "role": "caller"|"agent", "text": ...}
                     binary frames  = MP3 audio of the agent's reply
"""
from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agent.graph import respond
from app.agent.prompts import GREETING
from app.audio.preprocess import clean_pcm
from app.audio.stt import stt
from app.audio.tts import synthesize
from app.audio.vad import UtteranceDetector
from app.config import STATIC_DIR
from app.db.database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("voice-agent")

app = FastAPI(title="Everline Voice Agent")


@app.on_event("startup")
async def startup() -> None:
    init_db()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.websocket("/ws/call")
async def call_socket(ws: WebSocket) -> None:
    await ws.accept()
    call_id = uuid.uuid4().hex
    detector = UtteranceDetector()
    logger.info("Call %s connected", call_id[:8])

    async def speak(text: str) -> None:
        await ws.send_json({"type": "transcript", "role": "agent", "text": text})
        await ws.send_json({"type": "state", "value": "speaking"})
        audio = await synthesize(text)
        if audio:
            await ws.send_bytes(audio)
        await ws.send_json({"type": "state", "value": "listening"})

    try:
        await ws.send_json({"type": "ready", "greeting": GREETING})
        await speak(GREETING)

        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if (pcm := message.get("bytes")) is not None:
                for utterance in detector.feed(pcm):
                    await _handle_utterance(ws, call_id, utterance, speak)
            elif (text := message.get("text")) is not None:
                if '"end_call"' in text:
                    break

    except WebSocketDisconnect:
        pass
    finally:
        logger.info("Call %s ended", call_id[:8])


async def _handle_utterance(ws: WebSocket, call_id: str,
                            utterance_pcm: bytes, speak) -> None:
    await ws.send_json({"type": "state", "value": "thinking"})

    enhanced = await asyncio.to_thread(clean_pcm, utterance_pcm)
    transcript = await stt.transcribe(enhanced)

    if not transcript:
        await speak("Sorry, I didn't quite catch that. Could you say it again?")
        return

    logger.info("Call %s caller: %s", call_id[:8], transcript)
    await ws.send_json({"type": "transcript", "role": "caller", "text": transcript})

    reply = await respond(call_id, transcript)
    logger.info("Call %s agent:  %s", call_id[:8], reply)
    await speak(reply)
