# Everline Voice Agent — 24/7 AI Dispatcher for Home Services

An AI voice call agent for an HVAC / plumbing / electrical company. It answers
calls, triages emergencies (with safety instructions), verifies the service
area, quotes only diagnostic fees and documented price ranges, books
appointments, and dispatches on-call technicians — end to end, hands-free.

Everything runs on free tooling. The only key needed is a Google Gemini API key.

## Architecture

```
Browser mic (AudioWorklet, 16 kHz PCM)
        │  WebSocket (binary audio up / MP3 + JSON events down)
        ▼
FastAPI server
        │
        ├─ WebRTC VAD ──► utterance segmentation (with pre-speech buffer)
        ├─ Enhancement ──► high/low-pass filters, adaptive noise reduction,
        │                  de-clipping, loudness normalization
        ├─ STT (dual engine, automatic failover)
        │     primary : Gemini native audio understanding
        │     fallback: faster-whisper (local, offline, int8)
        ├─ LangGraph agent (Gemini 2.0 Flash + tools, per-call memory)
        │     tools: search_knowledge_base (Chroma RAG)
        │            check_service_area
        │            get_available_slots / book_appointment (SQLite)
        │            dispatch_emergency (SQLite)
        └─ TTS: edge-tts neural voice ──► MP3 back to the browser
```

## Why these choices

- **STT robustness** (the top requirement): every utterance is VAD-segmented,
  denoised, de-clipped, and loudness-normalized *before* transcription; then
  Gemini's audio model — one of the strongest available for accented and noisy
  speech — transcribes it, with a locally-run Whisper model as automatic
  fallback so speech recognition survives network/API failures.
- **Chroma** as the vector DB: embedded (no server to run), persistent on
  disk, metadata filtering, first-class LangChain integration. FAISS lacks the
  persistence conveniences; Qdrant/Weaviate need a running service.
- **LangGraph**: explicit agent ⇄ tools graph with a checkpointer, so each
  call has isolated multi-turn memory keyed by call id.
- **edge-tts**: free Microsoft neural voices, natural enough for a phone line.
- **Browser call over WebSocket**: real phone numbers require paid telephony
  (Twilio, etc.). The audio pipeline is transport-agnostic — a Twilio media
  stream adapter can replace the browser client without touching the agent.

## Setup

```powershell
# 1. Create the environment
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 2. Configure your key
copy .env.example .env    # then edit .env and paste your Gemini key

# 3. Build the knowledge base (one time, and after editing knowledge_base/*.md)
.venv\Scripts\python -m app.rag.ingest

# 4. Run
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000, press **Start Call**, and speak.

## Try these calls

- "My AC just died and it's 100 degrees outside, my mother is 80 and lives with us."
  → critical triage, safety guidance, fee disclosure, emergency dispatch.
- "I smell gas near my water heater." → evacuation instructions before anything else.
- "How much does a new water heater cost?" → range + "depends on diagnosis" framing.
- "Can you fix my refrigerator?" → polite refusal + referral (out of scope).
- "I'm in San Marcos" → out of service area, no booking.
- "I'd like to schedule an AC tune-up next week." → slot offer → booking with
  confirmation number.

## Project layout

```
app/
  main.py            FastAPI + WebSocket call loop
  config.py          all tunables (models, VAD timing, service area)
  audio/             preprocess.py, vad.py, stt.py (dual engine), tts.py
  agent/             graph.py (LangGraph), tools.py, prompts.py
  rag/               ingest.py, retriever.py (Chroma)
  db/                database.py (SQLite slots/bookings/dispatches)
knowledge_base/      markdown docs embedded into Chroma
static/              browser call client
scripts/             test utilities
```

## Configuration notes

- `STT_PRIMARY=whisper` in `.env` flips the engine order (fully offline STT).
- `WHISPER_MODEL_SIZE`: `base` is the default; `small` or `medium` raise
  accuracy at the cost of CPU time; `large-v3` is the most accent-robust.
- Business facts (service area ZIPs, fees, protocols) live in `app/config.py`
  and `knowledge_base/*.md` — edit and re-run the ingest step.
