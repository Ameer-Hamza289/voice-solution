"""Central configuration loaded from environment / .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# --- API keys -------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
if GOOGLE_API_KEY:
    # langchain-google-genai and google-genai both read this variable.
    os.environ.setdefault("GOOGLE_API_KEY", GOOGLE_API_KEY)

# --- Models ---------------------------------------------------------------
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
# Free-tier quotas are PER MODEL, so on 429/quota errors the agent fails over
# down this list, multiplying the effective free request budget.
GEMINI_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-2.5-flash-lite,gemini-3-flash-preview,gemini-3.5-flash",
    ).split(",")
    if m.strip()
]
# Transcription is a lighter task; the lite model preserves the bigger
# models' quota for the agent itself.
GEMINI_STT_MODEL = os.getenv("GEMINI_STT_MODEL", "gemini-2.5-flash-lite")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
STT_PRIMARY = os.getenv("STT_PRIMARY", "gemini").lower()  # "gemini" | "whisper"

# --- Voice ----------------------------------------------------------------
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-JennyNeural")
TTS_RATE = os.getenv("TTS_RATE", "+8%")

# --- Audio pipeline -------------------------------------------------------
SAMPLE_RATE = 16_000          # PCM sample rate expected from the client
VAD_AGGRESSIVENESS = 3        # 0-3, 3 = most aggressive filtering of non-speech
VAD_FRAME_MS = 30             # webrtcvad supports 10/20/30 ms frames
SILENCE_END_MS = 900          # how much trailing silence ends an utterance
MIN_UTTERANCE_MS = 300        # ignore blips shorter than this
MAX_UTTERANCE_S = 30          # hard cap per utterance

# --- Paths ----------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
DB_PATH = DATA_DIR / "dispatch.db"
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge_base"
STATIC_DIR = PROJECT_ROOT / "static"

DATA_DIR.mkdir(exist_ok=True)

# --- Business -------------------------------------------------------------
BUSINESS_NAME = "Everline Home Services"
SERVICE_ZIP_CODES = {
    "78701", "78702", "78703", "78704", "78705", "78721", "78722",
    "78723", "78724", "78727", "78728", "78729", "78731", "78741",
    "78744", "78745", "78746", "78748", "78749", "78750", "78751",
    "78752", "78753", "78756", "78757", "78758", "78759",
}
SERVICE_CITIES = {"austin", "round rock", "pflugerville", "cedar park"}
