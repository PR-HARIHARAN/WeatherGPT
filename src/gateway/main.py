"""Single entrypoint that wires every component together.

Run it with:

    uv run uvicorn gateway.main:app --reload --port 8000

Env vars of note (see .env.example):
    LLM_MODELS               default "groq:openai/gpt-oss-120b"
    WEATHER_BACKEND_URL      default "http://localhost:8000/backend" (this
                              process, via the mounted weather_backend app)
    IMD_API_KEY               optional, enables the IMD tier in weather_backend
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Optional

import json

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# `src` and `src/agents` both need to be importable: the agent package
# (`weathergpt_agent`) and the backend package (`weather_backend`) live under
# `src/agents`, everything else lives directly under `src`.
_AGENTS_PATH = os.path.join(os.path.dirname(__file__), "..", "agents")
if _AGENTS_PATH not in sys.path:
    sys.path.insert(0, _AGENTS_PATH)

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("LLM_MODELS", "groq:openai/gpt-oss-120b")
os.environ.setdefault("WEATHER_BACKEND_URL", "http://127.0.0.1:8000/backend")

from src.alerts.database import init_db as init_alerts_db  # noqa: E402
from src.alerts.database import Alert, SessionLocal  # noqa: E402
from src.alerts.routes import router as alerts_router  # noqa: E402
from src.multilingual_system.translation_manager import MultilingualSystem  # noqa: E402
from weather_gpt.voice_service import VoiceService  # noqa: E402
from weather_backend.main import app as backend_app  # noqa: E402
from weather_backend import store as backend_store  # noqa: E402
from weathergpt_agent import WeatherGPTAgent, message_text  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("gateway")

agent: Optional[WeatherGPTAgent] = None
translator: Optional[MultilingualSystem] = None
voice: Optional[VoiceService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, translator, voice
    logger.info("starting gateway: alerts DB, agent, translator, voice service")
    init_alerts_db()
    translator = MultilingualSystem()
    voice = VoiceService()
    agent = WeatherGPTAgent()
    async with backend_app.router.lifespan_context(backend_app):
        yield
    await agent.aclose()
    logger.info("gateway shut down")


app = FastAPI(
    title="WeatherGPT",
    version="0.1.0",
    description="Integrated WeatherGPT: agent, IMD/Open-Meteo backend, alerts, multilingual, voice.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# The agent's tools call WEATHER_BACKEND_URL, which defaults to this same
# process at /backend, so mounting (not include_router) is required: mounting
# preserves weather_backend as an independent ASGI app with its own lifespan,
# which is already driven above via lifespan_context.
app.mount("/backend", backend_app)

# Alerts/advisory: unchanged paths (`/alerts/{district}`, `/advisory`, `/ws/alerts`).
app.include_router(alerts_router)


# --------------------------------------------------------------------------- #
# /api/v1/alerts/active - alias the frontend already calls
# --------------------------------------------------------------------------- #

@app.get("/api/v1/alerts/active")
async def get_active_alerts_alias(district: str) -> list[dict[str, Any]]:
    """Non-expired alerts for a district, under the path the frontend expects."""
    from datetime import datetime, timezone

    def _query() -> list[dict[str, Any]]:
        db = SessionLocal()
        try:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            rows = (
                db.query(Alert)
                .filter(Alert.district.ilike(district.strip()))
                .filter(Alert.valid_until > now)
                .order_by(Alert.valid_until.asc())
                .all()
            )
            return [
                {
                    "event": "weather_alert",
                    "alert_id": a.external_warning_id,
                    "district": a.district,
                    "severity": a.severity,
                    "title": a.title,
                    "message": a.message,
                    "action": a.action,
                    "valid_until": a.valid_until.isoformat(),
                }
                for a in rows
            ]
        finally:
            db.close()

    from fastapi.concurrency import run_in_threadpool

    return await run_in_threadpool(_query)


# --------------------------------------------------------------------------- #
# /api/chat, /api/chat/voice - the contract the frontend already calls
# --------------------------------------------------------------------------- #

class ChatRequest(BaseModel):
    message: str
    language: str = "auto"
    crop: str = "other"
    session_id: str | None = None
    latitude: float | None = None
    longitude: float | None = None


def _resolve_reply_language(requested: str, detected: str) -> str:
    """Detected language wins unless the caller explicitly overrode it.

    `"auto"` (or empty) means "trust detection"; any other value is a manual
    pick from the language dropdown and takes priority over detection.
    """
    if requested and requested != "auto":
        return requested
    return detected or "en"


def _persona_for_crop(crop: str) -> str:
    return "general" if crop in (None, "", "other") else "farmer"


@app.post("/api/chat")
async def chat(body: ChatRequest) -> dict[str, Any]:
    """Text chat: translate to English, run the agent, translate the reply back."""
    english_text, detected_lang = translator.process_user_input(body.message)
    reply_language = _resolve_reply_language(body.language, detected_lang)

    reply = await agent.achat(
        english_text,
        session_id=body.session_id,
        language=reply_language,
        persona=_persona_for_crop(body.crop),
        latitude=body.latitude,
        longitude=body.longitude,
    )
    answer_en = message_text(reply)
    answer = translator.process_llm_response(answer_en, reply_language)
    meta = reply.response_metadata.get("weathergpt_agent") or reply.response_metadata.get(
        "weather_gpt", {}
    )

    return {
        "headline": "WeatherGPT",
        "summary": answer,
        "session_id": meta.get("session_id"),
        "language": reply_language,
        "detected_language": detected_lang,
        "tools_used": meta.get("tools_used", []),
        "degraded": meta.get("degraded", False),
    }


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.post("/api/chat/stream")
async def chat_stream(body: ChatRequest):
    """Server-Sent Events version of /api/chat: live tool calls + token stream.

    Event shapes (one JSON object per `data:` line):
        {"type": "tool_start", "tool": "get_forecast"}
        {"type": "tool_end", "tool": "get_forecast", "degraded": false}
        {"type": "token", "text": "..."}               (English only, see below)
        {"type": "final", "text": "...", "tools_used": [...], "degraded": false,
         "session_id": "...", "language": "hi"}
        {"type": "error", "message": "..."}

    Token-by-token text is only streamed when the reply stays in English: the
    translator works on complete sentences, so a non-English reply is
    translated once the agent finishes and delivered as a single `final`
    event instead of word-by-word garbled partial translations.
    """
    english_text, detected_lang = translator.process_user_input(body.message)
    reply_language = _resolve_reply_language(body.language, detected_lang)
    persona = _persona_for_crop(body.crop)

    async def event_source():
        buffer: list[str] = []
        try:
            async for event in agent.astream_events(
                english_text,
                session_id=body.session_id,
                language=reply_language,
                persona=persona,
                latitude=body.latitude,
                longitude=body.longitude,
            ):
                etype = event["type"]

                if etype == "tool_start":
                    yield _sse({"type": "tool_start", "tool": event["tool"]})

                elif etype == "tool_end":
                    yield _sse(
                        {"type": "tool_end", "tool": event["tool"], "degraded": event["degraded"]}
                    )

                elif etype == "token":
                    buffer.append(event["text"])
                    if reply_language == "en":
                        yield _sse({"type": "token", "text": event["text"]})

                elif etype == "done":
                    full_text_en = "".join(buffer)
                    final_text = translator.process_llm_response(full_text_en, reply_language)
                    yield _sse(
                        {
                            "type": "final",
                            "text": final_text,
                            "tools_used": event["tools_used"],
                            "degraded": event["degraded"],
                            "session_id": event["session_id"],
                            "language": reply_language,
                            "detected_language": detected_lang,
                        }
                    )

                elif etype == "error":
                    yield _sse({"type": "error", "message": event["message"]})
        except Exception as exc:  # noqa: BLE001
            logger.exception("chat stream failed")
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat/voice")
async def chat_voice(
    file: UploadFile,
    language: str = "auto",
    crop: str = "other",
    session_id: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
) -> dict[str, Any]:
    """Voice chat: speech-to-text, run the same pipeline as /api/chat, then TTS the reply."""
    audio_bytes = await file.read()
    # STT needs a concrete language hint; "auto" isn't one of gTTS/speech_recognition's
    # codes, so guess English for transcription and let text-based detection in
    # chat() pick the real reply language from the transcript itself.
    stt_language = language if language and language != "auto" else "en"
    transcript = voice.speech_to_text(audio_bytes, language=stt_language)

    chat_result = await chat(
        ChatRequest(
            message=transcript,
            language=language,
            crop=crop,
            session_id=session_id,
            latitude=latitude,
            longitude=longitude,
        )
    )
    chat_result["userTranscript"] = transcript
    return chat_result


class TTSRequest(BaseModel):
    text: str
    language: str = "en"


@app.post("/api/tts")
async def tts(body: TTSRequest):
    """Text-to-speech via gTTS, in whatever language the caller detected.

    Returns MP3 audio bytes on success, or a 503 JSON error if gTTS is
    unavailable/failed so the frontend can fall back to the browser's
    SpeechSynthesis API instead of showing a broken audio player.
    """
    from fastapi.responses import Response
    from fastapi.concurrency import run_in_threadpool

    lang = body.language if body.language and body.language != "auto" else "en"
    audio_bytes = await run_in_threadpool(voice.text_to_speech, body.text, lang)
    if not audio_bytes:
        return JSONResponse(status_code=503, content={"detail": "TTS unavailable"})
    return Response(content=audio_bytes, media_type="audio/mpeg")


@app.get("/health", tags=["System"])
async def health_check() -> dict[str, Any]:
    return {"status": "healthy", "service": "WeatherGPT gateway"}


@app.get("/", tags=["System"])
async def root() -> dict[str, Any]:
    return {"message": "WeatherGPT gateway", "docs": "/docs", "health": "/health"}
