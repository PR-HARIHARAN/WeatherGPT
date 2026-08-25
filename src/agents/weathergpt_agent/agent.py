"""WeatherGPT agent: the whole agent layer, in one file.

`create_agent` is LangChain's prebuilt agent harness — it owns the model/tool
loop, tool execution, error handling, streaming and checkpointing. What is left
for us is genuinely ours and nothing more:

1. which model to talk to (Groq first, Gemini as fallback),
2. one system prompt per persona (`prompts.py`),
3. the tools themselves (`tools/`).

There is no intent router: the model chooses tools, which is what tool-calling
is for. There is no custom reply type either: `achat` returns the final
`AIMessage`, with our few extras in its `response_metadata`, so the gateway
serialises a standard LangChain message instead of a bespoke dataclass.

Gateway usage:

    agent = WeatherGPTAgent()          # once, at app startup
    message = await agent.achat("will it rain in Coimbatore tomorrow",
                                session_id=sid, persona="farmer")
    message_text(message)                       # -> answer for Bhashini TTS
    message.response_metadata["weather_gpt"]    # -> tools used, degraded, latency
    message.model_dump()                        # -> JSON for the API response
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, AsyncIterator, Literal, Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    SummarizationMiddleware,
)
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.checkpoint.memory import InMemorySaver

from .clients import close_client
from .config import get_settings
from .prompts import build_context_note, build_system_prompt
from .tools import ALL_TOOLS

logger = logging.getLogger(__name__)

Persona = Literal["farmer", "general"]

IST = timezone(timedelta(hours=5, minutes=30))
# Substrings that mean "this answer is not official IMD data". The agent-side
# Open-Meteo fallback emits the first; the backend emits the others when it has no
# IMD key or had to derive a hazard assessment itself.
FALLBACK_MARKERS = (
    "fallback global model",
    "no IMD key configured",
    "NOT an official IMD",
    "UNOFFICIAL derived",
    "not an IMD agromet",
    "not an IMD bulletin",
)

# Key under which we stash our own metadata on the returned AIMessage.
META_KEY = "weather_gpt"


class NoLLMConfigured(RuntimeError):
    """Neither GROQ_API_KEY nor GOOGLE_API_KEY is set."""


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _provider_chain() -> tuple[BaseChatModel, ...]:
    """Configured models in preference order: Groq first, Gemini second.

    `init_chat_model` takes "provider:model" strings, so switching providers is
    an env change. `InMemoryRateLimiter` paces Groq under its ~30 req/min free
    tier; the failover itself is `ModelFallbackMiddleware` inside the agent, not
    `.with_fallbacks` here, because `create_agent` needs a real chat model it can
    call `bind_tools` on and a fallback-wrapped runnable is not one.

    API keys are not passed here on purpose; the integrations read GROQ_API_KEY
    and GOOGLE_API_KEY from the environment themselves.
    """
    settings = get_settings().llm
    chain: list[BaseChatModel] = []

    for spec in settings.available_models:
        provider = spec.split(":", 1)[0]
        kwargs: dict[str, Any] = {
            "temperature": settings.temperature,
            "timeout": settings.request_timeout,
        }
        kwargs["max_tokens"] = settings.max_tokens
        kwargs["max_retries"] = settings.max_retries
        if provider == "groq" and settings.groq_requests_per_second > 0:
            # Groq's free tier is ~30 req/min and a judging queue will hit it.
            kwargs["rate_limiter"] = InMemoryRateLimiter(
                requests_per_second=settings.groq_requests_per_second,
                check_every_n_seconds=0.1,
                max_bucket_size=settings.rate_limit_burst,
            )
        try:
            chain.append(init_chat_model(spec, **kwargs))
        except Exception as exc:  # noqa: BLE001 - missing provider package, bad spec
            logger.warning("skipping model %s: %s", spec, exc)

    if not chain:
        raise NoLLMConfigured(
            "No usable model. Set LLM_MODELS and the matching API key, e.g. "
            "LLM_MODELS=groq:openai/gpt-oss-120b with GROQ_API_KEY set."
        )
    return tuple(chain)


def get_primary_model() -> BaseChatModel:
    """The model the agent talks to first."""
    return _provider_chain()[0]


def get_fallback_models() -> list[BaseChatModel]:
    """Models to fail over to, in order. Empty if only one provider is configured."""
    return list(_provider_chain()[1:])


# --------------------------------------------------------------------------- #
# transcript helpers
# --------------------------------------------------------------------------- #

def message_text(message: AnyMessage) -> str:
    """Plain text of a message, via v1 `content_blocks` with a string fallback."""
    blocks = getattr(message, "content_blocks", None)
    if blocks:
        text = "".join(
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
        if text:
            return text
    content = getattr(message, "content", "")
    return content.strip() if isinstance(content, str) else ""


def _turn_start(messages: Sequence[AnyMessage]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return i
    return 0


def tools_used(messages: Sequence[AnyMessage]) -> list[str]:
    """Names of every tool called during this turn, for the debug panel."""
    names: list[str] = []
    for message in messages[_turn_start(messages):]:
        if isinstance(message, AIMessage):
            names.extend(call["name"] for call in message.tool_calls or [])
    return names


def was_degraded(messages: Sequence[AnyMessage]) -> bool:
    """True if any tool result this turn was not official IMD data."""
    for message in messages[_turn_start(messages):]:
        content = str(getattr(message, "content", ""))
        if any(marker in content for marker in FALLBACK_MARKERS):
            return True
    return False


def _final_message(messages: Sequence[AnyMessage]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls and message_text(message):
            return message
    return None


# --------------------------------------------------------------------------- #
# facade
# --------------------------------------------------------------------------- #

class WeatherGPTAgent:
    """One instance serves every concurrent user.

    Conversation state lives in the checkpointer keyed by `session_id`, and both
    persona agents share it, so switching persona mid-session keeps the history.
    """

    def __init__(self, checkpointer: Any | None = None) -> None:
        self._settings = get_settings()
        self._checkpointer = checkpointer or InMemorySaver()
        self._model = get_primary_model()
        self._agents: dict[str, Any] = {}

    # -- internals ---------------------------------------------------------- #

    def _middleware(self) -> list[Any]:
        """Prebuilt middleware, in place of hand-written equivalents.

        - ModelFallbackMiddleware: Groq rate-limits or 5xx roll to Gemini mid-run.
        - ModelCallLimitMiddleware: caps the loop and ends the run *gracefully*
          (`exit_behavior="end"`), so a stuck conversation still answers instead
          of raising a recursion error at the user.
        - SummarizationMiddleware: a session is a long-lived thread, and without
          this the checkpointed history grows until every turn drags the whole
          conversation through the model. `trigger` must be passed explicitly;
          summarisation does not fire without it.
        """
        stack: list[Any] = []

        fallbacks = get_fallback_models()
        if fallbacks:
            stack.append(ModelFallbackMiddleware(*fallbacks))

        stack.append(
            ModelCallLimitMiddleware(
                run_limit=self._settings.max_tool_iterations + 1,
                exit_behavior="end",
            )
        )
        stack.append(
            SummarizationMiddleware(
                model=self._model,
                trigger=("tokens", self._settings.summarize_after_tokens),
                keep=("messages", self._settings.keep_recent_messages),
            )
        )
        return stack

    def _agent_for(self, persona: str):
        """One prebuilt agent per persona, built on first use."""
        if persona not in self._agents:
            self._agents[persona] = create_agent(
                model=self._model,
                tools=ALL_TOOLS,
                system_prompt=build_system_prompt(persona),
                middleware=self._middleware(),
                checkpointer=self._checkpointer,
            )
        return self._agents[persona]

    def _user_message(
        self,
        message: str,
        language: str,
        location: str | None,
        latitude: float | None,
        longitude: float | None,
    ) -> HumanMessage:
        coords = (
            f"{latitude}, {longitude}"
            if latitude is not None and longitude is not None
            else "not provided"
        )
        return HumanMessage(
            content=build_context_note(
                question=message,
                # IST, not UTC: "tomorrow" must mean tomorrow for the user, and
                # near midnight UTC that is a different day in India.
                now=datetime.now(IST).strftime("%A %d %B %Y, %H:%M IST"),
                location=location or self._settings.default_location,
                coords=coords,
                language=language or self._settings.default_language,
            )
        )

    def _config(self, session_id: str, user_id: str | None, persona: str) -> dict[str, Any]:
        return {
            "configurable": {
                "thread_id": session_id,
                # Profile tools read the authenticated user id from config, so
                # the model is never in a position to supply one.
                "user_id": user_id or "anonymous",
            },
            # Picked up by LangSmith when tracing is on, so demo-day latency can
            # be sliced by persona without adding any logging of our own.
            "run_name": f"weathergpt-{persona}",
            "tags": [f"persona:{persona}"],
            "metadata": {"session_id": session_id, "persona": persona},
            # No recursion_limit override. ModelCallLimitMiddleware is the cap and
            # it ends gracefully; a hand-derived recursion_limit raced it and won,
            # which turned a graceful stop into a GraphRecursionError at the user.
            # LangGraph's own default stays as the backstop.
        }

    # -- public API --------------------------------------------------------- #

    async def achat(
        self,
        message: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        language: str = "en",
        location: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        persona: Persona = "general",
    ) -> AIMessage:
        """Answer one turn.

        Never raises: on failure the caller still gets an AIMessage, carrying the
        exception under `response_metadata["weather_gpt"]["error"]`.
        """
        session_id = session_id or str(uuid.uuid4())
        started = time.perf_counter()

        def elapsed_ms() -> float:
            return round((time.perf_counter() - started) * 1000, 1)

        try:
            result = await self._agent_for(persona).ainvoke(
                {
                    "messages": [
                        self._user_message(message, language, location, latitude, longitude)
                    ]
                },
                config=self._config(session_id, user_id, persona),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("agent turn failed for session %s", session_id)
            return AIMessage(
                content=(
                    # Deliberately does not name the weather service: this also
                    # catches model failures, and in testing a dead Ollama produced
                    # "could not reach the weather service", sending debugging the
                    # wrong way for several minutes.
                    "Something went wrong on my side and I could not answer that. "
                    "Please try again in a moment. For an urgent warning, check "
                    "mausam.imd.gov.in or call the NDMA helpline 1078."
                ),
                response_metadata={
                    META_KEY: {
                        "session_id": session_id,
                        "language": language,
                        "tools_used": [],
                        "degraded": False,
                        "latency_ms": elapsed_ms(),
                        "error": str(exc),
                    }
                },
            )

        messages = result.get("messages", [])
        answer = _final_message(messages) or AIMessage(
            content=(
                "I did not get a usable answer from the weather service. Could you ask "
                "again with the place name, for example 'rain in Nashik tomorrow'?"
            )
        )
        # Copy rather than mutate: that message is the same object the checkpointer
        # is holding, and metadata is merged rather than replaced so the provider's
        # own fields (token usage, finish reason, which model answered) survive.
        return answer.model_copy(
            update={
                "response_metadata": {
                    **(answer.response_metadata or {}),
                    META_KEY: {
                        "session_id": session_id,
                        "language": language,
                        "tools_used": tools_used(messages),
                        "degraded": was_degraded(messages),
                        "latency_ms": elapsed_ms(),
                        "error": None,
                    },
                }
            }
        )

    async def astream(
        self,
        message: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        language: str = "en",
        location: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        persona: Persona = "general",
    ) -> AsyncIterator[str]:
        """Yield answer tokens as they arrive, for the chat UI.

        Only assistant text is emitted: tool-call argument deltas and tool
        results are skipped, so the user never sees raw JSON mid-stream. For a
        stream that also reports which tool is running, see `astream_events`.
        """
        session_id = session_id or str(uuid.uuid4())

        try:
            async for chunk, _metadata in self._agent_for(persona).astream(
                {
                    "messages": [
                        self._user_message(message, language, location, latitude, longitude)
                    ]
                },
                config=self._config(session_id, user_id, persona),
                stream_mode="messages",
            ):
                if not isinstance(chunk, AIMessage):
                    continue
                if getattr(chunk, "tool_call_chunks", None):
                    continue
                text = chunk.content
                if isinstance(text, str) and text:
                    yield text
        except Exception as exc:  # noqa: BLE001
            logger.exception("agent stream failed for session %s", session_id)
            yield "\n\nI lost the connection to the weather service. Please try again."
            logger.debug("stream error detail: %s", exc)

    async def astream_events(
        self,
        message: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        language: str = "en",
        location: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        persona: Persona = "general",
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream the turn as a sequence of small typed events, for a live UI.

        Yields dicts, always one of:
            {"type": "tool_start", "tool": <name>}
            {"type": "tool_end", "tool": <name>, "degraded": <bool>}
            {"type": "token", "text": <str>}
            {"type": "done", "session_id": ..., "tools_used": [...], "degraded": <bool>}
            {"type": "error", "message": <str>}

        Built on LangGraph's `astream_events` (v2) rather than `astream`: that
        API only exposes token deltas, with no signal for when a tool starts
        or finishes, so a UI built on it alone cannot show "checking IMD..."
        while a tool call is in flight.
        """
        session_id = session_id or str(uuid.uuid4())
        tools_seen: list[str] = []
        degraded = False

        try:
            async for event in self._agent_for(persona).astream_events(
                {
                    "messages": [
                        self._user_message(message, language, location, latitude, longitude)
                    ]
                },
                config=self._config(session_id, user_id, persona),
                version="v2",
            ):
                kind = event.get("event")
                name = event.get("name")

                if kind == "on_tool_start" and name:
                    tools_seen.append(name)
                    yield {"type": "tool_start", "tool": name}

                elif kind == "on_tool_end" and name:
                    output = event.get("data", {}).get("output")
                    content = output.content if hasattr(output, "content") else str(output or "")
                    if any(marker in str(content) for marker in FALLBACK_MARKERS):
                        degraded = True
                    yield {"type": "tool_end", "tool": name, "degraded": degraded}

                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    text = getattr(chunk, "content", None)
                    # Skip chunks that are only tool-call argument deltas: those
                    # carry no `content`, only `tool_call_chunks`, and printing
                    # them would leak raw JSON into the answer stream.
                    if isinstance(text, str) and text and not getattr(chunk, "tool_call_chunks", None):
                        yield {"type": "token", "text": text}

            yield {
                "type": "done",
                "session_id": session_id,
                "tools_used": tools_seen,
                "degraded": degraded,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("agent event stream failed for session %s", session_id)
            yield {
                "type": "error",
                "message": (
                    "Something went wrong on my side and I could not answer that. "
                    "Please try again in a moment. For an urgent warning, check "
                    "mausam.imd.gov.in or call the NDMA helpline 1078."
                ),
            }
            logger.debug("stream error detail: %s", exc)

    async def aclose(self) -> None:
        """Release the shared HTTP pool. Call from the FastAPI shutdown hook."""
        await close_client()


__all__ = [
    "WeatherGPTAgent",
    "Persona",
    "META_KEY",
    "get_primary_model",
    "get_fallback_models",
    "message_text",
    "tools_used",
    "was_degraded",
]
