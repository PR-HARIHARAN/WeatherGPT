"""Central configuration for the WeatherGPT agent layer.

Everything the agent needs is read from the environment exactly once, at import
time, so no tool call ever pays for env parsing. Keep secrets in `.env` (see
`.env.example`) and never commit them.

Note on API keys: we deliberately do *not* pass keys into the model
constructors. `init_chat_model` hands kwargs to the provider class, and both
ChatGroq and ChatGoogleGenerativeAI already read GROQ_API_KEY / GOOGLE_API_KEY
from the environment themselves. We only detect presence here, to decide which
providers can be built at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


def _env(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key, default)
    if value is not None:
        value = value.strip() or None
    return value


def _env_float(key: str, default: float) -> float:
    raw = _env(key)
    try:
        return float(raw) if raw is not None else default
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = _env(key)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


# Providers that need an API key, and the variable it lives in.
PROVIDER_KEY_ENV: dict[str, str] = {
    "groq": "GROQ_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
}


@dataclass(frozen=True)
class LLMSettings:
    """Models in preference order, as `init_chat_model` "provider:model" specs.

    One list instead of a field per provider: the first entry is the model the
    agent talks to, the rest are what it fails over to. Adding Cerebras or
    switching to a local model is an env change with no code edit.

        LLM_MODELS=groq:openai/gpt-oss-120b                # default
        LLM_MODELS=groq:openai/gpt-oss-120b,google_genai:gemini-2.0-flash

    Entries whose API key is missing are dropped, so the same value can be
    committed for the whole team and each machine uses what it has.
    """

    models: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            spec.strip()
            for spec in (
                _env("LLM_MODELS", "groq:openai/gpt-oss-120b,google_genai:gemini-2.0-flash")
                or ""
            ).split(",")
            if spec.strip()
        )
    )

    temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.2))
    max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 700))
    # Hard timeout per LLM call. Judges see latency directly, so fail fast and
    # let the fallback model answer rather than hanging on a slow provider.
    request_timeout: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT", 12.0))
    max_retries: int = field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 1))

    # Groq's free tier is ~30 requests/minute. 0.45 req/s leaves headroom, and
    # LangChain's InMemoryRateLimiter paces us instead of us eating 429s during
    # judging. Set to 0 to disable pacing.
    groq_requests_per_second: float = field(
        default_factory=lambda: _env_float("GROQ_REQUESTS_PER_SECOND", 0.45)
    )
    rate_limit_burst: int = field(default_factory=lambda: _env_int("RATE_LIMIT_BURST", 5))

    @property
    def available_models(self) -> tuple[str, ...]:
        """Configured specs whose credentials are actually present."""
        usable = []
        for spec in self.models:
            provider = spec.split(":", 1)[0]
            key_var = PROVIDER_KEY_ENV.get(provider)
            if key_var is None or _env(key_var):
                usable.append(spec)
        return tuple(usable)


@dataclass(frozen=True)
class BackendSettings:
    """The FastAPI service owned by the backend lead (shared OpenAPI contract).

    The agent never talks to IMD directly: it calls our own wrappers so that
    caching, retries and response-shape normalisation live in one place.
    """

    base_url: str = field(
        default_factory=lambda: (
            _env("WEATHER_BACKEND_URL", "http://localhost:8000") or ""
        ).rstrip("/")
    )
    api_key: str | None = field(default_factory=lambda: _env("WEATHER_BACKEND_API_KEY"))
    # 20s, not 8s: a cold climate-trend request makes the backend pull ten years of
    # daily ERA5 values, which is slow once and then cached. An 8s ceiling turned
    # that first call into a spurious "data unavailable".
    timeout: float = field(default_factory=lambda: _env_float("BACKEND_TIMEOUT", 20.0))
    # Direct Open-Meteo call used only if our backend is unreachable mid-demo.
    open_meteo_url: str = "https://api.open-meteo.com/v1/forecast"
    open_meteo_geocode_url: str = "https://geocoding-api.open-meteo.com/v1/search"
    enable_fallback: bool = field(
        default_factory=lambda: _env_bool("ENABLE_OPEN_METEO_FALLBACK", True)
    )


@dataclass(frozen=True)
class AgentSettings:
    llm: LLMSettings = field(default_factory=LLMSettings)
    backend: BackendSettings = field(default_factory=BackendSettings)

    # Safety rail on the tool-calling loop: 3 hops is enough for
    # "resolve location -> get forecast -> get advisory". Floored at 1, because 0
    # means the model cannot even answer once and the run can only fail.
    max_tool_iterations: int = field(
        default_factory=lambda: max(1, _env_int("MAX_TOOL_ITERATIONS", 4))
    )

    # Conversation compaction (SummarizationMiddleware). A session is a long-lived
    # thread, so history has to be bounded or every turn drags the whole
    # conversation through the model.
    summarize_after_tokens: int = field(
        default_factory=lambda: _env_int("SUMMARIZE_AFTER_TOKENS", 4000)
    )
    keep_recent_messages: int = field(
        default_factory=lambda: _env_int("KEEP_RECENT_MESSAGES", 8)
    )

    # No tool cache setting here on purpose. Caching IMD responses is the backend
    # service's job (it owns the IMD wrappers), so the agent would only be a
    # second cache in front of a cache, with a second TTL to reason about.

    default_location: str = field(default_factory=lambda: _env("DEFAULT_LOCATION", "Coimbatore"))
    default_language: str = field(default_factory=lambda: _env("DEFAULT_LANGUAGE", "en"))
    debug: bool = field(default_factory=lambda: _env_bool("AGENT_DEBUG", False))


@lru_cache(maxsize=1)
def get_settings() -> AgentSettings:
    """Process-wide singleton. Call this, don't instantiate AgentSettings."""
    return AgentSettings()
