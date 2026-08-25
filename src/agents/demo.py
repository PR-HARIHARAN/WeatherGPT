"""Terminal demo of the WeatherGPT agent layer.

Imports and drives every public component. Nothing is implemented here: each
section calls into the package and prints what came back.

    # terminal 1
    uv run --extra backend uvicorn weather_backend.main:app --port 8000

    # terminal 2
    $env:LLM_MODELS="groq:openai/gpt-oss-120b"
    $env:WEATHER_BACKEND_URL="http://127.0.0.1:8000"
    uv run python scripts/demo.py
"""

from __future__ import annotations

import asyncio
import os
import sys

# This is a multilingual product, so the console has to be UTF-8 or printing a
# Tamil or Hindi reply raises UnicodeEncodeError on a default Windows terminal.
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# --- the entire public surface of the agent layer --------------------------- #
from weathergpt_agent import (  # noqa: E402
    META_KEY,
    WeatherGPTAgent,
    compose_alert,
    get_settings,
    message_text,
    tools_used,
    was_degraded,
)
from weathergpt_agent.agent import get_fallback_models, get_primary_model  # noqa: E402
from weathergpt_agent.prompts import build_context_note, build_system_prompt  # noqa: E402
from weathergpt_agent.tools import ALL_TOOLS, TOOLS_BY_NAME  # noqa: E402

SESSION = "demo-session"
USER = "demo-user"
RULE = "=" * 78

# A warning record shaped like the alerts lead's polling job would deliver it.
DEMO_WARNING = {
    "severity": "orange",
    "hazard": "heavy rainfall",
    "district": "Thanjavur",
    "valid_from": "today 18:00 IST",
    "valid_until": "tomorrow 08:00 IST",
    "description": "Heavy to very heavy rain likely, 70 to 110 millimetres in 24 hours.",
}


def show_config() -> None:
    print(RULE)
    print("1. CONFIG  (weather_gpt.get_settings)")
    settings = get_settings()
    print(f"   models configured : {settings.llm.models}")
    print(f"   models usable now : {settings.llm.available_models}")
    print(f"   backend           : {settings.backend.base_url}")
    print(f"   open-meteo fallback: {settings.backend.enable_fallback}")
    print(f"   tool loop cap     : {settings.max_tool_iterations} model calls per turn")
    print(
        f"   compaction        : summarise past {settings.summarize_after_tokens} tokens, "
        f"keep {settings.keep_recent_messages} messages"
    )


def show_models() -> None:
    print(RULE)
    print("2. MODELS  (weather_gpt.agent.get_primary_model / get_fallback_models)")
    primary = get_primary_model()
    print(f"   primary   : {type(primary).__name__} / {primary.model}")
    fallbacks = get_fallback_models()
    print(
        f"   fallbacks : {[m.model for m in fallbacks] or 'none configured'}"
        "   (ModelFallbackMiddleware)"
    )


def show_prompt() -> None:
    print(RULE)
    print("3. PROMPT  (weather_gpt.prompts.build_system_prompt)")
    prompt = build_system_prompt("farmer")
    print(f"   {len(prompt.split())} words, constant per persona\n")
    for line in prompt.splitlines():
        print(f"   | {line}")
    print("\n   per-turn context rides on the user message (build_context_note):")
    note = build_context_note(
        question="will it rain tomorrow",
        now="Monday 24 August 2026, 14:30 IST",
        location="Coimbatore",
        coords="not provided",
        language="en",
    )
    for line in note.splitlines():
        print(f"   | {line}")


def show_tools() -> None:
    print(RULE)
    print("4. TOOLS  (weather_gpt.tools.ALL_TOOLS)")
    for tool in ALL_TOOLS:
        print(f"   {tool.name:28} args={list(tool.args) or '[] (injected config only)'}")
    sample = TOOLS_BY_NAME["get_district_warnings"]
    print(f"\n   TOOLS_BY_NAME['{sample.name}'] description:")
    print(f"   | {sample.description[:150]}...")


async def stream_turn(agent: WeatherGPTAgent, question: str, **kwargs) -> None:
    print(RULE)
    print("5. STREAMING TURN  (WeatherGPTAgent.astream)")
    print(f"   you> {question}")
    print("   bot> ", end="", flush=True)
    async for token in agent.astream(question, session_id=SESSION, user_id=USER, **kwargs):
        print(token, end="", flush=True)
    print()


async def buffered_turn(agent: WeatherGPTAgent, label: str, question: str, **kwargs):
    print(RULE)
    print(f"{label}  (WeatherGPTAgent.achat -> AIMessage)")
    print(f"   you> {question}")
    reply = await agent.achat(question, session_id=SESSION, user_id=USER, **kwargs)
    print(f"   bot> {message_text(reply)}")
    meta = (reply.response_metadata or {}).get(META_KEY, {})
    print(f"   tools_used   : {meta.get('tools_used')}")
    print(f"   degraded     : {meta.get('degraded')}  (True = not IMD data)")
    print(f"   latency_ms   : {meta.get('latency_ms')}")
    print(f"   error        : {meta.get('error')}")
    return reply


async def show_alert() -> None:
    print(RULE)
    print("6. PUSH ALERT  (weather_gpt.compose_alert)")
    print(f"   warning in : {DEMO_WARNING['severity']} / {DEMO_WARNING['hazard']}")
    print(f"   alert out  : {await compose_alert(DEMO_WARNING, persona='farmer', language='ta')}")


def show_transcript_helpers(reply) -> None:
    print(RULE)
    print("7. TRANSCRIPT HELPERS  (weather_gpt.tools_used / was_degraded)")
    print("   Both take a message list. Over the final answer alone they are empty by")
    print("   design: the tool calls live in earlier messages of the turn, which is")
    print("   exactly why achat pre-computes them into response_metadata for you.")
    print(f"   tools_used([final_reply])   : {tools_used([reply])}")
    print(f"   was_degraded([final_reply]) : {was_degraded([reply])}")
    meta = (reply.response_metadata or {}).get(META_KEY, {})
    print(f"   response_metadata['{META_KEY}'] : {meta.get('tools_used')}")


async def main() -> None:
    show_config()
    show_models()
    show_prompt()
    show_tools()

    agent = WeatherGPTAgent()
    try:
        await stream_turn(
            agent, "will it rain in Coimbatore tomorrow", location="Coimbatore", persona="farmer"
        )
        await buffered_turn(
            agent,
            "5b. FOLLOW-UP (same session, checkpointer memory)",
            "and the day after?",
            location="Coimbatore",
            persona="farmer",
        )
        await buffered_turn(
            agent,
            "5c. HAZARD (no fallback source, must not guess)",
            "is it safe to travel in Thanjavur tonight",
            location="Thanjavur",
            persona="farmer",
        )
        last = await buffered_turn(
            agent,
            "5d. PROFILE WRITE (user_id injected from config, not from the model)",
            "save my farm location as Erode",
            persona="farmer",
        )
        await show_alert()
        show_transcript_helpers(last)
    finally:
        # Releases the shared httpx pool (weather_gpt.clients.close_client).
        await agent.aclose()
        print(RULE)
        print("closed HTTP pool. demo complete.")


if __name__ == "__main__":
    asyncio.run(main())
