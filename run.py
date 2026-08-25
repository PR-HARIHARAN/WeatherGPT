"""Run the whole WeatherGPT project (backend gateway + frontend dev server) with one command.

    python run.py                  # backend on :8000, frontend on :5173
    python run.py --no-frontend    # backend only
    python run.py --backend-port 8001 --frontend-port 5174

What it does:
- Puts `src` and `src/agents` on PYTHONPATH (so `gateway`, `weathergpt_agent`,
  `weather_backend`, `weather_gpt`, `src.alerts`, `src.multilingual_system` all
  import correctly) and launches `uvicorn gateway.main:app`.
- Runs `npm install` in `frontend/` if `node_modules` is missing, copies
  `.env.example` to `.env` if no `.env` exists yet, then launches `npm run dev`.
- Streams both processes' output with a prefix so it's clear which is which,
  and shuts both down cleanly on Ctrl+C.

Requires: `uv` (for the backend) and `npm` (for the frontend) on PATH.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
AGENTS_SRC = SRC / "agents"
FRONTEND = ROOT / "frontend"


def _stream(pipe, prefix: str) -> None:
    for line in iter(pipe.readline, ""):
        if not line:
            break
        print(f"[{prefix}] {line.rstrip()}")
    pipe.close()


def _start(cmd: list[str], cwd: Path, env: dict, prefix: str) -> subprocess.Popen:
    print(f"[{prefix}] starting: {' '.join(cmd)}")
    # On Windows, npm/uv etc. are `.cmd`/`.bat` shims; shutil.which() finds the
    # real path (including the extension) but Popen won't launch a shim
    # without shell=True unless given that resolved path explicitly.
    resolved = shutil.which(cmd[0])
    if resolved:
        cmd = [resolved, *cmd[1:]]
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threading.Thread(target=_stream, args=(proc.stdout, prefix), daemon=True).start()
    return proc


def _backend_env() -> dict:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    parts = [str(SRC), str(AGENTS_SRC)] + ([existing] if existing else [])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env.setdefault("LLM_MODELS", "groq:openai/gpt-oss-120b")
    return env


def _ensure_frontend_ready() -> None:
    if not (FRONTEND / "node_modules").exists():
        print("[frontend] node_modules missing, running npm install ...")
        subprocess.run(["npm", "install"], cwd=str(FRONTEND), check=True)
    env_file = FRONTEND / ".env"
    env_example = FRONTEND / ".env.example"
    if not env_file.exists() and env_example.exists():
        shutil.copy(env_example, env_file)
        print("[frontend] created frontend/.env from .env.example")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--frontend-port", type=int, default=5173)
    parser.add_argument("--no-frontend", action="store_true", help="run the backend only")
    parser.add_argument("--no-reload", action="store_true", help="disable uvicorn --reload")
    args = parser.parse_args()

    for tool in ("uv",) if args.no_frontend else ("uv", "npm"):
        if shutil.which(tool) is None:
            print(f"error: '{tool}' not found on PATH", file=sys.stderr)
            return 1

    procs: list[subprocess.Popen] = []
    try:
        backend_cmd = [
            "uv", "run", "uvicorn", "gateway.main:app",
            "--host", "0.0.0.0", "--port", str(args.backend_port),
        ]
        if not args.no_reload:
            backend_cmd.append("--reload")
        procs.append(_start(backend_cmd, ROOT, _backend_env(), "backend"))

        if not args.no_frontend:
            _ensure_frontend_ready()
            frontend_cmd = ["npm", "run", "dev", "--", "--port", str(args.frontend_port), "--host"]
            procs.append(_start(frontend_cmd, FRONTEND, os.environ.copy(), "frontend"))

        print(
            f"\nWeatherGPT running:\n"
            f"  backend  -> http://localhost:{args.backend_port} (docs at /docs)\n"
            + (f"  frontend -> http://localhost:{args.frontend_port}\n" if not args.no_frontend else "")
            + "Press Ctrl+C to stop.\n"
        )

        for proc in procs:
            proc.wait()
        return 0
    except KeyboardInterrupt:
        print("\nshutting down ...")
        return 0
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
