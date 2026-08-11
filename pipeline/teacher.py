"""Shared client for the teacher / judge endpoint.

Used by stage 3 (trajectories), stage 4 (judge), and stage 6 (eval), so cost
accounting, retries, and fixture replay live in exactly one place.

Endpoint is any OpenAI-compatible chat-completions API:

    TEACHER_BASE_URL   default https://openrouter.ai/api/v1
    TEACHER_MODEL      default meta/muse-glimmer-30b
    JUDGE_MODEL        scoring model (stage 4) — deliberately NOT the teacher
    OPENROUTER_API_KEY / LLM_API_KEY

Fixture mode records every (request -> response) pair to disk and replays it,
so the whole pipeline is testable at zero cost and with no key at all. Replay
is keyed on a hash of the request, so a changed prompt is a cache miss rather
than a silently wrong answer.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from pipeline.config import PROJECT_ROOT, RAW

# Loaded lazily so importing this module never requires a .env to exist.
_ENV_LOADED = False

FIXTURE_DIR = RAW / "teacher_fixtures"

# OpenRouter list price for meta/muse-glimmer-30b, verified 2026-08-11.
# Used only for the local cost estimate printed after a run — the provider's
# invoice is authoritative.
PRICE_PER_M_INPUT = 0.35
PRICE_PER_M_OUTPUT = 1.50


def _load_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def base_url() -> str:
    _load_env()
    return os.environ.get("TEACHER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")


def api_key() -> str | None:
    _load_env()
    for name in ("OPENROUTER_API_KEY", "TEACHER_API_KEY", "LLM_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def teacher_model() -> str:
    _load_env()
    return os.environ.get("TEACHER_MODEL", "meta/muse-glimmer-30b")


def judge_model() -> str:
    _load_env()
    # Defaulting the judge to the teacher would let the model grade its own
    # work; callers should set JUDGE_MODEL explicitly.
    return os.environ.get("JUDGE_MODEL", "").strip() or teacher_model()


class TeacherError(RuntimeError):
    pass


class MissingKey(TeacherError):
    pass


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hits: int = 0
    errors: int = 0
    seconds: float = 0.0

    def add(self, other: "Usage") -> None:
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_hits += other.cache_hits
        self.errors += other.errors
        self.seconds += other.seconds

    def estimated_cost(self) -> float:
        return (self.input_tokens / 1e6 * PRICE_PER_M_INPUT
                + self.output_tokens / 1e6 * PRICE_PER_M_OUTPUT)

    def summary(self) -> str:
        return (
            f"{self.calls} calls "
            f"({self.cache_hits} replayed) | "
            f"in {self.input_tokens:,} tok, out {self.output_tokens:,} tok | "
            f"~${self.estimated_cost():.3f} | {self.seconds:.1f}s"
        )


@dataclass
class TeacherClient:
    model: str = field(default_factory=teacher_model)
    mode: str = "live"              # live | record | replay
    timeout: float = 180.0
    max_retries: int = 3
    usage: Usage = field(default_factory=Usage)
    fixture_dir: Path = field(default_factory=lambda: FIXTURE_DIR)

    def __post_init__(self) -> None:
        if self.mode in ("record", "replay"):
            self.fixture_dir.mkdir(parents=True, exist_ok=True)

    # -- fixtures ---------------------------------------------------------
    def _fixture_path(self, payload: dict) -> Path:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]
        return self.fixture_dir / f"{digest}.json"

    # -- main entry point -------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1600,
        reasoning_strength: str | None = None,
    ) -> dict:
        """Return the assistant message dict. Raises TeacherError on failure."""
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
        if reasoning_strength:
            # Glimmer renders this into the system turn via its chat template.
            payload["reasoning_strength"] = reasoning_strength

        path = self._fixture_path(payload)

        if self.mode == "replay":
            if not path.exists():
                raise TeacherError(
                    f"No fixture for this request ({path.name}). Record one with "
                    f"--mode record, or the prompt changed since recording."
                )
            self.usage.calls += 1
            self.usage.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["message"]

        if self.mode == "record" and path.exists():
            self.usage.calls += 1
            self.usage.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["message"]

        key = api_key()
        if not key:
            raise MissingKey(
                "No API key. Put OPENROUTER_API_KEY in C:\\Lumen Ember\\.env, "
                "or run with --mode replay to use recorded fixtures."
            )

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            # OpenRouter attribution headers; harmless elsewhere.
            "HTTP-Referer": "https://github.com/patmakesapps/Lumen-Ember",
            "X-Title": "Lumen-Ember",
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            started = time.monotonic()
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(
                        f"{base_url()}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                elapsed = time.monotonic() - started
                self.usage.seconds += elapsed

                if resp.status_code == 401:
                    raise MissingKey("Endpoint rejected the API key (401).")
                if resp.status_code == 402:
                    raise TeacherError(
                        "Out of credits (402). Top up the account and rerun; "
                        "recorded fixtures are reused so nothing is lost."
                    )
                if resp.status_code in (429, 500, 502, 503, 529):
                    last_error = TeacherError(
                        f"{resp.status_code} from endpoint: {resp.text[:200]}")
                    time.sleep(2 ** attempt)
                    continue
                if resp.status_code != 200:
                    raise TeacherError(
                        f"{resp.status_code} from endpoint: {resp.text[:400]}")

                data = resp.json()
                usage = data.get("usage") or {}
                self.usage.calls += 1
                self.usage.input_tokens += int(usage.get("prompt_tokens") or 0)
                self.usage.output_tokens += int(usage.get("completion_tokens") or 0)

                choices = data.get("choices") or []
                if not choices:
                    raise TeacherError(f"No choices in response: {str(data)[:300]}")
                message = choices[0].get("message") or {}

                if self.mode == "record":
                    path.write_text(
                        json.dumps({"request": payload, "message": message,
                                    "usage": usage}, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                return message

            except (MissingKey, TeacherError):
                raise
            except Exception as e:                      # network-level
                last_error = e
                self.usage.seconds += time.monotonic() - started
                time.sleep(2 ** attempt)

        self.usage.errors += 1
        raise TeacherError(f"Endpoint failed after {self.max_retries} attempts: {last_error}")


def preflight() -> dict:
    """Cheap connectivity + auth check. One tiny request."""
    client = TeacherClient()
    msg = client.chat(
        [{"role": "user", "content": "Reply with the single word: ready"}],
        max_tokens=8,
    )
    return {
        "model": client.model,
        "base_url": base_url(),
        "reply": (msg.get("content") or "").strip()[:40],
        "usage": client.usage.summary(),
    }


if __name__ == "__main__":
    _load_env()
    print(f"base_url : {base_url()}")
    print(f"teacher  : {teacher_model()}")
    print(f"judge    : {judge_model()}")
    print(f"api key  : {'present' if api_key() else 'MISSING'}")
    if api_key():
        try:
            info = preflight()
            print(f"preflight: ok — reply={info['reply']!r}")
            print(f"           {info['usage']}")
        except Exception as e:
            print(f"preflight: FAILED — {e}")
