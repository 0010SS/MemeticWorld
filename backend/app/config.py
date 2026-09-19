"""Runtime configuration. Everything provider-specific comes from environment variables.

Values are read from the process environment, falling back to the repo-root `.env`.
See `.env.example` for the full list.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent

load_dotenv(REPO_ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    llm_provider: str  # "openai" (any OpenAI-compatible API) | "mock"
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_temperature: float | None
    llm_max_tokens: int | None
    llm_json_mode: bool
    llm_max_concurrency: int
    llm_timeout: float
    llm_cache: bool
    embed_provider: str  # "openai" | "hash"
    embed_base_url: str
    embed_api_key: str
    embed_model: str
    db_path: str
    cors_origins: list[str]

    @property
    def model_label(self) -> str:
        return "mock" if self.llm_provider == "mock" else self.llm_model


def load_settings() -> Settings:
    api_key = _env("LLM_API_KEY") or _env("OPENAI_API_KEY")
    provider = _env("LLM_PROVIDER") or ("openai" if api_key else "mock")
    base_url = _env("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    embed_provider = _env("EMBED_PROVIDER") or ("openai" if provider == "openai" else "hash")
    max_tokens = _env("LLM_MAX_TOKENS")
    # "none" omits the parameter entirely (some reasoning models reject it).
    temperature_raw = _env("LLM_TEMPERATURE", "0.9")
    temperature = None if temperature_raw.lower() in {"", "none"} else float(temperature_raw)
    return Settings(
        llm_provider=provider,
        llm_base_url=base_url,
        llm_api_key=api_key,
        llm_model=_env("LLM_MODEL", "gpt-4o-mini"),
        llm_temperature=temperature,
        llm_max_tokens=int(max_tokens) if max_tokens else None,
        llm_json_mode=_flag("LLM_JSON_MODE", False),
        llm_max_concurrency=int(_env("LLM_MAX_CONCURRENCY", "8")),
        llm_timeout=float(_env("LLM_TIMEOUT", "60")),
        llm_cache=_flag("LLM_CACHE", True),
        embed_provider=embed_provider,
        embed_base_url=(_env("EMBED_BASE_URL") or base_url).rstrip("/"),
        embed_api_key=_env("EMBED_API_KEY") or api_key,
        embed_model=_env("EMBED_MODEL", "text-embedding-3-small"),
        db_path=_env("MEMETIC_DB_PATH") or str(BACKEND_DIR / "data" / "memeticworld.db"),
        cors_origins=[o.strip() for o in _env("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()],
    )


settings = load_settings()
