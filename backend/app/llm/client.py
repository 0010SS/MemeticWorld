"""OpenAI-compatible chat + embeddings client with caching, retries and an offline mock.

Works with anything that speaks `/chat/completions` and `/embeddings` (OpenAI, OpenRouter,
Together, vLLM, Ollama, LM Studio, ...). With LLM_PROVIDER=mock no network calls are made.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import zlib
from typing import Any, TypeVar

import httpx
import numpy as np
from pydantic import BaseModel, ValidationError

from app.config import Settings, settings as default_settings
from app.llm.mock import MockLLM

T = TypeVar("T", bound=BaseModel)

HASH_DIM = 256
REPAIR_MESSAGE = ("That reply was not a valid JSON object in the requested format. "
                  "Reply again with only the JSON object.")


class LLMError(RuntimeError):
    pass


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (tolerates code fences and chatter)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in reply")
    obj = json.loads(text[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("reply is not a JSON object")
    return obj


def hash_embed(text: str, dim: int = HASH_DIM) -> np.ndarray:
    """Deterministic bag-of-words + character-trigram embedding. Offline stand-in for a real model."""
    vec = np.zeros(dim, dtype=np.float32)
    words = re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*", text.lower())
    features = [(w, 2.0) for w in words]
    for w in words:
        padded = f"#{w}#"
        features += [(padded[i:i + 3], 1.0) for i in range(len(padded) - 2)]
    for feature, weight in features:
        h = zlib.crc32(feature.encode())
        vec[h % dim] += weight if (h >> 16) & 1 else -weight
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


class LLMClient:
    def __init__(self, conn=None, cfg: Settings = default_settings, cache_salt: str = ""):
        self.cfg = cfg
        self.conn = conn
        self.cache_salt = cache_salt  # e.g. the run seed, so different seeds don't share cached replies
        self.mock = MockLLM() if cfg.llm_provider == "mock" else None
        self._sem = asyncio.Semaphore(cfg.llm_max_concurrency)
        self._http: httpx.AsyncClient | None = None
        self.stats = {"calls": 0, "cache_hits": 0, "failures": 0,
                      "prompt_tokens": 0, "completion_tokens": 0, "embedded_texts": 0}

    # --- chat -----------------------------------------------------------------------------

    async def chat_json(self, messages: list[dict], model_cls: type[T],
                        mock_ctx: dict | None = None) -> tuple[T | None, str, str | None]:
        """Ask for a JSON object and validate it. Returns (parsed or None, raw text, error)."""
        raw = ""
        try:
            raw = await self._complete(messages, mock_ctx)
            try:
                return model_cls.model_validate(extract_json(raw)), raw, None
            except (ValueError, ValidationError):
                repair = messages + [{"role": "assistant", "content": raw},
                                     {"role": "user", "content": REPAIR_MESSAGE}]
                raw = await self._complete(repair, mock_ctx)
                return model_cls.model_validate(extract_json(raw)), raw, None
        except (ValueError, ValidationError, LLMError) as e:
            self.stats["failures"] += 1
            return None, raw, f"{type(e).__name__}: {str(e)[:300]}"

    async def _complete(self, messages: list[dict], mock_ctx: dict | None) -> str:
        if self.mock:
            self.stats["calls"] += 1
            return self.mock.complete(messages, mock_ctx or {})

        payload: dict[str, Any] = {"model": self.cfg.llm_model, "messages": messages}
        if self.cfg.llm_temperature is not None:
            payload["temperature"] = self.cfg.llm_temperature
        if self.cfg.llm_max_tokens:
            payload["max_tokens"] = self.cfg.llm_max_tokens
        if self.cfg.llm_json_mode:
            payload["response_format"] = {"type": "json_object"}

        key = _sha([payload, self.cache_salt])
        if self.cfg.llm_cache and self.conn is not None:
            row = self.conn.execute("SELECT value FROM llm_cache WHERE key = ?", (key,)).fetchone()
            if row:
                self.stats["cache_hits"] += 1
                return row["value"]

        data = await self._post(f"{self.cfg.llm_base_url}/chat/completions", payload, self.cfg.llm_api_key)
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response shape: {str(data)[:300]}") from e
        usage = data.get("usage") or {}
        self.stats["calls"] += 1
        self.stats["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        self.stats["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        if self.cfg.llm_cache and self.conn is not None:
            self.conn.execute("INSERT OR REPLACE INTO llm_cache (key, value) VALUES (?, ?)", (key, text))
        return text

    # --- embeddings -----------------------------------------------------------------------

    @property
    def uses_hash_embeddings(self) -> bool:
        return self.cfg.embed_provider != "openai"

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Unit-length embeddings, shape (len(texts), dim)."""
        if not texts:
            return np.zeros((0, HASH_DIM), dtype=np.float32)
        self.stats["embedded_texts"] += len(texts)
        if self.uses_hash_embeddings:
            return np.stack([hash_embed(t) for t in texts])

        keys = [_sha([self.cfg.embed_model, t]) for t in texts]
        found: dict[str, np.ndarray] = {}
        if self.conn is not None:
            unique = list(set(keys))
            for i in range(0, len(unique), 500):
                chunk = unique[i:i + 500]
                rows = self.conn.execute(
                    f"SELECT key, vector FROM embedding_cache WHERE key IN ({','.join('?' * len(chunk))})", chunk)
                for row in rows:
                    found[row["key"]] = np.frombuffer(row["vector"], dtype=np.float32)

        missing = list(dict.fromkeys(i for i, k in enumerate(keys) if k not in found))
        for start in range(0, len(missing), 96):
            batch = missing[start:start + 96]
            data = await self._post(f"{self.cfg.embed_base_url}/embeddings",
                                    {"model": self.cfg.embed_model, "input": [texts[i] for i in batch]},
                                    self.cfg.embed_api_key)
            for item in data.get("data", []):
                vec = np.asarray(item["embedding"], dtype=np.float32)
                vec /= max(float(np.linalg.norm(vec)), 1e-8)
                key = keys[batch[item["index"]]]
                found[key] = vec
                if self.conn is not None:
                    self.conn.execute("INSERT OR REPLACE INTO embedding_cache (key, vector) VALUES (?, ?)",
                                      (key, vec.tobytes()))
        if any(k not in found for k in keys):
            raise LLMError("embedding response was missing vectors")
        return np.stack([found[k] for k in keys])

    # --- transport ------------------------------------------------------------------------

    async def _post(self, url: str, payload: dict, api_key: str) -> dict:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.cfg.llm_timeout)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        delay, error = 1.0, "unknown error"
        for _ in range(5):
            async with self._sem:
                try:
                    resp = await self._http.post(url, json=payload, headers=headers)
                except httpx.HTTPError as e:
                    error = f"{type(e).__name__}: {e}"
                else:
                    if resp.status_code < 400:
                        return resp.json()
                    error = f"HTTP {resp.status_code} from {url}: {resp.text[:300]}"
                    if resp.status_code < 500 and resp.status_code not in (408, 409, 429):
                        raise LLMError(error)
            await asyncio.sleep(delay)
            delay *= 2
        raise LLMError(error)

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
