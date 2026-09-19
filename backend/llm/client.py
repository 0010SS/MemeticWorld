"""LLM access layer with record/replay caching.

Every LLM call made by the simulator (including the ones issued from inside the
vendored Generative Agents prompt functions) goes through `LLMClient.complete`.

Determinism: each call is keyed by (scope, sha256(model|system|prompt), n) where
`scope` is a deterministic job identifier set by the simulator (e.g.
"d1t014:conv:maya|ethan") and `n` is the occurrence count of that prompt inside
the scope. Recording stores every response; replay reads them back, so a run
can be re-executed bit-for-bit without network access.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

_local = threading.local()


def current_scope() -> str:
    return getattr(_local, "scope", None) or "global"


def current_purpose() -> str:
    return getattr(_local, "purpose", None) or "unspecified"


def current_agent() -> Optional[str]:
    return getattr(_local, "agent", None)


@contextlib.contextmanager
def llm_scope(scope: str):
    prev = getattr(_local, "scope", None)
    _local.scope = scope
    try:
        yield
    finally:
        _local.scope = prev


@contextlib.contextmanager
def llm_purpose(purpose: str, agent: Optional[str] = None):
    prev_p, prev_a = getattr(_local, "purpose", None), getattr(_local, "agent", None)
    _local.purpose = purpose
    if agent is not None:
        _local.agent = agent
    try:
        yield
    finally:
        _local.purpose = prev_p
        _local.agent = prev_a


class ReplayMiss(RuntimeError):
    pass


class Backend:
    name = "base"

    def generate(self, prompt: str, system: Optional[str], max_tokens: int,
                 temperature: float) -> str:
        raise NotImplementedError


class ClaudeCLIBackend(Backend):
    """Uses the locally authenticated `claude` CLI in print mode.

    The CLI does not expose temperature or stop sequences; stop sequences are
    applied post hoc by LLMClient.
    """
    name = "claude_cli"

    def __init__(self, model: str = "haiku", timeout: int = 45):
        self.model = model
        self.timeout = timeout

    def generate(self, prompt, system, max_tokens, temperature):
        cmd = ["claude", "-p", "--model", self.model, "--output-format", "json",
               "--tools", "", "--no-session-persistence", "--setting-sources", "",
               "--disable-slash-commands", "--strict-mcp-config",
               # extended thinking is on by default in the CLI; it multiplies latency ~3-10x
               "--settings", '{"alwaysThinkingEnabled": false}',
               "--system-prompt", system or "You are a helpful assistant."]
        last_err = None
        for attempt in range(4):
            try:
                p = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                                   timeout=self.timeout, cwd="/tmp")
                data = json.loads(p.stdout)
                if data.get("is_error"):
                    raise RuntimeError(str(data.get("result"))[:300])
                return data["result"]
            except Exception as e:  # noqa: BLE001 - retried below
                last_err = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"claude CLI failed: {last_err}")


class AnthropicBackend(Backend):
    name = "anthropic"

    def __init__(self, model: str = "claude-haiku-4-5"):
        import anthropic  # noqa: WPS433
        self.client = anthropic.Anthropic()
        self.model = model

    def generate(self, prompt, system, max_tokens, temperature):
        kw = dict(model=self.model, max_tokens=max_tokens, temperature=temperature,
                  messages=[{"role": "user", "content": prompt}])
        if system:
            kw["system"] = system
        msg = self.client.messages.create(**kw)
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class LLMClient:
    def __init__(self, backend: Backend, cache_path: Path, mode: str = "record",
                 replay_path: Optional[Path] = None, fallback: Optional[Backend] = None,
                 on_call: Optional[Callable[[dict], None]] = None):
        """mode: 'record' (call backend, store), 'replay' (read only from replay_path)."""
        self.backend = backend
        self.mode = mode
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, str], int] = {}
        self._replay: dict[str, str] = {}
        self.on_call = on_call
        self.fallback = fallback
        self.stats = {"calls": 0, "cached": 0, "errors": 0, "seconds": 0.0}
        if replay_path:
            with open(replay_path) as f:
                for line in f:
                    rec = json.loads(line)
                    self._replay[rec["key"]] = rec["response"]
        self._fh = open(self.cache_path, "a")

    def close(self):
        with self._lock:
            self._fh.close()

    @staticmethod
    def _hash(model: str, system: Optional[str], prompt: str) -> str:
        h = hashlib.sha256()
        h.update((model or "").encode())
        h.update(b"\x00")
        h.update((system or "").encode())
        h.update(b"\x00")
        h.update(prompt.encode())
        return h.hexdigest()[:24]

    def complete(self, prompt: str, *, system: Optional[str] = None, max_tokens: int = 400,
                 temperature: float = 0.7, stop: Optional[list[str]] = None,
                 purpose: Optional[str] = None) -> str:
        purpose = purpose or current_purpose()
        scope = current_scope()
        model = getattr(self.backend, "model", self.backend.name)
        ph = self._hash(model, system, prompt)
        with self._lock:
            n = self._counters.get((scope, ph), 0)
            self._counters[(scope, ph)] = n + 1
        key = f"{scope}|{ph}|{n}"
        t0 = time.time()
        cached = False
        if key in self._replay:
            text = self._replay[key]
            cached = True
        elif self.mode == "replay":
            if self.fallback is None:
                raise ReplayMiss(key)
            text = self.fallback.generate(prompt, system, max_tokens, temperature)
        else:
            try:
                text = self.backend.generate(prompt, system, max_tokens, temperature)
            except Exception as e:  # noqa: BLE001
                with self._lock:
                    self.stats["errors"] += 1
                text = f"LLM_ERROR: {e}"
        if stop:
            for s in stop:
                if s and s in text:
                    text = text.split(s)[0]
        dt = time.time() - t0
        rec = {"key": key, "scope": scope, "purpose": purpose, "agent": current_agent(),
               "model": model, "system": system, "prompt": prompt, "response": text,
               "cached": cached, "seconds": round(dt, 3)}
        with self._lock:
            self.stats["calls"] += 1
            self.stats["cached"] += int(cached)
            self.stats["seconds"] += dt
            self._fh.write(json.dumps(rec) + "\n")
            self._fh.flush()
        if self.on_call:
            self.on_call(rec)
        return text


def make_backend(cfg: dict) -> Backend:
    kind = cfg.get("backend", "mock")
    if kind == "claude_cli":
        return ClaudeCLIBackend(cfg.get("model", "haiku"))
    if kind == "anthropic":
        return AnthropicBackend(cfg.get("model", "claude-haiku-4-5"))
    if kind in ("mock", "replay"):
        from backend.llm.mock import MockBackend
        return MockBackend(seed=cfg.get("mock_seed", 0))
    if kind == "auto":
        if os.environ.get("ANTHROPIC_API_KEY"):
            return AnthropicBackend(cfg.get("model", "claude-haiku-4-5"))
        return ClaudeCLIBackend(cfg.get("model", "haiku"))
    raise ValueError(f"unknown llm backend {kind}")
