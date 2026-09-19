"""LLM access layer with record/replay caching.

Every LLM call made by the simulator (including the ones issued from inside the
vendored Generative Agents prompt functions) goes through `LLMClient.complete`.

Determinism: each call is keyed by (scope, sha256(model|system|prompt), n) where
`scope` is a deterministic job identifier set by the simulator (e.g.
"d1t014:conv:maya|ethan") and `n` is the occurrence count of that prompt inside
the scope. Recording stores every response; replay reads them back, so a run
can be re-executed bit-for-bit without network access.

v3 (docs/ONTOLOGY_V3.md §6.1): prefix mode (`replay_until_tick`: serve the parent's cache strictly before
tick T, raise PrefixDivergence on a pre-T miss, never serve after T), fail-fast (backend exceptions are
never recorded as answers; LLMUnavailable after retries and pauses), `observer_client` for probe/coder
caches, `pause_manifest` / `resume_llm_overlay` for pause and resume. Simulation scopes must be "seed" or
start with "t{tick:04d}:" so prefix mode can place them in time.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import subprocess
import tempfile
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


class ProviderFailure(RuntimeError):
    """A provider failure must not be interpreted as an in-world statement."""


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
                                   encoding="utf-8", timeout=self.timeout, cwd=tempfile.gettempdir())
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


class PrefixDivergence(RuntimeError):
    """Prefix mode (v3 §6.1): a call whose scope lies strictly before `replay_until_tick` (or is "seed")
    has no cached response in the parent's llm_calls.jsonl, so the branch's prefix is not the parent's."""


class LLMUnavailable(RuntimeError):
    """Fail-fast (v3 §6.1): the backend kept failing through every retry and pause. The run must stop and
    mark its manifest `status: paused` (see `pause_manifest`); resume by prefix replay (`resume_llm_overlay`)."""


# v3 §8.2 contract defaults for llm.fail_fast
FAIL_FAST_DEFAULTS = {"max_consecutive_errors": 3, "pause_seconds": 600, "max_pauses": 6}

_TICK_RES = (re.compile(r"^t(\d+):"), re.compile(r"^d\d+t(\d+)"))


def scope_tick(scope: Optional[str]) -> Optional[int]:
    """Tick a simulation scope belongs to: "seed" -> -1 (before tick 0); "t0014:..." -> 14 (also the legacy
    "d1t014..." form); anything else (probe/analysis/"global") -> None."""
    if scope == "seed":
        return -1
    for rx in _TICK_RES:
        m = rx.match(scope or "")
        if m:
            return int(m.group(1))
    return None


def _key_scope(rec: dict) -> str:
    """Scope of a cached record (scopes may contain '|', so prefer the stored field over the key)."""
    if "scope" in rec and rec["scope"] is not None:
        return rec["scope"]
    return rec["key"].rsplit("|", 2)[0]


def _read_jsonl(path: Path):
    """Records of a (possibly partial) llm_calls.jsonl; a torn last line from a crash is skipped."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


class LLMClient:
    def __init__(self, backend: Backend, cache_path: Path, mode: str = "record",
                 replay_path: Optional[Path] = None, fallback: Optional[Backend] = None,
                 on_call: Optional[Callable[[dict], None]] = None,
                 replay_until_tick: Optional[int] = None, fail_fast: Optional[dict] = None,
                  record_cached: bool = True, sleep: Optional[Callable[[float], None]] = None,
                  raise_on_error: bool = False):
        """mode: 'record' (call backend, store), 'replay' (read only from replay_path).

        v3 additions (all inert when absent, so v2 behaviour is unchanged):
        - `replay_until_tick=T` with `replay_path` -> 'prefix' mode: cached responses are served only for
          scopes "seed" or tick < T; a miss there raises PrefixDivergence; scopes with tick >= T (or no
          tick) are always live and recorded, never served from the cache.
        - `fail_fast={max_consecutive_errors, pause_seconds, max_pauses}`: a backend exception is never
          recorded as a response; failures are retried, then paused, then LLMUnavailable is raised.
          None keeps the legacy behaviour (an "LLM_ERROR: ..." text is returned and recorded).
        - `record_cached=False`: responses served from the cache are not re-appended (observer caches
          that reuse their own file).
        - If `replay_path` is the cache file itself (resume), the partial file is moved aside first
          (to `<stem>.resume<k>.jsonl`), unless `record_cached=False` (then new calls are appended in place).
        """
        self.backend = backend
        self.raise_on_error = raise_on_error
        self.replay_until_tick = None if replay_until_tick is None else int(replay_until_tick)
        if self.replay_until_tick is not None and not replay_path:
            raise ValueError("replay_until_tick needs replay_path (the parent's llm_calls.jsonl)")
        self.mode = "prefix" if self.replay_until_tick is not None else mode
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, str], int] = {}
        self._replay: dict[str, str] = {}
        self.on_call = on_call
        self.fallback = fallback
        self.record_cached = record_cached
        self.fail_fast = None if fail_fast is None else {**FAIL_FAST_DEFAULTS, **(fail_fast or {})}
        self._halt = threading.Event()        # set once LLMUnavailable is raised: wakes paused threads
        self._sleep = sleep or self._halt.wait
        self._unavailable: Optional[str] = None
        self.error_log: list[dict] = []
        self.stats = {"calls": 0, "cached": 0, "errors": 0, "seconds": 0.0}
        if self.mode == "prefix":
            self.stats.update({"prefix_served": 0, "live": 0, "pauses": 0})
        elif self.fail_fast is not None:
            self.stats["pauses"] = 0
        self.replay_source: Optional[Path] = None
        if replay_path:
            replay_path = Path(replay_path)
            if record_cached and replay_path.exists() and self.cache_path.exists() and \
                    replay_path.resolve() == self.cache_path.resolve():
                k = 0
                while (aside := self.cache_path.with_name(f"{self.cache_path.stem}.resume{k}.jsonl")).exists():
                    k += 1
                os.replace(self.cache_path, aside)
                replay_path = aside
            self.replay_source = replay_path
            for rec in _read_jsonl(replay_path):
                if rec.get("error"):
                    continue
                if self.mode == "prefix":
                    t = scope_tick(_key_scope(rec))
                    if t is None or t >= self.replay_until_tick:
                        continue           # never served after T
                    if str(rec.get("response", "")).startswith("LLM_ERROR"):
                        continue           # a legacy error text is not an answer: a pre-T miss, not a replay
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

    def _error(self, key: str, scope: str, purpose: str, e: Exception, attempt: int):
        rec = {"key": key, "scope": scope, "purpose": purpose, "error": str(e)[:500], "attempt": attempt,
               "wall": round(time.time(), 3)}
        with self._lock:
            self.stats["errors"] += 1
            self.error_log.append(rec)
            del self.error_log[:-50]
            try:
                with open(self.cache_path.with_name("llm_errors.jsonl"), "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            except OSError:
                pass

    def _live(self, key: str, scope: str, purpose: str, prompt: str, system: Optional[str],
              max_tokens: int, temperature: float) -> str:
        """One live call. Fail-fast: retry up to max_consecutive_errors, then pause; after max_pauses
        pauses raise LLMUnavailable. The exception text is never returned as a response."""
        if self.raise_on_error:
            try:
                return self.backend.generate(prompt, system, max_tokens, temperature)
            except Exception as exc:
                self._error(key, scope, purpose, exc, 1)
                # Strict legacy runtimes require a diagnostic call record, never error text in memory.
                record = {"key": key, "scope": scope, "purpose": purpose, "agent": current_agent(),
                          "model": getattr(self.backend, "model", self.backend.name),
                          "system": system, "prompt": prompt, "response": "", "error": str(exc), "cached": False}
                with self._lock:
                    self.stats["calls"] += 1
                    self._fh.write(json.dumps(record) + "\n")
                    self._fh.flush()
                raise ProviderFailure(str(exc)) from exc
        if self.fail_fast is None:
            try:
                return self.backend.generate(prompt, system, max_tokens, temperature)
            except Exception as e:  # noqa: BLE001 - legacy v2 behaviour
                with self._lock:
                    self.stats["errors"] += 1
                if self.raise_on_error:
                    raise ProviderFailure(str(e)) from e
                return f"LLM_ERROR: {e}"
        ff = self.fail_fast
        failures, pauses, attempt = 0, 0, 0
        while True:
            if self._unavailable:
                raise LLMUnavailable(self._unavailable)
            attempt += 1
            try:
                return self.backend.generate(prompt, system, max_tokens, temperature)
            except Exception as e:  # noqa: BLE001 - retried / paused below
                self._error(key, scope, purpose, e, attempt)
                failures += 1
                if failures < int(ff["max_consecutive_errors"]):
                    continue
                if pauses >= int(ff["max_pauses"]):
                    msg = (f"LLM backend unavailable after {pauses} pauses x {ff['pause_seconds']}s "
                           f"(scope {scope}): {e}")
                    with self._lock:
                        self._unavailable = self._unavailable or msg
                    self._halt.set()
                    raise LLMUnavailable(msg) from e
                pauses += 1
                with self._lock:
                    self.stats["pauses"] = self.stats.get("pauses", 0) + 1
                self._sleep(float(ff["pause_seconds"]))
                failures = 0

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
        if self.mode == "prefix":
            t = scope_tick(scope)
            if t is not None and t < self.replay_until_tick:
                if key not in self._replay:
                    raise PrefixDivergence(f"no cached response for pre-T call {key} (T={self.replay_until_tick}, "
                                           f"purpose={purpose}) in {self.replay_source}")
                text = self._replay[key]
                cached = True
                with self._lock:
                    self.stats["prefix_served"] += 1
            else:
                text = self._live(key, scope, purpose, prompt, system, max_tokens, temperature)
                with self._lock:
                    self.stats["live"] += 1
        elif key in self._replay:
            text = self._replay[key]
            cached = True
        elif self.mode == "replay":
            if self.fallback is None:
                with self._lock:
                    self.stats["errors"] += 1
                raise ReplayMiss(key)
            text = self.fallback.generate(prompt, system, max_tokens, temperature)
        else:
            text = self._live(key, scope, purpose, prompt, system, max_tokens, temperature)
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
            if not (cached and not self.record_cached):
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


# ---------------------------------------------------------------------------------------------- v3 helpers
def client_from_config(llm_cfg: dict, cache_path: Path, replay_from: Optional[Path] = None,
                       backend: Optional[Backend] = None, on_call=None) -> LLMClient:
    """The simulation's client from the `llm` config block (v3 §6.1), replacing the engine's inline setup:
    - `replay_until_tick` set (with `replay_from`) -> prefix mode;
    - `replay_from` alone -> v2 replay (mock/live fallback exactly as before);
    - `fail_fast` -> contract defaults when the key is absent; `fail_fast: null|false` keeps legacy errors."""
    llm_cfg = llm_cfg or {}
    backend = backend or make_backend(llm_cfg)
    replay_from = replay_from or llm_cfg.get("replay_from")
    T = llm_cfg.get("replay_until_tick")
    ff = llm_cfg.get("fail_fast", FAIL_FAST_DEFAULTS)
    ff = None if ff in (None, False) else ({} if ff is True else dict(ff))
    if T is not None:
        if not replay_from:
            raise ValueError("llm.replay_until_tick needs llm.replay_from")
        return LLMClient(backend, cache_path, replay_path=Path(replay_from), replay_until_tick=int(T),
                         fail_fast=ff, on_call=on_call)
    mode = "replay" if replay_from else "record"
    return LLMClient(backend, cache_path, mode=mode, replay_path=Path(replay_from) if replay_from else None,
                     fallback=None if mode == "replay" and llm_cfg.get("backend") != "mock" else backend,
                     fail_fast=ff, on_call=on_call)


OBSERVER_KINDS = ("probe", "coder", "coder2")


def observer_client(kind: str, path: Path, cfg: Optional[dict] = None,
                    backend: Optional[Backend] = None) -> LLMClient:
    """A client for observer calls (probes, coding) with its OWN cache file, never a simulation cache.

    kind: "probe" (cfg.probe.{backend,model}), "coder" (cfg.analysis.coder.{backend,model}) or "coder2"
    (the second coder: cfg.analysis.coder.second). `path` is the observer cache, e.g.
    <run>/probes/C4/llm_calls.jsonl or <run>/analysis/coding_llm_calls.jsonl. If it exists it is reused as a
    cache (hits are served and not re-appended; misses go live), so re-running a probe is idempotent.
    Refuses a run's simulation cache (<run>/llm_calls.jsonl next to manifest.json)."""
    if kind not in OBSERVER_KINDS:
        raise ValueError(f"observer kind must be one of {OBSERVER_KINDS}, got {kind!r}")
    path = Path(path)
    if path.name == "llm_calls.jsonl" and (path.parent / "manifest.json").exists():
        raise ValueError(f"{path} is a simulation cache; observer calls need their own file")
    cfg = cfg or {}
    if kind == "probe":
        sub = dict(cfg.get("probe") or {"backend": "claude_cli", "model": "haiku"})
    else:
        coder = (cfg.get("analysis") or {}).get("coder") or {"backend": "claude_cli", "model": "sonnet",
                                                               "second": "haiku"}
        sub = {"backend": coder.get("backend", "claude_cli"),
               "model": coder.get("second", "haiku") if kind == "coder2" else coder.get("model", "sonnet")}
    sub.setdefault("mock_seed", (cfg.get("llm") or {}).get("mock_seed", 0))
    backend = backend or make_backend(sub)
    ff = (cfg.get("llm") or {}).get("fail_fast", FAIL_FAST_DEFAULTS)
    ff = None if ff in (None, False) else ({} if ff is True else dict(ff))
    path.parent.mkdir(parents=True, exist_ok=True)
    return LLMClient(backend, path, mode="record", replay_path=path if path.exists() else None,
                     fail_fast=ff, record_cached=False)


def pause_manifest(run_dir: Path, last_complete_tick: Optional[int], error: str = "") -> dict:
    """Mark a run paused after LLMUnavailable (v3 §6.1/§6.5): manifest `status: paused`,
    `last_complete_tick` and `llm.errors`. Returns the updated manifest."""
    mp = Path(run_dir) / "manifest.json"
    man = json.load(open(mp)) if mp.exists() else {}
    man["status"] = "paused"
    man["last_complete_tick"] = last_complete_tick
    llm = man.get("llm") if isinstance(man.get("llm"), dict) else {}
    errs = [json.loads(l) for l in open(Path(run_dir) / "llm_errors.jsonl")] \
        if (Path(run_dir) / "llm_errors.jsonl").exists() else []
    llm.update({"errors": len(errs), "paused_reason": str(error)[:500]})
    man["llm"] = llm
    json.dump(man, open(mp, "w"), indent=1, default=str)
    return man


def resume_llm_overlay(run_dir: Path) -> dict:
    """The `llm` overlay that resumes a paused/crashed run in place by prefix replay (v3 §6.1):
    replay_from = its own partial llm_calls.jsonl, replay_until_tick = last_complete_tick + 1
    (0 when no tick completed: only the "seed" scope is replayed)."""
    man = json.load(open(Path(run_dir) / "manifest.json"))
    last = man.get("last_complete_tick")
    return {"replay_from": str(Path(run_dir) / "llm_calls.jsonl"),
            "replay_until_tick": 0 if last is None else int(last) + 1}
