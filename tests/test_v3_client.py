"""v3 step 1 (docs/ONTOLOGY_V3.md §6.1, §8.1): LLM client prefix mode, fail-fast, observer client, resume;
per-tick trace digests."""
import json
from pathlib import Path

import pytest

from backend.llm.client import (Backend, LLMClient, LLMUnavailable, PrefixDivergence, client_from_config,
                                llm_purpose, llm_scope, observer_client, pause_manifest, resume_llm_overlay,
                                scope_tick)
from backend.tracing.logger import TraceLogger, compare_digests, prefix_digest, read_digests


class Echo(Backend):
    name = "echo"
    model = "echo"

    def __init__(self, tag="live"):
        self.tag, self.calls = tag, 0

    def generate(self, prompt, system, max_tokens, temperature):
        self.calls += 1
        return f"{self.tag}:{prompt}"


class Flaky(Backend):
    """Fails the first `fail` calls, then answers."""
    name = "flaky"
    model = "echo"

    def __init__(self, fail):
        self.fail, self.calls = fail, 0

    def generate(self, prompt, system, max_tokens, temperature):
        self.calls += 1
        if self.calls <= self.fail:
            raise RuntimeError("rate limited")
        return f"ok:{prompt}"


def _lines(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def _parent(tmp_path, ticks=(0, 1, 2, 3)):
    """A parent cache: one call in 'seed' and one per tick."""
    c = LLMClient(Echo("parent"), tmp_path / "parent" / "llm_calls.jsonl")
    with llm_scope("seed"):
        c.complete("s")
    for t in ticks:
        with llm_scope(f"t{t:04d}:02agent:maya"):
            c.complete(f"p{t}")
    c.close()
    return tmp_path / "parent" / "llm_calls.jsonl"


def test_scope_tick():
    assert scope_tick("seed") == -1
    assert scope_tick("t0014:02agent:maya") == 14
    assert scope_tick("d1t014:conv:maya|ethan") == 14
    assert scope_tick("probe:c1:maya") is None and scope_tick("global") is None


def test_prefix_serves_before_T_and_never_after(tmp_path):
    parent = _parent(tmp_path)
    live = Echo("child")
    c = LLMClient(live, tmp_path / "child" / "llm_calls.jsonl", replay_path=parent, replay_until_tick=2)
    assert c.mode == "prefix"
    with llm_scope("seed"):
        assert c.complete("s") == "parent:s"
    for t in (0, 1):
        with llm_scope(f"t{t:04d}:02agent:maya"):
            assert c.complete(f"p{t}") == f"parent:p{t}"
    assert live.calls == 0
    # post-T: the parent has the exact key cached, but it is never served
    for t in (2, 3):
        with llm_scope(f"t{t:04d}:02agent:maya"):
            assert c.complete(f"p{t}") == f"child:p{t}"
    assert live.calls == 2
    assert c.stats["prefix_served"] == 3 and c.stats["live"] == 2
    c.close()
    recs = _lines(tmp_path / "child" / "llm_calls.jsonl")
    assert [r["cached"] for r in recs] == [True, True, True, False, False]   # the full prefix is recorded


def test_prefix_pre_T_miss_raises(tmp_path):
    parent = _parent(tmp_path)
    c = LLMClient(Echo("child"), tmp_path / "child" / "llm_calls.jsonl", replay_path=parent, replay_until_tick=3)
    with llm_scope("t0001:02agent:maya"), pytest.raises(PrefixDivergence):
        c.complete("a different prompt")
    with llm_scope("t0001:02agent:maya"):
        assert c.complete("p1") == "parent:p1"
        with pytest.raises(PrefixDivergence):       # second occurrence of the same prompt: not in the parent
            c.complete("p1")
    with llm_scope("seed"), pytest.raises(PrefixDivergence):
        c.complete("unseen seed prompt")


def test_prefix_does_not_serve_legacy_error_texts(tmp_path):
    p = tmp_path / "parent.jsonl"
    p.write_text(json.dumps({"key": "t0000:x|h|0", "scope": "t0000:x", "response": "LLM_ERROR: boom"}) + "\n")
    c = LLMClient(Echo(), tmp_path / "c.jsonl", replay_path=p, replay_until_tick=5)
    assert c._replay == {}


def test_prefix_requires_replay_path(tmp_path):
    with pytest.raises(ValueError):
        LLMClient(Echo(), tmp_path / "c.jsonl", replay_until_tick=3)


def test_errors_are_never_recorded_and_fail_fast(tmp_path):
    sleeps = []
    b = Flaky(fail=10 ** 9)
    c = LLMClient(b, tmp_path / "run" / "llm_calls.jsonl",
                  fail_fast={"max_consecutive_errors": 3, "pause_seconds": 600, "max_pauses": 2},
                  sleep=sleeps.append)
    with llm_scope("t0000:02agent:maya"), llm_purpose("react_decision"), pytest.raises(LLMUnavailable):
        c.complete("hello")
    assert b.calls == 3 * 3 and sleeps == [600.0, 600.0]
    assert c.stats["errors"] == 9 and c.stats["calls"] == 0
    c.close()
    assert _lines(tmp_path / "run" / "llm_calls.jsonl") == []          # nothing recorded as an answer
    errs = _lines(tmp_path / "run" / "llm_errors.jsonl")
    assert len(errs) == 9 and all("rate limited" in e["error"] for e in errs)
    # once unavailable, further calls fail immediately
    with llm_scope("t0000:02agent:ethan"), pytest.raises(LLMUnavailable):
        c.complete("x")
    assert b.calls == 9


def test_fail_fast_recovers_after_pause(tmp_path):
    sleeps = []
    c = LLMClient(Flaky(fail=4), tmp_path / "llm_calls.jsonl", fail_fast={"pause_seconds": 1}, sleep=sleeps.append)
    with llm_scope("t0000:x"):
        assert c.complete("hi") == "ok:hi"
    c.close()
    assert sleeps == [1.0]
    recs = _lines(tmp_path / "llm_calls.jsonl")
    assert len(recs) == 1 and recs[0]["response"] == "ok:hi"
    assert not any(r["response"].startswith("LLM_ERROR") for r in recs)


def test_legacy_error_behaviour_without_fail_fast(tmp_path):
    c = LLMClient(Flaky(fail=1), tmp_path / "llm_calls.jsonl")
    assert c.complete("hi").startswith("LLM_ERROR")
    assert c.stats["errors"] == 1


def test_client_from_config(tmp_path):
    parent = _parent(tmp_path)
    c = client_from_config({"backend": "mock"}, tmp_path / "a.jsonl")
    assert c.mode == "record" and c.fail_fast["max_consecutive_errors"] == 3
    c = client_from_config({"backend": "mock", "fail_fast": None}, tmp_path / "b.jsonl")
    assert c.fail_fast is None
    c = client_from_config({"backend": "mock", "replay_from": str(parent)}, tmp_path / "c.jsonl")
    assert c.mode == "replay" and c.fallback is not None
    c = client_from_config({"backend": "mock", "replay_from": str(parent), "replay_until_tick": 2},
                           tmp_path / "d.jsonl")
    assert c.mode == "prefix" and c.replay_until_tick == 2
    with pytest.raises(ValueError):
        client_from_config({"backend": "mock", "replay_until_tick": 2}, tmp_path / "e.jsonl")


def test_resume_in_place(tmp_path):
    """A paused run resumes with replay_from = its own partial cache and T = last_complete_tick + 1."""
    run = tmp_path / "run"
    run.mkdir()
    cache = run / "llm_calls.jsonl"
    c = LLMClient(Echo("first"), cache)
    for t in (0, 1, 2):                       # tick 2 is only partly done when the backend goes away
        with llm_scope(f"t{t:04d}:02agent:maya"):
            c.complete(f"p{t}")
    c.close()
    with open(cache, "a") as fh:
        fh.write('{"key": "torn')              # a torn last line from the crash
    json.dump({"status": "running"}, open(run / "manifest.json", "w"))
    man = pause_manifest(run, last_complete_tick=1, error="LLMUnavailable")
    assert man["status"] == "paused" and man["last_complete_tick"] == 1 and "errors" in man["llm"]
    ov = resume_llm_overlay(run)
    assert ov == {"replay_from": str(cache), "replay_until_tick": 2}
    live = Echo("second")
    c = client_from_config({"backend": "mock", **ov}, cache, backend=live)
    for t in (0, 1, 2, 3):
        with llm_scope(f"t{t:04d}:02agent:maya"):
            out = c.complete(f"p{t}")
            assert out == (f"first:p{t}" if t < 2 else f"second:p{t}")
    c.close()
    assert live.calls == 2
    assert (run / "llm_calls.resume0.jsonl").exists()
    assert [r["response"] for r in _lines(cache)] == ["first:p0", "first:p1", "second:p2", "second:p3"]


def test_observer_client_separate_idempotent_cache(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text("{}")
    (run / "llm_calls.jsonl").write_text("")
    with pytest.raises(ValueError):
        observer_client("probe", run / "llm_calls.jsonl")
    with pytest.raises(ValueError):
        observer_client("judge", run / "probes" / "C4" / "llm_calls.jsonl")
    path = run / "probes" / "C4" / "llm_calls.jsonl"
    cfg = {"probe": {"backend": "mock", "model": "haiku"},
           "analysis": {"coder": {"backend": "mock", "model": "sonnet", "second": "haiku"}}}
    b = Echo("probe")
    c = observer_client("probe", path, cfg, backend=b)
    with llm_scope("probe:C4:maya:q1"):
        a1 = c.complete("q1")
    c.close()
    c = observer_client("probe", path, cfg, backend=b)
    with llm_scope("probe:C4:maya:q1"):
        assert c.complete("q1") == a1
    c.close()
    assert b.calls == 1 and len(_lines(path)) == 1              # served from its own cache, not re-appended
    assert (run / "llm_calls.jsonl").read_text() == ""           # the simulation cache is untouched
    coder = observer_client("coder", run / "analysis" / "coding_llm_calls.jsonl", cfg)
    assert coder.backend.name == "mock"


# -------------------------------------------------------------------------------------------- digests
def _trace(run_dir, ticks, change_at=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    tr = TraceLogger(run_dir)
    with llm_scope("seed"):
        tr.log("memory_encoded", text="seed")
    tr.flush()
    for t in range(ticks):
        tr.tick, tr.time = t, f"2024-01-01T08:{t:02d}"
        for aid in ("maya", "ethan"):
            with llm_scope(f"t{t:04d}:02agent:{aid}"):
                tr.log("decision", agent=aid, x=("changed" if t == change_at else t))
        tr.flush()
    tr.close()
    return run_dir


def test_per_tick_digests(tmp_path):
    a = _trace(tmp_path / "a", 5)
    b = _trace(tmp_path / "b", 3)                  # a shorter twin: the same first 3 ticks
    c = _trace(tmp_path / "c", 5, change_at=3)
    da = read_digests(a)
    assert sorted(da) == [0, 1, 2, 3, 4] and len(set(da.values())) == 5
    assert compare_digests(a, b, 3) == [] and prefix_digest(a, 3) == prefix_digest(b, 3)
    assert compare_digests(a, c, 3) == [] and compare_digests(a, c, 5) == [3]
    assert prefix_digest(a, 4) != prefix_digest(c, 4)
    n = [json.loads(l)["n"] for l in open(a / "digests.jsonl")]
    assert n == [3, 2, 2, 2, 2]                    # tick 0 also holds the seed record


def test_paused_close_leaves_unfinished_tick_without_digest(tmp_path):
    tr = TraceLogger(tmp_path)
    for t in range(3):
        tr.tick = t
        tr.log("x", v=t)
        tr.flush()
    tr.close(end_tick=False)
    assert sorted(read_digests(tmp_path)) == [0, 1]


# ------------------------------------------------------------------------------ engine (mock) prefix run
def test_mock_sim_prefix_branch_has_identical_prefix(tmp_path, monkeypatch):
    """A mock parent run, then a child replaying ticks < T from the parent's cache and running live after:
    digests match, pre-T calls are all served from the cache, post-T calls are all live."""
    from conftest import run_sim
    from backend.simulation import engine
    parent = run_sim(tmp_path, name="parent")
    assert (tmp_path / "parent" / "digests.jsonl").exists()
    T = parent.clock.total_ticks // 2
    made = []

    def factory(backend, cache_path, mode="record", replay_path=None, fallback=None, on_call=None):
        c = LLMClient(backend, cache_path, replay_path=replay_path, replay_until_tick=T, fail_fast={})
        made.append(c)
        return c

    replay = str(tmp_path / "parent" / "llm_calls.jsonl")
    if not hasattr(engine, "client_from_config"):    # until the integrator wires client_from_config
        monkeypatch.setattr(engine, "LLMClient", factory)
    child = run_sim(tmp_path, {"llm": {"replay_from": replay, "replay_until_tick": T}}, name="child")
    c = made[0] if made else child.llm
    assert c.mode == "prefix" and c.replay_until_tick == T
    assert c.stats["prefix_served"] > 0 and c.stats["live"] > 0
    ticks = parent.clock.total_ticks
    assert compare_digests(tmp_path / "parent", tmp_path / "child", ticks) == []   # mock is deterministic
    recs = _lines(tmp_path / "child" / "llm_calls.jsonl")
    for r in recs:
        t = scope_tick(r["scope"])
        assert r["cached"] == (t is not None and t < T)
