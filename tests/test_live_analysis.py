"""Live (LLM-free) meme analysis, the pluggable judge and the meme report (mock backend only).

* synthetic traces pin the status definitions: a same-conversation repeat is "echo", reuse after
  exposure in another conversation is "spreading" (1 carried adopter) / "emerged" (2+), world wording is
  flagged even when it spreads, the planted control is "planted";
* a v2 all-on mock day (plus the planted control): cumulative counts are monotone in t, the timeline
  agrees with the snapshots, and a copy whose trace is cut mid-line (a run still being written) works
  and catches up when the rest of the file arrives;
* a v3 co-op mock run with turnover and a regime change: hidden-derived metrics are marked and listed,
  and strip_hidden removes all of them;
* the judge registry, the verdict schema, judge_run's output location, the pipeline's routing of the
  classifier through the default judge, and the report with and without analysis.json.
"""
import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest
import yaml

from backend.analysis import judge as JG
from backend.analysis.live import clear_cache, live_snapshot, live_timeline, strip_hidden
from backend.analysis.report import demo_report, meme_report
from tests.conftest import ROOT, run_sim

PLANTED = {"agent": "maya", "habit": 'Maya has a habit of calling any mess "a full pickle".'}
# mirrors tests/test_integration_v2.py ALL_ON (every v2 mechanism on), plus the planted control
ALL_ON = {
    "run_name": "live_all_on", "seed": 7, "simulation_days": 1,
    "llm": {"backend": "mock", "max_workers": 6},
    "latent_events": {"event_rate": 0.35, "link_visibility": 0.3, "referents": {"enabled": True},
                      "assignment": {"mode": "balanced"}, "schedule": "balanced"},
    "conversation": {"catchup": {"enabled": True}, "group": {"enabled": True, "prob": 1.0}},
    "need": {"enabled": True, "min_importance": 3},
    "reminding": {"enabled": True},
    "memory": {"verbatim": {"enabled": True}},
    "priming": {"enabled": True},
    "controls": {"planted_phrase": PLANTED},
}
# configs/v3_base.yaml shortened to two days: a wave of turnover, a binder wipe and a regime change on day 2
V3_TURN = {
    "run_name": "live_v3", "seed": 11, "world_seed": 11, "simulation_days": 2, "day_end": "19:30",
    "llm": {"backend": "mock", "max_workers": 6},
    "records": {"transitions": [{"day": 2, "mode": "wipe"}]},
    "turnover": {"waves": [{"day": 2, "depart": {"am_crew": 1, "pm_crew": 1, "stores": 1}}]},
    "regimes": {"schedule": [{"day": 1, "regime": "A"}, {"day": 2, "regime": "B"}],
                "cues": [{"day": 1, "time": "16:45", "arena": "Stockroom", "text_key": "new_supplier"}]},
    "comm": {"meeting": {"enabled": True, "days": [1]}},
}
RUN_FILES = ("manifest.json", "config.resolved.yaml", "trace.jsonl", "events.jsonl", "world_script.jsonl")


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_cache()
    yield
    clear_cache()


def _copy_run(src: Path, dst: Path, files=RUN_FILES) -> Path:
    dst.mkdir(parents=True)
    for f in files:
        if (Path(src) / f).exists():
            shutil.copy(Path(src) / f, dst / f)
    return dst


def _digest(d: Path) -> dict:
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*")) if p.is_file()}


# ------------------------------------------------------------------------------------------------ synthetic run
def _time(tick):
    return f"2026-09-14T{7 + (30 + 15 * tick) // 60:02d}:{(30 + 15 * tick) % 60:02d}:00"


def _conv(cid, tick, turns):
    """turns: [(speaker, listeners, text)] -> conversation + utterance trace records."""
    names = {a: f"{a.title()} Test" for a in ("maya", "leo", "dev", "hana")}
    us = [{"type": "utterance", "tick": tick, "time": _time(tick), "id": f"{cid}.u{i}", "conversation_id": cid, "idx": i, "speaker": s,
           "listeners": ls, "text": t, "retrieved": []} for i, (s, ls, t) in enumerate(turns)]
    conv = {"type": "conversation", "tick": tick, "id": cid, "participants": sorted({s for s, *_ in turns} |
                                                                                  {l for _, ls, _ in turns for l in ls}),
            "trigger": None, "transcript": [[names[s], t] for s, _, t in turns]}
    return [conv, *us]


def _remark(tick, speaker, text, n):
    return {"type": "utterance", "tick": tick, "time": _time(tick), "id": f"t{tick:04d}:remark:{speaker}:x{n}", "conversation_id": None,
            "idx": 0, "speaker": speaker, "listeners": [], "text": text, "retrieved": []}


def _synthetic(d: Path, planted=True) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    agents = {a: {"id": a, "name": f"{a.title()} Test", "background": "", "habits": [], "routine": []}
              for a in ("maya", "leo", "dev", "hana")}
    man = {"run_id": d.name, "status": "finished", "ticks": 120, "ticks_per_day": 60, "tick_minutes": 15,
           "start": "2026-09-14T07:30:00", "agents": agents, "groups": {}, "config": {},
           "world": {"graph": {"Dorm": []}, "arenas": {"Dorm": ["Room"]}},
           "population_lexicon": {"tokens": ["dining", "hall", "lounge", "sofa"]}}
    cfg = {"seed": 3, "population": "configs/population/homewood8.yaml", "llm": {"backend": "mock", "model": "m"},
           "controls": {"planted_phrase": PLANTED if planted else None}}
    fact = {"id": "ev000.b0.f0", "text": "Maya dropped the purple kettle behind the lounge sofa.", "salience": 0.55,
            "kind": "action", "visibility": "all", "involves": ["maya"]}
    event = {"id": "ev000", "latent_type": "E1", "start_tick": 2, "referents": [],
             "beats": [{"idx": 0, "tick": 2, "location": "Dorm", "arena": "Room", "facts": [fact]}]}
    trace = [
        {"type": "event_beat", "tick": 2, "event_id": "ev000", "latent_type": "E1", "beat": 0, "location": "Dorm",
         "arena": "Room", "facts": [fact]},
        {"type": "observation", "tick": 2, "agent": "maya", "source_type": "perception", "event_id": "ev000",
         "facts": [fact], "originating_event_ids": ["ev000"]},
        *_conv("c1", 5, [("maya", ["leo"], "Honestly the quibbly wobble again."),
                         ("leo", ["maya"], "Yes the quibbly wobble!")]),                 # echo only
        *_conv("c2", 10, [("maya", ["leo"], "That snorkel dance was wild.")]),
        *_conv("c9", 12, [("maya", ["leo"], "I saw the purple kettle.")]),
        *_conv("c3", 20, [("leo", ["dev"], "Maya did the snorkel dance.")]),              # 1 carried adopter
        *_conv("c10", 22, [("leo", ["dev"], "Where is the purple kettle now?")]),        # world wording
        *_conv("c4", 30, [("maya", ["leo"], "Grab me a glimmer toast.")]),
        *_conv("c5", 35, [("leo", ["dev"], "I want a glimmer toast.")]),
        *_conv("c6", 40, [("dev", ["hana"], "One glimmer toast please.")]),              # 2 carried adopters
        *_conv("c7", 45, [("maya", ["leo"], "What a full pickle.")]),
        _remark(50, "maya", "Frabjous, simply frabjous.", 1),
        _remark(52, "maya", "So frabjous.", 2),                                        # new
        *_conv("c8", 55, [("leo", ["dev"], "Total full pickle here.")]),                # planted
    ]
    json.dump(man, open(d / "manifest.json", "w"))
    yaml.safe_dump(cfg, open(d / "config.resolved.yaml", "w"))
    (d / "events.jsonl").write_text(json.dumps(event) + "\n")
    (d / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in trace))
    return d


def _find(snap_or_cards, phrase):
    items = snap_or_cards["expressions"] if isinstance(snap_or_cards, dict) else snap_or_cards
    return next((e for e in items if e["phrase"] == phrase or phrase in e["variants"]), None)


def test_statuses_follow_the_exposure_definitions(tmp_path):
    d = _synthetic(tmp_path / "syn")
    s = live_snapshot(d, top=20)
    echo, spread, emerged = _find(s, "quibbly wobble"), _find(s, "snorkel dance"), _find(s, "glimmer toast")
    world, new, planted = _find(s, "purple kettle"), _find(s, "frabjous"), _find(s, "full pickle")
    assert echo["status"] == "echo" and echo["emergence"]["n_adopters"] == 1 and echo["emergence"]["n_adopters_carried"] == 0
    assert spread["status"] == "spreading" and spread["emergence"]["carried_adopters"] == ["leo"]
    assert emerged["status"] == "emerged" and emerged["emergence"]["carried_adopters"] == ["dev", "leo"]
    assert emerged["emergence"]["n_independent"] == 0 and emerged["adoption_curve"] == [[30, 1], [35, 2], [40, 3]]
    assert world["status"] == "world_wording" and world["flags"] == ["world_wording", "spreading"]
    assert world["world_wording"]["in_world_text"] and world["world_wording"]["world_tick"] == 2
    assert new["status"] == "new" and new["n_speakers"] == 1
    assert planted["status"] == "planted" and planted["flags"] == ["planted", "spreading"] and planted["planted"]
    assert s["planted"] == {"agent": "maya", "phrase": "a full pickle", "uses": 2, "status": "planted"}
    assert emerged["first_use"]["speaker"] == "maya" and emerged["first_use"]["utterance_id"] == "c4.u0"
    # transmission edges: speaker -> listener who later used it, with the exchange it was heard in
    edges = {(e["from"], e["to"], e["kind"]) for e in s["transmission"]["edges"] if e["expression"] == "glimmer toast"}
    assert edges == {("maya", "leo", "carried"), ("leo", "dev", "carried")}
    assert ("maya", "leo", "echo") in {(e["from"], e["to"], e["kind"]) for e in s["transmission"]["edges"]
                                         if e["expression"] == "quibbly wobble"}
    json.dumps(s)


def test_status_evolves_with_t(tmp_path):
    d = _synthetic(tmp_path / "syn")
    assert _find(live_snapshot(d, tick=34), "glimmer toast") is None          # one use: not yet recurring
    assert _find(live_snapshot(d, tick=36), "glimmer toast")["status"] == "spreading"
    assert _find(live_snapshot(d, tick=40), "glimmer toast")["status"] == "emerged"
    early = live_snapshot(d, tick=1)
    assert early["n_events"] == 0 and early["funnel"]["cumulative"]["events_witnessed"] == 0
    s = live_snapshot(d, tick=12)
    assert s["n_events"] == 1 and s["funnel"]["cumulative"]["events_witnessed"] == 1
    assert s["funnel"]["cumulative"]["events_discussed"] == 1                   # "purple kettle" at tick 12
    # the timeline re-evaluates the final pool per bucket
    tl = live_timeline(d, step=10)
    by = {r["tick"]: r for r in tl["series"]}
    assert by[30]["status_counts"]["emerged"] == 0 and by[40]["status_counts"]["emerged"] == 1
    assert [r["tick"] for r in tl["series"]] == list(range(0, 51, 10)) + [55]


def test_no_planted_control_means_no_planted_status(tmp_path):
    s = live_snapshot(_synthetic(tmp_path / "syn", planted=False))
    assert s["planted"] is None and s["status_counts"]["planted"] == 0
    assert _find(s, "full pickle")["status"] == "spreading"


# ------------------------------------------------------------------------------------------------ v2 mock day
@pytest.fixture(scope="module")
def v2_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("live_v2")
    return run_sim(tmp, ALL_ON, name="all_on").run_dir


def test_snapshot_counts_are_monotone_in_t(v2_run):
    latest = live_snapshot(v2_run)["latest_tick"]
    assert latest > 5
    prev = None
    keys = ("n_utterances", "n_conversations", "n_events")
    for t in range(0, latest + 1):
        s = live_snapshot(v2_run, tick=t)
        assert s["tick"] == t
        json.dumps(s)
        if prev:
            for k in keys:
                assert s[k] >= prev[k], (t, k)
            for k, v in s["funnel"]["cumulative"].items():
                if k != "reuse_after_exposure":          # pool-dependent (expressions re-extracted per t)
                    assert v >= prev["funnel"]["cumulative"][k], (t, k)
            for e in prev["expressions"]:   # the same group (same n-grams) never loses uses or speakers; grouping
                now = _find(s, e["phrase"])  # itself may change as t grows, so only identical groups are compared
                if now and now["phrase"] == e["phrase"] and set(now["variants"]) == set(e["variants"]):
                    assert now["uses"] >= e["uses"] and now["n_speakers"] >= e["n_speakers"]
                    assert now["first_use"]["tick"] <= e["first_use"]["tick"]
        prev = s
    final = live_snapshot(v2_run)
    tr = [json.loads(l) for l in open(v2_run / "trace.jsonl")]
    assert final["n_utterances"] == sum(r["type"] == "utterance" for r in tr)
    assert final["n_conversations"] == sum(r["type"] == "conversation" for r in tr)
    assert final["expressions"] and all(e["status"] in ("planted", "system_wording", "world_wording", "emerged",
                                                        "spreading", "echo", "new") for e in final["expressions"])
    assert all(e["tier"] in ("candidate", "spreading", "convention") for e in final["expressions"])
    assert final["planted"]["phrase"] == "a full pickle"       # control configured (the mock never says it)
    assert "v3" not in final and final["hidden_fields"]


def test_timeline_agrees_with_snapshots(v2_run):
    tl = live_timeline(v2_run, step=4)
    json.dumps(tl)
    for row in tl["series"]:
        s = live_snapshot(v2_run, tick=row["tick"])
        for k in ("n_utterances", "n_conversations", "n_events", "n_agents_active"):
            assert row[k] == s[k], (row["tick"], k)
        for k, v in s["funnel"]["cumulative"].items():
            if k != "reuse_after_exposure":
                assert row["funnel"][k] == v, (row["tick"], k)
    assert tl["series"][-1]["tick"] == tl["latest_tick"]


def test_trace_cut_mid_line_does_not_crash_and_catches_up(v2_run, tmp_path):
    d = _copy_run(v2_run, tmp_path / "running")
    full = (d / "trace.jsonl").read_bytes()
    cut = len(full) // 2
    while full[cut - 1:cut] == b"\n":
        cut += 1
    (d / "trace.jsonl").write_bytes(full[:cut])                    # a torn last line, as while the run writes
    lines = full[:cut].split(b"\n")[:-1]
    n_utt = sum(json.loads(l)["type"] == "utterance" for l in lines)
    s = live_snapshot(d)
    assert s["n_utterances"] == n_utt and s["latest_tick"] == max(json.loads(l)["tick"] for l in lines)
    live_timeline(d)
    meme_report(d)
    # a torn events.jsonl and a half-written manifest are survived too (the previous manifest is kept)
    (d / "manifest.json").write_text((d / "manifest.json").read_text()[:50])
    assert live_snapshot(d, tick=3)["tick"] == 3
    shutil.copy(v2_run / "manifest.json", d / "manifest.json")
    # the writer finishes the file: the incremental reader picks up exactly the rest
    with open(d / "trace.jsonl", "ab") as f:
        f.write(full[cut:])
    ref = live_snapshot(v2_run)
    now = live_snapshot(d)
    assert now["n_utterances"] == ref["n_utterances"] and now["latest_tick"] == ref["latest_tick"]
    assert [e["phrase"] for e in now["expressions"]] == [e["phrase"] for e in ref["expressions"]]
    # a rewritten (shorter, different) trace resets the reader
    (d / "trace.jsonl").write_bytes(b"".join(l + b"\n" for l in full.split(b"\n")[:5]))
    assert live_snapshot(d)["n_utterances"] <= 5


# ------------------------------------------------------------------------------------------------ v3 co-op run
@pytest.fixture(scope="module")
def v3_run(tmp_path_factory):
    from backend.config import load_config
    from backend.simulation.engine import Simulation
    cfg = load_config("configs/v3_base.yaml", V3_TURN)
    sim = Simulation(cfg, tmp_path_factory.mktemp("live_v3") / "coop", progress=False)
    sim.run()
    return sim.run_dir


def test_v3_block_and_demo_strip(v3_run):
    s = live_snapshot(v3_run)
    v3 = s["v3"]
    assert v3["jobs"]["started"] == 16 and v3["binder"]["reads"] > 0
    assert [(t["day"], t["mode"]) for t in v3["binder"]["transitions"]] == [(2, "wipe")]
    assert len(v3["roster"]["departed"]) == 3 and set(v3["roster"]["active_by_cohort"]) == {"founder", "W1"}
    hidden = {k for k, v in v3.items() if isinstance(v, dict) and v.get("hidden") is True}
    assert hidden == {p.split(".", 1)[1] for p in s["hidden_fields"]}
    assert v3["regime"]["current"] == "B" and v3["old_regime_response"]["applicable"]
    assert set(v3["first_attempt_accuracy"]["by_day"]) == {1, 2}
    demo = strip_hidden(s)
    assert set(demo["v3"]) == set(v3) - hidden
    blob = json.dumps(demo)
    for bad in ('"regime"', '"mapping"', '"K1"', '"K2"', '"shift_day"', '"cause"', "u_attempt"):
        assert bad not in blob, bad
    tl = live_timeline(v3_run)
    assert tl["hidden_fields"] == ["series[].v3_hidden"] and all("v3_hidden" in r for r in tl["series"])
    assert all("v3_hidden" not in r for r in strip_hidden(tl)["series"])
    rep = meme_report(v3_run)
    assert "co-op" in rep["summary"]["text"] and "accuracy" not in rep["summary"]["text"]
    assert "accuracy" in meme_report(v3_run, debug=True)["summary"]["text"]
    assert '"regime"' not in json.dumps(demo_report(v3_run))


def test_v3_mid_run_snapshot_sees_only_the_past(v3_run):
    tpd = live_snapshot(v3_run)["ticks_per_day"]
    s = live_snapshot(v3_run, tick=tpd - 1)       # end of day 1: no turnover, no wipe, regime A
    assert s["day"] == 1 and s["v3"]["roster"]["departed"] == [] and s["v3"]["binder"]["transitions"] == []
    assert s["v3"]["regime"]["current"] == "A" and not s["v3"]["old_regime_response"]["applicable"]
    assert s["v3"]["jobs"]["started"] == 8


# ------------------------------------------------------------------------------------------------ judges
def test_registry_and_verdict_schema(tmp_path):
    assert isinstance(JG.get_judge({"provider": "mock"}), JG.MockJudge)
    assert isinstance(JG.get_judge("mock"), JG.MockJudge)
    assert isinstance(JG.get_judge({"analysis": {"judge": {"provider": "mock"}}, "llm": {}}), JG.MockJudge)
    default = JG.get_judge(None, run_dir=tmp_path)
    assert isinstance(default, JG.LLMJudge) and default.describe() == {
        "judge_id": "claude_cli-sonnet-v1", "provider": "claude_cli", "model": "sonnet", "prompt_version": "v1"}
    assert JG.get_judge({"provider": "anthropic", "model": "sonnet"}, run_dir=tmp_path).model == "claude-sonnet-5"
    with pytest.raises(ValueError):
        JG.get_judge({"provider": "nope"})
    expr = {"phrase": "glimmer toast", "variants": ["glimmer toast"], "n_uses": 3, "n_speakers": 3}
    ctx = ["Maya: Grab me a glimmer toast.", "Leo: I want a glimmer toast."]
    v = JG.MockJudge().judge(expr, ctx)
    assert JG.validate_verdict(v) == [] and v == JG.MockJudge().judge(expr, ctx)
    assert JG.MockJudge().judge(expr, ctx, {"world_wording": True})["is_convention"] is False
    assert JG.validate_verdict({**v, "function": "slogan"}) and JG.validate_verdict({**v, "confidence": 2})
    # an LLM judge on the mock backend: schema-valid, cached in its own file, idempotent
    lj = JG.LLMJudge({"provider": "mock", "model": "mock", "prompt_version": "v1"}, run_dir=tmp_path)
    v1 = lj.judge(expr, ctx)
    lj.close()
    assert JG.validate_verdict(v1) == [] and v1["judge_id"] == "mock-mock-v1"
    calls = (tmp_path / "judge_llm_calls.jsonl").read_text().splitlines()
    assert len(calls) == 1 and "glimmer toast" in json.loads(calls[0])["prompt"]
    lj2 = JG.LLMJudge({"provider": "mock", "model": "mock", "prompt_version": "v1"}, run_dir=tmp_path)
    assert lj2.judge(expr, ctx) == v1
    lj2.close()
    assert len((tmp_path / "judge_llm_calls.jsonl").read_text().splitlines()) == 1
    bad = JG.MockJudge().verdict(None, "no json here")
    assert JG.validate_verdict(bad) == [] and bad["is_convention"] is False


def test_judge_prompt_is_an_observer_template_without_coinage_wording():
    body = (ROOT / "backend/prompts/judge_convention_v1.txt").read_text().split(
        "<commentblockmarker>###</commentblockmarker>")[-1]
    for w in ("meme", "slang", "coin", "invent", "nickname", "new word", "name for"):   # test_invariants.py
        assert w not in body.lower(), w
    banned = re.compile(r"\b(invent\w*|coin(s|ed|ing)?|label\w*|terms?|slang|memes?|nicknames?|shorthand|rules?|"
                        r"tips?|procedures?|jot\w*)\b", re.I)                       # test_v3_invariants.py
    assert not banned.search(body)
    prompt, _ = JG.render_prompt("v1", {"phrase": "glimmer toast", "variants": ["glimmer toasts"], "n_uses": 3,
                                        "n_speakers": 2}, ["A: x", "B: y"])
    assert "!<INPUT" not in prompt and " | ".join(JG.FUNCTIONS) in prompt and '"glimmer toasts"' in prompt
    assert "used 3 times by 2 different speakers" in prompt


def test_judge_run_writes_only_under_analysis_judgements(tmp_path):
    d = _synthetic(tmp_path / "syn")
    before = _digest(d)
    doc = JG.judge_run(d, {"provider": "mock"}, top=10)
    after = _digest(d)
    new = set(after) - set(before)
    assert new == {"analysis_judgements/mock-mock-v1.json"}
    assert all(after[k] == v for k, v in before.items())
    assert doc["source"] == "live" and doc["n_verdicts"] == len(doc["verdicts"]) > 0
    assert all(JG.validate_verdict(r["verdict"]) == [] for r in doc["verdicts"])
    # a plugged-in LLM judge (mock backend): the verdict file plus its own call cache, nothing else
    JG.register_judge("test_llm", lambda spec, run_dir: JG.LLMJudge(dict(spec, provider="mock", id="test_llm"),
                                                                      run_dir=run_dir))
    try:
        JG.judge_run(d, "test_llm", top=5)
    finally:
        JG.PROVIDERS.pop("test_llm")
    final = _digest(d)
    assert set(final) - set(after) == {"analysis_judgements/test_llm.json", "judge_llm_calls.jsonl"}
    assert all(final[k] == v for k, v in before.items())
    info = JG.available_judges(d)
    assert {r["judge_id"] for r in info["ran"]} == {"mock-mock-v1", "test_llm"} and "v1" in info["prompt_versions"]


# ------------------------------------------------------------------------------------------------ pipeline + report
def test_pipeline_routes_classifier_through_default_judge(v2_run, tmp_path):
    from backend.analysis.pipeline import analyze
    d = _copy_run(v2_run, tmp_path / "an", RUN_FILES + ("memory_meta.json",))
    out = analyze(d, llm_backend="mock", probes=False, verbose=False)
    # an explicit mock override: the judge follows the observer spec (backend mock, the run's model name)
    jid = out["judge"]["judge_id"]
    assert out["judge"]["provider"] == "mock" and out["judge"]["model"] == out["observer"]["model"]
    assert jid == f"mock-{out['observer']['model']}-v1" and out["judge"]["prompt_version"] == "v1"
    judged = [c for c in out["candidates"] if "judgements" in c]
    assert judged
    for c in judged:
        assert set(c["llm"]) >= {"is_convention", "gloss", "confidence", "raw"}           # legacy fields kept
        v = c["judgements"][jid]
        # a mock verdict is a placeholder: recorded with its provenance, never a convention
        assert JG.validate_verdict(v) == [] and c["llm"]["placeholder"]["is_convention"] == v["is_convention"]
        assert c["llm"]["is_convention"] is None and c["llm"]["real_judge"] is False and c["llm"]["judge_id"] == jid
        assert c["tier"] != "convention" and c["card"]["is_convention"] is None
    assert out["summary"]["n_llm_conventions"] == 0 and out["summary"]["n_conventions"] == 0
    assert out["summary"]["n_llm_conventions_placeholder"] == sum(v["is_convention"] for c in judged
                                                                  for v in c["judgements"].values())
    assert out["summary"]["judge_status"] == "mock_only" and "no real judge has run" in out["summary"]["judge_note"]
    assert not (d / "judge_llm_calls.jsonl").exists()        # the mock judge makes no LLM calls


def test_legacy_classifier_path_is_prompt_v0(tmp_path):
    """llm_classify without a judge keeps the pre-judge prompt, purpose and fields."""
    from backend.analysis.candidates import CLASSIFY, llm_classify
    from backend.llm.client import LLMClient, make_backend
    client = LLMClient(make_backend({"backend": "mock"}), tmp_path / "calls.jsonl")
    c = {"id": "m00", "canonical_form": "glimmer toast", "variants": ["glimmer toast"], "speakers": ["a", "b"],
         "usage_count": 2, "usages": [{"context": "A: grab a glimmer toast"}, {"context": "B: one glimmer toast\nA: ok"}]}
    llm_classify([c], client)
    client.close()
    rec = json.loads((tmp_path / "calls.jsonl").read_text().splitlines()[0])
    assert rec["purpose"] == "analysis_classifier"
    assert rec["prompt"] == CLASSIFY.format(expr="glimmer toast", uses="1. A: grab a glimmer toast\n2. B: one glimmer toast / A: ok")
    assert set(c["llm"]) >= {"is_convention", "gloss", "confidence", "raw"}
    assert c["llm"]["is_convention"] is None and c["llm"]["provider"] == "mock"      # mock backend: placeholder
    assert c["judgements"]["mock-mock-v0"]["prompt_version"] == "v0"


def test_report_with_and_without_analysis_json(v2_run, tmp_path):
    from backend.analysis.pipeline import analyze
    d = _copy_run(v2_run, tmp_path / "rep", RUN_FILES + ("memory_meta.json",))
    r0 = meme_report(d)
    json.dumps(r0)
    assert not r0["sources"]["analysis_json"] and r0["cards"] and r0["summary"]["text"]
    assert r0["judge_slot"]["ran"] == [] and "No LLM judge" in r0["summary"]["text"]
    assert r0["judge_slot"]["configured"]["provider"] == "claude_cli"         # analysis.judge defaults
    c0 = r0["cards"][0]
    for k in ("phrase", "status", "first_use", "adopters_timeline", "transmission_tree", "contexts", "world_wording",
              "judgements"):
        assert k in c0
    assert len(c0["contexts"]) <= 6 and c0["analysis"] is None
    root = c0["transmission_tree"]["root"]
    assert root["agent"] == c0["first_use"]["speaker"] and root["role"] == "originator"

    analyze(d, llm_backend="mock", probes=False, verbose=False)
    JG.judge_run(d, {"provider": "mock", "model": "mock2"}, top=10)
    r1 = meme_report(d)
    json.dumps(r1)
    assert r1["sources"]["analysis_json"] and any(c["analysis"] for c in r1["cards"])
    ids = {r["judge_id"] for r in r1["judge_slot"]["ran"]}
    assert ids == {json.load(open(d / "analysis.json"))["judge"]["judge_id"], "mock-mock2-v1"}
    judged = [c for c in r1["cards"] if c["judgements"]]
    assert judged
    for c in judged:
        ts = [j["judged_at"] or "" for j in c["judgements"]]
        assert ts == sorted(ts, reverse=True) and c["latest_verdict"] == c["judgements"][0]
    with_an = [c for c in r1["cards"] if c["analysis"]]
    assert all("grounding" not in c["analysis"] for c in with_an)
    assert any("grounding" in c["analysis"] for c in meme_report(d, debug=True)["cards"] if c["analysis"])
    assert all("latent_alignment" not in (c["analysis"]["card"] or {}) for c in with_an)


def test_report_meaning_over_time_is_hidden(tmp_path):
    d = _synthetic(tmp_path / "syn")
    (d / "probes").mkdir()
    json.dump({"X": "glimmer toast"}, open(d / "probes" / "targets.json", "w"))
    json.dump({"checkpoints": {"C1": {"checkpoint": "C1", "regime": "A", "sri": {"X": 0.25}, "know": 0.0,
                                      "ext": {}, "n_responses": 6}}}, open(d / "meaning.json", "w"))
    rep = meme_report(d)
    card = _find(rep["cards"], "glimmer toast")
    mot = card["meaning_over_time"]
    assert mot["hidden"] is True and mot["cue"] == "X" and mot["checkpoints"][0]["sri"] == 0.25
    assert "cards[].meaning_over_time" in rep["hidden_fields"]
    assert "meaning_over_time" not in _find(demo_report(d)["cards"], "glimmer toast")
    assert card["status"] == "emerged" and card["transmission_tree"]["root"]["children"][0]["agent"] == "leo"
    assert 'The planted control "a full pickle" (maya) has been used 2 times' in rep["summary"]["text"]
