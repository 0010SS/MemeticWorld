"""Trace registry (backend/tracing/schema.py), frames and the API views built on them. Mock backend only.

* Every record type a mock v2 all-on day, a mock v3 co-op day and a mock v3 turnover day emit is registered, and
  every record carries its registered required fields (and the envelope); the v2/v3 mechanisms' types all occur.
* Demo mode: `schema.strip` and the API's `_strip` drop every hidden type and remove every hidden field.
* Frames carry the v3 state (active, role, cohort, open matters, wordings, away, coop.binder, coop.jobs) and no
  hidden truth; a pending attempt's outcome is never shown before it is released.
* GET /api/schema, GET /api/runs/<id>/coop and the trace filters work via the FastAPI TestClient, including for
  runs written before these fields existed.
* docs/TRACE_SCHEMA.md is current.
"""
from __future__ import annotations

import json
import re
import shutil
import warnings
from pathlib import Path

import pytest

from tests.conftest import ROOT
from tests.test_integration_v2 import ALL_ON
from tests.test_v3_invariants import V3_DAY

from backend.tracing import schema as TS

# a two-day co-op with a turnover wave, a wipe and a regime change on day 2 (roster, farewell, onboarding)
TURN = {"run_name": "coop_turn", "population": "configs/population/coop13.yaml", "population_size": 13,
        "seed": 11, "world_seed": 11, "simulation_days": 2, "day_end": "19:30",
        "llm": {"backend": "mock", "max_workers": 6},
        "turnover": {"waves": [{"day": 2, "depart": {"am_crew": 1, "pm_crew": 1, "stores": 1}}]},
        "records": {"transitions": [{"day": 2, "mode": "wipe"}]},
        "regimes": {"schedule": [{"day": 1, "regime": "A"}, {"day": 2, "regime": "B"}],
                    "cues": [{"day": 1, "time": "16:45", "arena": "Stockroom", "text_key": "new_supplier"}]},
        "comm": {"meeting": {"days": [1]}}}
HIDDEN_STR = [re.compile(p) for p in (r'"K[0-3]"', r'"(LENS|DAMP|BELT|AIR|WARP)"', r'"M[12]"', r'"regime"',
                                      r'"mapping"', r'"u_attempt"', r'"job_truth"', r'"regime_active"',
                                      r'"event_beat"', r'"world_event_start"', r'"latent_type"')]
JOB_KEYS = {"id", "day", "slot", "shift", "operator", "project", "symptom", "attempt", "status", "tried"}
SNAPSHOT_KEYS = {"active", "role", "cohort", "open_matters", "wordings"}


def _run(tmp: Path, base: str | None, over: dict, name: str) -> Path:
    from backend.config import deep_merge, load_config
    from backend.simulation.engine import Simulation
    cfg = load_config(base, deep_merge({}, over))
    sim = Simulation(cfg, tmp / name, progress=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        sim.run()
    return tmp / name


def _trace(d: Path) -> list[dict]:
    return [json.loads(l) for l in open(d / "trace.jsonl")]


def _frames(d: Path) -> list[dict]:
    return [json.loads(l) for l in open(d / "frames.jsonl")]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("trace_schema")
    return {"v2": _run(tmp, "configs/baseline.yaml", ALL_ON, "v2_all_on"),
            "v3": _run(tmp, "configs/baseline.yaml", V3_DAY, "v3_day"),
            "turn": _run(tmp, "configs/v3_base.yaml", TURN, "v3_turn")}


# ------------------------------------------------------------------------------------------------ registry
def test_registry_is_well_formed():
    for t, s in TS.TRACE_TYPES.items():
        assert s["layer"] in TS.LAYERS, t
        assert s["description"] and s["section"] and s["since"] in ("v1", "v2", "v3"), t
        assert not set(s["required"]) & set(s["optional"]), t
        assert not set(s["required"]) & set(TS.ENVELOPE) - {"id"}, t
        for p in s["hidden_fields"]:
            assert p.split(".")[0] in set(s["required"]) | set(s["optional"]), (t, p)
        for f, (other, _v) in s["required_if"].items():
            assert f in s["optional"] and other in s["required"], (t, f)
    assert TS.HIDDEN_TYPES == {"world_event_start", "event_beat", "job_truth", "regime_active"}
    from backend.api.server import HIDDEN_TRACE_TYPES
    assert TS.HIDDEN_TYPES <= HIDDEN_TRACE_TYPES


def test_schema_doc_is_current():
    import importlib.util
    spec = importlib.util.spec_from_file_location("trace_schema_doc", ROOT / "scripts" / "trace_schema_doc.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (ROOT / "docs" / "TRACE_SCHEMA.md").read_text() == mod.render(), \
        "docs/TRACE_SCHEMA.md is stale: run scripts/trace_schema_doc.py"


def test_validate_and_strip_helpers():
    ok = {"id": "x#0", "type": "roster_change", "tick": 48, "time": "", "day": 2, "agent": "wei", "kind": "arrive",
          "role": "am_crew", "replaces": "hana"}
    assert TS.validate(ok) == []
    bad = {k: v for k, v in ok.items() if k not in ("replaces", "time")}
    errs = TS.validate(bad)
    assert any("'time'" in e for e in errs) and any("'replaces'" in e for e in errs)
    assert TS.validate({**ok, "type": "no_such_type"}) == ["unregistered trace type 'no_such_type'"]
    conv = {"id": "c", "type": "conversation", "tick": 3, "time": "", "participants": ["a", "b"],
            "trigger": {"agent": "a", "topic": "clarify", "event_ids": ["j01.1"], "job": "j01.1"}}
    s = TS.strip(conv)
    assert s["trigger"] == {"agent": "a", "topic": "clarify", "job": "j01.1"} and "event_ids" in conv["trigger"]
    assert TS.strip({"type": "job_truth", "tick": 6, "class": "K1"}) is None
    assert TS.strip({"type": "unknown", "tick": 1, "x": 1}) == {"type": "unknown", "tick": 1, "x": 1}


def test_trace_logger_warns_on_unregistered_types(tmp_path):
    from backend.tracing.logger import TraceLogger
    (tmp_path / "checked").mkdir()
    (tmp_path / "quiet").mkdir()
    tl = TraceLogger(tmp_path / "checked", check=True)
    quiet = TraceLogger(tmp_path / "quiet", check=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tl.log("mystery", x=1)
        tl.log("move", agent="maya")
        tl.log("mystery", x=2)                         # warned once per problem
        quiet.log("mystery")
    msgs = [str(x.message) for x in w if issubclass(x.category, RuntimeWarning)]
    assert sum("unregistered trace type 'mystery'" in m for m in msgs) == 1
    assert any("move: missing required field 'frm'" in m for m in msgs)
    tl.close()
    quiet.close()
    assert [json.loads(l)["type"] for l in open(tmp_path / "checked" / "trace.jsonl")] == ["mystery", "move", "mystery"]


# ---------------------------------------------------------------------------------------- emitted records
def test_every_emitted_type_is_registered_with_its_fields(runs):
    for name, d in runs.items():
        problems = sorted({p for r in _trace(d) for p in TS.validate(r)})
        assert not problems, (name, problems[:10])


def test_mechanisms_emit_their_types(runs):
    seen = {n: {r["type"] for r in _trace(d)} for n, d in runs.items()}
    assert {"memory_link", "reminding", "wording", "priming", "open_matter", "open_matter_focal", "viewpoint",
            "world_event_start", "event_beat", "conversation", "exposure", "invitation"} <= seen["v2"]
    v3 = {"job_start", "job_truth", "job_decision", "job_attempt", "job_end", "tally", "cue_event", "regime_active",
          "record_read", "record_write", "record_transition", "handover", "meeting", "coop_day", "checkpoint",
          "coop_fact"}
    assert v3 <= seen["v3"] and v3 <= seen["turn"]
    assert {"roster_change", "onboarding", "farewell", "clarification"} <= seen["turn"]
    assert not {"coop_fact", "farewell", "job_start"} & seen["v2"]
    assert set().union(*seen.values()) <= set(TS.TRACE_TYPES)


def test_coop_facts_and_farewells(runs):
    tr = _trace(runs["turn"])
    beats = [r for r in tr if r["type"] == "event_beat" and r["latent_type"] == "coop" and r["facts"]]
    facts = [r for r in tr if r["type"] == "coop_fact"]
    assert len(facts) == len(beats)
    assert [(r["ref"], r["beat"], [f["text"] for f in r["facts"]]) for r in facts] == \
           [(r["event_id"], r["beat"], [f["text"] for f in r["facts"]]) for r in beats]
    roles = {f["role"] for r in facts for f in r["facts"]}
    assert {"job_start", "symptom", "outcome", "tally", "farewell"} <= roles and not {"code", "oddity"} & roles
    fw = [r for r in tr if r["type"] == "farewell"]
    dep = {r["agent"]: r for r in tr if r["type"] == "roster_change" and r["kind"] == "depart"}
    assert {r["agent"] for r in fw} == set(dep)
    for r in fw:
        assert r["day"] == dep[r["agent"]]["day"] - 1 and r["role"] == dep[r["agent"]]["role"]


def test_every_encoded_memory_has_its_observation(runs):
    """Job episodes and binder reads are traced as observations too (they were not before), so every memory's
    observation id resolves (the causal chain view needs it)."""
    for name, d in runs.items():
        tr = _trace(d)
        obs = {r["observation_id"] for r in tr if r["type"] == "observation"}
        miss = [r["observation_id"] for r in tr if r["type"] == "memory_encoded" and r.get("observation_id")
                and r["observation_id"] not in obs]
        assert not miss, (name, miss[:5])
    tr = _trace(runs["v3"])
    assert any(r.get("episode") == "job" for r in tr if r["type"] == "observation")
    assert any(r["source_type"] == "record" for r in tr if r["type"] == "observation")


# --------------------------------------------------------------------------------------------- demo strip
def _has_path(obj, path: list[str]) -> bool:
    if isinstance(obj, list):
        return any(_has_path(x, path) for x in obj)
    if not isinstance(obj, dict) or path[0] not in obj:
        return False
    return len(path) == 1 or _has_path(obj[path[0]], path[1:])


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from _keys(x)


def test_demo_strip_removes_every_hidden_type_and_field(runs):
    from backend.api.server import HIDDEN_KEYS, _strip
    for name, d in runs.items():
        tr = _trace(d)
        for r in tr:
            s = TS.strip(r)
            spec = TS.TRACE_TYPES[r["type"]]
            if spec["hidden"]:
                assert s is None
                continue
            for p in spec["hidden_fields"]:
                assert not _has_path(s, p.split(".")), (r["type"], p)
        out = _strip(tr)
        assert not {r["type"] for r in out} & TS.HIDDEN_TYPES
        for r in out:
            for p in TS.TRACE_TYPES[r["type"]]["hidden_fields"]:
                assert not _has_path(r, p.split(".")), (name, r["type"], p)
        assert not set(_keys(out)) & HIDDEN_KEYS
        if name != "v2":
            s = json.dumps(out)
            assert not [p.pattern for p in HIDDEN_STR if p.search(s)], name


# --------------------------------------------------------------------------------------------------- frames
def test_frames_carry_v3_state(runs):
    tr = _trace(runs["turn"])
    fr = _frames(runs["turn"])
    tpd = len(fr) // 2
    dep = {r["agent"] for r in tr if r["type"] == "roster_change" and r["kind"] == "depart"}
    arr = {r["agent"] for r in tr if r["type"] == "roster_change" and r["kind"] == "arrive"}
    for f in fr:
        for aid, a in f["agents"].items():
            assert SNAPSHOT_KEYS <= set(a), aid
            assert a["role"] in ("am_crew", "pm_crew", "stores")
            assert a["cohort"] == ("newcomer" if aid in arr else "founder")
        assert f["away"] == sorted(aid for aid, a in f["agents"].items() if not a["active"])
        assert set(f["coop"]) == {"binder", "jobs"}
    assert fr[0]["away"] == [] and set(fr[tpd]["away"]) == dep and not arr & set(fr[0]["agents"])
    assert all(fr[tpd]["agents"][a]["location"] == "Away" for a in dep)
    # jobs: public keys only, world text, a pending attempt's outcome is not shown before it happens
    sym = {r["ref"]: next(f["text"] for f in r["facts"] if f["role"] == "symptom")
           for r in tr if r["type"] == "coop_fact" and r["kind"] == "job" and r["beat"] == 0}
    ends = {r["job"]: r for r in tr if r["type"] == "job_end"}
    shown_end = {}
    statuses = set()
    for f in fr:
        for j in f["coop"]["jobs"]:
            assert set(j) == JOB_KEYS
            assert j["symptom"] == sym[j["id"]]
            statuses.add(j["status"])
            if j["status"] == "running":
                assert len(j["tried"]) == j["attempt"] - 1
            if j["status"] in ("delivered", "defer", "failed"):
                assert j["id"] not in shown_end                      # a finished job is shown once
                shown_end[j["id"]] = f["tick"]
                assert f["tick"] == ends[j["id"]]["tick"] and j["status"] == ends[j["id"]]["result"]
                assert [t["outcome"] for t in j["tried"]] == [a["outcome"] for a in ends[j["id"]]["attempts"]]
    assert set(shown_end) == set(ends) and {"running", "delivered"} <= statuses
    s = json.dumps([f["coop"] for f in fr])
    assert not [p.pattern for p in HIDDEN_STR if p.search(s)]
    # v2: the same agent fields, nothing co-op
    for f in _frames(runs["v2"]):
        assert "coop" not in f and f["away"] == []
        for a in f["agents"].values():
            assert SNAPSHOT_KEYS <= set(a) and a["active"] and a["role"] is None and a["cohort"] is None
    v2 = _frames(runs["v2"])
    assert max(a["open_matters"] for f in v2 for a in f["agents"].values()) > 0
    assert max(a["wordings"] for f in v2 for a in f["agents"].values()) > 0
    # a co-op day without the roster: the workshop's crews, no cohort
    f = _frames(runs["v3"])[10]
    assert {a["role"] for a in f["agents"].values()} == {"am_crew", "pm_crew", "stores"}
    assert {a["cohort"] for a in f["agents"].values()} == {None}


# ------------------------------------------------------------------------------------------------------ API
@pytest.fixture(scope="module")
def api(runs, tmp_path_factory):
    from fastapi.testclient import TestClient
    from backend.api import server
    root = tmp_path_factory.mktemp("schema_api_root").resolve()
    for name, d in runs.items():
        shutil.copytree(d, root / f"ts_{name}")
    # a co-op run from before coop_fact / the new frame fields existed
    old = root / "ts_old_v3"
    shutil.copytree(runs["turn"], old)
    tr = [r for r in _trace(old) if r["type"] not in ("coop_fact", "farewell")]
    (old / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tr))
    fr = _frames(old)
    for f in fr:
        f.pop("away")
        f["coop"].pop("jobs")
        for a in f["agents"].values():
            for k in SNAPSHOT_KEYS:
                a.pop(k)
    (old / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in fr))
    saved = server.RUNS
    server.RUNS = root
    yield TestClient(server.app)
    server.RUNS = saved


def test_api_schema(api):
    demo = api.get("/api/schema").json()
    assert set(demo["types"]) == set(TS.TRACE_TYPES) - TS.HIDDEN_TYPES
    assert demo["envelope"] == list(TS.ENVELOPE) and "agent" in demo["frame"]
    s = json.dumps(demo)
    assert '"hidden' not in s and "event_ids" not in s and '"regime"' not in s
    assert "forced_by_event" not in demo["types"]["move"]["required"]
    dbg = api.get("/api/schema?debug=1").json()
    assert set(dbg["types"]) == set(TS.TRACE_TYPES) and dbg["types"]["job_truth"]["hidden"] is True
    assert dbg["types"]["observation"]["hidden_fields"] == ["event_id", "originating_event_ids"]


def test_api_coop_view(api, runs):
    tr = _trace(runs["turn"])
    res = api.get("/api/runs/ts_turn/coop")
    assert res.status_code == 200
    c = res.json()
    assert c["enabled"] and c["mechanisms"] == {"workshop": True, "records": True, "roster": True, "turnover": True}
    starts = [r for r in tr if r["type"] == "job_start"]
    assert [j["job"] for j in c["jobs"]] == [r["job"] for r in sorted(starts, key=lambda r: (r["tick"], r["job"]))]
    for j in c["jobs"]:
        assert j["symptom"] and j["end"] is not None
        assert [a["attempt"] for a in j["attempts"]] == list(range(1, len(j["attempts"]) + 1))
        assert all(a["text"] for a in j["attempts"])
        assert "truth" not in j
    writes = [r for r in tr if r["type"] == "record_write"]
    tl = c["binder"]["timeline"]
    assert [e for e in tl if e["kind"] == "write"] and len([e for e in tl if e["kind"] == "write"]) == len(writes)
    assert [e["kind"] for e in tl if e["kind"] == "transition"] == ["transition"]
    st = c["binder"]["state"]
    assert st["binder_id"] == "binder-2" and st["archived"] and st["archived"][0]["binder_id"] == "binder-1"
    last = _frames(runs["turn"])[-1]["coop"]["binder"]
    assert st["binder_id"] == last["binder_id"] and [e["text"] for e in st["log"][::-1][:5]] == \
        [e["text"] for e in last["log"]]
    assert (st["front"][-1]["text"] if st["front"] else None) == ((last["front"] or {}).get("text"))
    assert any(j["writes"] for j in c["jobs"])
    assert len(c["roster_changes"]) == 6 and len(c["farewells"]) == 3 and len(c["onboarding"]) == 3
    assert c["tallies"] and c["cues"] and c["handovers"] and c["meetings"] and c["days"] and c["checkpoints"]
    convs = {r["id"]: r for r in tr if r["type"] == "conversation"}
    talk = [(q["conversation_id"], "clarify") for j in c["jobs"] for q in j["clarifications"]] + \
        [(h["conversation_id"], "handover") for h in c["handovers"]] + \
        [(m["conversation_id"], "meeting") for m in c["meetings"]]
    assert talk and all(convs[cid]["trigger"]["topic"] == topic for cid, topic in talk)
    assert "regimes" not in c
    s = json.dumps(c)
    assert not [p.pattern for p in HIDDEN_STR if p.search(s)]
    dbg = api.get("/api/runs/ts_turn/coop?debug=1").json()
    truth = {r["job"]: r["class"] for r in tr if r["type"] == "job_truth"}
    assert {j["job"]: j["truth"]["class"] for j in dbg["jobs"]} == truth
    assert [r["regime"] for r in dbg["regimes"]] == ["A", "B"]
    # a v2 run: same shape, nothing in it
    v2 = api.get("/api/runs/ts_v2/coop").json()
    assert v2["enabled"] is False and v2["jobs"] == [] and v2["binder"]["timeline"] == []
    # an older co-op run (no coop_fact): symptoms and outcomes come from perceived facts
    old = api.get("/api/runs/ts_old_v3/coop").json()
    assert [j["symptom"] for j in old["jobs"]] == [j["symptom"] for j in c["jobs"]]
    assert old["farewells"] == []


def test_api_trace_filters(api, runs):
    tr = _trace(runs["turn"])
    got = api.get("/api/runs/ts_turn/trace?type=job_start,job_end&tick_from=40&tick_to=60").json()
    want = [r for r in tr if r["type"] in ("job_start", "job_end") and 40 <= r["tick"] <= 60]
    assert [r["id"] for r in got] == [r["id"] for r in want] and got
    assert api.get("/api/runs/ts_turn/trace?types=job_start&start=40&end=60").json() == \
        [r for r in got if r["type"] == "job_start"]                                  # older parameter names
    assert len(api.get("/api/runs/ts_turn/trace?type=move&limit=7").json()) == 7
    assert api.get("/api/runs/ts_turn/trace?type=job_truth,regime_active").json() == []
    assert api.get("/api/runs/ts_turn/trace?type=job_truth&debug=1").json()
    newcomer = next(r["agent"] for r in tr if r["type"] == "roster_change" and r["kind"] == "arrive")
    mine = api.get(f"/api/runs/ts_turn/trace?agent={newcomer}&limit=100000").json()
    assert mine and all(newcomer in json.dumps(r) for r in mine)
    assert {"roster_change", "onboarding"} <= {r["type"] for r in mine}
    op = next(r["operator"] for r in tr if r["type"] == "job_start")
    ops = api.get(f"/api/runs/ts_turn/trace?type=job_start&agent={op}").json()
    assert ops and [r["id"] for r in ops] == [r["id"] for r in tr if r["type"] == "job_start" and r["operator"] == op]
    cid = next(r["id"] for r in tr if r["type"] == "conversation")
    one = api.get(f"/api/runs/ts_turn/trace?conversation={cid}").json()
    assert {r["type"] for r in one} >= {"conversation", "utterance"} and \
        all(r["id"] == cid or r.get("conversation_id") == cid for r in one)
    conv = api.get("/api/runs/ts_turn/trace?type=conversation").json()
    assert conv and not any("event_ids" in (r.get("trigger") or {}) for r in conv)


def test_api_serves_old_and_new_frames(api):
    for rid in ("ts_turn", "ts_old_v3", "ts_v2", "ts_v3"):
        fr = api.get(f"/api/runs/{rid}/frames")
        assert fr.status_code == 200, rid
        assert api.get(f"/api/runs/{rid}/agent/dev?tick=30").status_code == 200, rid
        assert api.get(f"/api/runs/{rid}/manifest").status_code == 200, rid
    f = api.get("/api/runs/ts_turn/frames").json()[-1]
    assert f["away"] and all("latent_type" not in b and "event_id" not in b for b in f["beats"])
    assert f["coop"]["binder"]["binder_id"] == "binder-2"
