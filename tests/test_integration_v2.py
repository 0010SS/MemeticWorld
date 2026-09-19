"""End-to-end checks of the v2 engine with its mechanisms switched on (mock backend only).

* A full day with EVERY v2 mechanism on (group talk, catch-ups, reminding, need, priming, verbatim,
  referents, balanced assignment and schedule, link visibility) is recorded in one process and replayed
  in another with a different PYTHONHASHSEED: the trace, frames, events, world script and every
  memory artifact must come out byte-identical, and every LLM call must come from the recording.
* Common random numbers inside the engine: switching a social mechanism on that never fires, or making
  a private fact public, must not shift any other draw.
* Group talk: NEED for participants, one draw per eligible set and window, a group for each circle's
  staggered dinner. Forced movers are not narrated.
* The API server lists and serves design runs (runs/<design>/<cell>/s<seed>), strips every hidden v2
  field in demo mode, and shows reminding thoughts in the causal chain and the agent inspector.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.conftest import ROOT, run_sim

ALL_ON = {
    "run_name": "all_on", "seed": 7, "simulation_days": 1,
    "llm": {"backend": "mock", "max_workers": 6},
    "latent_events": {"event_rate": 0.35, "link_visibility": 0.3, "referents": {"enabled": True},
                      "assignment": {"mode": "balanced"}, "schedule": "balanced"},
    "conversation": {"catchup": {"enabled": True}, "group": {"enabled": True, "prob": 1.0}},
    "need": {"enabled": True, "min_importance": 3},   # low, so group-talk memories reliably qualify
    "reminding": {"enabled": True},
    "memory": {"verbatim": {"enabled": True}},
    "priming": {"enabled": True},
}
BYTE_FILES = ("trace.jsonl", "frames.jsonl", "events.jsonl", "world_script.jsonl", "memory_meta.json")


def _cli(args, hashseed, cwd=ROOT):
    env = {**os.environ, "PYTHONHASHSEED": str(hashseed)}
    return subprocess.run([sys.executable, "-m", "backend.cli", *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=600)


def _trace(run_dir):
    return [json.loads(l) for l in open(Path(run_dir) / "trace.jsonl")]


@pytest.fixture(scope="module")
def all_on(tmp_path_factory):
    """(recorded run dir, replayed run dir): record under PYTHONHASHSEED=11, replay under 999."""
    tmp = tmp_path_factory.mktemp("all_on")
    cfg = tmp / "all_on.yaml"
    cfg.write_text(yaml.safe_dump({"extends": "configs/baseline.yaml", **ALL_ON}))
    rec, rep = tmp / "rec", tmp / "rep"
    r = _cli(["run", "--config", str(cfg), "--out", str(rec)], 11)
    assert r.returncode == 0, r.stderr[-3000:]
    r = _cli(["replay", str(rec), "--out", str(rep)], 999)
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-3000:])      # exit 0 = identical trace digest
    return rec, rep


# ------------------------------------------------------------------ replay with everything on
def test_replay_with_all_mechanisms_is_bit_identical_across_processes(all_on):
    from backend.simulation.engine import trace_digest
    rec, rep = all_on
    assert trace_digest(rec) == trace_digest(rep)
    for f in BYTE_FILES:
        assert (rec / f).read_bytes() == (rep / f).read_bytes(), f
    agents = sorted(p.name for p in (rec / "agents_final").iterdir())
    assert len(agents) == 8
    for a in agents:
        for f in ("nodes.json", "kw_strength.json", "embeddings.json"):
            sub = Path("agents_final") / a / "associative_memory" / f
            assert (rec / sub).read_bytes() == (rep / sub).read_bytes(), sub
    man = json.load(open(rep / "manifest.json"))
    assert man["status"] == "finished"
    assert man["stats"]["llm"]["calls"] > 500 and man["stats"]["llm"]["cached"] == man["stats"]["llm"]["calls"]
    assert man["trace_sha256"] == json.load(open(rec / "manifest.json"))["trace_sha256"]


def test_all_on_run_exercises_every_mechanism(all_on):
    rec, _ = all_on
    tr = _trace(rec)
    types = {r["type"] for r in tr}
    convs = [r for r in tr if r["type"] == "conversation"]
    assert any((c.get("trigger") or {}).get("topic") == "catchup" for c in convs)
    assert any(len(c["participants"]) >= 3 for c in convs)
    assert any(r["type"] == "reminding" and r.get("node_id") for r in tr)
    assert {"open_matter", "open_matter_focal", "priming", "wording", "memory_link"} <= types
    world = [json.loads(l) for l in open(rec / "world_script.jsonl")]
    assert world and all(e.get("circle") for e in world)                       # balanced assignment
    assert any(r.get("reused") for e in world for r in e.get("referents") or [])
    assert any(f["kind"] == "inner" and f["visibility"] == "all" for e in world for b in e["beats"]
               for f in b["facts"])                                           # link visibility


def test_group_participants_open_matters(all_on):
    """NEED.note runs for group-conversation participants exactly as for dyads (D60 has no exception)."""
    rec, _ = all_on
    tr = _trace(rec)
    cfg = yaml.safe_load(open(rec / "config.resolved.yaml"))
    thr = cfg["need"]["min_importance"]
    grp = re.compile(r"g\d+\.obs\.")
    enc = [r for r in tr if r["type"] == "memory_encoded" and r["source_type"] == "conversation"
           and grp.search(r.get("observation_id") or "")]
    assert enc, "no group-conversation memories"
    opened = {r["observation_id"] for r in tr if r["type"] == "open_matter" and r["action"] in ("open", "refresh")}
    eligible = [r for r in enc if r["importance"] >= thr]
    assert eligible, "no group memory reached need.min_importance"
    assert all(r["observation_id"] in opened for r in eligible)


def test_forced_movers_get_neutral_activities(all_on):
    rec, _ = all_on
    tr = _trace(rec)
    forced = [r for r in tr if r["type"] == "move" and r.get("forced_by_event")]
    assert forced
    plans = {(r["agent"], r["day"]): r["plan"] for r in tr if r["type"] == "day_plan"}
    for r in forced:
        assert r["activity"] == f"stopping by the {r['to'][0]}" or any(
            e["activity"] == r["activity"] for e in plans[(r["agent"], 1)]) or r["activity"].startswith(
            ("hanging out with", "heading to the")), r
    for f in ("trace.jsonl", "llm_calls.jsonl", "frames.jsonl"):
        assert "something unexpected" not in (rec / f).read_text()


def test_busy_agents_are_distracted_and_overhear_less(all_on):
    """A conversation runs into the next tick's perception (busy_factor, 'distracted' vantage), and people
    in another conversation in the same room are bystanders who overhear at overhear_prob x busy_factor."""
    rec, _ = all_on
    tr = _trace(rec)
    vantages = {f["vantage"] for r in tr if r["type"] == "viewpoint" for f in r["facts"]}
    assert "distracted" in vantages
    frames = [json.loads(l) for l in open(rec / "frames.jsonl")]
    for r in tr:
        if r["type"] == "viewpoint" and any(f["vantage"] == "distracted" for f in r["facts"]):
            prev = frames[r["tick"] - 1]["agents"][r["agent"]]
            assert prev["conversation"]                              # was talking in the previous tick
    in_conv = {}
    for c in (r for r in tr if r["type"] == "conversation"):
        for p in c["participants"]:
            in_conv[(c["tick"], p)] = c["id"]
    busy_heard = [(r["tick"], x) for r in tr if r["type"] == "exposure" and r.get("conversation_id")
                  for x in r["listener_ids"] if in_conv.get((r["tick"], x)) not in (None, r["conversation_id"])]
    assert busy_heard


# ------------------------------------------------------------------ common random numbers
def _digest(sim):
    from backend.simulation.engine import trace_digest
    return trace_digest(sim.run_dir)


def test_idle_social_mechanisms_do_not_shift_other_draws(tmp_path):
    """Catch-ups and group talk switched on with prob 0 never fire, so nothing else may change: each social
    purpose has its own stream, and conversations are seeded by their participants, not list position."""
    base = {"day_end": "20:00", "seed": 1}
    a = run_sim(tmp_path, base, "off")
    b = run_sim(tmp_path, {**base, "conversation": {"catchup": {"enabled": True, "prob": 0.0},
                                                    "group": {"enabled": True, "prob": 0.0,
                                                              "min_participants": 1}}}, "idle")
    assert _digest(a) == _digest(b)
    assert (a.run_dir / "frames.jsonl").read_bytes() == (b.run_dir / "frames.jsonl").read_bytes()


def test_world_seed_alone_fixes_plans_and_events(tmp_path):
    """Routine plans are part of the world (agreement a): with world_seed held fixed, changing `seed`
    changes neither the day plans nor the world script, so every beat happens at the same place."""
    runs = [run_sim(tmp_path, {"seed": s, "world_seed": 5}, f"s{s}") for s in (1, 2)]
    plans = [[r for r in _trace(x.run_dir) if r["type"] == "day_plan"] for x in runs]
    assert plans[0] and [r["plan"] for r in plans[0]] == [r["plan"] for r in plans[1]]
    assert runs[0].world_sha == runs[1].world_sha
    assert (runs[0].run_dir / "world_script.jsonl").read_bytes() == (runs[1].run_dir / "world_script.jsonl").read_bytes()
    other = run_sim(tmp_path, {"seed": 1, "world_seed": 6}, "w6")
    assert [r["plan"] for r in _trace(other.run_dir) if r["type"] == "day_plan"] != [r["plan"] for r in plans[0]]


def test_link_visibility_does_not_shift_bystander_perception(tmp_path):
    """Making a private fact public adds nothing but that fact: until the runs diverge in where people are
    or who is talking, every agent notices exactly the same other facts."""
    runs = {lv: run_sim(tmp_path, {"seed": 2, "latent_events": {"link_visibility": lv}}, f"lv{lv}")
            for lv in (0.0, 1.0)}
    frames = {lv: [json.loads(l) for l in open(s.run_dir / "frames.jsonl")] for lv, s in runs.items()}

    def state(f):
        return {a: (x["location"], x["arena"], x["conversation"]) for a, x in f["agents"].items()}
    n = min(len(frames[0.0]), len(frames[1.0]))
    split = next((t for t in range(n) if state(frames[0.0][t]) != state(frames[1.0][t])), n)
    # perception at tick t depends on positions at t and conversations at t-1
    until = split if split < n and {a: s[:2] for a, s in state(frames[0.0][split]).items()} != \
        {a: s[:2] for a, s in state(frames[1.0][split]).items()} else split + 1
    inner = {f["id"] for s in runs.values() for e in s.world for b in e.beats for f in b["facts"]
             if f["kind"] == "inner"}

    def seen(sim):
        return sorted((r["tick"], r["agent"], f["id"]) for r in _trace(sim.run_dir)
                      if r["type"] == "observation" and r["source_type"] == "perception" and r["tick"] < until
                      for f in r["facts"] if f["id"] not in inner)
    assert seen(runs[0.0]) == seen(runs[1.0])
    # the check has teeth: some bystander stood by a public private fact before the split
    public = [(b["tick"], b["location"], e.beats[0]["movers"]) for e in runs[1.0].world for b in e.beats
              if b["tick"] < until and any(f["kind"] == "inner" and f["visibility"] == "all" for f in b["facts"])]
    assert any(frames[1.0][t]["agents"][a]["location"] == loc and a not in movers
               for t, loc, movers in public for a in frames[1.0][t]["agents"])
    assert seen(runs[0.0])


# ------------------------------------------------------------------ group talk
def test_each_circle_gets_its_staggered_dinner_group(tmp_path):
    """Generated topology staggers the circles' shared dinners at one venue (D62): each circle's arrival is
    a new eligible set with its own draw, so with prob 1 both circles talk as a group, and nobody is in
    two groups in one window."""
    sim = run_sim(tmp_path, {"day_end": "20:00", "topology": {"mode": "generated"},
                             "latent_events": {"event_rate": 0.0},
                             "conversation": {"group": {"enabled": True, "prob": 1.0}}}, "dinners")
    convs = [r for r in _trace(sim.run_dir) if r["type"] == "conversation" and len(r["participants"]) >= 3]
    evening = [c for c in convs if c["time"][11:16] >= "18:00"]
    for cid, members in sim.circles.items():
        assert any(len(set(c["participants"]) & set(members)) >= 2 for c in evening), (cid, evening)
    for window in (("12:00", "13:30"), ("18:00", "19:30")):
        ps = [p for c in convs if window[0] <= c["time"][11:16] <= window[1] for p in c["participants"]]
        assert len(ps) == len(set(ps)), window


def test_group_prob_is_per_eligible_set_not_per_tick(tmp_path, monkeypatch):
    """One draw per eligible set and (day, window, venue), whatever its outcome."""
    from backend.simulation import engine as E
    draws = []
    real = E.seed_rng

    def spy(*parts):
        g = real(*parts)
        if len(parts) > 1 and parts[1] == "group":
            class G:
                def random(self_inner):
                    draws.append(parts[2])
                    return g.random()

                def permutation(self_inner, n):
                    return g.permutation(n)
            return G()
        return g
    monkeypatch.setattr(E, "seed_rng", spy)
    sim = run_sim(tmp_path, {"day_end": "20:00", "latent_events": {"event_rate": 0.0},
                             "conversation": {"group": {"enabled": True, "prob": 0.0}}}, "gprob")
    n_agents = len(sim.agents)
    assert draws
    for key, seen in sim.group_seen.items():
        assert len(seen) <= n_agents
    # every draw consumed at least min_participants never-seen agents, so draws per window are bounded
    per_window = {}
    for t in draws:
        per_window.setdefault((sim.clock.day_of(t), sim.clock.time_of(t).hour < 15), []).append(t)
    for ts in per_window.values():
        assert len(ts) <= n_agents // 3


# ------------------------------------------------------------------ API server
HIDDEN_VALUE_WHITELIST = {("config", "latent_events", "families"), ("config", "latent_events", "family_weights")}


def _synthetic_analysis(rec: Path) -> dict:
    """An analysis.json carrying every hidden observer field (independent of the observer code)."""
    utt = next(r for r in _trace(rec) if r["type"] == "utterance")
    cand = {"id": "m00", "canonical_form": "the thing", "display_form": "the thing", "variants": [],
            "usage_count": 3, "speakers": ["maya", "leo"], "status": "candidate",
            "usages": [{"utterance_id": utt["id"], "latent_types": {"E1": 1.0}, "retrieved_event_ids": ["ev000"]}],
            "llm": {"is_convention": True, "gloss": "x"}, "card": {"latent_alignment": 1.0, "users": 2}}
    return {"run_id": rec.name, "summary": {"n_events": 1, "n_conversations": 1, "n_utterances": 1},
            "latent_type_base_rates": {"E1": 0.25}, "candidates": [cand],
            "transmission": {"m00": {"depth": 1, "edges": []}},
            "semantics": {"m00": {"coherence": 0.5, "within_group": {"dorm_a": 0.4}}},
            "probes": {"m00": {"options": [{"letter": "A", "family": "E1", "text": "x"}],
                               "agents": {"maya": {"match_choice": "A", "match_family": "E1"}}}},
            "evaluation": {"m00": {"alignment": 1.0, "lift": 2.0, "permutation": {"p_value": 0.01}}},
            "grounding": {"candidates": {"m00": {"best_family": "E1"}}}, "funnel": {"metrics": {}},
            "outcomes": {"n_grounded": 1}}


@pytest.fixture(scope="module")
def api(all_on, tmp_path_factory):
    """A runs root with a top-level run and a design run, served by the API app."""
    from fastapi.testclient import TestClient
    from backend.api import server
    rec, _ = all_on
    root = tmp_path_factory.mktemp("runs_root").resolve()
    design_id = "demo_design/need-on__link-on/s7"
    for rid in ("20260101-000000_plain_s7", design_id):
        shutil.copytree(rec, root / rid)
        json.dump(_synthetic_analysis(rec), open(root / rid / "analysis.json", "w"))
    man = json.load(open(root / design_id / "manifest.json"))
    man["condition"] = {"design": "demo_design", "cell": "need-on__link-on", "seed": 7,
                        "levels": {"need": "on", "structure": "scrambled"}, "control": None}
    json.dump(man, open(root / design_id / "manifest.json", "w"))
    old = server.RUNS
    server.RUNS = root
    yield TestClient(server.app), design_id, rec
    server.RUNS = old


def _hidden_values(rec: Path) -> set:
    world = [json.loads(l) for l in open(rec / "world_script.jsonl")]
    man = json.load(open(rec / "manifest.json"))
    vals = {"E0", "E1", "E2", "E3", "E4", "scrambled"} | set(man.get("circles") or {})
    for e in world:
        vals |= {e.get("schema"), e.get("skin"), e.get("circle")}
        vals |= {c.get("skin") for c in e.get("composed_from") or []}
    return {v for v in vals if v}


def _scan(obj, hidden_vals, path=()):
    from backend.api.server import HIDDEN_KEYS
    if path[:3] in HIDDEN_VALUE_WHITELIST:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert k not in HIDDEN_KEYS and k not in hidden_vals, path + (k,)
            _scan(v, hidden_vals, path + (k,))
    elif isinstance(obj, list):
        for x in obj:
            _scan(x, hidden_vals, path)
    elif isinstance(obj, str):
        assert obj not in hidden_vals, (path, obj)


def test_api_lists_and_serves_design_runs(api):
    client, design_id, _ = api
    ids = [r["run_id"] for r in client.get("/api/runs").json()]
    assert design_id in ids and "20260101-000000_plain_s7" in ids
    for ep in ("manifest", "frames", "trace?limit=50", "analysis", "llm_calls?limit=3"):
        assert client.get(f"/api/runs/{design_id}/{ep}").status_code == 200, ep
    assert client.get(f"/api/runs/{design_id}/agent/maya?tick=30").status_code == 200
    assert client.get(f"/api/runs/{design_id}/events").status_code == 403
    assert client.get(f"/api/runs/{design_id}/events?debug=1").status_code == 200
    assert client.get("/api/runs/demo_design/../../etc/manifest").status_code == 404
    assert client.get("/api/runs/demo_design/manifest").status_code == 404        # not a run dir
    rows = client.get("/api/compare").json()
    assert design_id in {r["run_id"] for r in rows}


def test_api_demo_mode_strips_every_hidden_v2_field(api):
    client, design_id, rec = api
    hidden = _hidden_values(rec)
    utt = next(r for r in _trace(rec) if r["type"] == "utterance")
    for url in ("manifest", "frames", "trace", "analysis", "agent/maya?tick=59", f"chain/{utt['id']}"):
        res = client.get(f"/api/runs/{design_id}/{url}")
        assert res.status_code == 200, url
        _scan(res.json(), hidden)
    man = client.get(f"/api/runs/{design_id}/manifest").json()
    assert "structure" not in man["config"]["latent_events"]
    assert "levels" not in man["condition"] and "cell" not in man["condition"]
    assert "free" not in man["topology"] and "bridges" not in man["topology"]
    for row in client.get("/api/compare").json():
        _scan(row, hidden)
        assert not {"n_grounded", "n_aligned_p05", "mean_alignment", "cell", "levels"} & set(row)
    # debug mode keeps everything
    dbg = client.get(f"/api/runs/{design_id}/manifest?debug=1").json()
    assert dbg["circles"] and dbg["condition"]["levels"]["structure"] == "scrambled"


def test_api_generated_topology_hides_circle_groups():
    from backend.api.server import _strip_manifest
    man = {"groups": {"c1": ["a", "b", "c"]}, "circles": {"c1": ["a", "b", "c"]}, "config": {}}
    assert "groups" not in _strip_manifest(man) and "circles" not in _strip_manifest(man)
    man = {"groups": {"dorm_a": ["a", "b"]}, "circles": {"c_lab": ["a", "b", "c"]}}
    assert _strip_manifest(man)["groups"] == {"dorm_a": ["a", "b"]}


def test_api_shows_reminding_thoughts(api):
    client, design_id, rec = api
    tr = _trace(rec)
    rem = {r["node_id"]: r for r in tr if r["type"] == "reminding" and r.get("node_id")}
    utt = next(u for u in tr if u["type"] == "utterance" and set(u["retrieved"]) & set(rem))
    c = client.get(f"/api/runs/{design_id}/chain/{utt['id']}").json()
    nodes = [m for m in c["retrieved"] if m["node_id"] in rem]
    assert nodes
    for m in nodes:
        assert not m.get("missing") and m["kind"] == "thought" and m["source_type"] == "reminding"
        r = rem[m["node_id"]]
        assert [x["node_id"] for x in m["from_memories"]] == [r["new_node"], r["reminded_of"]]
        assert not any(x.get("missing") for x in m["from_memories"])
    # agent inspector: counted, listed, and resolvable among the last retrieved memories
    aid = rem[nodes[0]["node_id"]]["agent"]
    last = max(r["tick"] for r in tr)
    d = client.get(f"/api/runs/{design_id}/agent/{aid}?tick={last}").json()
    mine = [n for n, r in rem.items() if r["agent"] == aid]
    forgotten = {r["node_id"] for r in tr if r["type"] == "memory_forgotten"}
    assert {m["node_id"] for m in d["remindings"]} == {n for n in mine if n not in forgotten} or len(d["remindings"]) == 30
    assert all(m["importance"] is not None for m in d["remindings"])
    assert any(m["source_type"] == "reminding" for m in d["memories"])
    n_alive = len({r["node_id"] for r in tr if r["type"] in ("memory_encoded", "reflection") and r["agent"] == aid}
                  | set(mine)) - len({n for n in forgotten if n.startswith(aid + ":")})
    assert d["n_memories"] == n_alive
