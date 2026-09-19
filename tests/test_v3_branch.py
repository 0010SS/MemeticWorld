"""v3 checkpoints, branches and design trees (docs/ONTOLOGY_V3.md §5.1, §6.2-§6.4). Unit level: the whole mock
tree run (G0) needs the integrated engine (CoopWorld, prefix-mode client wiring, day-end/branch hooks)."""
import json
import os
import stat
from pathlib import Path

import pytest
import yaml

from backend import cli
from backend.config import load_config
from backend.experiment import branch as B
from backend.experiment import checkpoint as CK
from backend.experiment import design as D

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "configs/designs/records_boundaries_v3.yaml"
SMOKE = ROOT / "configs/designs/smoke_v3_mock.yaml"
CAL = ROOT / "configs/designs/v3_calibration.yaml"


# ------------------------------------------------------------------------------------ overlay whitelist
PARENT = {"regimes": {"schedule": [{"day": 1, "regime": "A"}], "cues": []},
          "records": {"transitions": [{"day": 4, "mode": "keep"}], "history_from_day": None},
          "turnover": {"waves": [{"day": 4, "depart": {"am_crew": 1}}]}, "simulation_days": 4,
          "day_start": "07:30", "day_end": "22:30", "tick_minutes": 15, "start_date": "2026-09-14"}


def test_overlay_whitelist_accepts_study_overlays():
    B.check_overlay({"regimes": {"schedule": [{"day": 1, "regime": "A"}, {"day": 5, "regime": "B"}]},
                     "records": {"transitions": [{"day": 4, "mode": "keep"}, {"day": 5, "mode": "wipe"}]},
                     "branch": {"salt": 1}}, 5, PARENT)
    B.check_overlay({"records": {"transitions": [{"day": 4, "mode": "wipe"}]}}, 4, PARENT)
    B.check_overlay({"records": {"history_from_day": 5}, "comm": {"meeting": {"from_day": 6}},
                     "simulation_days": 7, "checkpoints": {"days": [5, 6]}}, 5, PARENT)


@pytest.mark.parametrize("overlay", [
    {"records": {"display": "history"}},                        # not a branch key
    {"workshop": {"p_fault": 0.9}},
    {"seed": 3},
    {"llm": {"model": "sonnet"}},
    {"records": {"transitions": [{"day": 4, "mode": "wipe"}, {"day": 5, "mode": "keep"}]}},   # pre-T entry changed
    {"regimes": {"schedule": [{"day": 1, "regime": "B"}]}},     # rewrites day 1
    {"records": {"history_from_day": 3}},                       # dated before at_day
    {"records": {"transitions": [{"mode": "wipe"}]}},           # undated entry
])
def test_overlay_whitelist_rejects(overlay):
    with pytest.raises(B.BranchError):
        B.check_overlay(overlay, 5, PARENT)


def test_branch_tick_and_salt_leave_prefix_unchanged():
    cfg = load_config(None, {"branch": {"at_day": 5, "salt": 1}})
    T = B.branch_tick(cfg, 5)
    assert T == 240 and B.branch_tick(cfg, 4) == 180           # 60 ticks a day (07:30-22:30, 15 min)
    assert B.salted_parts(cfg, T - 1, 7, "talk", T - 1) == (7, "talk", T - 1)
    assert B.salted_parts(cfg, T, 7, "talk", T) == (7, "talk", T, "salt", 1)
    assert B.salted_parts(load_config(), T + 5, 7, "talk") == (7, "talk")   # no salt, no branch


def test_apply_salt_rederives_agent_streams():
    from backend.agents.agent import Agent
    from backend.agents.profile import load_population
    cfg = load_config()
    profiles, _ = load_population(cfg["population"], 2)
    a, b = (Agent(p, cfg, None, 5) for p in list(profiles.values())[:1] * 2)
    x0 = a.stream("talk").random()
    assert x0 == b.stream("talk").random()                     # same seed -> same stream
    B.apply_salt({"a": a}, 5, 1)
    assert a.stream("talk").random() != b.stream("talk").random()
    assert a.rng is a._streams["legacy"]


# ---------------------------------------------------------------------------------------- trees
def test_study_tree_expands_in_dependency_order(tmp_path):
    d = D.load_design(STUDY, runs_root=tmp_path)
    assert isinstance(d, D.TreeDesign) and list(d.nodes)[0] == "trunk"
    runs = D.expand(d)
    assert len(runs) == 8 * 5 + 2                               # 42 segments (§7.3)
    by = {(r.node, r.seed): r for r in runs}
    assert ("keep_shift_rep", 13) not in by and ("keep_shift_rep", 12) in by
    for s in d.seeds:                                           # parents precede their branches
        seq = [r.node for r in runs if r.seed == s]
        assert seq[0] == "trunk"
    trunk = by[("trunk", 11)]
    t = trunk.config()
    assert t["seed"] == t["world_seed"] == 11 and t["simulation_days"] == 4
    assert t["llm"].get("replay_until_tick") is None and t["records"]["transitions"] == [{"day": 4, "mode": "keep"}]
    assert trunk.run_dir == tmp_path / "records_boundaries_v3" / "trunk" / "s11"
    w = by[("wipe4", 11)].config()
    assert w["simulation_days"] == 4 and w["llm"]["replay_until_tick"] == 180
    assert w["llm"]["replay_from"] == str(trunk.run_dir / "llm_calls.jsonl")
    assert w["records"]["transitions"] == [{"day": 4, "mode": "wipe"}]
    ks = by[("keep_shift", 12)].config()
    assert ks["simulation_days"] == 6 and ks["llm"]["replay_until_tick"] == 240
    assert ks["regimes"]["schedule"][-1] == {"day": 5, "regime": "B"} and ks["branch"]["at_day"] == 5
    assert ks["_condition"]["node"] == "keep_shift" and ks["branch"]["salt"] is None
    assert by[("keep_shift_rep", 11)].config()["branch"]["salt"] == 1
    # trunk prompts never depend on branch-only config: the trunk config is identical whatever the branches say
    assert "keep_shift" not in json.dumps({k: v for k, v in t.items() if k != "_condition"})


def test_tree_rejects_bad_overlay_at_load(tmp_path):
    raw = yaml.safe_load(SMOKE.read_text())
    raw["tree"]["wipe2"]["set"] = {"records": {"display": "none"}}
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(B.BranchError):
        D.load_design(p, runs_root=tmp_path)
    raw["tree"]["wipe2"] = {"parent": "nope", "at_day": 2, "days": 1}
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        D.load_design(p, runs_root=tmp_path)


def test_calibration_and_smoke_trees_load(tmp_path):
    cal = D.expand(D.load_design(CAL, runs_root=tmp_path))
    assert {(r.node, r.seed) for r in cal} == {("cal_scrambled", 10), ("cal_shiftmini", 9)}
    assert next(r for r in cal if r.node == "cal_scrambled").config()["workshop"]["causal"] == "scrambled"
    sm = D.expand(D.load_design(SMOKE, runs_root=tmp_path))
    assert [r.node for r in sm] == ["trunk", "wipe2", "keep_shift", "wipe_shift", "keep_shift_rep"]
    assert sm[0].config()["llm"]["backend"] == "mock"


def test_tree_status_waiting_parent_and_dry_run(tmp_path):
    d = D.load_design(SMOKE, runs_root=tmp_path)
    runs = D.expand(d)
    st = {r["cell"]: r["status"] for r in D.status(d, cells=runs)}
    assert st["trunk"] == "missing" and st["wipe2"] == "waiting_parent"
    plan = D.run_tree(d, runs, dry_run=True, log=lambda *_: None)
    assert [p["cell"] for p in plan][0] == "trunk" and plan[0]["cmd"][3:6] == ["design", "run-cell", str(d.path)]
    probes = D.probe_plan(d, runs, "C2")
    assert {(p["node"], p["checkpoint"]) for p in probes} == {("trunk", "C2"), ("wipe2", "C2")}


def test_tree_contrasts(tmp_path):
    d = D.load_design(STUDY, runs_root=tmp_path)
    rows = [{"cell": "trunk", "seed": 11, "v3.continuity.acc_mem_w1_c4": 0.8},
            {"cell": "wipe4", "seed": 11, "v3.continuity.acc_mem_w1_c4": 0.5},
            {"cell": "trunk", "seed": 12, "v3.continuity.acc_mem_w1_c4": 0.6},
            {"cell": "wipe4", "seed": 12, "v3.continuity.acc_mem_w1_c4": 0.6}]
    c = {x["name"]: x for x in D.tree_contrasts(d, rows)}["continuity"]
    assert c["n"] == 2 and c["mean_diff"] == pytest.approx(0.15)


def test_cli_tree_expand(capsys, tmp_path):
    cli.main(["design", "expand", str(SMOKE), "--runs-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert "keep_shift_rep" in out and "5 runs" in out


# ------------------------------------------------------------------------------- make_branch / prefix
def _fake_parent(tmp: Path, name="parent", status="finished", errors=0, digests=None, cv=None) -> Path:
    d = tmp / name
    d.mkdir(parents=True)
    cfg = load_config(None, {"simulation_days": 4, "seed": 11, "world_seed": 11,
                             "records": {"transitions": [{"day": 4, "mode": "keep"}]}})
    yaml.safe_dump(cfg, open(d / "config.resolved.yaml", "w"), sort_keys=False)
    cv = cv or {"git_sha": "abc", "prompt_hashes": {"p.txt": "1"}}
    (d / "manifest.json").write_text(json.dumps({"status": status, "code_version": cv}))
    with open(d / "llm_calls.jsonl", "w") as fh:
        fh.write(json.dumps({"key": "seed|x|0", "scope": "seed", "response": "ok"}) + "\n")
        for _ in range(errors):
            fh.write(json.dumps({"key": "t0001:x|y|0", "scope": "t0001", "response": "LLM_ERROR: boom"}) + "\n")
    with open(d / "digests.jsonl", "w") as fh:
        for t, h in (digests or {t: f"h{t}" for t in range(4 * 60)}).items():
            fh.write(json.dumps({"tick": t, "sha256": h, "n": 1}) + "\n")
    with open(d / "world_script.jsonl", "w") as fh:
        for day in range(1, 5):
            fh.write(json.dumps({"id": f"j0{day}.1", "kind": "job", "day": day}) + "\n")
    return d


def test_make_branch_writes_prefix_mode_config(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "code_version", lambda: {"git_sha": "abc", "prompt_hashes": {"p.txt": "1"}})
    parent = _fake_parent(tmp_path)
    ov = {"regimes": {"schedule": [{"day": 1, "regime": "A"}, {"day": 5, "regime": "B"}]},
          "records": {"transitions": [{"day": 4, "mode": "keep"}, {"day": 5, "mode": "keep"}]}}
    out, cfg = B.make_branch(parent, 5, ov, days=2, out=tmp_path / "child")
    assert cfg["simulation_days"] == 6 and cfg["llm"]["replay_until_tick"] == 240
    assert cfg["llm"]["replay_from"] == str(parent.resolve() / "llm_calls.jsonl")
    assert yaml.safe_load(open(out / "config.resolved.yaml")) == cfg
    rec = json.loads((out / "branch.json").read_text())
    assert rec["at_tick"] == 240 and rec["overlay_sha"] == B.sha256_json(ov) and rec["git_sha"] == "abc"
    with pytest.raises(B.BranchError):
        B.make_branch(parent, 5, {"records": {"display": "none"}}, out=tmp_path / "c2")


@pytest.mark.parametrize("kw,cur", [({"status": "running"}, None), ({"errors": 1}, None),
                                    ({}, {"git_sha": "zzz", "prompt_hashes": {"p.txt": "1"}}),
                                    ({}, {"git_sha": "abc", "prompt_hashes": {"p.txt": "2"}})])
def test_make_branch_refuses_bad_parents(tmp_path, monkeypatch, kw, cur):
    monkeypatch.setattr(B, "code_version", lambda: cur or {"git_sha": "abc", "prompt_hashes": {"p.txt": "1"}})
    parent = _fake_parent(tmp_path, **kw)
    with pytest.raises(B.BranchError):
        B.make_branch(parent, 5, {}, out=tmp_path / "child")


def test_check_prefix_digest_identity(tmp_path):
    parent = _fake_parent(tmp_path)
    child = _fake_parent(tmp_path, name="child", digests={**{t: f"h{t}" for t in range(240)},
                                                          **{t: f"other{t}" for t in range(240, 360)}})
    cfg = yaml.safe_load(open(child / "config.resolved.yaml"))
    cfg["branch"] = {"parent": str(parent), "at_day": 5, "salt": 1}
    yaml.safe_dump(cfg, open(child / "config.resolved.yaml", "w"))
    res = B.check_prefix(parent, child, 5)
    assert res["ok"], res["failures"]
    assert res["prefix_digest"] == res["parent_prefix_digest"]
    from backend.tracing import logger as L
    if hasattr(L, "prefix_digest"):
        assert res["prefix_digest"] == L.prefix_digest(parent, 240)    # the tracer's definition
    assert json.loads((child / "branch.json").read_text())["prefix_check"]["ok"]
    # one differing pre-T tick fails; a differing world-script day < at_day fails
    bad = _fake_parent(tmp_path, name="bad", digests={**{t: f"h{t}" for t in range(240)}, 17: "x"})
    yaml.safe_dump(cfg, open(bad / "config.resolved.yaml", "w"))
    assert not B.check_prefix(parent, bad, 5)["ok"]
    with open(child / "world_script.jsonl", "a") as fh:
        fh.write(json.dumps({"id": "j02.9", "kind": "job", "day": 2}) + "\n")
    assert not B.check_prefix(parent, child, 5, write=False)["ok"]
    with pytest.raises(B.PrefixMismatch):
        B.assert_prefix(parent, child, 5, write=False)


# ------------------------------------------------------------------------------------------ checkpoints
def _sim(tmp: Path, name: str):
    from backend.simulation.engine import Simulation
    cfg = load_config("configs/baseline.yaml", {"simulation_days": 1, "day_end": "09:00", "population_size": 3,
                                                "llm": {"backend": "mock", "max_workers": 2},
                                                "latent_events": {"event_rate": 0.3},
                                                "checkpoints": {"enabled": True, "days": "all"}})
    sim = Simulation(cfg, tmp / name, progress=False)
    real_finish = sim.finish
    sim.finish = lambda seconds: None                          # keep the tracer open for the checkpoint
    sim.run()
    return sim, real_finish


@pytest.fixture(scope="module")
def two_sims(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ck")
    out = []
    for name in ("a", "b"):
        sim, fin = _sim(tmp, name)
        rec = CK.maybe_checkpoint(sim, sim.clock.total_ticks - 1)
        fin(0.0)
        out.append((sim, rec))
    return out


def test_checkpoint_contents_and_deterministic_bytes(two_sims):
    (sa, ra), (sb, rb) = two_sims
    assert ra["id"] == "C1" and ra["tick"] == sa.clock.ticks_per_day - 1
    assert ra["files"] == rb["files"]                          # byte-identical across identical runs
    ca = sa.run_dir / "checkpoints" / "C1"
    names = set(ra["files"])
    for f in ("memory_meta.json", "agent_state.json", "binder.json", "roster.json", "world_state.json"):
        assert f in names
    assert any(n.startswith("agents/") and n.endswith("associative_memory/nodes.json") for n in names)
    st = json.loads((ca / "agent_state.json").read_text())
    aid = sorted(st)[0]
    assert {"role", "cohort", "arrival_day", "relationships", "scratch", "rng_streams"} <= set(st[aid])
    ws = json.loads((ca / "world_state.json").read_text())
    assert ws["regime"] == "A" and ws["day"] == 1
    trace = [json.loads(l) for l in open(sa.run_dir / "trace.jsonl")]
    ck = [r for r in trace if r["type"] == "checkpoint"]
    assert len(ck) == 1 and ck[0]["files"] == ra["files"]


def test_checkpoint_readonly_write_once_and_load(two_sims):
    (sa, ra), _ = two_sims
    ca = sa.run_dir / "checkpoints" / "C1"
    for rel in ra["files"]:
        assert not os.stat(ca / rel).st_mode & stat.S_IWUSR
    assert CK.verify_checkpoint(ca)
    with pytest.raises(FileExistsError):
        CK.write_checkpoint(sa, 1)
    ck = CK.load_checkpoint(ca, load_memories=True)
    assert ck["agents"] == sorted(sa.agents) and set(ck["memories"]) == set(sa.agents)
    n = len(next(iter(ck["memories"].values())).id_to_node)
    assert n == len(sa.agents[ck["agents"][0]].a_mem.id_to_node)
    assert CK.checksums(ca) == ra["files"]                     # loading never writes
    assert CK.list_checkpoints(sa.run_dir) == ["C1"]


def test_checkpoint_days_config():
    assert not CK.enabled_for(load_config(), 1)
    assert CK.enabled_for(load_config(None, {"checkpoints": {"enabled": True}}), 3)
    assert CK.enabled_for(load_config(None, {"checkpoints": {"enabled": True, "days": [3, 4]}}), 4)
    assert not CK.enabled_for(load_config(None, {"checkpoints": {"enabled": True, "days": [3, 4]}}), 5)


@pytest.mark.skip(reason="needs the integrated v3 engine (CoopWorld, prefix-mode client, day-end/branch hooks)")
def test_whole_mock_tree_runs(tmp_path):
    d = D.load_design(SMOKE, runs_root=tmp_path)
    res = D.run_tree(d, D.expand(d), analyze=False, log=lambda *_: None)
    assert all(r["rc"] == 0 for r in res)
