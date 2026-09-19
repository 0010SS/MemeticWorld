"""Research plumbing: fresh identities, matched configurations, replay and failure."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.analysis.pipeline import analyze
from backend.config import load_config
from backend.llm.client import current_purpose
from backend.llm.mock import MockBackend
from backend.simulation.engine import Simulation, trace_digest


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def commons_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("commons") / "recorded"
    sim = Simulation(load_config("configs/commons_smoke.yaml"), directory, progress=False)
    sim.run()
    analyze(directory, verbose=False)
    return sim


def test_complete_run_has_actions_records_fresh_newcomers_and_observer_boundaries(commons_run):
    sim = commons_run
    trace = rows(sim.run_dir / "trace.jsonl")
    analysis = json.loads((sim.run_dir / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["summary"]["projects_completed"] > 0
    assert analysis["summary"]["record_versions"] > 0
    assert analysis["summary"]["record_reads"] > 0
    initialized = [e for e in trace if e["type"] == "newcomer_initialized"]
    assert {e["agent"] for e in initialized} == {"riley", "alex"}
    assert all(e["n_memories"] == e["inherited_relationships"] == 0 for e in initialized)
    assert all(not sim.agents[aid].profile.relationships for aid in ("riley", "alex"))
    decisions = [e for e in trace if e["type"] == "decision" and e["kind"] == "commons"]
    for e in decisions:
        context = json.loads(e["prompt"].split("LOCAL_CONTEXT_JSON\n", 1)[1])
        view = json.dumps(context["view"])
        assert all(key not in view for key in ("bench_mode", "change_enabled", "change_tick", "research_design"))
        if e["tick"] >= sim.commons.turnover_tick:
            assert e["agent"] not in {"maya", "ethan"}
    assert "not evaluated" in analysis["question_status"]["RQ1"]
    assert analysis["probes"] == {}  # no legacy E1--E4 probes in this world
    assert not (sim.run_dir / "analysis_llm_calls.jsonl").exists()
    snapshots = sorted((sim.run_dir / "checkpoints").glob("*/snapshot.json"))
    assert len(snapshots) == 4
    after = json.loads(snapshots[1].read_text(encoding="utf-8"))
    assert "riley" in after["active_agents"] and "maya" not in after["active_agents"]


def test_commons_replay_matches_trace_and_final_world(commons_run, tmp_path):
    replay = Simulation(commons_run.cfg, tmp_path / "replay", replay_from=commons_run.run_dir / "llm_calls.jsonl", progress=False)
    replay.run()
    assert trace_digest(replay.run_dir) == trace_digest(commons_run.run_dir)
    assert (replay.run_dir / "commons_final.json").read_bytes() == (commons_run.run_dir / "commons_final.json").read_bytes()
    assert replay.llm.stats["cached"] == replay.llm.stats["calls"]


def test_factorial_configs_change_only_declared_factors():
    paths = ("commons", "commons_records_stable", "commons_no_records_change", "commons_no_records_stable")
    normalized, treatments = [], set()
    for name in paths:
        cfg = load_config(f"configs/{name}.yaml")
        treatments.add((cfg["commons"].pop("records_enabled"), cfg["commons"].pop("change_enabled")))
        cfg.pop("run_name")
        normalized.append(cfg)
    assert len(treatments) == 4
    assert all(cfg == normalized[0] for cfg in normalized)


@pytest.mark.parametrize("records,change", [(True, False), (False, True), (False, False)])
def test_remaining_factorial_cells_run_with_the_same_calendar(tmp_path, records, change):
    cfg = load_config("configs/commons_smoke.yaml", {"commons": {"records_enabled": records, "change_enabled": change}})
    sim = Simulation(cfg, tmp_path / "cell", progress=False)
    sim.run()
    result = analyze(sim.run_dir, verbose=False)
    assert result["summary"]["projects_released"] == 8
    assert {n["joined_tick"] for n in result["newcomers"]} == {16}
    assert result["phases"]["after_boundary"]["start_tick"] == 32
    assert result["design_executed"]["intervention_boundary_reached"]
    assert sim.commons.outdoor_condition == ("humid" if change else "dry")
    if not records:
        assert result["summary"]["record_reads"] == result["summary"]["record_versions"] == 0


def test_strict_replay_rejects_missing_wrapped_llm_response(commons_run, tmp_path):
    calls = rows(commons_run.run_dir / "llm_calls.jsonl")
    # GA's retry wrapper can catch exceptions. The runtime must still invalidate the run.
    dropped = next(i for i, call in enumerate(calls) if call["purpose"].startswith("poignancy"))
    calls.pop(dropped)
    incomplete = tmp_path / "incomplete.jsonl"
    incomplete.write_text("\n".join(json.dumps(c) for c in calls) + "\n", encoding="utf-8")
    sim = Simulation(commons_run.cfg, tmp_path / "missing", replay_from=incomplete, progress=False)
    with pytest.raises(RuntimeError):
        sim.run()
    manifest = json.loads((sim.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed" and sim.llm.stats["errors"] > 0


def test_provider_errors_fail_run_without_entering_agent_memory(tmp_path, monkeypatch):
    original = MockBackend.generate

    def failing(self, prompt, system, max_tokens, temperature):
        if current_purpose() == "commons_action":
            raise RuntimeError("TEST_PROVIDER_UNAVAILABLE")
        return original(self, prompt, system, max_tokens, temperature)
    monkeypatch.setattr(MockBackend, "generate", failing)
    sim = Simulation(load_config("configs/commons_smoke.yaml"), tmp_path / "failure", progress=False)
    with pytest.raises(RuntimeError):
        sim.run()
    assert json.loads((sim.run_dir / "manifest.json").read_text())["status"] == "failed"
    assert any(call.get("error") for call in rows(sim.run_dir / "llm_calls.jsonl"))
    for agent in sim.agents.values():
        assert all("TEST_PROVIDER_UNAVAILABLE" not in node.description and "LLM_ERROR" not in node.description
                   for node in agent.a_mem.all_nodes())
    with pytest.raises(ValueError, match="finished"):
        analyze(sim.run_dir, verbose=False)


def test_recording_is_not_overwritten(commons_run):
    before = (commons_run.run_dir / "trace.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        Simulation(commons_run.cfg, commons_run.run_dir, progress=False)
    assert (commons_run.run_dir / "trace.jsonl").read_bytes() == before


def test_api_supports_world_frames_and_hides_physical_answer(commons_run, monkeypatch):
    from backend.api import server
    monkeypatch.setattr(server, "RUNS", commons_run.run_dir.parent)
    server._load.cache_clear()
    with TestClient(server.app) as client:
        prefix = "/api/runs/" + commons_run.run_dir.name
        response = client.get(prefix + "/frames")
        assert response.status_code == 200
        assert "bench_mode" not in response.text
        assert response.json()[-1]["commons"]["records"]
        debug = client.get(prefix + "/frames?debug=1")
        assert debug.json()[-1]["commons"]["research"]["bench_mode"] in ("A", "B")
        result = client.get(prefix + "/analysis").json()
        assert result["mode"] == "commons" and result["summary"]["record_reads"] > 0
        assert client.get(prefix + "/agent/riley?tick=63").status_code == 200
