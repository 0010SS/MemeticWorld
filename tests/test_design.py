"""Experiment control (docs/ONTOLOGY_V2.md §5): design expansion, run status, tables, CLI."""
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import yaml

from backend import cli
from backend.config import ConfigKeyWarning, load_config, observer_spec
from backend.experiment import design as D

ROOT = Path(__file__).resolve().parents[1]
BOTTLENECK = ROOT / "configs/designs/bottleneck_factorial.yaml"
SMOKE = ROOT / "configs/designs/smoke_mock.yaml"


def _write(tmp: Path, name: str, spec: dict) -> Path:
    p = tmp / f"{name}.yaml"
    p.write_text(yaml.safe_dump({"name": name, **spec}, sort_keys=False))
    return p


# ------------------------------------------------------------------------------------------ expansion
def test_bottleneck_expands_to_full_factorial_plus_controls(tmp_path):
    d = D.load_design(BOTTLENECK, runs_root=tmp_path)
    cells = D.expand(d)
    ids = list(dict.fromkeys(c.cell_id for c in cells))
    assert len(ids) == 2 ** 4 + 2 and len(cells) == 18 * 3
    assert "need-on__link-off__wording-on__structure-real" in ids
    assert ids[-2:] == ["control-no_events", "control-planted"]
    assert [c.seed for c in cells[:18]] == [1] * 18            # seed-major: complete replicates first
    c = next(c for c in cells if c.cell_id == "need-on__link-off__wording-on__structure-scrambled" and c.seed == 2)
    assert c.levels == {"need": "on", "link": "off", "wording": "on", "structure": "scrambled"}
    assert c.run_dir == tmp_path / "bottleneck_factorial" / c.cell_id / "s2"
    cfg = c.config()
    assert cfg["seed"] == cfg["world_seed"] == 2 and cfg["simulation_days"] == 2
    assert cfg["_condition"]["design"] == "bottleneck_factorial" and cfg["_condition"]["cell"] == c.cell_id
    assert cfg["_condition"]["levels"] == c.levels and cfg["_condition"]["seed"] == 2
    assert cfg["analysis"]["observer"] == {"backend": "claude_cli", "model": "sonnet"}
    assert cfg["llm"] == {**load_config("configs/baseline.yaml")["llm"], "backend": "claude_cli", "model": "haiku",
                          "max_workers": 4}
    # NEED on, LINK off, WORDING on, scrambled structure; everything else stays at the baseline default
    le = cfg["latent_events"]
    assert le["referents"]["enabled"] and le["assignment"] == {"mode": "balanced", "strength": 0.7}
    assert cfg["conversation"]["catchup"]["enabled"] and cfg["conversation"]["group"]["enabled"]
    assert cfg["need"]["enabled"] and not cfg["reminding"]["enabled"] and le["link_visibility"] == 0.0
    assert cfg["memory"]["verbatim"]["enabled"] and cfg["priming"]["enabled"]
    assert cfg["retrieval"]["source_weights"] == {"seed": 0.5, "ambient": 0.5}
    assert le["structure"] == "scrambled" and le["event_rate"] == 0.12
    base = next(x for x in cells if x.cell_id == "need-off__link-off__wording-off__structure-real").config()
    assert not any([base["need"]["enabled"], base["reminding"]["enabled"], base["memory"]["verbatim"]["enabled"],
                    base["priming"]["enabled"], base["latent_events"]["referents"]["enabled"]])
    assert base["latent_events"]["assignment"]["mode"] == "none" and base["retrieval"]["source_weights"] == {}


def test_bottleneck_controls_and_common_random_numbers(tmp_path):
    cells = D.expand(D.load_design(BOTTLENECK, runs_root=tmp_path))
    by = {(c.cell_id, c.seed): c for c in cells}
    ne = by[("control-no_events", 1)].config()
    assert ne["latent_events"]["event_rate"] == 0 and ne["reminding"]["enabled"] and ne["need"]["enabled"]
    assert ne["_condition"]["control"] == "no_events"
    assert by[("control-no_events", 1)].levels == {"need": "on", "link": "on", "wording": "on", "structure": "real"}
    pl = by[("control-planted", 3)].config()
    assert pl["controls"]["planted_phrase"]["agent"] == "maya"
    assert '"a full pickle"' in pl["controls"]["planted_phrase"]["habit"]
    assert pl["latent_events"]["event_rate"] == 0.12
    # every cell of a seed gets the same seed and world seed; no factorial cell carries the planted habit
    for s in (1, 2, 3):
        assert {(c.config()["seed"], c.config()["world_seed"]) for c in cells if c.seed == s} == {(s, s)}
    assert all(c.config()["controls"]["planted_phrase"] is None for c in cells if c.control != "planted")


def test_backend_override_relocates_runs_and_switches_observer(tmp_path):
    d = D.load_design(BOTTLENECK, runs_root=tmp_path)
    c = D.expand(d, backend_override="mock")[0]
    cfg = c.config()
    assert cfg["llm"]["backend"] == "mock" and cfg["analysis"]["observer"]["backend"] == "mock"
    assert cfg["_condition"]["backend_override"] == "mock"
    assert c.run_dir.parts[-3] == "bottleneck_factorial__mock"
    s = D.load_design(SMOKE, runs_root=tmp_path)                 # already mock: same directory
    assert D.expand(s, backend_override="mock")[0].run_dir == D.expand(s)[0].run_dir


def test_smoke_mock_design(tmp_path):
    d = D.load_design(SMOKE, runs_root=tmp_path)
    cells = D.expand(d)
    assert [c.cell_id for c in cells] == ["link-off__wording-off", "link-off__wording-on", "link-on__wording-off",
                                          "link-on__wording-on", "control-no_events"]
    cfg = cells[-1].config()
    assert cfg["llm"]["backend"] == "mock" and cfg["analysis"]["observer"]["backend"] == "mock"
    assert cfg["simulation_days"] == 1 and cfg["day_end"] == "11:00" and cfg["latent_events"]["event_rate"] == 0
    assert cells[-1].levels == {"link": None, "wording": None}    # plain control: base + overlay only
    assert not cfg["reminding"]["enabled"] and not cfg["memory"]["verbatim"]["enabled"]


def test_runs_root_precedence(tmp_path, monkeypatch):
    p = _write(tmp_path, "rr", {"seeds": [1], "runs_root": str(tmp_path / "from_file")})
    assert D.load_design(p).runs_root == tmp_path / "from_file"
    monkeypatch.setenv("MEMEWORLD_RUNS_ROOT", str(tmp_path / "from_env"))
    assert D.load_design(p).runs_root == tmp_path / "from_env"
    assert D.load_design(p, runs_root=tmp_path / "arg").runs_root == tmp_path / "arg"
    assert [c.cell_id for c in D.expand(D.load_design(p))] == ["base"]      # no factors: one base cell


@pytest.mark.parametrize("spec, msg", [
    ({"seeds": [1], "factor": {}}, "unknown design keys"),
    ({"seeds": []}, "seeds"),
    ({"seeds": [1], "factors": {"a": {"x": {"reminding": {"enabled": True}}},
                                "b": {"y": {"reminding": {"enabled": False}}}}}, "disjoint"),
    ({"seeds": [1], "factors": {"a": {"x": {"retrieval": {"source_weights": {}}}},
                                "b": {"y": {"retrieval": {"source_weights": {"seed": 0.5}}}}}}, "disjoint"),
    ({"seeds": [1], "factors": {"bad-name": {"x": {}}}}, "single underscores"),
    ({"seeds": [1], "factors": {"a": {"x": {"seed": 3}}}}, "set by the design runner"),
    ({"seeds": [1], "factors": {"a": {"x": {}}}, "controls": {"c": {"levels": {"a": "z"}}}}, "unknown level"),
    ({"seeds": [1], "factors": {"seed": {"x": {}}}}, "reserved"),
    ({"seeds": [1], "observer": {"backend": "mock", "temperature": 0}}, "observer"),
])
def test_invalid_designs_are_rejected(tmp_path, spec, msg):
    with pytest.raises(ValueError, match=msg):
        D.load_design(_write(tmp_path, "bad", spec))


def test_yaml_boolean_level_names_become_on_off(tmp_path):
    p = tmp_path / "b.yaml"
    p.write_text("name: b\nseeds: [1]\nfactors:\n  link: {off: {}, on: {reminding: {enabled: true}}}\n"
                 "controls:\n  all: {levels: {link: on}}\n")
    cells = D.expand(D.load_design(p))
    assert [c.cell_id for c in cells] == ["link-off", "link-on", "control-all"]
    assert cells[2].config()["reminding"]["enabled"]


def test_unknown_overlay_keys_warn(tmp_path):
    p = _write(tmp_path, "typo", {"seeds": [1], "factors": {"a": {"x": {"latent_events": {"holdout_from_day": 2}}}}})
    with pytest.warns(ConfigKeyWarning, match="holdout_frac"):
        D.load_design(p)
    with pytest.warns(ConfigKeyWarning, match="memory.verbatim.enabeld"):
        load_config("configs/baseline.yaml", {"memory": {"verbatim": {"enabeld": True}}})
    with warnings.catch_warnings():                     # open maps and private keys are fine
        warnings.simplefilter("error")
        load_config("configs/baseline.yaml", {"retrieval": {"source_weights": {"seed": 0.5}}, "_condition": {"a": 1}})
        for f in ROOT.glob("configs/*.yaml"):
            load_config(f)
    with pytest.warns(ConfigKeyWarning, match="dropped in v2"):     # world_script.generate raises on it
        load_config(None, {"latent_events": {"generator": "llm"}})


def test_observer_spec():
    cfg = {"analysis": {"observer": {"backend": "claude_cli", "model": "sonnet"}}}
    assert observer_spec(cfg) == ("claude_cli", "sonnet")
    assert observer_spec(cfg, "mock") == ("mock", "sonnet")
    assert observer_spec({}, None, None) == (None, None)


def test_cli_analyze_defaults_to_config_observer(tmp_path, monkeypatch):
    import backend.analysis.pipeline as P
    seen = {}
    monkeypatch.setattr(P, "analyze", lambda run_dir, **kw: seen.update(kw))
    (tmp_path / "config.resolved.yaml").write_text(yaml.safe_dump(
        {"analysis": {"observer": {"backend": "claude_cli", "model": "sonnet"}}}))
    cli.main(["analyze", str(tmp_path)])
    assert seen["llm_backend"] == "claude_cli" and seen["llm_model"] == "sonnet"
    cli.main(["analyze", str(tmp_path), "--backend", "mock", "--no-probes"])
    assert seen["llm_backend"] == "mock" and seen["llm_model"] == "sonnet" and seen["probes"] is False


# ------------------------------------------------------------------------------------ status + tables
def _fake_run(cell, manifest_status=None, side=None, outcomes=None):
    cell.run_dir.mkdir(parents=True, exist_ok=True)
    if manifest_status:
        (cell.run_dir / "manifest.json").write_text(json.dumps({"status": manifest_status}))
    if side is not None:
        (cell.run_dir / D.SIDECAR).write_text(json.dumps({"host": D.socket.gethostname(), **side}))
    if outcomes is not None:
        (cell.run_dir / "outcomes.json").write_text(json.dumps(outcomes))


def test_cell_status_and_prepare(tmp_path):
    cells = D.expand(D.load_design(SMOKE, runs_root=tmp_path))
    a, b, c, d, e = cells
    dead = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True,
                          text=True).stdout.strip()
    assert D.cell_status(a) == "missing"
    _fake_run(a, "running", {"pid": os.getpid()})
    assert D.cell_status(a) == "running"
    _fake_run(b, "running", {"pid": int(dead)})
    assert D.cell_status(b) == "failed"
    _fake_run(c, "running", {"pid": os.getpid(), "error": "RuntimeError: boom", "ended": "x"})
    assert D.cell_status(c) == "failed"
    _fake_run(d, "finished", {"pid": os.getpid(), "ended": "x"})
    assert D.cell_status(d) == "finished"
    _fake_run(d, outcomes={"observer": {"backend": "mock"}})
    assert D.cell_status(d) == "analyzed"
    _fake_run(e, "finished", outcomes={"observer": {"backend": "claude_cli", "model": "sonnet"}})
    assert D.cell_status(e) == "stale"                   # analyzed by another observer -> re-analyze
    assert D.prepare(c) == "missing" and not c.run_dir.exists()
    assert (c.run_dir.parent / "s1.failed1" / D.SIDECAR).exists()
    counts = [r["status"] for r in D.status(D.load_design(SMOKE, runs_root=tmp_path))]
    assert counts == ["running", "failed", "missing", "analyzed", "stale"]


def test_table_summary_markdown_csv(tmp_path):
    d = D.load_design(SMOKE, runs_root=tmp_path)
    d.seeds = [1, 2]
    cells = D.expand(d)
    for c in cells:
        if c.cell_id == "link-on__wording-on":
            _fake_run(c, "finished", outcomes={"observer": {"backend": "mock", "model": None},
                                               "n_candidates": 5 + c.seed, "n_emerged": c.seed, "n_grounded": 0,
                                               "max_emerged_adoption": 0.5, "emerged": ["a"] * c.seed,
                                               "validity": {"reminding": "inactive", "verbatim": "active"}})
    rows = D.table(d)
    assert len(rows) == 2
    r = rows[0]
    assert r["cell"] == "link-on__wording-on" and r["link"] == "on" and r["wording"] == "on" and r["seed"] == 1
    assert r["n_candidates"] == 6 and r["n_emerged"] == 1 and r["inactive"] == "reminding"
    summ = {s["cell"]: s for s in D.summarize(d, rows)}
    assert summ["link-on__wording-on"]["n"] == 2 and summ["link-off__wording-off"]["n"] == 0
    mean, sd, n = summ["link-on__wording-on"]["n_emerged"]
    assert (mean, round(sd, 3), n) == (1.5, 0.707, 2)
    md = D.markdown(d, rows)
    assert "1.50 ± 0.71" in md and "| link-off__wording-off | off | off | 0 |" in md
    D.write_csv(d, rows, tmp_path / "t.csv")
    lines = (tmp_path / "t.csv").read_text().splitlines()
    assert lines[0].startswith("cell,control,link,wording,seed,n_candidates") and len(lines) == 3


def test_experiment_controller_is_not_imported_by_agent_side_code():
    for sub in ["backend/agents", "backend/memory", "backend/simulation", "backend/modules", "backend/analysis"]:
        for f in (ROOT / sub).rglob("*.py"):
            assert "backend.experiment" not in f.read_text(), f


# ----------------------------------------------------------------------------------------- end to end
def _cli(*args, env=None):
    return subprocess.run([sys.executable, "-m", "backend.cli", *args], cwd=ROOT, capture_output=True, text=True,
                          env={**os.environ, **(env or {})}, timeout=600)


def test_cli_expand_and_status_on_smoke_mock(tmp_path):
    env = {"MEMEWORLD_RUNS_ROOT": str(tmp_path)}
    out = _cli("design", "expand", str(SMOKE), env=env)
    assert out.returncode == 0, out.stderr
    assert "5 cells x 1 seeds = 5 runs" in out.stdout and "control-no_events" in out.stdout
    js = json.loads(_cli("design", "expand", str(SMOKE), "--json", env=env).stdout)
    assert js[0]["overrides"]["_condition"]["cell"] == "link-off__wording-off"
    out = _cli("design", "status", str(SMOKE), env=env)
    assert out.returncode == 0 and "missing 5 (of 5 runs)" in out.stdout
    out = _cli("design", "run", str(SMOKE), "--dry-run", "--only", "control", env=env)
    assert "control-no_events s1: missing -> simulate+analyze" in out.stdout and not list(tmp_path.iterdir())


def test_design_run_mock_end_to_end(tmp_path):
    """`design run --backend mock` into a tmp runs root, then `design table`. Probes are switched off in a
    copy of smoke_mock: this test checks the controller, and the observer's own tests cover probes."""
    spec = yaml.safe_load(SMOKE.read_text())
    spec["common"]["analysis"] = {"probes": False}
    p = tmp_path / "smoke_mock.yaml"
    p.write_text(yaml.safe_dump(spec, sort_keys=False))
    root = tmp_path / "runs"
    out = _cli("design", "run", str(p), "--backend", "mock", "--parallel", "3", "--runs-root", str(root))
    logs = "\n".join(f.read_text()[-3000:] for f in root.rglob(D.LOG))
    assert out.returncode == 0, out.stdout + out.stderr + logs
    d = D.load_design(p, runs_root=root)
    cells = D.expand(d, backend_override="mock")
    for c in cells:
        man = json.loads((c.run_dir / "manifest.json").read_text())
        assert man["status"] == "finished" and man["config"]["_condition"]["cell"] == c.cell_id
        assert man["config"]["seed"] == man["config"]["world_seed"] == 1
        assert (c.run_dir / "analysis.json").exists(), "analysis did not run"
        assert json.loads((c.run_dir / D.SIDECAR).read_text())["stage"] == "done"
    assert {r["status"] for r in D.status(d, "mock")} <= {"finished", "analyzed"}
    # second invocation only re-analyzes what has no outcomes.json yet; nothing is simulated again
    mtime = (cells[0].run_dir / "trace.jsonl").stat().st_mtime
    _cli("design", "run", str(p), "--backend", "mock", "--only", cells[0].cell_id, "--runs-root", str(root))
    assert (cells[0].run_dir / "trace.jsonl").stat().st_mtime == mtime
    out = _cli("design", "table", str(p), "--backend", "mock", "--runs-root", str(root), "--csv",
               str(tmp_path / "t.csv"))
    assert out.returncode == 0, out.stderr
    if not all((c.run_dir / "outcomes.json").exists() for c in cells):
        pytest.skip("observer does not write outcomes.json yet (backend/analysis/outcomes.py)")
    assert all(r["status"] == "analyzed" for r in D.status(d, "mock"))
    rows = D.table(d, "mock")
    assert len(rows) == 5 and all(isinstance(r["n_candidates"], int) for r in rows)
    assert "control-no_events" in out.stdout and "5 analyzed runs" in out.stdout
