"""End-to-end software and evidence-contract checks; no claims about live LLM accuracy."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from backend.config import load_config
from backend.research.common import digest, read_json, write_json, write_jsonl
from backend.research.evidence import Evidence, packets, precedes
from backend.research.observer import apply_judgment, observe


@pytest.fixture
def society(tmp_path):
    run = tmp_path / "society"
    run.mkdir()
    cfg = load_config("configs/memetics.yaml", {"llm": {"backend": "mock"}, "analysis": {"observer": {"backend": "mock"}}})
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    write_json(run / "manifest.json", {"status": "finished", "ticks": 21, "ticks_per_day": 10,
        "agents": {a: {"background": "Initial role", "habits": ""} for a in ("ada", "ben", "cy")},
        "groups": {"crew": ["ada", "ben"], "new": ["cy"]}})
    rows = [
        {"id": "u0", "type": "utterance", "tick": 0, "speaker": "ada", "listeners": ["ben"], "conversation_id": "c1", "idx": 0, "text": "Keep the blue door open: let anyone ask for help."},
        {"id": "u1", "type": "utterance", "tick": 0, "speaker": "ben", "listeners": ["ada"], "conversation_id": "c1", "idx": 1, "text": "Leave room for everyone to ask for help."},
        {"id": "m1", "type": "memory_encoded", "tick": 1, "agent": "ben", "text": "The blue door means including others."},
        {"id": "r1", "type": "record_write", "tick": 4, "agent": "ben", "rev_id": "rev1", "text": "The blue door protects people who report mistakes."},
        {"id": "read1", "type": "record_read", "tick": 5, "agent": "cy", "rev_ids": ["rev1"]},
        {"id": "u2", "type": "utterance", "tick": 10, "speaker": "cy", "listeners": ["ben"], "conversation_id": "c2", "idx": 0, "text": "We need a blue door: nobody should be punished for reporting a mistake."},
        {"id": "u3", "type": "utterance", "tick": 20, "speaker": "ben", "listeners": ["cy"], "conversation_id": "c3", "idx": 0, "text": "I reject using blue door to promise immunity for every mistake."},
        {"id": "hidden1", "type": "job_truth", "tick": 10, "text": "SECRET HIDDEN WORLD CAUSE"},
        {"id": "future", "type": "utterance", "tick": 25, "speaker": "ada", "text": "Uncommitted future statement"},
    ]
    write_jsonl(run / "trace.jsonl", rows)
    write_jsonl(run / "digests.jsonl", [{"tick": t, "sha256": str(t), "n": 1} for t in range(21)])
    write_jsonl(run / "frames.jsonl", [{"tick": t, "agents": {a: {"active": a != "cy" or t >= 5} for a in ("ada", "ben", "cy")}} for t in range(21)])
    return run


def judgment(events):
    ids = {e["id"]: e for e in events}
    meanings = {"u0": "open access to help", "u1": "open access to help", "m1": "open access to help",
                "r1": "protection for reporting mistakes", "u2": "protection for reporting mistakes", "u3": "unconditional immunity"}
    return {"threads": [{"id": None, "key": "door", "label": "The blue door", "description": "A revisable image of access and protection.",
        "occurrences": [{"event_id": eid, "quote": ids[eid]["text"], "interpretation": meaning, "sense": meaning,
                          "stance": "rejects" if eid == "u3" else "supports", "function": "evaluation", "uncertainty": "Synthetic fixture"}
                         for eid, meaning in meanings.items() if eid in ids]}], "relationships": [],
        "changes": [{"thread": "door", "before": ["u0"], "after": ["u2"], "kind": "expanded implications",
                     "description": "Access develops into protection from punishment.", "uncertainty": "Possible related uses rather than a replacement."}],
        "observations": [{"description": "A later speaker rejects an expansive reading.", "evidence_ids": ["u3"], "uncertainty": "Disagreement, not consensus."}]}


class ScriptedJudge:
    name = "scripted_fixture"
    model = "fixture"

    def __init__(self):
        self.calls = []

    def generate(self, prompt, system, max_tokens, temperature):
        payload = json.loads(prompt.split("EVIDENCE_JSON\n", 1)[1])
        self.calls.append(payload)
        if "MEMETIC_DISCOVERY_V1" in prompt:
            return json.dumps(judgment(payload["events"]))
        if "MEMETIC_RETRIEVAL_V1" in prompt:
            return json.dumps({"searches": ["punished reporting mistake"], "thread_ids": [], "rationale": "Search reinterpretations"})
        if "MEMETIC_AUDIT_V1" in prompt:
            return json.dumps({"verdict": "supported", "reason": "The fixture explicitly expresses this interpretation."})
        ids = [e["id"] for e in payload["evidence"]]
        return json.dumps({"answer": "The proposed idea had multiple expressed readings.", "claims": [{"description": "The supplied evidence includes an interpretation of the blue door.", "evidence_ids": ids[:1], "uncertainty": "Fixture only"}], "followups": ["Did rejection affect later usage?"]})


def test_committed_evidence_origin_and_privacy(society):
    e = Evidence(society)
    meta = e.build()
    assert meta["through_tick"] == 20
    by = {r["id"]: r for r in e.events()}
    assert "future" not in by
    assert by["hidden1"]["channel"] == "hidden" and not by["hidden1"]["text"]
    assert by["m1"]["channel"] == "private"
    assert by["initial:cy"]["tick"] == 5
    assert {"source": "r1", "actor": "cy", "tick": 5, "via": "read"} in e.exposures()
    assert precedes(by["u0"], by["u1"])
    assert not precedes(by["u0"], {**by["u1"], "conversation": "parallel"})
    with (society / "trace.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"unfinished":')
    assert e.build()["fingerprint"] == meta["fingerprint"]


def test_discovery_metrics_reports_and_no_feedback(society):
    before = {p.name: digest(p.read_bytes()) for p in society.iterdir() if p.is_file()}
    backend = ScriptedJudge()
    result = observe(society, judge_backend=backend)
    assert result["status"] == "complete" and len(result["threads"]) == 1
    assert result["coverage"]["processed_events"] == result["coverage"]["eligible_events"]
    assert "SECRET" not in json.dumps(backend.calls)
    t = result["threads"][0]
    m = result["measurements"]["threads"][t["id"]]
    assert m["public_users"] == 3 and m["private_occurrences"] == 1
    assert m["senses"] == 3 and len(m["changes"]) == 1
    assert m["uptake"]["ben"]["prior_exposure"] and m["uptake"]["cy"]["later_exchange"]
    assert any(e["via"] == "read" for e in m["possible_transmission"])
    assert m["stances"]["rejects"] == 1
    out = society / "analyses" / result["analysis_id"]
    assert all((out / f).exists() for f in ("report.html", "report.md", "occurrences.csv", "evidence.json", "figures/public_use.svg"))
    assert observe(society, judge_backend=backend)["analysis_id"] == result["analysis_id"]
    assert len(backend.calls) == 1
    assert {name: digest((society / name).read_bytes()) for name in before} == before


def test_bad_quotes_and_future_evidence_do_not_commit(society):
    e = Evidence(society)
    e.build()
    rows = e.events(text_only=True)
    answer = judgment(rows)
    answer["threads"][0]["occurrences"][0]["quote"] = "Fabricated quote"
    registry, revisions = {}, []
    with pytest.raises(ValueError, match="non-verbatim"):
        apply_judgment(answer, registry, {r["id"]: r for r in rows}, window_id="w", revision_log=revisions)
    assert registry == {} and revisions == []
    answer = judgment(rows)
    answer["changes"][0]["after"] = ["future"]
    with pytest.raises(ValueError, match="supplied evidence"):
        apply_judgment(answer, registry, {r["id"]: r for r in rows}, window_id="w", revision_log=revisions)
    assert registry == {}


def test_inquiry_retrieval_scope_and_saved_audit(society):
    from backend.research.inquiries import inquire
    from backend.research.audit import audit
    judge = ScriptedJudge()
    observe(society, judge_backend=judge)
    result = inquire(society, "Initial role", limit=3, judge_backend=judge)
    assert result["plan"]["supplied_events"] == 3
    assert "u2" in result["plan"]["supplied_ids"]  # expansion survives even when first search fills budget
    assert (society / "inquiries" / result["id"] / "report.html").exists()
    scoped = inquire(society, "door", actor="cy", start=10, end=10, judge_backend=judge)
    assert scoped["plan"]["supplied_ids"] == ["u2"]
    checked = audit(society, sample=3, seed=7, judge_backend=judge)
    assert checked["reviewed"] == 3 and checked["counts"]["supported"] == 3
    assert audit(society, sample=3, seed=7, judge_backend=judge) == checked


def test_mock_is_explicitly_not_a_semantic_result(society):
    result = observe(society, backend="mock")
    assert result["synthetic"] and result["threads"] == []
    assert result["summary"]["mean_agent_agreement"] is None
    assert "No semantic LLM judgments" in (society / "analyses" / result["analysis_id"] / "report.html").read_text(encoding="utf-8")


def test_all_study_presets_and_independent_agent_replicas(tmp_path):
    from backend.research.experiments import catalog, resolve
    from backend.experiment.design import expand
    assert len(catalog()) == 11
    for item in catalog():
        d = resolve(item["id"], tmp_path)
        cells = expand(d)
        assert d.questions and len({c.run_dir for c in cells}) == len(cells)
        for cell in cells:
            cfg = cell.config()
            assert cfg["analysis"]["pipeline"] == "memetics"
            assert cfg["world_seed"] == cell.seed
    cells = expand(resolve("contingency", tmp_path))
    group = [c for c in cells if c.seed == 11 and c.cell_id == cells[0].cell_id]
    assert len({c.config()["seed"] for c in group}) == 2
    assert len({c.config()["world_seed"] for c in group}) == 1


def test_api_evidence_exports_and_path_confinement(society, monkeypatch):
    from backend.api import server
    observe(society, judge_backend=ScriptedJudge())
    monkeypatch.setattr(server, "RUNS", society.parent)
    client = TestClient(server.app)
    assert len(client.get("/api/research/experiments").json()) == 11
    analysis = client.get("/api/research/analysis", params={"run": "society"}).json()
    assert analysis["threads"][0]["occurrences"][0]["event_id"] == "u0"
    assert client.get("/api/research/evidence", params={"run": "society", "event_id": "hidden1"}).json() == []
    assert client.get("/api/research/evidence", params={"run": "society", "event_id": "u0"}).json()[0]["text"].startswith("Keep")
    report = f"society/analyses/{analysis['analysis_id']}/report.html"
    assert client.get("/api/research/files/" + report).status_code == 200
    assert client.get("/api/research/artifact", params={"path": "../README.md"}).status_code == 404
    assert client.post("/api/research/jobs", json={"action": "inquire", "run": "society", "question": ""}).status_code == 400
    assert client.get("/api/compare").status_code == 200


def test_real_engine_through_full_pipeline_and_continuation(tmp_path):
    from backend.experiment.design import load_design, expand
    from backend.research.experiments import run
    from backend.experiment.execution import continue_run
    from backend.tracing.logger import compare_digests
    design_path = tmp_path / "fixture.yaml"
    design_path.write_text(yaml.safe_dump({"name": "pipeline_fixture", "base": "configs/memetics.yaml", "seeds": [11], "days": 1,
        "common": {"llm": {"backend": "mock"}}, "observer": {"backend": "mock", "model": "sonnet"},
        "questions": ["What ideas developed?"], "outcomes": ["memetics.n_threads", "memetics.n_changes"]}), encoding="utf-8")
    design = load_design(design_path, tmp_path / "runs")
    result = run(design, parallel=1)
    assert result["status"] == "complete", result
    assert (Path(result["report_dir"]) / "report.html").exists()
    cell = expand(design)[0]
    analysis = read_json(cell.run_dir / "analysis.json")
    assert analysis["synthetic"] and analysis["coverage"]["eligible_events"] > 0
    assert list((cell.run_dir / "inquiries").glob("*/report.html"))
    child = continue_run(cell.run_dir, tmp_path / "continued", days=2, progress=False)
    manifest = read_json(child / "manifest.json")
    assert manifest["status"] == "finished" and manifest["continuation"]["prefix_verified"]
    assert not compare_digests(cell.run_dir, child, read_json(cell.run_dir / "manifest.json")["ticks"])


def test_charts_leave_unobserved_interpretations_missing():
    from backend.research.reports import chart
    svg = chart([("sense", [(1, 1), (2, None), (3, .5)])], "Missing observations")
    assert "nan" not in svg and svg.count("M") >= 3


def test_windows_cover_large_event_fragments():
    rows = [{"id": "long", "text": "x" * 10000, "tick": 0, "channel": "speech"}]
    groups = list(packets(rows, max_chars=2000, overlap=1))
    assert len(groups) > 1
    assert sum(len(e["text"]) for group, _ in groups for e in group) >= 10000
