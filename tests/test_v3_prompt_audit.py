"""Rendered-prompt audit over a mock v3 run of the real engine (ontology v3 §8.1 step 10, §5.11 item 13).

Every prompt in the simulation's llm_calls.jsonl must be free of hidden ids (K0-K3, causes, mappings), job ids,
node / arm names, `regime` / `mapping`, content-pack ids, E1-E4 and uniforms; no observer purpose or scope may
ever appear in a simulation cache; every world fact released to agents is free of the same ids.

Skipped until engine.py wires CoopWorld (tests/test_v3_invariants.py runs the same audit through a harness).
"""
from __future__ import annotations

import json
import re

import pytest

from tests.conftest import ROOT
from tests.test_v3_invariants import V3_DAY, audit_text, engine_wired

pytestmark = pytest.mark.skipif(not engine_wired(), reason="CoopWorld is not wired into engine.py yet")

OBSERVER_SCOPE = re.compile(r"^(probe|dcf|retest|synth|fresh|code):")
OBSERVER_PURPOSE = re.compile(r"^(probe_|code_)")
TWO_DAYS = {**V3_DAY, "run_name": "coop_audit", "simulation_days": 2,
            "records": {"enabled": True, "transitions": [{"day": 2, "mode": "wipe"}]},
            "regimes": {"schedule": [{"day": 1, "regime": "A"}, {"day": 2, "regime": "B"}],
                        "cues": [{"day": 1, "time": "16:45", "arena": "Stockroom", "text_key": "new_supplier"}]},
            "comm": {"clarify": {"enabled": True}, "handover": {"enabled": True},
                     "meeting": {"enabled": True, "days": [1, 2]}}}


@pytest.fixture(scope="module")
def audit_run(tmp_path_factory):
    from backend.config import deep_merge, load_config
    from backend.simulation.engine import Simulation
    base = "configs/v3_base.yaml" if (ROOT / "configs" / "v3_base.yaml").exists() else "configs/baseline.yaml"
    over = dict(TWO_DAYS)
    if base.endswith("v3_base.yaml"):
        over.pop("records")                       # keep v3_base's binder settings; only shorten the run
        over["turnover"] = {"waves": []}
    cfg = load_config(base, deep_merge({}, over))
    run_dir = tmp_path_factory.mktemp("audit") / "run"
    Simulation(cfg, run_dir, progress=False).run()
    return run_dir


def test_every_rendered_prompt_is_free_of_hidden_ids(audit_run):
    bad, n, purposes = [], 0, set()
    for line in open(audit_run / "llm_calls.jsonl"):
        r = json.loads(line)
        n += 1
        purposes.add(r.get("purpose"))
        hits = audit_text(r.get("prompt") or "")
        if hits:
            bad.append((r.get("purpose"), hits))
    assert n > 0 and "job_decision" in purposes
    assert not bad, bad[:10]


def test_no_observer_call_in_the_simulation_cache(audit_run):
    for line in open(audit_run / "llm_calls.jsonl"):
        r = json.loads(line)
        assert not OBSERVER_SCOPE.match(str(r.get("scope") or "")), r.get("scope")
        assert not OBSERVER_PURPOSE.match(str(r.get("purpose") or "")), r.get("purpose")


def test_world_facts_and_binder_views_are_free_of_hidden_ids(audit_run):
    bad = []
    for line in open(audit_run / "trace.jsonl"):
        r = json.loads(line)
        texts = []
        if r["type"] == "event_beat":
            texts = [f["text"] for f in r.get("facts") or []]
        elif r["type"] in ("observation", "memory_encoded", "utterance"):
            texts = [r.get("text") or ""] + [f.get("text", "") for f in r.get("facts") or [] if isinstance(f, dict)]
        for t in texts:
            hits = [h for h in audit_text(t) if "regime" not in h and "mapping" not in h]   # ordinary English
            if hits:
                bad.append((r["type"], hits, t[:120]))
    assert not bad, bad[:10]
