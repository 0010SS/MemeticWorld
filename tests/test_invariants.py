"""Scientific invariants: no meme state in agents, no hidden labels in prompts,
observer never imported by agent-side code."""
import dataclasses
import json
import re
from pathlib import Path

from backend.agents.profile import FORBIDDEN_FIELDS, AgentProfile
from backend.simulation.latent_events import LATENT_TYPES

ROOT = Path(__file__).resolve().parents[1]
AGENT_SIDE = ["backend/agents", "backend/memory", "backend/simulation", "backend/modules", "backend/prompts"]


def test_profile_has_no_cultural_state():
    names = {f.name for f in dataclasses.fields(AgentProfile)}
    assert not names & FORBIDDEN_FIELDS


def test_agent_objects_have_no_cultural_state(mock_run):
    for a in mock_run.agents.values():
        attrs = set(vars(a)) | set(vars(a.state)) | set(vars(a.scratch))
        assert not attrs & FORBIDDEN_FIELDS
        assert not any("meme" in x.lower() for x in attrs)


def test_prompts_never_contain_hidden_labels(mock_run):
    from backend.simulation.skins import SKINS
    bad = [r"\bE[0-4]\b", r"latent", r"\bmeme", r"slang", r"\binvent", r"\bev\d{3}\b", r"\bc_(lab|dorm)\b",
           r"\bregime\b", r"scrambled"]
    bad += [re.escape(s["key"]) for s in SKINS] + [re.escape(v["name"]) for v in LATENT_TYPES.values()]
    n = 0
    for line in open(mock_run.run_dir / "llm_calls.jsonl"):
        r = json.loads(line)
        n += 1
        for b in bad:
            assert not re.search(b, r["prompt"], re.I), (b, r["purpose"])
    assert n > 50


def test_agent_side_code_never_imports_observer():
    for d in AGENT_SIDE:
        for p in (ROOT / d).rglob("*.py"):
            src = p.read_text()
            assert "backend.analysis" not in src and "from backend import analysis" not in src, p


def test_agent_prompt_templates_do_not_ask_for_naming():
    for p in (ROOT / "backend/prompts").glob("*.txt"):
        if p.name.startswith("probe_"):
            continue  # observer probes, run after the simulation
        body = p.read_text().split("<commentblockmarker>###</commentblockmarker>")[-1].lower()
        for w in ("meme", "slang", "coin", "invent", "nickname", "new word", "name for"):
            assert w not in body, (p.name, w)


def test_memory_nodes_hold_no_event_ids(mock_run):
    for a in mock_run.agents.values():
        for n in a.a_mem.all_nodes():
            assert not re.search(r"\bev\d{3}\b", n.description)
            assert not hasattr(n, "originating_event_id")
