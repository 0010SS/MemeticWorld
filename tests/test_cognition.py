"""Perception, lossy memory, stochastic retrieval, latent events, modules."""
import numpy as np

from backend.agents.perception import observe
from backend.memory.retrieval import retrieve
from backend.simulation import latent_events as LE
from tests.conftest import run_sim


def _beat(agents):
    return {"tick": 5, "location": "Library", "arena": "Study Tables", "facts": [
        {"id": "x.f0", "text": "Someone dropped a stack of books.", "salience": 0.5, "visibility": "all", "kind": "action", "involves": []},
        {"id": "x.f1", "text": "Maya forwarded the wrong room number.", "salience": 0.3, "visibility": ["maya"], "kind": "cause", "involves": []},
        {"id": "x.f2", "text": "A librarian sighed loudly.", "salience": 0.2, "visibility": "all", "kind": "action", "involves": []},
    ]}


def test_partial_perception_differs_between_agents(mock_run):
    a, b = mock_run.agents["priya"], mock_run.agents["leo"]
    for ag in (a, b):
        ag.state.location, ag.state.arena, ag.state.in_conversation = "Library", "Study Tables", None
    seen_a, seen_b = set(), set()
    for s in range(40):
        oa = observe(a, _beat(None), "x", mock_run.cfg, np.random.default_rng(s), "o")
        ob = observe(b, _beat(None), "x", mock_run.cfg, np.random.default_rng(1000 + s), "o")
        seen_a.add(tuple(f["id"] for f in oa.facts) if oa else ())
        seen_b.add(tuple(f["id"] for f in ob.facts) if ob else ())
        # visibility-restricted fact is never perceived by non-privy agents
        assert not oa or "x.f1" not in [f["id"] for f in oa.facts]
    assert len(seen_a) > 1 and len(seen_b) > 1


def test_retrieval_is_stochastic_and_tau0_is_deterministic(mock_run):
    a = mock_run.agents["maya"]
    q = "something went wrong this morning"
    picks = {tuple(n.node_id for n in retrieve(a, [q], rng=np.random.default_rng(s), touch=False)[q].nodes)
             for s in range(15)}
    assert len(picks) > 1
    a.cfg = {**a.cfg, "retrieval": {**a.cfg["retrieval"], "temperature": 0.0}}
    det = {tuple(n.node_id for n in retrieve(a, [q], rng=np.random.default_rng(s), touch=False)[q].nodes)
           for s in range(5)}
    a.cfg = mock_run.cfg
    assert len(det) == 1


def test_lossy_vs_perfect_memory(tmp_path):
    lossy = run_sim(tmp_path, {"memory": {"encoding_noise": 0.6}}, "lossy")
    perfect = run_sim(tmp_path, {"memory": {"encoding_noise": 0.0}, "retrieval": {"temperature": 0.0}}, "perfect")
    import json
    enc = lambda s: [json.loads(l) for l in open(s.run_dir / "trace.jsonl") if '"memory_encoded"' in l]
    pm = [r for r in enc(perfect) if r["source_type"] == "perception"]
    assert pm and all(r["prompt"] is None for r in pm)          # verbatim, no LLM rewrite
    for r in pm:
        for fact in r["observation"].split("\n"):
            assert fact in r["text"]
    lm = [r for r in enc(lossy) if r["source_type"] == "perception"]
    assert lm and all(r["prompt"] for r in lm)


def test_same_event_different_memories(mock_run):
    import collections
    import json
    by_event = collections.defaultdict(dict)
    for l in open(mock_run.run_dir / "trace.jsonl"):
        r = json.loads(l)
        if r["type"] == "memory_encoded" and r["source_type"] == "perception":
            for e in r["originating_event_ids"]:
                by_event[e][r["agent"]] = r["text"]
    multi = [v for v in by_event.values() if len(v) >= 2]
    assert multi and any(len(set(v.values())) > 1 for v in multi)


def test_all_scenarios_instantiate_without_labels(mock_run):
    rng = np.random.default_rng(0)
    for fam in LE.LATENT_TYPES:
        assert LE.scenarios_for(fam, False) and LE.scenarios_for(fam, True)
    for scn in LE.SCENARIOS:
        inst = LE.instantiate(scn, "ev999", 2, rng, mock_run.agents, set(), 60)
        assert inst is not None
        for b in inst.beats:
            for f in b["facts"]:
                assert "{" not in f["text"]
                assert scn["family"] not in f["text"] and scn["key"] not in f["text"]


def test_modules_toggle_via_config_only(tmp_path):
    sim = run_sim(tmp_path, {"modules": {"emotion": True, "social_reward": True, "prestige_bias": True,
                                         "conformity": True}}, "mods")
    assert {m.name for m in sim.ctx.mods.modules} == {"emotion", "social_reward", "prestige_bias", "conformity"}
    import json
    frames = [json.loads(l) for l in open(sim.run_dir / "frames.jsonl")]
    assert "emotion" in frames[-1]["agents"]["maya"]["modules"]
    assert type(sim.agents["maya"]).__name__ == "Agent"


def test_replay_is_deterministic(tmp_path):
    import yaml
    from backend.simulation.engine import Simulation, trace_digest
    a = run_sim(tmp_path, None, "orig")
    cfg = yaml.safe_load(open(a.run_dir / "config.resolved.yaml"))
    b = Simulation(cfg, tmp_path / "replay", replay_from=a.run_dir / "llm_calls.jsonl", progress=False)
    b.run()
    assert trace_digest(a.run_dir) == trace_digest(b.run_dir)
    assert b.llm.stats["cached"] == b.llm.stats["calls"]


def test_memory_roundtrip_with_forgetting_gaps(tmp_path):
    import datetime as dt
    from backend.memory.store import MemoryStream
    ms = MemoryStream("ethan")
    t = dt.datetime(2026, 9, 14, 9)
    for i in range(60):
        ms.add("event", t, "Ethan", "saw", "x", f"memory number {i}", {"x"}, 3, [0.1] * 4)
    for i in range(1, 50, 2):
        ms.remove(f"ethan:m{i}")  # leave gaps, as forgetting does
    ms.save_ga(tmp_path / "mem")
    back = MemoryStream.load_ga("ethan", tmp_path / "mem")
    assert set(back.id_to_node) == set(ms.id_to_node)
    assert {n.node_id for n in back.all_nodes()} == set(ms.id_to_node)
