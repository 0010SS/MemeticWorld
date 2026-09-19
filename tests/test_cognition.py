"""Perception, lossy memory, stochastic retrieval, latent events, modules."""
import numpy as np
import pytest

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


def test_world_events_carry_no_labels_in_agent_visible_text(mock_run):
    import json
    for l in open(mock_run.run_dir / "events.jsonl"):
        e = json.loads(l)
        for b in e["beats"]:
            for f in b["facts"]:
                assert "{" not in f["text"]
                assert e["latent_type"] not in f["text"] and (e.get("skin") or "@@") not in f["text"]


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


def test_viewpoints_differ_and_hide_unknown_names(mock_run):
    import json
    vps = [json.loads(l) for l in open(mock_run.run_dir / "trace.jsonl") if '"type": "viewpoint"' in l]
    assert vps
    vantages = {f["vantage"] for r in vps for f in r["facts"]}
    assert "participant" in vantages and len(vantages) >= 2
    for r in vps:
        me = mock_run.agents[r["agent"]]
        for f in r["facts"]:
            for other, a in mock_run.agents.items():
                if other != me.id and me.profile.rel(other).familiarity < 0.3:
                    assert a.profile.first_name.lower() not in f["perceived"].lower().split()


def test_participants_remember_their_own_experience(mock_run):
    import json
    enc = [json.loads(l) for l in open(mock_run.run_dir / "trace.jsonl") if '"memory_encoded"' in l]
    own = [r for r in enc if r.get("self_experience")]
    assert own and all("experienced first-hand" in r["prompt"] for r in own if r["prompt"])


def test_memory_lens_is_random_but_seeded():
    import numpy as np
    from backend.memory.encoder import sample_lens
    lenses = [sample_lens(0.3, 1.0, "Maya", np.random.default_rng(s))["text"] for s in range(40)]
    assert len(set(lenses)) >= 15
    assert sample_lens(0.3, 1.0, "Maya", np.random.default_rng(7)) == sample_lens(0.3, 1.0, "Maya", np.random.default_rng(7))
    assert sample_lens(0.3, 0.0, "Maya", np.random.default_rng(7))["text"] == ""


def test_home_assignment_casts_from_circles_and_is_hidden(tmp_path):
    import json
    sim = run_sim(tmp_path, {"simulation_days": 2, "day_end": "22:30",
                             "latent_events": {"event_rate": 0.4, "assignment": {"mode": "home", "strength": 1.0}}}, "clu")
    evs = [json.loads(l) for l in open(sim.run_dir / "events.jsonl")]
    assert evs and all("cast_from_home" in e and "circle" in e for e in evs)
    home = [e for e in evs if e["cast_from_home"]]
    assert len(home) / len(evs) >= 0.6                  # busy circle members fall back to anyone
    for e in home:
        assert e["roles"]["P"]["agent"] in sim.circles[e["circle"]]
    man = json.load(open(sim.run_dir / "manifest.json"))
    from backend.api.server import _strip
    stripped = json.dumps(_strip(man))
    # These are hidden metadata keys; ordinary vocabulary can include "assignment".
    assert '"circles":' not in stripped and '"assignment":' not in stripped


def test_catchup_conversations_between_close_friends(tmp_path):
    import json
    sim = run_sim(tmp_path, {"day_end": "21:00", "conversation": {"catchup": {"enabled": True}}}, "cu")
    convs = [json.loads(l) for l in open(sim.run_dir / "trace.jsonl") if '"type": "conversation"' in l]
    cu = [c for c in convs if (c.get("trigger") or {}).get("topic") == "catchup"]
    assert cu
    for c in cu:
        a, b = c["participants"]
        assert sim.agents[a].profile.rel(b).familiarity >= 0.6


def test_reminding_links_two_of_the_agents_own_memories(tmp_path):
    import json
    sim = run_sim(tmp_path, {"reminding": {"enabled": True}}, "rem")
    rem = [json.loads(l) for l in open(sim.run_dir / "trace.jsonl") if '"type": "reminding"' in l]
    hits = [r for r in rem if r["node_id"]]
    assert rem and hits
    for r in rem:
        assert all(c.startswith(r["agent"] + ":") for c in r["candidates"])
        for bad in ("E1", "E2", "E3", "E4", "cascade", "coincidence", "latent"):
            assert bad not in r["prompt"]


def test_distinctive_wording_sticks_verbatim(mock_run):
    import numpy as np
    from backend.memory.encoder import distinctive_phrases, sticky_phrases
    a = mock_run.agents["leo"]
    a.cfg = {**mock_run.cfg, "memory": {**mock_run.cfg["memory"], "verbatim": {**mock_run.cfg["memory"]["verbatim"], "enabled": True}}}
    ph = [p for p, _ in distinctive_phrases(a, "Maya said it was the whole zamboni situation again", 3.6)]
    assert any("zamboni" in p.lower() for p in ph) and not any("maya" in p.lower() for p in ph)
    facts = [{"text": 'Maya: "the zamboni situation again"'}]
    hits = sum(bool(sticky_phrases(a, facts, np.random.default_rng(s))) for s in range(200))
    assert 30 < hits < 110                               # base 0.3 per phrase, <= 2 phrases


def test_source_weights_demote_seed_memories(mock_run):
    import numpy as np
    from backend.memory.retrieval import retrieve
    a = mock_run.agents["maya"]
    q = "Priya Raman"
    def seed_share(w):
        a.cfg = {**mock_run.cfg, "retrieval": {**mock_run.cfg["retrieval"], "source_weights": w}}
        n = tot = 0
        for s in range(30):
            for nd in retrieve(a, [q], rng=np.random.default_rng(s), touch=False)[q].nodes:
                tot += 1
                n += a.ctx.meta.get(nd.node_id).source_type == "seed"
        return n / tot
    try:
        assert seed_share({"seed": 0.3}) < seed_share({})
    finally:
        a.cfg = mock_run.cfg


def test_perception_takes_one_draw_per_fact_whatever_its_visibility(mock_run):
    """A private fact made public (link_visibility) must not change which OTHER facts a bystander notices."""
    import copy
    a = mock_run.agents["priya"]
    a.state.location, a.state.arena, a.state.in_conversation = "Library", "Study Tables", None
    private = _beat(None)
    private["facts"].insert(0, private["facts"].pop(1))            # the private fact comes first
    public = copy.deepcopy(private)
    public["facts"][0]["visibility"] = "all"
    for s in range(40):
        ids = []
        for beat in (private, public):
            o = observe(a, beat, "x", mock_run.cfg, np.random.default_rng(s), "o")
            ids.append([f["id"] for f in (o.facts if o else []) if f["id"] != "x.f1"])
        assert ids[0] == ids[1], s


def test_busy_observer_is_distracted_and_notices_less(mock_run):
    from backend.agents.perception import attention_prob, vantage
    a = mock_run.agents["leo"]
    beat, fact = _beat(None), _beat(None)["facts"][0]
    a.state.location, a.state.arena = "Library", "Study Tables"
    try:
        a.state.in_conversation = None
        free = attention_prob(a, fact, beat, mock_run.cfg)
        assert vantage(a, fact, beat) == "near"
        a.state.in_conversation = "d1t0010c0"
        busy = attention_prob(a, fact, beat, mock_run.cfg)
        assert vantage(a, fact, beat) == "distracted"
        assert busy == pytest.approx(free * mock_run.cfg["perception"]["busy_factor"])   # nobody involved
    finally:
        a.state.in_conversation = None


def test_forced_mover_activity_is_neutral():
    from backend.simulation.engine import forced_activity
    plan = {"location": "Library", "arena": "Study Tables", "activity": "studying for a quiz"}
    assert forced_activity(plan, {"location": "Library", "arena": "Stacks"}) == "studying for a quiz"
    assert forced_activity(plan, {"location": "Quad", "arena": "Lawn"}) == "stopping by the Quad"
    asleep = {"location": "Dorm", "arena": "Room", "activity": "sleeping"}
    assert forced_activity(asleep, {"location": "Dorm", "arena": "Room"}) == "stopping by the Dorm"


def test_conversation_carries_into_next_ticks_perception(tmp_path):
    """Whoever talked last tick and is still there with a partner is busy while perceiving; the
    conversation phase clears it. Someone whose partner left is free again."""
    from backend.config import deep_merge, load_config
    from backend.simulation.engine import Simulation
    from tests.conftest import SHORT
    sim = Simulation(load_config("configs/baseline.yaml", deep_merge(SHORT, {"latent_events": {"event_rate": 0.0}})),
                     tmp_path / "carry", progress=False)
    try:
        A = sim.agents
        for a in A.values():
            a.set_time(sim.clock.time_of(5))
        sim.tracer.tick = 5
        spot = {"location": "Cafe", "arena": "Counter", "activity": "having coffee", "until": 10}
        for x in ("maya", "priya", "leo"):
            A[x].day_plan = []
            A[x].state.location, A[x].state.arena = "Cafe", "Counter"
            sim.overrides[x] = dict(spot)
        A["maya"].state.in_conversation = A["priya"].state.in_conversation = "c0"
        A["leo"].state.in_conversation = A["dev"].state.in_conversation = "c1"   # dev will not be there
        sim._move(5, [])
        assert A["maya"].state.in_conversation == A["priya"].state.in_conversation == "c0"
        assert A["leo"].state.in_conversation is None
        sim.tick_utts = []
        sim._conversations(5, [], {})
        assert all(a.state.in_conversation is None or a.state.in_conversation.startswith("d1t0005")
                   for a in A.values())
    finally:
        sim.pool.shutdown()
        sim.frames_fh.close()
        sim.events_fh.close()


def test_code_version_covers_vendored_ga(monkeypatch):
    from backend.simulation import engine as E
    calls = []

    class R:
        stdout = ""
    monkeypatch.setattr(E.subprocess, "run", lambda args, **kw: calls.append(args) or R())
    cv = E._code_version()
    status = next(c for c in calls if "status" in c)
    assert "third_party" in status and "backend" in status and "configs" in status
    assert "v2/decide_to_talk_v2.txt" in {k[3:] for k in cv["prompt_hashes"] if k.startswith("ga/")}
    assert "group_chat_v1.txt" in cv["prompt_hashes"]


def test_memory_files_are_written_with_sorted_keys(tmp_path):
    import datetime as dt
    import json
    from backend.memory.store import MemoryStream
    ms = MemoryStream("ethan")
    t = dt.datetime(2026, 9, 14, 9)
    for i, kws in enumerate([{"zebra", "apple"}, {"mango", "kiwi", "apple"}]):
        ms.add("event", t, "Ethan", "saw", "x", f"memory {i}", kws, 3, [0.1] * 4)
    ms.save_ga(tmp_path)
    for f in ("kw_strength.json", "embeddings.json"):
        raw = (tmp_path / f).read_text()
        assert raw == json.dumps(json.loads(raw), sort_keys=True), f
