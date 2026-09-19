"""Ontology v2 §3 (social) + §5 planted-phrase control: generated topology, multi-party talk,
independent pressure modules with manipulation checks.

The group-conversation tests build a minimal simulation context directly (agents, mock LLM, tracer)
instead of running the engine, so they exercise run_group_conversation on its own."""
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest

from backend import ga_compat
from backend.agents import topology as TOPO
from backend.agents.profile import FORBIDDEN_FIELDS, AgentProfile, apply_planted, load_population
from backend.config import deep_merge, load_config
from backend.modules.base import ModuleStack, build_modules

POP = "configs/population/homewood8.yaml"
PLANTED = {"agent": "maya", "habit": "Maya has a habit of calling any mess \"a full pickle\"."}


def _cfg(over=None):
    return load_config("configs/baseline.yaml", deep_merge({"llm": {"backend": "mock"}}, over or {}))


def _profiles(n=8):
    return load_population(POP, n)[0]


# ------------------------------------------------------------------ topology

def test_topology_deterministic_disjoint_and_roommates_together():
    cfg = _cfg({"topology": {"mode": "generated"}})
    a = TOPO.generate(_profiles(), cfg, np.random.default_rng(7))
    b = TOPO.generate(_profiles(), cfg, np.random.default_rng(7))
    assert a == b
    others = [TOPO.generate(_profiles(), cfg, np.random.default_rng(s)) for s in range(8)]
    assert len({json.dumps(o["circles"], sort_keys=True) for o in others}) > 1
    for t in [a] + others:
        members = [m for ms in t["circles"].values() for m in ms]
        assert len(members) == len(set(members)) == 8          # disjoint partition of everyone
        assert len(t["circles"]) == 2 and all(len(ms) >= 3 for ms in t["circles"].values())
        circle_of = {m: c for c, ms in t["circles"].items() for m in ms}
        assert circle_of["maya"] == circle_of["priya"] and circle_of["ethan"] == circle_of["leo"]  # share a room
        assert all(k[0] < k[1] for k in t["relationships"])


def test_topology_metrics():
    cfg = _cfg({"topology": {"mode": "generated"}})
    t = TOPO.generate(_profiles(), cfg, np.random.default_rng(3))
    m = t["metrics"]
    g = cfg["topology"]["generated"]
    assert m["mode"] == "generated" and m["n_agents"] == 8 and m["free"] == []
    assert m["density_within"] == 1.0
    assert m["mean_familiarity_within"] == pytest.approx(g["familiarity_within"])
    assert m["n_bridges"] == g["n_bridges"]
    assert all(b["circles"][0] != b["circles"][1] for b in m["bridges"])
    assert m["density_between"] < m["density_within"] and m["modularity"] > 0.2
    json.dumps(m)                                                # manifest-ready
    # no between ties and no bridges -> two disconnected cliques
    iso = TOPO.generate(_profiles(), _cfg({"topology": {"generated": {"p_between": 0.0, "n_bridges": 0}}}),
                        np.random.default_rng(3))["metrics"]
    assert iso["density_between"] == 0.0 and iso["n_bridges"] == 0 and iso["modularity"] == pytest.approx(0.5)


def test_topology_small_populations():
    cfg = _cfg({"topology": {"generated": {"n_circles": 3}}})
    t5 = TOPO.generate(_profiles(5), cfg, np.random.default_rng(0))
    assert len(t5["circles"]) == 1 and all(len(ms) >= 3 for ms in t5["circles"].values())
    t2 = TOPO.generate(_profiles(2), cfg, np.random.default_rng(0))
    assert t2["circles"] == {} and t2["routine_additions"] == {} and t2["metrics"]["free"] == ["maya", "priya"]


def test_file_topology_metrics():
    m = TOPO.metrics(_profiles(), {"c_lab": ["maya", "dev", "hana"], "c_dorm": ["ethan", "leo", "jordan"]})
    assert m["mode"] == "file" and m["free"] == ["priya", "sofia"]
    assert m["density_within"] == 1.0 and m["modularity"] is not None
    assert {(b["a"], b["b"]) for b in m["bridges"]} == {("jordan", "maya")}   # classmates, familiarity 0.6


def test_apply_rewrites_ties_and_colocates_circles_at_dinner():
    from backend.simulation.scheduler import plan_day, routine_position
    from backend.simulation.world import Clock
    cfg = _cfg({"topology": {"mode": "generated"}, "routine": {"jitter_minutes": 0, "deviation_prob": 0.0}})
    profiles = _profiles()
    t = TOPO.generate(profiles, cfg, np.random.default_rng(11))
    TOPO.apply(profiles, t)
    ids = sorted(profiles)
    for a in ids:
        assert set(profiles[a].relationships) == set(ids) - {a}
        for b in ids:
            if a != b:
                assert profiles[a].rel(b) == profiles[b].rel(a)
                assert profiles[a].rel(b) is not profiles[b].rel(a)
    slots = t["metrics"]["shared_meals"]
    assert len(slots) == 2 and len({s["time"] for s in slots}) == 2          # circles dine at different times
    clock = Clock(cfg)
    start = int(clock.day_start.total_seconds() // 60)
    for s in slots:
        members = t["circles"][s["circle"]]
        for a in members:
            dinners = [r for r in profiles[a].routine if "dinner" in r.activity]
            assert [(r.time, r.location, r.arena) for r in dinners] == [(s["time"], s["location"], s["arena"])]
            assert s["location"] in profiles[a].known_locations
        k = (TOPO._mins(s["time"]) + 30 - start) // clock.tick_minutes     # mid-meal
        k_end = (TOPO._mins(s["time"]) + TOPO.MEAL_MINUTES - start) // clock.tick_minutes
        pos, after = set(), set()
        for a in members:
            agent = SimpleNamespace(profile=profiles[a], cfg=cfg)
            agent.day_plan = plan_day(agent, clock, np.random.default_rng(0))
            p = routine_position(agent, k)
            pos.add((p["location"], p["arena"]))
            after.add(routine_position(agent, k_end)["location"])
        assert pos == {(s["location"], s["arena"])}
        assert s["location"] not in after                                  # the meal ends; nobody lingers
    # a plan that fell inside the meal hour starts right after it (dev: 18:00 run -> after an 18:00 dinner)
    dev_slot = next(s for s in slots if "dev" in t["circles"][s["circle"]])
    if dev_slot["time"] == "18:00":
        assert any(r.time == "19:00" and r.location == "Gym" for r in profiles["dev"].routine)


# ------------------------------------------------------------------ planted phrase

def test_planted_phrase_applied_and_noop_when_null():
    profiles = _profiles()
    before = {a: list(p.habits) for a, p in profiles.items()}
    assert apply_planted(profiles, _cfg()) is None
    assert {a: p.habits for a, p in profiles.items()} == before
    rec = apply_planted(profiles, _cfg({"controls": {"planted_phrase": PLANTED}}))
    assert rec == {"agent": "maya", "habit": 'has a habit of calling any mess "a full pickle"'}
    assert profiles["maya"].habits == before["maya"] + [rec["habit"]]
    assert all(profiles[a].habits == before[a] for a in profiles if a != "maya")
    assert profiles["maya"].ga_lifestyle().endswith('; has a habit of calling any mess "a full pickle".')
    import dataclasses
    assert not {f.name for f in dataclasses.fields(AgentProfile)} & FORBIDDEN_FIELDS
    with pytest.raises(ValueError):
        apply_planted(_profiles(3), _cfg({"controls": {"planted_phrase": {"agent": "hana", "habit": "x"}}}))


# ------------------------------------------------------------------ minimal world for talk + modules

def _world(tmp_path, over=None, where=("Dining Hall", "Main Floor"), n=8):
    from backend.agents.agent import Agent
    from backend.llm.client import LLMClient
    from backend.llm.embeddings import make_embedder
    from backend.llm.mock import MockBackend
    from backend.memory.encoder import add_simple_event
    from backend.memory.store import SimMemoryMeta
    from backend.simulation.world import Clock
    from backend.tracing.logger import TraceLogger
    cfg = _cfg(over)
    tmp_path.mkdir(parents=True, exist_ok=True)
    llm = LLMClient(MockBackend(0), tmp_path / "llm_calls.jsonl")
    embed = make_embedder(cfg.get("embedding"))
    ga_compat.load(llm, embed)
    clock = Clock(cfg)
    ctx = SimpleNamespace(cfg=cfg, llm=llm, embed=embed, meta=SimMemoryMeta(), tracer=TraceLogger(tmp_path),
                          clock=clock, agents={}, mods=None)
    for pid, prof in load_population(cfg["population"], n)[0].items():
        a = Agent(prof, cfg, None, cfg["seed"])
        a.ctx = ctx
        ctx.agents[pid] = a
    ctx.mods = build_modules(cfg, ctx)
    now = clock.time_of(43)                                     # day 1, 18:15
    for a in ctx.agents.values():
        a.mods = ctx.mods
        a.set_time(now)
        a.state.location, a.state.arena, a.state.activity = where[0], where[1], "eating dinner"
        a.sync_scratch()
    for a in ctx.agents.values():                               # a little to remember
        add_simple_event(a, "The dryer in the dorm basement ate someone's laundry card again.", 5, [])
        add_simple_event(a, f"{a.profile.first_name} missed the shuttle this morning and was late.", 6, [a.id])
    return ctx


def _trace(ctx):
    ctx.tracer.flush()
    return [json.loads(l) for l in open(ctx.tracer.run_dir / "trace.jsonl")]


def _talking(ctx, ids, cid):
    for x in ids:
        a = ctx.agents[x]
        a.state.in_conversation = cid
        a.state.pre_chat_activity = a.state.activity
        a.state.activity = "chatting"
        a.sync_scratch()


def test_group_conversation_record_trace_and_memories(tmp_path):
    from backend.agents.conversation import run_conversation
    from backend.agents.group_conversation import run_group_conversation
    from backend.llm.client import llm_scope
    ctx = _world(tmp_path)
    A = ctx.agents
    parts = ["dev", "ethan", "hana", "maya"]
    _talking(ctx, parts, "g0")
    _talking(ctx, ["leo", "jordan"], "c0")
    with llm_scope("group"):
        g = run_group_conversation("g0", [A[x] for x in parts], [A["priya"], A["sofia"]],
                                   np.random.default_rng(5), topic="meal")
    with llm_scope("dyad"):
        d = run_conversation("c0", A["leo"], A["jordan"], [], np.random.default_rng(5))
    assert set(g) == set(d)                                                   # same record shape
    assert g["participants"] == parts and g["trigger"] == {"agent": None, "text": None, "topic": "meal"}
    assert set(g["relationship_summaries"]) == set(parts)
    us = g["utterances"]
    assert len(parts) <= len(us) <= ctx.cfg["conversation"]["group"]["max_utterances"]
    speakers = [u["speaker"] for u in us]
    s0 = parts.index(speakers[0])
    assert speakers == [parts[(s0 + i) % len(parts)] for i in range(len(us))]  # round-robin
    for u in us:
        assert set(u["listeners"]) >= set(parts) - {u["speaker"]}
        assert set(u["listeners"]) <= set(parts + ["priya", "sofia"]) - {u["speaker"]}
        assert u["source"] == "group_chat_utterance"
    # trace records have the dyadic schema
    tr = _trace(ctx)

    def schema(typ, cid):
        recs = [r for r in tr if r["type"] == typ and cid in (r.get("conversation_id"), r.get("id"))]
        assert recs, (typ, cid)
        return {frozenset(r) for r in recs}
    for typ in ("utterance", "exposure", "conversation"):
        assert schema(typ, "g0") == schema(typ, "c0"), typ
    exp = [r for r in tr if r["type"] == "exposure" and r["conversation_id"] == "g0"]
    assert [r["listener_ids"] for r in exp] == [u["listeners"] for u in us]
    # every participant encodes one conversation memory attributed to the others
    for x in parts:
        others = [o for o in parts if o != x]
        chats = A[x].a_mem.seq_chat
        assert chats and chats[0].object == ", ".join(A[o].name for o in others)
        meta = ctx.meta.get(chats[0].node_id)
        assert meta.source_type == "conversation" and meta.speakers == others
        assert set(meta.source_ids) >= {f"g0.obs.{x}"} | {u["id"] for u in us}
        assert A[x].state.talks_today == 1 and set(A[x].state.last_talk) == set(others)
    # overhearing bystanders get observations of exactly what they heard
    assert g["overheard"]
    for o in g["overheard"]:
        heard = [u["id"] for u in us if o.agent_id in u["listeners"]]
        assert o.source_type == "overheard" and o.utterance_ids == heard and o.agent_id in ("priya", "sofia")
    # the prompt: GA-style, the whole table named, nothing hidden
    calls = [json.loads(l) for l in open(tmp_path / "llm_calls.jsonl")]
    gp = [c["prompt"] for c in calls if c["purpose"] == "group_chat_utterance"]
    assert len(gp) == len(us)
    assert "Dev Patel, Ethan Brooks, Hana Okafor and Maya Chen are chatting together" in gp[0]
    for p in gp:
        for bad in (r"\bcircle", r"\bc[12]\b", r"\bE[1-4]\b", r"latent", r"\bmeme", r"slang", r"\binvent"):
            assert not re.search(bad, p, re.I), bad


def test_group_conversation_is_deterministic(tmp_path):
    from backend.agents.group_conversation import run_group_conversation
    from backend.llm.client import llm_scope
    out = []
    for run in ("a", "b"):
        ctx = _world(tmp_path / run)
        parts = ["dev", "hana", "maya"]
        _talking(ctx, parts, "g0")
        with llm_scope("group"):
            g = run_group_conversation("g0", [ctx.agents[x] for x in parts], [], np.random.default_rng(9))
        out.append([(u["speaker"], u["text"]) for u in g["utterances"]])
    assert out[0] == out[1]


def test_group_conversation_module_lines_cover_the_table(tmp_path):
    from backend.agents.group_conversation import _context
    ctx = _world(tmp_path, {"modules": {"emotion": True, "prestige_bias": True},
                            "module_params": {"prestige_bias": {"scores_source": "file",
                                                                "scores": {"hana": 0.9}}}})
    emo = ctx.mods.get("emotion")
    emo.st("dev").valence = -0.8
    A = ctx.agents
    text = _context(A["maya"], [A["dev"], A["hana"]], "meal")
    assert "Dev seems down." in text and "Hana is widely admired" in text
    assert "sitting at the same table over a meal" in text


# ------------------------------------------------------------------ modules

def _conv(parts, lines):
    return {"id": "x", "participants": parts,
            "utterances": [{"speaker": s, "text": t, "listeners": [p for p in parts if p != s]} for s, t in lines]}


def test_social_reward_alone_is_independent_of_emotion(tmp_path):
    ctx = _world(tmp_path, {"modules": {"social_reward": True}})
    assert [m.name for m in ctx.mods.modules] == ["social_reward"]
    sr = ctx.mods.get("social_reward")
    A = ctx.agents
    lines = ctx.mods.modify_prompt(A["maya"], "chat", [], target=A["dev"])
    assert len(lines) == 1 and "values making the people around them feel better" in lines[0]
    assert not any(re.search(r"is feeling|seems (down|upbeat|stressed|excited)", l) for l in lines)
    conv = _conv(["maya", "dev"], [("maya", "Ugh, I missed the shuttle."), ("dev", "Honestly that was awful."),
                                   ("maya", "Anyway."), ("dev", "Thanks, that helped, this is great!")])
    ctx.mods.on_conversation(conv, A)
    r = conv["module_state"]["social_reward"]
    assert r["maya"] > 0                          # dev cheered up over the conversation
    assert "valence_before" not in conv["module_state"]
    draft = {"text": "m", "importance": 4, "salience": 0.6, "source_type": "conversation", "speakers": ["dev"],
             "involves": ["dev"]}
    assert ctx.mods.modify_memory(A["maya"], dict(draft))["importance"] > 4
    assert ctx.mods.modify_memory(A["maya"], dict(draft, source_type="perception"))["importance"] == 4
    mc = ctx.mods.manipulation_checks()["social_reward"]
    assert mc["active"] and mc["valence_source"] == "lexicon"
    assert mc["counters"]["rewarded_conversations"] == 1 and mc["counters"]["memory.conversation"] == 1
    assert mc["counters"]["prompt.chat"] == 1
    # talk-probability boost toward someone who came across as down
    sr.apparent["leo"] = -0.6
    assert ctx.mods.modify_utility(A["maya"], "talk_prob", 0.2, target=A["leo"]) > \
        ctx.mods.modify_utility(A["maya"], "talk_prob", 0.2, target=A["hana"])


def test_social_reward_uses_emotion_valence_when_both_on(tmp_path):
    ctx = _world(tmp_path, {"modules": {"emotion": True, "social_reward": True}})
    assert [m.name for m in ctx.mods.modules] == ["emotion", "social_reward"]
    conv = _conv(["maya", "dev", "hana"], [("maya", "Great news, we won!"), ("dev", "Awesome, love it."),
                                           ("hana", "So happy for you.")])
    ctx.mods.on_conversation(conv, ctx.agents)
    ms = conv["module_state"]
    assert "valence_before" in ms and set(ms["social_reward"]) == {"maya", "dev", "hana"}
    exp = {i: round(sum(ms["valence_after"][j] - ms["valence_before"][j] for j in ms["valence_before"] if j != i) / 2, 4)
           for i in ms["valence_before"]}
    assert ms["social_reward"] == exp
    assert ctx.mods.manipulation_checks()["social_reward"]["valence_source"] == "emotion"


def test_module_counters_and_manipulation_checks(tmp_path):
    ctx = _world(tmp_path, {"modules": {"emotion": True, "social_reward": True, "prestige_bias": True,
                                        "conformity": True}})
    assert ModuleStack([]).manipulation_checks() == {}
    A = ctx.agents
    checks = ctx.mods.manipulation_checks()
    assert set(checks) == {"emotion", "social_reward", "prestige_bias", "conformity"}
    assert all(not c["active"] and not any(c["counters"].values()) for c in checks.values())
    fact = {"id": "f", "text": "Hana won a prize.", "salience": 0.6, "involves": ["hana"]}
    ctx.mods.modify_attention(A["maya"], fact, 0.4)
    ctx.mods.on_observation(A["maya"], [fact])
    ctx.mods.modify_prompt(A["maya"], "react", [])
    c = ctx.mods.manipulation_checks()
    assert c["prestige_bias"]["counters"]["attention"] == 1 and c["prestige_bias"]["active"]
    assert c["emotion"]["counters"]["appraisal.observation"] == 1
    assert c["social_reward"]["counters"]["prompt.react"] == 1
    json.dumps(c)
    # a hook that changes nothing is not an effect
    fact0 = {"id": "g", "text": "Someone walked by.", "salience": 0.2, "involves": []}
    ctx.mods.modify_attention(A["maya"], fact0, 0.4)
    assert ctx.mods.manipulation_checks()["prestige_bias"]["counters"]["attention"] == 1


def test_conformity_counts_group_partners_separately(tmp_path):
    ctx = _world(tmp_path, {"modules": {"conformity": True}})
    a = ctx.agents["maya"]
    t = a.scratch.curr_time
    text = "the laundry card got eaten by the dryer again"
    emb = ctx.embed(text)
    a.a_mem.add("chat", t, a.name, "chat with", "Dev Patel, Hana Okafor", text, {"dev patel"}, 5, emb)
    node = a.a_mem.add("event", t, a.name, "saw", "x", text, {"x"}, 5, emb)
    out = ctx.mods.modify_retrieval(a, [node], {node.node_id: 1.0}, "laundry")
    assert out[node.node_id] == pytest.approx(1.0 + 0.6 * np.log(2))
    assert ctx.mods.manipulation_checks()["conformity"]["counters"] == {"nodes_boosted": 1, "retrieval": 1}


def test_prestige_score_sources(tmp_path):
    from backend.modules.prestige import PrestigeBias, degree_scores
    ctx = _world(tmp_path)
    deg = degree_scores(ctx.agents)
    p = PrestigeBias({"scores_source": "degree"}, ctx)
    assert p.scores == deg and max(deg.values()) == 1.0
    r = PrestigeBias({"scores_source": "random"}, ctx)
    assert sorted(r.scores.values()) == sorted(deg.values()) and set(r.scores) == set(deg)
    assert r.scores == PrestigeBias({"scores_source": "random"}, ctx).scores      # seeded
    f = PrestigeBias({"scores_source": "file", "scores": {"leo": 0.9}}, ctx)
    assert f.p("leo") == 0.9 and f.p("maya") == 0.0
    assert PrestigeBias({"scores": {"leo": 0.9}}, ctx).source == "file"          # pre-v2 configs
    for bad in ({"scores_source": "file"}, {"scores_source": "degree", "scores": {"leo": 1}},
                {"scores_source": "fame"}):
        with pytest.raises(ValueError):
            PrestigeBias(bad, ctx)
