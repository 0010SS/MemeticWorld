"""Ontology v2 §2 (agent cognition): random substreams, verbatim fixes, Link records and reminding,
Wording records and production priming, open matters (NEED).

Built on light fakes (real Agent / memory / mock LLM, no engine), so these tests do not depend on the
engine wiring."""
import datetime as dt
import json
import re
import types

import numpy as np
import pytest

from backend import ga_compat
from backend.agents.agent import Agent
from backend.agents.perception import AgentObservation
from backend.agents.profile import load_population
from backend.config import deep_merge, load_config
from backend.llm.client import LLMClient, current_purpose
from backend.llm.embeddings import make_embedder
from backend.llm.mock import MockBackend
from backend.memory import need, wording
from backend.memory.encoder import distinctive_phrases, encode, is_function_word, sticky_phrases
from backend.memory.reminding import PROMPT as REMINDING_PROMPT, _candidates, maybe_remind
from backend.memory.store import SimMemoryMeta
from backend.modules.base import ModuleStack

T0 = dt.datetime(2026, 9, 14, 9, 0)


class ListTracer:
    tick, time = 0, ""

    def __init__(self):
        self.records = []

    def log(self, type_, **fields):
        self.records.append({"type": type_, **fields})
        return str(len(self.records))

    def of(self, t):
        return [r for r in self.records if r["type"] == t]


class Scripted(MockBackend):
    """Mock LLM with fixed answers for some purposes."""

    def __init__(self, replies):
        super().__init__()
        self.replies = replies

    def generate(self, prompt, system, max_tokens, temperature):
        r = self.replies.get(current_purpose())
        if r is None:
            return super().generate(prompt, system, max_tokens, temperature)
        return r(prompt) if callable(r) else r


def world(tmp_path, overrides=None, backend=None, name="w"):
    cfg = load_config("configs/baseline.yaml", deep_merge({"llm": {"backend": "mock"}}, overrides or {}))
    res = load_population(cfg["population"], cfg.get("population_size"))
    profiles = res[0] if isinstance(res, tuple) else res
    llm = LLMClient(backend or MockBackend(), tmp_path / name / "llm_calls.jsonl")
    embed = make_embedder(cfg.get("embedding"))
    ga_compat.load(llm, embed)
    ctx = types.SimpleNamespace(cfg=cfg, llm=llm, embed=embed, meta=SimMemoryMeta(), tracer=ListTracer(),
                                agents={}, mods=ModuleStack([]))
    for pid, prof in profiles.items():
        a = Agent(prof, cfg, ctx.mods, cfg["seed"])
        a.ctx = ctx
        a.set_time(T0)
        a.sync_scratch()
        ctx.agents[pid] = a
    return ctx


def at(ctx, t):
    for a in ctx.agents.values():
        a.set_time(t)


def conv_obs(ctx, me, partner, lines, oid, tick=0, events=()):
    """lines: [(speaker_id, text)] -> a conversation observation shaped like conversation.py's."""
    facts = [{"id": f"{oid}.u{i}", "text": f'{ctx.agents[s].profile.first_name}: "{t}"', "salience": 0.6,
              "involves": [s] if s != me else []} for i, (s, t) in enumerate(lines)]
    a = ctx.agents[me]
    return AgentObservation(id=oid, agent_id=me, tick=tick, location=a.state.location, arena=a.state.arena,
                            source_type="conversation", facts=facts, event_ids=list(events), speakers=[partner],
                            utterance_ids=[f["id"] for f in facts], partner=ctx.agents[partner].name)


def perc_obs(ctx, me, text, oid, involves=(), salience=0.6, events=("ev001",), tick=0):
    a = ctx.agents[me]
    facts = [{"id": f"{oid}.f0", "text": text, "salience": salience, "involves": list(involves),
              "visibility": "all", "kind": "action"}]
    return AgentObservation(id=oid, agent_id=me, tick=tick, location=a.state.location, arena=a.state.arena,
                            source_type="perception", facts=facts, event_ids=list(events))


VERBATIM_ON = {"memory": {"verbatim": {"enabled": True, "base": 1.0, "max": 1.0}, "merge_threshold": 1.1}}


# --------------------------------------------------------------------------- 2.1 streams
def test_streams_are_named_seeded_and_cached(tmp_path):
    ctx = world(tmp_path)
    a = ctx.agents["maya"]
    assert a.stream("lens") is a.stream("lens")
    assert a.rng is a.stream("legacy")
    b = Agent(a.profile, a.cfg, ctx.mods, a.seed)
    assert b.stream("lens").random(5).tolist() == Agent(a.profile, a.cfg, ctx.mods, a.seed).stream("lens").random(5).tolist()
    fresh = Agent(a.profile, a.cfg, ctx.mods, a.seed)
    assert fresh.stream("lens").random() != fresh.stream("verbatim").random()
    other = Agent(ctx.agents["leo"].profile, a.cfg, ctx.mods, a.seed)
    assert Agent(a.profile, a.cfg, ctx.mods, a.seed).stream("lens").random() != other.stream("lens").random()


def test_toggling_verbatim_does_not_change_lens_draws(tmp_path):
    lines = [[("maya", "the zamboni situation again, it wrecked the cafeteria"), ("leo", "no way")],
             [("maya", "that gremlin printer ate my thesis draft"), ("leo", "classic")],
             [("maya", "the kerfuffle at the quiz was unreal"), ("leo", "yeah")],
             [("maya", "my soggy croissant incident is back"), ("leo", "ha")]]
    lenses = {}
    for on in (True, False):
        ctx = world(tmp_path, VERBATIM_ON if on else {"memory": {"merge_threshold": 1.1}}, name=f"v{on}")
        for i, ls in enumerate(lines):
            at(ctx, T0 + dt.timedelta(hours=i))
            encode(ctx.agents["leo"], conv_obs(ctx, "leo", "maya", ls, f"c{i}"), np.random.default_rng(i))
        recs = ctx.tracer.of("memory_encoded")
        assert len(recs) == len(lines)
        keys = ("noise", "focus", "length", "style", "unsure", "distort", "associate")
        lenses[on] = [{k: r["lens"][k] for k in keys} for r in recs]
        if on:
            assert ctx.tracer.of("wording")                 # the toggle really did something
    assert lenses[True] == lenses[False]


# --------------------------------------------------------------------------- 2.2 verbatim fixes
def test_clean_spans_have_content_words_at_both_edges(tmp_path):
    a = world(tmp_path).agents["leo"]
    texts = ["I keep rewriting all those notes for the lab, honestly",
             "my handwriting mine's basically illegible now",
             "Maya said it was the whole zamboni situation again",
             "he weirdly vanished during the quiz. The centrifuge thing again"]
    got = [p for t in texts for p, _ in distinctive_phrases(a, t, 3.6)]
    assert "rewriting all those" not in [g.lower() for g in got]
    assert "handwriting mine's basically" not in [g.lower() for g in got]
    assert "zamboni situation" in got and "centrifuge thing" in got
    for p in got:
        toks = p.split()
        assert 2 <= len(toks) <= 3
        assert not is_function_word(toks[0]) and not is_function_word(toks[-1]), p
        assert not {"maya", "leo"} & {t.lower() for t in toks}
    # spans never cross a sentence boundary
    assert not any("quiz" in p.lower() and "centrifuge" in p.lower() for p in got)
    for w in ("all", "those", "basically", "really", "mine's", "don't", "honestly"):
        assert is_function_word(w)
    assert not is_function_word("printer's") and not is_function_word("zamboni")


def test_lexicon_mode_ignores_campus_vocabulary(tmp_path):
    a = world(tmp_path).agents["leo"]
    text = "the makerspace printer smelled funny"      # "makerspace" is rare but in everyone's routines
    lex = [p for p, _ in distinctive_phrases(a, text, 3.6, mode="lexicon")]
    zipf = [p for p, _ in distinctive_phrases(a, text, 3.6, mode="zipf")]
    assert "makerspace printer" in zipf and "makerspace printer" not in lex


def test_exclude_self_and_speaker_attribution(tmp_path):
    ctx = world(tmp_path, VERBATIM_ON)
    leo = ctx.agents["leo"]
    obs = conv_obs(ctx, "leo", "maya", [("leo", "the zamboni situation is back"),
                                        ("maya", "that gremlin printer again")], "c1")
    got = sticky_phrases(leo, obs.facts, np.random.default_rng(0))
    assert got and all(w["heard_from"] == "maya" for w in got)
    assert not any("zamboni" in w["phrase"].lower() for w in got)
    assert got[0]["utterance_id"] == "c1.u1"
    leo.cfg = deep_merge(leo.cfg, {"memory": {"verbatim": {"exclude_self": False}}})
    got = sticky_phrases(leo, obs.facts, np.random.default_rng(0))
    assert any(w["heard_from"] == "leo" for w in got)


@pytest.mark.parametrize("render", [True, False])
def test_stuck_wording_is_a_record_and_rendered_only_if_asked(tmp_path, render):
    ctx = world(tmp_path, deep_merge(VERBATIM_ON, {"memory": {"verbatim": {"render_in_text": render}}}))
    leo = ctx.agents["leo"]
    node = encode(leo, conv_obs(ctx, "leo", "maya", [("maya", "that gremlin printer ate my draft")], "c1", tick=7),
                  np.random.default_rng(0))
    ws = ctx.meta.get(node.node_id).wordings
    assert ws and ws[0]["heard_from"] == "maya" and ws[0]["utterance_id"] == "c1.u0"
    assert ws[0]["tick"] == 7 and ws[0]["self_produced"] is False
    tr = ctx.tracer.of("wording")
    assert tr and tr[0]["node_id"] == node.node_id and tr[0]["phrase"] == ws[0]["phrase"]
    enc = ctx.tracer.of("memory_encoded")[0]
    assert enc["lens"]["verbatim"] == [w["phrase"] for w in ws]
    assert ("The exact words" in enc["prompt"]) is render


def test_merged_experience_keeps_wording_and_records_a_merge_link(tmp_path):
    ctx = world(tmp_path, deep_merge(VERBATIM_ON, {"memory": {"merge_threshold": 0.3}}))
    leo = ctx.agents["leo"]
    first = encode(leo, conv_obs(ctx, "leo", "maya", [("maya", "that gremlin printer ate my draft")], "c1",
                                 events=["ev001"]), np.random.default_rng(0))
    at(ctx, T0 + dt.timedelta(minutes=30))
    again = encode(leo, conv_obs(ctx, "leo", "maya", [("maya", "that gremlin printer ate my draft")], "c2",
                                 events=["ev002"]), np.random.default_rng(1))
    assert again is first
    links = [r for r in ctx.tracer.of("memory_link") if r["mechanism"] == "merge"]
    assert links and links[0]["from"] == "c2" and links[0]["to"] == first.node_id
    assert links[0]["from_event_ids"] == ["ev002"] and "ev001" in links[0]["to_event_ids"]
    m = ctx.meta.get(first.node_id)
    assert any(l["mechanism"] == "merge" for l in m.links)
    assert len(m.wordings) >= 2                                   # heard twice -> two wording records


# --------------------------------------------------------------------------- 2.3 links + reminding
def test_lens_association_is_a_link(tmp_path):
    ctx = world(tmp_path, {"memory": {"encoding_variability": 3.0, "merge_threshold": 1.1}})   # always associate
    maya = ctx.agents["maya"]
    encode(maya, perc_obs(ctx, "maya", "Someone spilled coffee all over the counter.", "o1"), np.random.default_rng(0))
    at(ctx, T0 + dt.timedelta(hours=2))
    node = encode(maya, perc_obs(ctx, "maya", "A student spilled juice near the counter.", "o2", events=["ev002"]),
                  np.random.default_rng(1))
    links = [r for r in ctx.tracer.of("memory_link") if r["mechanism"] == "association"]
    assert links and links[0]["from"] == node.node_id and links[0]["agent"] == "maya"
    assert links[0]["to"] in maya.a_mem.id_to_node
    assert ctx.meta.get(node.node_id).links[0]["to"] == links[0]["to"]
    assert ctx.tracer.of("memory_encoded")[-1]["lens"]["association_node"] == links[0]["to"]


def _reminding_world(tmp_path, reply, overrides=None):
    over = deep_merge({"reminding": {"enabled": True}, "memory": {"merge_threshold": 1.1}}, overrides or {})
    ctx = world(tmp_path, over, backend=Scripted({"reminding": reply}))
    maya = ctx.agents["maya"]
    for i, t in enumerate(["Leo spilled a whole tray of soup and was late for class.",
                           "Maya missed the shuttle and had to walk in the rain."]):
        at(ctx, T0 + dt.timedelta(minutes=10 * i))
        encode(maya, perc_obs(ctx, "maya", t, f"old{i}", events=[f"ev00{i}"]), np.random.default_rng(i))
    at(ctx, T0 + dt.timedelta(hours=3))
    return ctx, maya


YES = json.dumps({"ordinary": False, "reminded_of": 1, "what_felt_alike": "one slip made everything late"})


def test_reminding_fires_on_conversation_and_records_a_link(tmp_path):
    ctx, maya = _reminding_world(tmp_path, YES)
    obs = conv_obs(ctx, "maya", "priya", [("priya", "I dropped my tray and then missed the lecture")], "c1",
                   events=["ev009"])
    new = encode(maya, obs, np.random.default_rng(0))
    budget = maya.scratch.importance_trigger_curr
    node = maybe_remind(maya, obs, new, None)
    assert node is not None and node.type == "thought"
    rem = ctx.tracer.of("reminding")[-1]
    assert rem["source_type"] == "conversation" and rem["reminded_of"] and rem["node_id"] == node.node_id
    assert "just talked with Priya" in rem["prompt"]
    link = [r for r in ctx.tracer.of("memory_link") if r["mechanism"] == "reminding"][-1]
    assert link["from"] == new.node_id and link["to"] == rem["reminded_of"] and link["via"] == node.node_id
    assert link["reason"] == "one slip made everything late"
    assert link["from_event_ids"] == ["ev009"] and link["to_event_ids"]
    assert ctx.meta.get(new.node_id).links[-1]["mechanism"] == "reminding"
    assert ctx.meta.get(node.node_id).source_type == "reminding"
    # a reminding thought does not draw down the reflection trigger (reminding.reflection_weight = 0)
    assert maya.scratch.importance_trigger_curr == budget


def test_reminding_reflection_weight_is_configurable(tmp_path):
    ctx, maya = _reminding_world(tmp_path, YES, {"reminding": {"reflection_weight": 0.5}})
    obs = conv_obs(ctx, "maya", "priya", [("priya", "I dropped my tray and then missed the lecture")], "c1")
    new = encode(maya, obs, np.random.default_rng(0))
    budget = maya.scratch.importance_trigger_curr
    node = maybe_remind(maya, obs, new, None)
    assert maya.scratch.importance_trigger_curr == pytest.approx(budget - 0.5 * node.poignancy)


def test_ordinary_means_nothing_comes_to_mind(tmp_path):
    ctx, maya = _reminding_world(tmp_path, json.dumps({"ordinary": True, "reminded_of": 1, "what_felt_alike": "x"}))
    obs = perc_obs(ctx, "maya", "Someone dropped a tray and was late.", "o9")
    assert maybe_remind(maya, obs, encode(maya, obs, np.random.default_rng(0)), None) is None
    rem = ctx.tracer.of("reminding")[-1]
    assert rem["ordinary"] is True and rem["reminded_of"] is None
    assert not [r for r in ctx.tracer.of("memory_link") if r["mechanism"] == "reminding"]


def test_on_sources_gate_skips_without_an_llm_call(tmp_path):
    ctx, maya = _reminding_world(tmp_path, YES, {"reminding": {"on_sources": ["perception"]}})
    obs = conv_obs(ctx, "maya", "priya", [("priya", "I dropped my tray")], "c1")
    new = encode(maya, obs, np.random.default_rng(0))
    assert maybe_remind(maya, obs, new, None) is None and not ctx.tracer.of("reminding")


def _pick_earlier_thought(prompt):
    """Answer with the listed option that is an earlier reminding thought, if any."""
    for line in prompt.split("earlier things")[-1].splitlines():
        m = re.match(r"^(\d+)\. .*It reminded", line)
        if m:
            return json.dumps({"ordinary": False, "reminded_of": int(m.group(1)), "what_felt_alike": "same again"})
    return YES


def test_reminding_chains_through_earlier_reminding_thoughts(tmp_path):
    ctx, maya = _reminding_world(tmp_path, _pick_earlier_thought, {"reminding": {"n_related": 20, "n_salient": 20}})
    obs = perc_obs(ctx, "maya", "Someone dropped a tray and then missed the bus.", "o5", events=["ev005"])
    first = maybe_remind(maya, obs, encode(maya, obs, np.random.default_rng(0)), None)
    assert first is not None and not ctx.tracer.of("reminding")[-1]["chained"]
    at(ctx, T0 + dt.timedelta(hours=6))                   # the first reminding thought is now old enough
    obs2 = perc_obs(ctx, "maya", "A student tripped on a cable and missed a quiz.", "o6", events=["ev006"])
    new2 = encode(maya, obs2, np.random.default_rng(1))
    assert first.node_id in {n.node_id for n in _candidates(maya, new2, obs2.text(), np.random.default_rng(0), 20, 20)}
    node = maybe_remind(maya, obs2, new2, None)
    rem = ctx.tracer.of("reminding")[-1]
    assert node is not None and rem["reminded_of"] == first.node_id and rem["chained"] is True
    link = ctx.tracer.of("memory_link")[-1]
    assert link["to"] == first.node_id and set(link["to_event_ids"]) >= {"ev005"}


def test_reminding_prompt_v2_defaults_to_nothing_and_names_nothing(tmp_path):
    assert REMINDING_PROMPT.endswith("reminding_v2.txt")
    body = open(REMINDING_PROMPT).read().split("<commentblockmarker>###</commentblockmarker>")[-1].lower()
    assert '"ordinary"' in body and "not enough" in body
    for w in ("meme", "slang", "coin", "invent", "nickname", "new word", "name for", "label", "e1", "latent"):
        assert w not in body


def test_reflection_evidence_is_linked(tmp_path):
    from backend.memory.reflection import reflect
    ctx = world(tmp_path, {"memory": {"merge_threshold": 1.1}})
    maya = ctx.agents["maya"]
    for i, t in enumerate(["Leo spilled soup at lunch.", "Maya missed the shuttle.", "Priya lost her planner.",
                           "Ethan was late to the lecture.", "Someone locked the lab door."]):
        at(ctx, T0 + dt.timedelta(minutes=20 * i))
        encode(maya, perc_obs(ctx, "maya", t, f"o{i}", events=[f"ev00{i}"]), np.random.default_rng(i))
    created = reflect(maya, np.random.default_rng(0))
    links = [r for r in ctx.tracer.of("memory_link") if r["mechanism"] == "reflection"]
    assert created and links
    assert {l["from"] for l in links} <= {n.node_id for n in created}


# --------------------------------------------------------------------------- 2.4 wording + priming
def test_priming_line_from_own_wordings(tmp_path):
    ctx = world(tmp_path, deep_merge(VERBATIM_ON, {"priming": {"enabled": True, "window_hours": 24, "max_phrases": 3}}))
    leo = ctx.agents["leo"]
    assert wording.priming_line(leo) is None                      # nothing heard yet
    encode(leo, conv_obs(ctx, "leo", "maya", [("maya", "that gremlin printer ate my draft")], "c1"),
           np.random.default_rng(0))
    line = wording.priming_line(leo, conversation_id="conv9")
    phrases = ctx.meta.get(leo.a_mem.all_nodes()[0].node_id).wordings
    assert line.startswith("Things Leo has heard people say lately: \"") and phrases[0]["phrase"] in line
    pr = ctx.tracer.of("priming")[-1]
    assert pr["agent"] == "leo" and pr["conversation_id"] == "conv9" and pr["phrases"]
    at(ctx, T0 + dt.timedelta(hours=30))                          # outside the window
    assert wording.priming_line(leo) is None
    leo.cfg = deep_merge(leo.cfg, {"priming": {"enabled": False}})
    at(ctx, T0)
    assert wording.priming_line(leo) is None


def test_recent_wordings_weight_recency_and_repetition(tmp_path):
    ctx = world(tmp_path)
    leo = ctx.agents["leo"]
    obs = conv_obs(ctx, "leo", "maya", [("maya", "x")], "c1")
    for i in range(6):
        at(ctx, T0 + dt.timedelta(minutes=i))
        n = leo.a_mem.add("chat", leo.scratch.curr_time, leo.name, "chat with", "Maya", f"memory {i}", {"m"}, 4,
                          ctx.embed(f"memory {i}"))
        from backend.memory.store import MemoryMeta
        ctx.meta.set(n.node_id, MemoryMeta("leo", "conversation"))
        wording.record(leo, n.node_id, ["gremlin printer" if i < 5 else "soggy croissant"], obs)
    wording.record(leo, n.node_id, [{"phrase": "my own words", "heard_from": "leo"}], obs)
    at(ctx, T0 + dt.timedelta(hours=1))
    firsts = [wording.recent_wordings(leo, leo.scratch.curr_time, 24, 1, np.random.default_rng(s))[0]
              for s in range(200)]
    assert firsts.count("gremlin printer") > 150 and "my own words" not in firsts
    assert set(wording.recent_wordings(leo, leo.scratch.curr_time, 24, 5, np.random.default_rng(0))) == \
        {"gremlin printer", "soggy croissant"}


# --------------------------------------------------------------------------- 2.5 open matters
def test_open_matters_note_focal_discussed_decay(tmp_path):
    ctx = world(tmp_path, {"need": {"enabled": True, "min_importance": 6, "half_life_hours": 24, "max_open": 2,
                                    "discuss_decay": 0.5}, "memory": {"merge_threshold": 1.1}})
    maya = ctx.agents["maya"]

    def enc(text, oid, involves=(), minutes=0):
        at(ctx, T0 + dt.timedelta(minutes=minutes))
        o = perc_obs(ctx, "maya", text, oid, involves=involves)
        n = encode(maya, o, np.random.default_rng(minutes))
        return o, n

    o1, n1 = enc("Maya's tray tipped over at the counter.", "o1", involves=["maya"])       # took part
    assert need.note(maya, o1, n1)["node_id"] == n1.node_id
    o2, n2 = enc("A pigeon walked by.", "o2", minutes=10)
    n2.poignancy = 2
    assert need.note(maya, o2, n2) is None                                                  # neither
    o3, n3 = enc("Someone's laptop caught fire in the library.", "o3", minutes=20)
    n3.poignancy = 8
    assert need.note(maya, o3, n3) is not None                                              # important
    o4, n4 = enc("Maya's card was declined at the cafe.", "o4", involves=["maya"], minutes=30)
    need.note(maya, o4, n4)
    assert [m["node_id"] for m in maya.open_matters] == [n1.node_id, n3.node_id, n4.node_id]
    now = T0 + dt.timedelta(minutes=30)
    top = need.focal(maya, now, conversation_id="conv1")
    assert top == [n4.description, n3.description]                  # max_open = 2, strongest (most recent) first
    rec = ctx.tracer.of("open_matter_focal")[-1]
    assert rec["conversation_id"] == "conv1" and rec["node_ids"] == [n4.node_id, n3.node_id]
    need.discussed(maya, [n4.node_id])
    assert need.focal(maya, now)[0] == n3.description               # talked through -> weaker
    m1 = maya.open_matters[0]
    assert need.strength(m1, T0 + dt.timedelta(hours=24), 24) == pytest.approx(0.5)
    assert need.focal(maya, T0 + dt.timedelta(days=5)) == []        # faded
    maya.a_mem.remove(n3.node_id)                                   # forgotten memories are not on one's mind
    assert n3.description not in need.focal(maya, now)
    types_ = {r["action"] for r in ctx.tracer.of("open_matter")}
    assert {"open", "discussed"} <= types_


def test_need_is_a_no_op_when_disabled(tmp_path):
    ctx = world(tmp_path)
    maya = ctx.agents["maya"]
    o = perc_obs(ctx, "maya", "Maya's tray tipped over.", "o1", involves=["maya"])
    n = encode(maya, o, np.random.default_rng(0))
    assert need.note(maya, o, n) is None and need.focal(maya, T0) == []
    need.discussed(maya, [n.node_id])
    assert not getattr(maya, "open_matters", []) and not ctx.tracer.of("open_matter")


# --------------------------------------------------------------------------- D42 name guard (viewpoint + encoder)
def _names(ctx, aid):
    a = ctx.agents[aid]
    return {a.name.lower(), a.profile.first_name.lower()}


def _mentions(text, names):
    low = (text or "").lower()
    return {n for n in names if re.search(rf"\b{re.escape(n)}\b", low)}


def seen_obs(ctx, me, facts, oid="o1", vantage="near"):
    """A perception observation shaped like perception.observe's (world_text, vantage, recognised)."""
    from backend.agents.perception import RECOGNISE_FAMILIARITY
    a = ctx.agents[me]
    out = []
    for i, (text, involves) in enumerate(facts):
        rec = [x for x in involves if x != me and a.profile.rel(x).familiarity >= RECOGNISE_FAMILIARITY]
        out.append({"id": f"{oid}.f{i}", "text": text, "world_text": text, "salience": 0.7, "involves": list(involves),
                    "visibility": "all", "kind": "action", "vantage": "participant" if me in involves else vantage,
                    "recognised": rec})
    return AgentObservation(id=oid, agent_id=me, tick=0, location=a.state.location, arena=a.state.arena,
                            source_type="perception", facts=out, event_ids=["ev001"])


SKETCH = [("Ethan left a spiral sketchbook on the ledge by the cafe tip jar.", ["ethan"]),
          ("Priya waved at Ethan Brooks from the counter.", ["priya", "ethan"])]


@pytest.mark.parametrize("reply", ["LLM_ERROR: timeout", "not json at all", "{}", '{"f0": null}', '{"other": 1}',
                                   '{"f0": 3, "f1": ["x"]}'])
def test_viewpoint_fallback_still_hides_unknown_names(tmp_path, reply):
    ctx = world(tmp_path, backend=Scripted({"viewpoint": reply}))
    from backend.agents.viewpoint import render
    sofia = ctx.agents["sofia"]                                    # knows Priya, not Ethan
    assert sofia.profile.rel("ethan").familiarity < 0.3 <= sofia.profile.rel("priya").familiarity
    obs = seen_obs(ctx, "sofia", SKETCH)
    render(sofia, obs)
    assert [f["id"] for f in obs.facts] == ["o1.f0", "o1.f1"]      # nothing lost, nothing named
    for f in obs.facts:
        assert not _mentions(f["text"], _names(ctx, "ethan")), f["text"]
        assert f["viewpoint_fallback"] is True
    assert "Priya" in obs.facts[1]["text"] and "sketchbook" in obs.facts[0]["text"]
    vp = ctx.tracer.of("viewpoint")[-1]
    assert vp["fallback"] == ["o1.f0", "o1.f1"] and vp["dropped"] == []
    assert all(not _mentions(f["perceived"], _names(ctx, "ethan")) for f in vp["facts"])


def test_viewpoint_guard_covers_names_bleeding_across_items(tmp_path):
    reply = json.dumps({"f0": "A guy in a hoodie left a sketchbook by the tip jar.",
                        "f1": "Priya waved at ethan, the sketchbook guy."})     # renderer leaked the name into f1
    ctx = world(tmp_path, backend=Scripted({"viewpoint": reply}))
    from backend.agents.viewpoint import render
    obs = seen_obs(ctx, "sofia", [SKETCH[0], ("Priya waved from the counter.", ["priya"])])
    render(ctx.agents["sofia"], obs)
    assert obs.facts[1]["text"] == "Priya waved at someone, the sketchbook guy."
    assert not any(f.get("viewpoint_fallback") for f in obs.facts) and ctx.tracer.of("viewpoint")[-1]["fallback"] == []


@pytest.mark.parametrize("reply", [None, "LLM_ERROR: boom"])
def test_encode_prompt_never_names_unrecognised_people(tmp_path, reply):
    """D42 follow-up: the encoder's relationship context lists only people the agent can name, so the
    stranger the renderer turned into "someone" is not re-bound to the incident."""
    replies = {} if reply is None else {"viewpoint": reply}
    ctx = world(tmp_path, {"memory": {"merge_threshold": 1.1}}, backend=Scripted(replies))
    from backend.agents.viewpoint import render
    sofia = ctx.agents["sofia"]
    obs = seen_obs(ctx, "sofia", SKETCH)
    render(sofia, obs)
    node = encode(sofia, obs, np.random.default_rng(0))
    enc = ctx.tracer.of("memory_encoded")[-1]
    ethan = _names(ctx, "ethan")
    assert enc["prompt"] and not _mentions(enc["prompt"], ethan), enc["prompt"]
    assert sofia.relationship_line(ctx.agents["priya"]) in enc["prompt"]   # people she knows keep their context
    assert not _mentions(" ".join(node.keywords) + " " + node.object, ethan)
    assert node.object == "Priya Raman"


def test_encode_context_without_viewpoints_or_after_generalisation(tmp_path):
    # viewpoints off: the world sentence names him, so his relationship line stays (control unchanged)
    ctx = world(tmp_path, {"memory": {"merge_threshold": 1.1, "entity_generalization": 0.0},
                           "perception": {"viewpoints": False}}, name="off")
    sofia = ctx.agents["sofia"]
    encode(sofia, seen_obs(ctx, "sofia", SKETCH[:1]), np.random.default_rng(0))
    assert sofia.relationship_line(ctx.agents["ethan"]) in ctx.tracer.of("memory_encoded")[-1]["prompt"]
    # the pre-filter generalised him to "a student": the context must not bring his name back
    ctx = world(tmp_path, {"memory": {"merge_threshold": 1.1, "entity_generalization": 10.0},
                           "perception": {"viewpoints": False}}, name="gen")
    sofia = ctx.agents["sofia"]
    node = encode(sofia, seen_obs(ctx, "sofia", SKETCH[:1]), np.random.default_rng(0))
    enc = ctx.tracer.of("memory_encoded")[-1]
    assert {"op": "generalize_entity", "who": "ethan"} in enc["encoding_ops"]
    assert not _mentions(enc["prompt"], _names(ctx, "ethan")) and node.object == sofia.state.location
    # facts without a `recognised` tag fall back on familiarity
    ctx = world(tmp_path, {"memory": {"merge_threshold": 1.1, "entity_generalization": 0.0}}, name="legacy")
    sofia = ctx.agents["sofia"]
    encode(sofia, perc_obs(ctx, "sofia", "Someone left a sketchbook; Priya picked it up.", "o2",
                           involves=["ethan", "priya"]), np.random.default_rng(0))
    p = ctx.tracer.of("memory_encoded")[-1]["prompt"]
    assert not _mentions(p, _names(ctx, "ethan")) and sofia.relationship_line(ctx.agents["priya"]) in p


def test_heard_sources_keep_the_speakers_context(tmp_path):
    ctx = world(tmp_path, {"memory": {"merge_threshold": 1.1}})
    sofia = ctx.agents["sofia"]
    encode(sofia, conv_obs(ctx, "sofia", "ethan", [("ethan", "the sketchbook was mine")], "c1"),
           np.random.default_rng(0))
    assert sofia.relationship_line(ctx.agents["ethan"]) in ctx.tracer.of("memory_encoded")[-1]["prompt"]


# --------------------------------------------------------------------------- retrieval noise vs pool (§2.1)
def _stocked(tmp_path, n=12):
    ctx = world(tmp_path, {"memory": {"merge_threshold": 1.1}})
    maya = ctx.agents["maya"]
    texts = ["Leo spilled soup at lunch.", "Maya missed the shuttle.", "Priya lost her planner.",
             "Ethan was late to the lecture.", "Someone locked the lab door.", "The printer jammed again.",
             "Hana sang at the open mic.", "Dev fixed the club website.", "It rained all afternoon.",
             "The cafe ran out of oat milk.", "Jordan printed thirty posters.", "Sofia found a lost key."]
    for i, t in enumerate(texts[:n]):
        at(ctx, T0 + dt.timedelta(minutes=15 * i))
        encode(maya, perc_obs(ctx, "maya", t, f"o{i}", events=[f"ev{i:03d}"]), np.random.default_rng(i))
    at(ctx, T0 + dt.timedelta(hours=6))
    return ctx, maya


def _ids(res, q):
    return [n.node_id for n in res[q].nodes]


def test_retrieval_takes_one_draw_per_call_whatever_the_pool(tmp_path):
    from backend.memory.retrieval import retrieve
    ctx, maya = _stocked(tmp_path)
    q = "something went wrong"
    after = []
    for kw in ({"focal_points": [q]}, {"focal_points": [q, "who was late", "open matter"]},
               {"focal_points": [q], "exclude_ids": [n.node_id for n in maya.a_mem.all_nodes()[:5]]},
               {"focal_points": [q], "kinds": ("thought",)}):               # empty pool
        rng = np.random.default_rng(5)
        retrieve(maya, rng=rng, touch=False, **kw)
        after.append(rng.random())
    maya.cfg = deep_merge(maya.cfg, {"retrieval": {"temperature": 0.0}})
    rng = np.random.default_rng(5)
    retrieve(maya, [q], rng=rng, touch=False)
    after.append(rng.random())
    assert len(set(after)) == 1                                     # same stream position afterwards


def test_retrieval_noise_is_invariant_to_unrelated_nodes(tmp_path):
    from backend.memory.retrieval import gumbel_noise, retrieve
    a = gumbel_noise(11, "q", ["a", "b", "c"])
    b = gumbel_noise(11, "q", ["z", "c", "a", "y", "b"])
    assert a.tolist() == [b[2], b[4], b[1]]
    assert gumbel_noise(12, "q", ["a"])[0] != a[0] and gumbel_noise(11, "q2", ["a"])[0] != a[0]
    ctx, maya = _stocked(tmp_path)
    q = "someone had a bad morning"
    k = 6
    before = {s: _ids(retrieve(maya, [q], k=k, rng=np.random.default_rng(s), touch=False), q) for s in range(12)}
    # a copy of an existing memory: same recency, importance and relevance, so no score's min-max range moves
    src = maya.a_mem.all_nodes()[3]
    extra = maya.a_mem.add("event", src.created, src.subject, src.predicate, src.object, src.description, set(),
                           src.poignancy, ctx.embed(src.description), [])
    extra.last_accessed = src.last_accessed
    moved = 0
    for s, old in before.items():
        new = _ids(retrieve(maya, [q], k=k, rng=np.random.default_rng(s), touch=False), q)
        rest = [i for i in new if i != extra.node_id]
        assert rest == old[:len(rest)]                              # the others keep their noise and their order
        moved += extra.node_id in new
    assert moved                                                    # and the extra node did take part


def test_hashed_gumbel_top1_matches_softmax():
    from backend.memory.retrieval import gumbel_noise, softmax
    s = np.array([0.0, 0.5, 1.0])
    tau = 0.5
    n = 20000
    wins = np.bincount([int(np.argmax(s / tau + gumbel_noise(b, "q", ["x", "y", "z"]))) for b in range(n)],
                       minlength=3) / n
    assert np.allclose(wins, softmax((s / tau).tolist()), atol=0.015)
    g = np.concatenate([gumbel_noise(b, "q", [f"n{i}" for i in range(40)]) for b in range(500)])
    assert abs(g.mean() - np.euler_gamma) < 0.03 and abs(g.var() - np.pi ** 2 / 6) < 0.1
