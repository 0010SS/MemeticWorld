"""`conversation.repair`: the channel through which a phrase carries its MEANING.

What these tests hold the mechanism to:

  * it is OFF by default, and a condition that carries the block but disables it is byte-identical to one
    that has never heard of it (no draw, no embedding, no LLM call, no trace line);
  * what an agent knows is what is in its MEMORY -- never a registry, a config list or a planted-phrase
    roster -- so giving the hearer a memory of the phrase silences the question and taking it away
    brings the question back;
  * the answer is the answering agent's own retrieved memory, and nothing a designer wrote: a gloss
    parked in config reaches neither the prompt nor the transcript;
  * the question never contains the phrase, so a hearer who does not know an expression is never
    recorded as having used it, and no repair question can itself be read as a coinage;
  * no canonical place name reaches any generated text (D75);
  * same seed -> same repair, byte for byte, and the cap holds.

Mock LLM only; no live calls and no simulation runs.
"""
from __future__ import annotations

import inspect
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest

from backend import ga_compat
from backend.agents import conversation as CONV
from backend.agents import repair as R
from backend.agents.profile import load_population
from backend.config import deep_merge, load_config

# The four phrases the 2x2 cohort is being built from. They are here as DATA the mechanism must be able
# to see, not as anything the mechanism knows about: nothing in backend/agents/repair.py mentions them.
COHORT = ["blue-tray", "hood-three", "running warm", "Tuesday list"]

PHRASE = "blue-tray"
USE = "Honestly, the whole thing was another blue-tray, if you ask me"
SPEAKER_MEMS = [
    "The trays came out of the wash still greasy even though they looked clean; people called it blue-tray.",
    "Someone got sick off a blue-tray plate that had looked perfectly fine on the rack.",
]
OTHER_MEM = "The dryer in the dorm basement ate someone's laundry card again."
ON = {"conversation": {"repair": {"enabled": True, "prob": 1.0}}}


def _cfg(over=None):
    return load_config("configs/baseline.yaml", deep_merge(
        {"llm": {"backend": "mock"}, "day_start": "08:30", "day_end": "17:00",
         "world": {"reference_mode": "situated"}}, over or {}))


def _world(tmp_path, over=None, n=4, tick=12):
    """Four agents in one room with a mock LLM: the smallest context `maybe_repair` needs."""
    from backend.agents.agent import Agent
    from backend.llm.client import LLMClient
    from backend.llm.embeddings import make_embedder
    from backend.llm.mock import MockBackend
    from backend.memory.store import SimMemoryMeta
    from backend.modules.base import build_modules
    from backend.simulation.world import Clock
    from backend.tracing.logger import TraceLogger
    cfg = _cfg(over)
    tmp_path.mkdir(parents=True, exist_ok=True)
    llm = LLMClient(MockBackend(0), tmp_path / "llm_calls.jsonl")
    embed = make_embedder(cfg.get("embedding"))
    ga_compat.load(llm, embed)
    clock = Clock(cfg)
    tracer = TraceLogger(tmp_path, check=True)
    tracer.tick = tick
    ctx = SimpleNamespace(cfg=cfg, llm=llm, embed=embed, meta=SimMemoryMeta(), tracer=tracer,
                          clock=clock, agents={}, mods=None)
    for pid, prof in load_population("configs/population/homewood8.yaml", n)[0].items():
        a = Agent(prof, cfg, None, cfg["seed"])
        a.ctx = ctx
        ctx.agents[pid] = a
    ctx.mods = build_modules(cfg, ctx)
    now = clock.time_of(tick)
    for a in ctx.agents.values():
        a.mods = ctx.mods
        a.set_time(now)
        a.day_plan = [{"k": 14, "location": "Dining Hall", "arena": "Main Floor", "activity": "having lunch"}]
        a.state.location, a.state.arena, a.state.activity = "Quad", "Lawn", "reading and preparing work"
        a.sync_scratch()
    return ctx


def _remember(agent, *texts, importance=5):
    from backend.memory.encoder import add_simple_event
    return [add_simple_event(agent, t, importance, []) for t in texts]


def _pair(ctx):
    ids = sorted(ctx.agents)[:2]
    return ctx.agents[ids[0]], ctx.agents[ids[1]]


def _utt(speaker, hearer, text=USE, uid="d1t0012c0.u0", idx=0, extra_listeners=()):
    """An utterance record shaped exactly as conversation.run_conversation writes one."""
    return {"id": uid, "conversation_id": uid.split(".")[0], "idx": idx, "speaker": speaker.id,
            "text": text, "listeners": [hearer.id, *extra_listeners], "location": speaker.state.location,
            "arena": speaker.state.arena, "retrieved": [], "retrieved_event_ids": [],
            "source": "chat_utterance"}


def _trace(ctx, type_=None):
    ctx.tracer.flush()
    recs = [json.loads(l) for l in open(ctx.tracer.run_dir / "trace.jsonl")]
    return [r for r in recs if type_ is None or r["type"] == type_]


def _calls(ctx):
    ctx.tracer.flush()
    path = ctx.tracer.run_dir / "llm_calls.jsonl"
    return [json.loads(l) for l in open(path)] if path.exists() else []


def _ready(tmp_path, over=None, hearer_mems=(OTHER_MEM,)):
    """A speaker who has `blue-tray` in memory and a hearer who has never met it."""
    ctx = _world(tmp_path, deep_merge(ON, over or {}))
    a, b = _pair(ctx)
    _remember(a, *SPEAKER_MEMS)
    _remember(b, *hearer_mems)
    return ctx, a, b


# ------------------------------------------------------------------------------------------ default off
def test_off_by_default_and_nothing_is_touched(tmp_path):
    ctx = _world(tmp_path)
    a, b = _pair(ctx)
    _remember(a, *SPEAKER_MEMS)
    _remember(b, OTHER_MEM)
    assert R.enabled(ctx.cfg) is False
    before = len(_calls(ctx))
    assert R.maybe_repair("c0", _utt(a, b), a, b, [], []) == []
    assert _trace(ctx, "repair") == []
    assert len(_calls(ctx)) == before          # no LLM call, so no embedding or retrieval either


def test_the_block_present_but_disabled_changes_no_conversation(tmp_path):
    """A condition that carries `conversation.repair` with enabled:false must produce the identical
    conversation to one that has never heard of the mechanism -- the same invariant mealtalk holds."""
    from backend.llm.client import llm_scope
    outs = []
    for tag, over in (("bare", None), ("block", {"conversation": {"repair": {"enabled": False, "prob": 1.0}}})):
        ctx = _world(tmp_path / tag, over)
        a, b = _pair(ctx)
        _remember(a, *SPEAKER_MEMS)
        _remember(b, OTHER_MEM)
        with llm_scope(tag):
            conv = CONV.run_conversation("c0", a, b, [], np.random.default_rng(5))
        outs.append([(u["speaker"], u["text"], u["idx"], tuple(u["retrieved"])) for u in conv["utterances"]])
    assert outs[0] and outs[0] == outs[1]


# ---------------------------------------------------------------------------------- the exchange itself
def test_a_phrase_new_to_the_hearer_is_asked_about_and_answered(tmp_path):
    ctx, a, b = _ready(tmp_path)
    turns = R.maybe_repair("d1t0012c0", _utt(a, b), a, b, [], [[a.name, USE]])
    assert [t["source"] for t in turns] == ["repair_question", "repair_answer"]
    assert turns[0]["speaker"] == b.id and turns[1]["speaker"] == a.id          # only the addressee asks
    assert [t["id"] for t in turns] == ["d1t0012c0.u0.r0", "d1t0012c0.u0.r1"]
    rec = _trace(ctx, "repair")
    assert len(rec) == 1
    assert (rec[0]["phrase"], rec[0]["asker"], rec[0]["answerer"]) == (PHRASE, b.id, a.id)
    assert rec[0]["kind"] == "ask" and rec[0]["competing"] is False and rec[0]["explained"] is True
    assert rec[0]["channel"] == "repeated" and rec[0]["turn_ids"] == [t["id"] for t in turns]
    # both turns are ordinary spoken lines as far as the rest of the simulation is concerned
    utts = {u["id"]: u for u in _trace(ctx, "utterance")}
    exps = {e["utterance_id"] for e in _trace(ctx, "exposure")}
    assert set(utts) == exps == {t["id"] for t in turns}


def test_only_the_addressees_ears_hear_it_and_no_new_draw_is_taken(tmp_path):
    """The exchange reaches exactly the ears that heard the line that triggered it -- the addressee plus
    whoever overheard that line -- so switching repair on takes no overhearing draw."""
    ctx, a, b = _ready(tmp_path)
    c = ctx.agents[sorted(ctx.agents)[2]]
    turns = R.maybe_repair("c0", _utt(a, b, extra_listeners=[c.id]), a, b, [], [])
    assert turns[0]["listeners"] == [a.id, c.id]        # the asker's question: speaker + the overhearer
    assert turns[1]["listeners"] == [b.id, c.id]        # the answer: the asker + the overhearer


def test_what_the_hearer_knows_comes_from_their_own_memory(tmp_path):
    """No registry, no planted-phrase list: one memory carrying the phrase is the whole difference."""
    naive, a, b = _ready(tmp_path / "naive")
    assert R.maybe_repair("c0", _utt(a, b), a, b, [], [])
    knows, a2, b2 = _ready(tmp_path / "knows", hearer_mems=(SPEAKER_MEMS[0],))
    assert R.recall(b2, PHRASE)
    assert R.maybe_repair("c0", _utt(a2, b2), a2, b2, [], []) == []


def test_a_stuck_wording_counts_as_knowing_it(tmp_path):
    """WORDING holds a phrase verbatim on a node's sidecar even when the memory text paraphrases it
    away; that still means the agent has met the expression."""
    from backend.memory.store import MemoryMeta
    ctx, a, b = _ready(tmp_path)
    node = _remember(b, "Someone said something odd about the trays at lunch.")[0]
    ctx.meta.set(node.node_id, MemoryMeta(agent_id=b.id, source_type="conversation",
                                          wordings=[{"phrase": PHRASE, "heard_from": a.id}]))
    assert R.recall(b, PHRASE) == [node.node_id]
    assert R.maybe_repair("c0", _utt(a, b), a, b, [], []) == []


def test_a_speaker_who_cannot_find_it_in_memory_is_not_repaired(tmp_path):
    """Repair hands over remembered meaning. An agent who has just improvised a phrase has nothing to
    hand over, and the requirement applies to both channels so they stay matched."""
    ctx = _world(tmp_path, ON)
    a, b = _pair(ctx)
    _remember(a, OTHER_MEM)
    _remember(b, OTHER_MEM)
    assert R.candidates(a, b, USE) == []
    assert R.maybe_repair("c0", _utt(a, b), a, b, [], []) == []


def test_probability_gates_the_exchange(tmp_path):
    ctx, a, b = _ready(tmp_path / "never", {"conversation": {"repair": {"prob": 0.0}}})
    assert R.maybe_repair("c0", _utt(a, b), a, b, [], []) == []
    assert _trace(ctx, "repair") == []


# ------------------------------------------------------------------------------------------- the cap
def test_the_cap_holds(tmp_path):
    ctx, a, b = _ready(tmp_path / "one")
    first = R.maybe_repair("c0", _utt(a, b), a, b, [], [])
    assert first
    # the cap counts EXCHANGES: one repair_question (or repair_collision) each
    assert R.maybe_repair("c0", _utt(a, b, uid="c0.u1", idx=1), a, b, first, []) == []
    ctx2, a2, b2 = _ready(tmp_path / "two", {"conversation": {"repair": {"max_followups": 2}}})
    got = R.maybe_repair("c0", _utt(a2, b2), a2, b2, [], [])
    assert R.maybe_repair("c0", _utt(a2, b2, uid="c0.u1", idx=1), a2, b2, got, [])
    assert len(_trace(ctx2, "repair")) == 2


# --------------------------------------------------------------------------------------- determinism
def test_same_seed_same_repair(tmp_path):
    from backend.llm.client import llm_scope
    runs = []
    for tag in ("a", "b"):
        ctx, a, b = _ready(tmp_path / tag)
        with llm_scope(tag):
            turns = R.maybe_repair("c0", _utt(a, b), a, b, [], [[a.name, USE]])
        rec = _trace(ctx, "repair")[0]
        runs.append(([(t["id"], t["speaker"], t["text"], t["idx"], tuple(t["listeners"])) for t in turns],
                     {k: v for k, v in rec.items() if k not in ("id", "prompt")}))
    assert runs[0][0] and runs[0] == runs[1]


def test_the_draw_does_not_depend_on_what_happened_earlier(tmp_path):
    """The draw is addressed by (hearer, utterance, phrase), so it never shifts because some other
    repair happened first -- common random numbers across conditions (v2 §2.1)."""
    ctx, a, b = _ready(tmp_path)
    first = R.draw(b, "c0.u3", PHRASE).random()
    for _ in range(5):
        R.draw(b, "c9.u0", "something else").random()
        b.stream("repair").random()
    assert R.draw(b, "c0.u3", PHRASE).random() == first


# --------------------------------------------------------- what reaches an agent's eyes and the record
def test_the_question_never_contains_the_phrase(tmp_path):
    """A hearer who does not know an expression must never be recorded as having used it."""
    ctx, a, b = _ready(tmp_path)
    turns = R.maybe_repair("c0", _utt(a, b), a, b, [], [])
    assert not R.mentions(turns[0]["text"], PHRASE)
    for q in R.QUESTIONS:
        assert not R.mentions(q, PHRASE)


@pytest.mark.parametrize("question", R.QUESTIONS)
def test_a_repair_question_can_never_be_read_as_a_coinage(question, tmp_path):
    """The openers are made of ordinary words only, so neither the agent-side detector nor an observer
    scanning for coined expressions can mistake one for a phrase being introduced."""
    ctx = _world(tmp_path, ON)
    a, _ = _pair(ctx)
    assert R.candidate_spans(a, question, R._zipf_max(a)) == []


def test_no_canonical_place_name_and_no_config_gloss_reaches_generated_text(tmp_path):
    """R3: every agent-facing string is a description. R7: the meaning of a phrase is never read out of
    config -- a gloss parked there reaches neither the prompt nor the transcript."""
    from backend.simulation.world import ARENAS, WORLD_GRAPH

    def leaks(text):
        keys = list(WORLD_GRAPH) + [x for v in ARENAS.values() for x in v]
        return {k for k in keys if re.search(rf"(?<!\w){re.escape(k)}(?!\w)", text or "")}

    gloss = "ZZGLOSSZZ anything that looks fine and is not"
    ctx, a, b = _ready(tmp_path, {"conversation": {"repair": {
        "_gloss": gloss, "_registry": [{"phrase": PHRASE, "means": gloss}]}}})
    turns = R.maybe_repair("c0", _utt(a, b), a, b, [], [[a.name, USE]])
    prompt = _trace(ctx, "repair")[0]["prompt"]
    for t in [prompt, *(x["text"] for x in turns), *(u["context"] for u in _trace(ctx, "utterance"))]:
        assert "ZZGLOSSZZ" not in t
    # the persona ISS is supplied by the profile and is the same string every other prompt already
    # carries, so what this part must add is NOTHING: the situation line, the transcript, the retrieved
    # memories and the place slot name no place (D75).
    assert leaks(prompt) - leaks(a.iss()) == set(), leaks(prompt) - leaks(a.iss())
    assert leaks(R.reference.describe(a.state.location, a.state.arena, agent=a, cfg=a.cfg)) == set()
    for t in (R.ASK_LINE, R.COLLIDE_LINE, R.EXPLAIN_FOCAL, *R.QUESTIONS):
        assert leaks(t) == set()


def test_the_answer_is_the_speakers_own_remembered_words(tmp_path):
    """The mock returns nothing usable for this purpose, which exercises the no-LLM fallback: the agent
    speaks its own retrieved memory back. Either way the content comes from the memory stream."""
    ctx, a, b = _ready(tmp_path)
    turns = R.maybe_repair("c0", _utt(a, b), a, b, [], [])
    rec = _trace(ctx, "repair")[0]
    said = turns[1]["text"].lower()
    mine = [a.a_mem.id_to_node[n].description.lower() for n in rec["retrieved"]]
    assert rec["retrieved"] and any(said[:40].strip(".") in m for m in mine), (said, mine)
    assert rec["fallback"] is True and rec["grounded_in_phrase"] is True


def test_the_record_carries_the_fields_the_registry_entry_will_require(tmp_path):
    ctx, a, b = _ready(tmp_path)
    R.maybe_repair("c0", _utt(a, b), a, b, [], [])
    rec = _trace(ctx, "repair")[0]
    required = {"conversation_id", "utterance_id", "kind", "phrase", "channel", "shape", "asker",
                "answerer", "explanation", "explained", "competing", "turn_ids"}
    optional = {"agreement", "asker_memories", "answerer_memories", "retrieved", "grounded_in_phrase",
                "fallback", "prompt", "response"}
    assert required <= set(rec) and optional <= set(rec)
    assert {"id", "type", "tick", "time"} <= set(rec)          # the TraceLogger envelope


# ----------------------------------------------------------------------- a competing understanding
def _collide_world(tmp_path, below):
    ctx = _world(tmp_path, deep_merge(ON, {"conversation": {"repair": {"collide_below": below}}}))
    a, b = _pair(ctx)
    _remember(a, *SPEAKER_MEMS)
    _remember(b, "Marcus keeps calling the wobbly shelf in the bike shed a blue-tray for some reason.")
    return ctx, a, b


def test_a_hearer_who_remembers_it_differently_says_so(tmp_path):
    ctx, a, b = _collide_world(tmp_path / "on", 0.95)
    turns = R.maybe_repair("c0", _utt(a, b), a, b, [], [[a.name, USE]])
    assert [t["source"] for t in turns] == ["repair_collision"]
    assert turns[0]["speaker"] == b.id                      # the hearer offers THEIR account, unasked
    rec = _trace(ctx, "repair")[0]
    assert rec["kind"] == "collide" and rec["competing"] is True
    assert rec["asker"] == b.id and rec["answerer"] == b.id
    assert isinstance(rec["agreement"], float) and rec["agreement"] < 0.95
    assert rec["asker_memories"] and rec["answerer_memories"]


def test_the_collision_channel_is_off_by_default(tmp_path):
    """Off on a MEASUREMENT, not a preference: under the run's default hash embedder a hearer who agrees
    about a phrase and one who does not score in the same range, so any threshold would fire on wording
    rather than on meaning. The test states the measurement rather than trusting the comment."""
    from backend.llm.embeddings import cos
    ctx, a, b = _collide_world(tmp_path / "off", R.DEFAULTS["collide_below"])
    assert R.DEFAULTS["collide_below"] <= 0
    assert R.collisions(a, b, USE) == [] and R.maybe_repair("c0", _utt(a, b), a, b, [], []) == []
    use = ctx.embed(R._clause_with(USE, PHRASE))
    agree = [cos(use, ctx.embed(m)) for m in SPEAKER_MEMS]                       # same understanding
    differ = [cos(use, ctx.embed(m)) for m in (
        "Marcus keeps calling the wobbly shelf in the bike shed a blue-tray for some reason.",
        "They stuck a blue-tray label on every bike rack by the west gate last term.")]
    # measured: agree 0.296 / 0.262, differ 0.258 / 0.245. The gap BETWEEN the two groups is smaller
    # than the spread WITHIN either of them, i.e. the score tracks wording, not meaning.
    gap = min(agree) - max(differ)
    assert gap < max(max(agree) - min(agree), max(differ) - min(differ)), (agree, differ)


# ------------------------------------------------------------------------- matching the meme cohort
@pytest.mark.parametrize("phrase", COHORT)
def test_every_cohort_phrase_can_reach_the_repair_channel(phrase, tmp_path):
    """R6: a meme whose phrase no channel can see would have no repair channel at all and would lose on
    the mechanism rather than on grounding or breadth. `detectable` is the check to run per registry
    entry BEFORE a run."""
    ctx = _world(tmp_path, ON)
    a, _ = _pair(ctx)
    d = R.detectable(a, phrase)
    assert d["channel"] is not None, (phrase, d["reason"])


def test_a_leading_article_hides_a_phrase_from_every_channel(tmp_path):
    """Why the registry must declare "Tuesday list", not "the Tuesday list": a phrase whose edge is a
    function word is invisible, though the core of it is still found inside an utterance."""
    ctx = _world(tmp_path, ON)
    a, _ = _pair(ctx)
    assert R.detectable(a, "the Tuesday list")["channel"] is None
    found = [c["phrase"] for c in R.candidate_spans(a, "Don't forget the Tuesday list.", R._zipf_max(a))]
    assert "Tuesday list" in found


def test_rarity_alone_cannot_find_a_coinage_built_from_ordinary_words(tmp_path):
    """Measured, not assumed: wordfreq scores a compound at the frequency of its commonest reading, so
    the rare-word channel would see none of the cohort. The common-ground channel is load-bearing."""
    ctx = _world(tmp_path, deep_merge(ON, {"conversation": {"repair": {"phrase_source": "rare"}}}))
    a, _ = _pair(ctx)
    assert all(R.detectable(a, p)["channel"] is None for p in COHORT)
    assert R.detectable(a, "glorpwhistle")["channel"] == "rare"      # a true neologism still gets through


def test_ordinary_talk_proposes_nothing(tmp_path):
    ctx = _world(tmp_path, ON)
    a, _ = _pair(ctx)
    for line in ("Hey, how is your day going?", "I am heading to lunch, want to come along?",
                 "Yeah, I'll see you later then."):
        assert R.candidate_spans(a, line, R._zipf_max(a)) == [], line


# --------------------------------------------------------------------------- slotting into a conversation
def test_repair_turns_slot_into_a_conversation(tmp_path):
    """The turns must behave like the conversation's own: unique ids, listeners drawn from the people
    who were there, and -- the one that matters for the result -- they must still sort into the order
    they were said under the observer's chronological key."""
    from backend.analysis.rundata import chron_key
    from backend.llm.client import llm_scope
    ctx, a, b = _ready(tmp_path)
    with llm_scope("conv"):
        conv = CONV.run_conversation("c0", a, b, [], np.random.default_rng(5))
    utterances, chat, spoken = [], [], []
    for u in conv["utterances"]:
        utterances.append(u)
        chat.append([ctx.agents[u["speaker"]].name, u["text"]])
        spoken.append(u)
        for ru in R.maybe_repair("c0", u, ctx.agents[u["speaker"]],
                                 ctx.agents[u["listeners"][0]], utterances, chat):
            utterances.append(ru)
            chat.append([ctx.agents[ru["speaker"]].name, ru["text"]])
            spoken.append(ru)
    assert any(u["source"].startswith("repair") for u in utterances), "no repair fired; test is vacuous"
    ids = [u["id"] for u in utterances]
    assert len(ids) == len(set(ids))
    assert all(set(u["listeners"]) <= {a.id, b.id} for u in utterances)
    # a repair turn carries its trigger's idx, so the id breaks the tie the right way round and the
    # observer's own sort reproduces the order the lines were actually said in
    rows = [dict(u, tick=ctx.tracer.tick) for u in utterances]
    assert [r["id"] for r in sorted(rows, key=chron_key)] == [u["id"] for u in spoken]


def test_the_conversation_hook_is_wired():
    """`conversation.py` belongs to another part, so this states what the hook there must keep true: it
    calls the mechanism, and it records a bystander's overhearing by POSITION in `utterances`. Keyed by
    the loop index instead, the first repair would shift every later line and a bystander would be
    handed somebody else's words."""
    src = inspect.getsource(CONV.run_conversation)
    if "maybe_repair" not in src:
        pytest.skip("conversation.run_conversation does not call repair.maybe_repair yet (see contract)")
    assert "heard_by[b.id].append(i)" not in src, "overhearing is keyed by the loop index, not by position"
    assert "len(utterances)" in src
