"""Ontology v3 §2: the co-op binder (records.py), RecordWrite (agents/record_write.py) and the encoder's
`record` source. Light fakes (real Agent / memory / mock LLM, no engine)."""
import json
import random
import re
import types
import datetime as dt

import numpy as np
import pytest

from backend import ga_compat
from backend.agents import record_write as RW
from backend.agents.agent import Agent
from backend.agents.profile import load_population
from backend.config import deep_merge, load_config
from backend.llm.client import LLMClient, current_purpose
from backend.llm.embeddings import make_embedder
from backend.llm.mock import MockBackend
from backend.memory.encoder import encode
from backend.memory.store import SimMemoryMeta
from backend.modules.base import ModuleStack
from backend.simulation import records as R

T0 = dt.datetime(2026, 9, 14, 9, 0)
HIDDEN = [r"\bK[0-3]\b", r"\bLENS\b", r"\bDAMP\b", r"\bBELT\b", r"\bAIR\b", r"\bWARP\b", r"\bM[12]\b",
          r"j\d\d\.\d", r"\bregime\b", r"\bmapping\b", r"\bwipe4\b", r"\btrunk\b", r"binder-\d", r"\b[er]\d+\b"]


class ListTracer:
    def __init__(self):
        self.records = []

    def log(self, type_, **fields):
        self.records.append({"type": type_, **fields})

    def of(self, t):
        return [r for r in self.records if r["type"] == t]


def cfg_(**rec):
    return {"records": {"enabled": True, **rec}}


def filled(b=None):
    b = b or R.Binder()
    b.rewrite("dev", "Dev", 9, "Wavy lines on the right: tightened the belt knob under the bed, fine after.")
    b.append("hana", "Hana", 12, "Tea edges again, wiped the lens and reran.", job_id="j03.1")
    b.append("maya", "Maya", 14, "Slow cut on birch, ran it twice.", job_id="j04.2")
    return b


def assert_clean(text):
    for pat in HIDDEN:
        assert not re.search(pat, text), (pat, text)


# --------------------------------------------------------------------------- rendering
def test_view_format_provenance_and_no_hidden_fields():
    b = filled()
    v = b.render("current", 5)
    lines = v.text.splitlines()
    assert lines[0] == R.HEADER
    assert lines[1].startswith("Front page (last rewritten Mon ") and lines[1].endswith(" by Dev):")
    assert lines[3] == "Log, newest first:"
    assert lines[4].startswith("  [Mon ") and ", Maya] Slow cut" in lines[4]   # newest first, name and time
    assert v.entry_ids == ["e2", "e1"] and v.rev_ids == ["r1"]
    assert_clean(v.text)
    assert b.view("none", 5) == ""
    assert len(b.render("current", 1).entry_ids) == 1


def test_empty_binder_renders_the_same_whatever_the_reason():
    never = R.Binder().view("current", 5)
    wiped = filled()
    wiped.transition("wipe", 20)
    assert wiped.view("current", 5) == never == f"{R.HEADER}\n{R.EMPTY_TEXT}"
    assert wiped.view("history", 5) == never


def test_display_parity_keep_equals_no_transition():
    a, b = filled(), filled()
    assert b.transition("keep", 20) is None
    for mode in ("current", "history"):
        assert a.view(mode, 5) == b.view(mode, 5)


def test_history_shows_up_to_two_superseded_versions_and_history_from_day():
    b = R.Binder()
    for i, who in enumerate(["Ana", "Ben", "Cy", "Dee"]):
        b.rewrite(who.lower(), who, 4 * i, f"version {i} of the page")
    v = b.render("history", 5)
    assert v.rev_ids == ["r4", "r3", "r2"]
    assert v.text.count("Earlier front page (") == 2 and ", replaced Mon " in v.text
    assert "version 0" not in v.text
    c = cfg_(display="history", history_from_day=5)
    assert R.display_mode(c, 4) == "current" and R.display_mode(c, 5) == "history"
    assert R.display_mode({"records": {"enabled": False}}, 5) == "none"
    assert R.consult_mode(cfg_(consult="on_choice", consult_from_day=5), 4) == "always"
    assert R.consult_mode(cfg_(consult="on_choice", consult_from_day=5), 5) == "on_choice"


def test_manager_signed_front_page():
    b = filled(R.Binder(authority="manager_signed"))
    assert "Front page (kept by the shop manager):" in b.view("current", 5)
    assert b.front[-1].author == "dev"


# --------------------------------------------------------------------------- reading / receipts
def test_new_only_reads_and_receipts():
    b, tr, c = filled(), ListTracer(), cfg_()
    v, obs = R.read(b, c, tr, "leo", 20, 1, "decision", location="Makerspace", arena="laser")
    assert obs.source_type == "record" and [f["record_id"] for f in obs.facts] == ["r1", "e2", "e1"]
    assert all(f["salience"] == 0.5 and f["involves"] == [] for f in obs.facts)
    assert obs.facts[1]["text"] == v.lines["e2"] and obs.facts[1]["speaker"] == "maya"
    assert obs.facts[1]["text"] in "\n".join(l.strip() for l in v.text.splitlines())   # formatted as displayed
    # a re-read is not re-encoded
    _, again = R.read(b, c, tr, "leo", 21, 1, "tally")
    assert again is None
    b.append("hana", "Hana", 22, "Fine all afternoon.")
    _, third = R.read(b, c, tr, "leo", 23, 1, "decision")
    assert [f["record_id"] for f in third.facts] == ["e3"]
    reads = tr.of("record_read")
    assert [r["new_ids"] for r in reads] == [["r1", "e2", "e1"], [], ["e3"]]
    assert reads[0]["context"] == "decision" and reads[0]["view_mode"] == "current" and reads[0]["view_sha"]
    assert b.receipts["leo"]["r1"] == 20 and b.receipts["leo"]["e3"] == 23
    # the author never gets their own entry as new
    _, own = R.read(b, c, tr, "hana", 24, 1, "decision")
    assert "e3" not in [f["record_id"] for f in own.facts]
    # encode_on_read: never keeps receipts but builds no observation
    _, none = R.read(b, cfg_(encode_on_read="never"), tr, "wei", 25, 1, "decision")
    assert none is None and "r1" in b.receipts["wei"]


def test_relevant_retrieval_orders_by_similarity():
    b = filled()
    emb = make_embedder({"backend": "hash"})
    v = b.render("current", 1, retrieval="relevant", query="Tea edges lens", embed=emb)
    assert v.entry_ids == ["e1"]


# --------------------------------------------------------------------------- transitions
def test_wipe_archives_and_matched_facts():
    b, tr = filled(), ListTracer()
    c = cfg_(transitions=[{"day": 4, "mode": "wipe"}])
    assert R.apply_transitions(b, c, tr, 3, 30) == []
    facts = R.apply_transitions(b, c, tr, 4, 40)
    assert facts[0]["text"] == R.TRANSITION_FACT["wipe"] and facts[0]["salience"] == 0.5
    assert b.is_empty() and b.binder_id == "binder-2" and b.created_tick == 40
    d = b.to_dict()
    assert d["archived"][0]["binder_id"] == "binder-1" and len(d["archived"][0]["log"]) == 2
    assert d["archived"][0]["log"][0]["job_id"] == "j03.1"          # world side keeps everything
    assert tr.of("record_transition") == [{"type": "record_transition", "day": 4, "mode": "wipe",
                                           "archived_binder_id": "binder-1"}]
    k = filled()
    kf = R.apply_transitions(k, cfg_(transitions=[{"day": 4, "mode": "keep"}]), tr, 4, 40)
    assert kf[0]["text"] == R.TRANSITION_FACT["keep"] and not k.archived and not k.is_empty()
    # new ids stay unique across binders
    b.append("leo", "Leo", 41, "first note in the new one")
    assert b.log[0].entry_id == "e3"


def test_roundtrip_serialisation():
    b = filled()
    R.read(b, cfg_(), ListTracer(), "leo", 20, 1, "decision")
    b.transition("wipe", 30)
    b.append("leo", "Leo", 31, "hello")
    d = b.to_dict()
    b2 = R.Binder.from_dict(json.loads(json.dumps(d)))
    assert b2.to_dict() == d and b2.view("current", 5) == b.view("current", 5)
    assert json.dumps(d, sort_keys=True) == json.dumps(b2.to_dict(), sort_keys=True)


# --------------------------------------------------------------------------- writing
def test_writes_apply_in_sorted_tick_agent_order_and_truncate():
    decs = [{"agent": a, "author_name": a.title(), "tick": t, "offer": "job", "choice": ch, "text": f"{a} at {t}"}
            for a, t, ch in [("zoe", 10, "front"), ("ana", 10, "front"), ("leo", 9, "log"), ("ana", 9, "none")]]
    decs.append({"agent": "bo", "author_name": "Bo", "tick": 10, "offer": "tally", "choice": "log",
                 "text": "word " * 80})
    random.Random(3).shuffle(decs)
    b, tr = R.Binder(), ListTracer()
    applied = R.apply_writes(b, cfg_(), tr, decs)
    assert [(x["tick"], x["agent"]) for x in applied] == [(9, "ana"), (9, "leo"), (10, "ana"), (10, "bo"), (10, "zoe")]
    assert b.front[-1].author == "zoe" and b.front[-1].revision_of == "r1" and len(b.front) == 2
    bo = [x for x in applied if x["agent"] == "bo"][0]
    assert bo["truncated"] and len(bo["text"]) <= 200 and bo["text"].endswith("word")
    writes = tr.of("record_write")
    assert len(writes) == 5 and writes[0]["choice"] == "none" and writes[1]["entry_id"] == "e1"
    assert b.log[1].context == "tally"
    # log only: a front choice becomes a log note
    b2 = R.Binder()
    R.apply_writes(b2, cfg_(revisable=False), tr, [dict(decs[0], choice="front", text="x y")])
    assert not b2.front and len(b2.log) == 1


def test_truncate_words():
    assert R.truncate_words("a b c", 10) == ("a b c", False)
    t, tr = R.truncate_words("alpha beta gamma delta", 12)
    assert tr and t == "alpha beta" and len(t) <= 12


def test_offer_enabled():
    assert R.offer_enabled(cfg_(), "job") and R.offer_enabled(cfg_(), "farewell")
    assert not R.offer_enabled(cfg_(write_after=["tally"]), "job")
    assert not R.offer_enabled({}, "tally")


# --------------------------------------------------------------------------- agents: RecordWrite + encoding
class Scripted(MockBackend):
    def __init__(self, replies):
        super().__init__()
        self.replies = replies

    def generate(self, prompt, system, max_tokens, temperature):
        r = self.replies.get(current_purpose())
        if r is None:
            return super().generate(prompt, system, max_tokens, temperature)
        return r(prompt) if callable(r) else r


def world(tmp_path, overrides=None, backend=None):
    cfg = load_config("configs/baseline.yaml", deep_merge({"llm": {"backend": "mock"}}, overrides or {}))
    res = load_population(cfg["population"], cfg.get("population_size"))
    profiles = res[0] if isinstance(res, tuple) else res
    llm = LLMClient(backend or MockBackend(), tmp_path / "llm_calls.jsonl")
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


FORBIDDEN = ("invent", "coin", "name", "label", "term", "slang", "meme", "nickname", "shorthand", "rule", "tip",
             "procedure", "jot")


def test_record_write_template_is_neutral():
    body = open(RW.PROMPT).read().split("<commentblockmarker>###</commentblockmarker>")[-1].lower()
    for w in FORBIDDEN:
        assert w not in body, w


@pytest.mark.parametrize("reply,expect", [
    ('{"choice": "log", "text": "Wiped the lens, fine after."}', ("log", "Wiped the lens, fine after.")),
    ('{"choice": "front", "text": "Check the knob."}', ("front", "Check the knob.")),
    ('{"choice": "none", "text": null}', ("none", None)),
    ('{"choice": "log", "text": null}', ("none", None)),
    ("LLM_ERROR: timeout", ("none", None)),
    ("garbage", ("none", None)),
])
def test_decide_write_parses_and_prompt_is_clean(tmp_path, reply, expect):
    ctx = world(tmp_path, backend=Scripted({"record_write": reply}))
    leo = ctx.agents["leo"]
    b = filled()
    d = RW.decide_write(leo, "job", b.view("current", 5), ["The cut came out wavy on the right."], tick=12,
                        job_id="j03.1")
    assert (d["choice"], d["text"]) == expect
    p = d["prompt"]
    assert "Leo just finished a job on the co-op's laser cutter." in p and "The cut came out wavy" in p
    assert "(up to 200 characters)" in p and "(up to 600 characters)" in p and R.HEADER in p
    assert_clean(p)
    body = p.lower().split("is there anything")[-1]
    for w in FORBIDDEN:
        assert w not in body, w


def test_decide_write_farewell_retrieves_and_mock_roundtrip(tmp_path):
    ctx = world(tmp_path)
    leo = ctx.agents["leo"]
    b = R.Binder()
    out = []
    for t in range(6):
        d = RW.decide_write(leo, "farewell", b.view("current", 5), None, tick=t)
        assert d["choice"] in RW.CHOICES and "last shift at the co-op" in d["prompt"]
        out.append(d)
    applied = R.apply_writes(b, cfg_(), ctx.tracer, out)
    for a in applied:
        RW.remember_write(leo, a)
    calls = [json.loads(l) for l in open(tmp_path / "llm_calls.jsonl")]
    assert any(c["purpose"] == "record_write" for c in calls)
    notlog = RW.decide_write(leo, "tally", "", ["A quiet day."], tick=9, rng=np.random.default_rng(0))
    assert R.EMPTY_TEXT in notlog["prompt"]
    ro = RW.options(deep_merge(leo.cfg, {"records": {"revisable": False}}), "Leo")
    assert "front" not in ro[1] and "rewrite" not in ro[0]


def test_author_gets_a_self_memory(tmp_path):
    ctx = world(tmp_path)
    leo = ctx.agents["leo"]
    b, tr = R.Binder(), ListTracer()
    [a] = R.apply_writes(b, cfg_(), tr, [{"agent": "leo", "author_name": "Leo", "tick": 3, "offer": "job",
                                           "choice": "log", "text": "Wiped the lens, fine after."}])
    node = RW.remember_write(leo, a)
    m = ctx.meta.get(node.node_id)
    assert m.source_type == "self" and m.record_ids == ["e1"]
    assert node.description.startswith("Leo") and "wrote in the log of the co-op binder" in node.description
    assert RW.remember_write(leo, {"choice": "none"}) is None


def test_record_observation_encodes_with_record_ids_and_verbatim(tmp_path):
    over = {"memory": {"verbatim": {"enabled": True, "base": 1.0, "max": 1.0,
                                    "on_sources": ["conversation", "overheard", "record"]},
                       "merge_threshold": 1.1}}
    ctx = world(tmp_path, over)
    leo = ctx.agents["leo"]
    b = R.Binder()
    b.append("maya", "Maya", 3, "The gremlin knob wobbled, so I tightened the belt knob.")
    b.append("leo", "Leo", 4, "Zamboni situation with the bed again.")
    b.receipts.pop("leo")                                   # pretend Leo forgot he read it (exclude_self test)
    _, obs = R.read(b, cfg_(), ctx.tracer, "leo", 5, 1, "decision", location=leo.state.location,
                    arena=leo.state.arena)
    node = encode(leo, obs, np.random.default_rng(0))
    m = ctx.meta.get(node.node_id)
    assert m.source_type == "record" and m.record_ids == ["e2", "e1"]
    enc = ctx.tracer.of("memory_encoded")[-1]
    assert enc["record_ids"] == ["e2", "e1"] and "read in the co-op binder" in enc["prompt"]
    assert node.type == "event"
    ws = ctx.tracer.of("wording")
    assert ws and all(w["heard_from"] == "maya" for w in ws)           # own entries never stick
    assert not any("zamboni" in w["phrase"].lower() for w in ws)
    # default on_sources (no record) -> nothing sticks from the binder
    ctx2 = world(tmp_path / "b", {"memory": {"verbatim": {"enabled": True, "base": 1.0, "max": 1.0},
                                             "merge_threshold": 1.1}})
    _, obs2 = R.read(filled(), cfg_(), ctx2.tracer, "leo", 5, 1, "decision")
    encode(ctx2.agents["leo"], obs2, np.random.default_rng(0))
    assert not ctx2.tracer.of("wording")


def test_records_module_never_imports_observer():
    import pathlib
    for p in ["backend/simulation/records.py", "backend/agents/record_write.py"]:
        src = pathlib.Path(p).read_text()
        assert "backend.analysis" not in src
