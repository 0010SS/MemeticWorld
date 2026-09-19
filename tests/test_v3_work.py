"""Part A (agent work): job decisions, job episodes, co-op talk helpers (ontology v3 §4.2-4.5). Mock only."""
from __future__ import annotations

import datetime as dt
import json
import re
import types

import pytest

from backend import ga_compat
from backend.agents import coop_talk
from backend.agents.agent import Agent
from backend.agents.profile import load_population
from backend.agents.work import (PROMPT, JobView, build_options, decide_job, encode_job, job_episode,
                                 make_job_view, parse_choice, remark_of)
from backend.config import deep_merge, load_config
from backend.llm.client import LLMClient, current_purpose
from backend.llm.embeddings import make_embedder
from backend.llm.mock import MockBackend
from backend.memory.store import SimMemoryMeta
from backend.modules.base import ModuleStack

T0 = dt.datetime(2026, 9, 17, 10, 0)
JOB = {"id": "j04.2", "kind": "job", "day": 4, "slot": 2, "shift": "am", "start_tick": 194, "operator": "dev",
       "project": "tube racks for the bio lab",
       "menu_order": [["dry", "rerun", "stop", "slow", "lens", "belt"], ["belt", "slow", "lens", "rerun", "dry", "stop"]],
       "hidden": {"fault": True, "class": {"A": "K1", "B": "K1"}, "cause": {"A": "LENS", "B": "DAMP"}}}
SYMPTOM = ("Dev's first sheet came out with the left third still attached, and the edges there were the color "
           "of weak tea.")
BINDER = ('The co-op binder next to the laser cutter.\nFront page (last rewritten Wed 11:20 by Dev):\n'
          '  "Wavy lines on the right: tightened the belt knob under the bed, fine after."\nLog, newest first:\n'
          '  [Wed 16:05, Hana] left side stuck again, wiped the lens.')
FORBIDDEN_TEMPLATE = r"\b(invent\w*|coin\w*|name|names|named|naming|label\w*|term|terms|slang|meme\w*|nickname\w*|" \
                     r"shorthand|rule\w*|tip|tips|procedure\w*|jot\w*)\b"
HIDDEN = re.compile(r"\bK[0-3]\b|LENS|DAMP|BELT|\bAIR\b|WARP|\bM[12]\b|j\d\d\.\d|regime|mapping|trunk|wipe4|"
                    r"keep_shift|noshift|placebo|laser_alpha|laser_beta")


class ListTracer:
    tick, time = 194, ""

    def __init__(self):
        self.records = []

    def log(self, type_, **fields):
        self.records.append({"type": type_, **fields})

    def of(self, t):
        return [r for r in self.records if r["type"] == t]


class Scripted(MockBackend):
    def __init__(self, replies):
        super().__init__()
        self.replies = replies
        self.calls = []

    def generate(self, prompt, system, max_tokens, temperature):
        p = current_purpose()
        self.calls.append((p, prompt))
        r = self.replies.get(p)
        if r is None:
            return super().generate(prompt, system, max_tokens, temperature)
        if isinstance(r, list):
            return r.pop(0) if len(r) > 1 else r[0]
        return r(prompt) if callable(r) else r


def world(tmp_path, overrides=None, backend=None, name="w"):
    base = {"llm": {"backend": "mock"},
            "records": {"enabled": True, "consult": "always"},
            "comm": {"clarify": {"enabled": True, "max_per_job": 1, "max_utterances": 4}},
            "workshop": {"encode_reason": True}}
    cfg = load_config("configs/baseline.yaml", deep_merge(base, overrides or {}))
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
        a.state.location, a.state.arena = "Research Lab", "Makerspace"
        a.sync_scratch()
        ctx.agents[pid] = a
    return ctx


def view(ctx, attempt=1, present=("maya", "hana"), asked=0):
    return make_job_view(JOB, attempt, [SYMPTOM, "The panel showed F4."], [ctx.agents[p] for p in present],
                         questions_asked=asked)


def menu_lines(prompt):
    return re.findall(r"^([A-L])\. (.+)$", prompt.split("do next?\n")[1].split("Answer with")[0], re.M)


# --------------------------------------------------------------------------- decision
def test_decide_job_menu_order_trace_and_no_hidden_ids(tmp_path):
    ctx = world(tmp_path)
    dev = ctx.agents["dev"]
    d = decide_job(dev, view(ctx), BINDER, dev.stream("work"))
    assert d["valid"] and d["action"] in {"rerun", "lens", "dry", "belt", "slow", "stop", "ask"}
    lines = menu_lines(d["prompt"])
    assert [t for _, t in lines[:6]] == ["dry the sheets on the heated rack for 15 minutes, then re-run",
                                         "re-run the sheet as it is", "stop and leave the job for later",
                                         "slow the cutting speed down, then re-run",
                                         "clean the focus lens, then re-run", "tighten the drive belt, then re-run"]
    assert [t for _, t in lines[6:]] == ["ask Hana something first", "ask Maya something first"]
    assert "It is Thursday 10:00. Dev is on the morning shift" in d["prompt"]
    assert "tube racks for the bio lab" in d["prompt"] and BINDER.splitlines()[1] in d["prompt"]
    assert "People here: Maya" in d["prompt"] or "People here: Hana" in d["prompt"]
    assert not HIDDEN.search(d["prompt"]), HIDDEN.search(d["prompt"])
    tr = ctx.tracer.of("job_decision")[-1]
    for k in ("agent", "job", "attempt", "menu_order", "choice", "action", "question", "says_aloud", "reason",
              "retrieved", "binder_view_sha", "binder_entry_ids", "prompt", "response"):
        assert k in tr
    assert tr["job"] == "j04.2" and tr["menu_order"] == JOB["menu_order"][0] and tr["binder_view_sha"]


def test_attempt_two_uses_second_order_and_is_deterministic(tmp_path):
    ctx1, ctx2 = world(tmp_path, name="a"), world(tmp_path, name="b")
    d1 = decide_job(ctx1.agents["dev"], view(ctx1, attempt=2), BINDER, ctx1.agents["dev"].stream("work"))
    d2 = decide_job(ctx2.agents["dev"], view(ctx2, attempt=2), BINDER, ctx2.agents["dev"].stream("work"))
    assert d1["prompt"] == d2["prompt"] and d1["action"] == d2["action"]
    assert menu_lines(d1["prompt"])[0][1] == "tighten the drive belt, then re-run"


@pytest.mark.parametrize("consult,shown", [("always", True), ("never", False)])
def test_binder_shown_only_when_consult_always(tmp_path, consult, shown):
    ctx = world(tmp_path, {"records": {"consult": consult}})
    d = decide_job(ctx.agents["dev"], view(ctx), {"text": BINDER, "entry_ids": ["e1"], "rev_ids": ["r1"]})
    assert (BINDER.splitlines()[0] in d["prompt"]) is shown
    assert d["binder_shown"] is shown and d["binder_entry_ids"] == (["e1", "r1"] if shown else [])


def test_records_disabled_hides_binder(tmp_path):
    ctx = world(tmp_path, {"records": {"enabled": False}})
    d = decide_job(ctx.agents["dev"], view(ctx), BINDER)
    assert "co-op binder" not in d["prompt"]


def test_on_choice_offers_binder_then_reasks_with_view(tmp_path):
    def reply(prompt):
        opts = dict((t, k) for k, t in menu_lines(prompt))
        if "look through the binder first" in opts:
            return json.dumps({"choice": opts["look through the binder first"], "question": None,
                               "says_aloud": None, "reason": "check"})
        return json.dumps({"choice": opts["clean the focus lens, then re-run"], "question": None,
                           "says_aloud": None, "reason": "edges"})
    be = Scripted({"job_decision": reply})
    ctx = world(tmp_path, {"records": {"consult": "on_choice"}}, backend=be)
    d = decide_job(ctx.agents["dev"], view(ctx), BINDER)
    prompts = [p for pur, p in be.calls if pur == "job_decision"]
    assert len(prompts) == 2 and "co-op binder" not in prompts[0] and "co-op binder" in prompts[1]
    assert d["action"] == "lens" and d["binder_chosen"] and d["binder_shown"]


def test_ask_option_rules(tmp_path):
    ctx = world(tmp_path)
    dev = ctx.agents["dev"]
    jv = view(ctx, present=())
    assert all(o["kind"] == "action" for o in build_options(jv, ask=False, binder_option=False))
    d = decide_job(dev, view(ctx, present=()), None)
    assert "ask " not in "\n".join(t for _, t in menu_lines(d["prompt"]))
    assert '"question": null' in d["prompt"]
    d = decide_job(dev, view(ctx, asked=1), None)          # max_per_job = 1 already used
    assert "ask " not in "\n".join(t for _, t in menu_lines(d["prompt"]))
    ctx2 = world(tmp_path, {"comm": {"clarify": {"enabled": False}}}, name="noclar")
    d = decide_job(ctx2.agents["dev"], view(ctx2), None)
    assert "ask " not in "\n".join(t for _, t in menu_lines(d["prompt"]))


def test_choice_g_returns_ask_and_clarify_talk(tmp_path):
    def reply(prompt):
        k = dict((t, k) for k, t in menu_lines(prompt))["ask Maya something first"]
        return json.dumps({"choice": k, "question": "Is it the lens again?", "says_aloud": None, "reason": "unsure"})
    ctx = world(tmp_path, backend=Scripted({"job_decision": reply}))
    dev = ctx.agents["dev"]
    d = decide_job(dev, view(ctx), BINDER)
    assert d["action"] == "ask" and d["ask"] == {"target": "maya", "question": "Is it the lens again?"}
    init, tgt, opening, trig = coop_talk.clarify_talk(dev, ctx.agents["maya"], d["ask"]["question"], "j04.2")
    assert (init, tgt, opening) == ("dev", "maya", "Is it the lens again?") and trig["topic"] == "clarify"
    assert trig["event_ids"] == ["j04.2"] and trig["text"] is None


def test_unparseable_retries_once_then_stop(tmp_path):
    be = Scripted({"job_decision": "I think I'd clean it"})
    ctx = world(tmp_path, backend=be)
    d = decide_job(ctx.agents["dev"], view(ctx), BINDER)
    assert d["action"] == "stop" and not d["valid"] and d["choice"] is None
    assert sum(1 for p, _ in be.calls if p == "job_decision") == 2
    assert ctx.tracer.of("job_decision")[-1]["valid"] is False


def test_retry_succeeds(tmp_path):
    be = Scripted({"job_decision": ["garbage", json.dumps({"choice": "B", "reason": "x"})]})
    ctx = world(tmp_path, backend=be)
    d = decide_job(ctx.agents["dev"], view(ctx), None)
    assert d["valid"] and d["action"] == "rerun"          # B in attempt-1 order is rerun


def test_parse_choice_variants():
    jv = JobView(job_id="x", attempt=1, menu_order=["rerun", "lens", "dry", "belt", "slow", "stop"], project="p")
    opts = build_options(jv, ask=False, binder_option=False)
    assert parse_choice('{"choice": "b."}', opts)["option"]["action"] == "lens"
    assert parse_choice('{"choice": "(C)"}', opts)["option"]["action"] == "dry"
    assert parse_choice('{"choice": "stop"}', opts)["option"]["action"] == "stop"
    assert parse_choice('{"choice": "Z"}', opts) is None
    assert parse_choice('{"choice": "A", "says_aloud": "null"}', opts)["says_aloud"] is None


def test_says_aloud_becomes_remark(tmp_path):
    be = Scripted({"job_decision": json.dumps({"choice": "A", "question": None, "says_aloud": "tea edges again?",
                                               "reason": "x"})})
    ctx = world(tmp_path, backend=be)
    d = decide_job(ctx.agents["dev"], view(ctx), None)
    r = remark_of(d)
    assert r["action"] == "REACT" and r["utterance"] == "tea edges again?"


def test_template_audit():
    text = open(PROMPT).read()
    assert not re.search(FORBIDDEN_TEMPLATE, text, re.I), re.search(FORBIDDEN_TEMPLATE, text, re.I)
    assert not HIDDEN.search(text)


# --------------------------------------------------------------------------- job episodes
def _fact(i, text, involves=(), rendered=True):
    return {"id": f"j04.2.b{i}.f0", "text": text, "world_text": text, "salience": 0.6, "involves": list(involves),
            "visibility": "arena", "vantage": "participant" if involves else "near", "recognised": [],
            "rendered": rendered}


def test_job_episode_one_obs_with_inner_reason(tmp_path):
    ctx = world(tmp_path)
    dev = ctx.agents["dev"]
    facts = [_fact(0, SYMPTOM, ["dev"]), _fact(1, "Dev cleaned the focus lens and re-ran the sheet; the cut went "
                                                   "all the way through.", ["dev"])]
    obs = job_episode(dev, "j04.2", facts, tick=196, reasons=["tea-colored edges", None])
    assert obs.source_type == "perception" and obs.event_ids == ["j04.2"] and len(obs.facts) == 3
    assert obs.facts[-1]["text"] == "Dev had thought: tea-colored edges" and obs.facts[-1]["kind"] == "inner"
    ctx2 = world(tmp_path, {"workshop": {"encode_reason": False}}, name="nr")
    obs2 = job_episode(ctx2.agents["dev"], "j04.2", facts, tick=196, reasons=["tea-colored edges"])
    assert len(obs2.facts) == 2
    assert job_episode(dev, "j04.2", [], tick=196) is None


def test_encode_job_links_memory_to_job(tmp_path):
    ctx = world(tmp_path)
    maya = ctx.agents["maya"]
    facts = [_fact(0, SYMPTOM, ["dev"], rendered=False),
             _fact(1, "Dev cleaned the focus lens and re-ran the sheet; the cut went all the way through.", ["dev"],
                   rendered=False)]
    obs, node = encode_job(maya, JOB, {"facts": facts, "tick": 197})
    assert node is not None and all(f["rendered"] for f in obs.facts)
    assert ctx.meta.get(node.node_id).originating_event_ids == ["j04.2"]
    assert ctx.tracer.of("viewpoint") and ctx.tracer.of("memory_encoded")


# --------------------------------------------------------------------------- co-op talk
def test_coop_talk_lines_and_budgets(tmp_path):
    ctx = world(tmp_path, {"comm": {"handover": {"enabled": True}, "meeting": {"enabled": True}}})
    dev, maya = ctx.agents["dev"], ctx.agents["maya"]
    assert coop_talk.dyad_topic_line("clarify", dev, maya) == \
        f"{dev.name} is running a job on the laser cutter and asks {maya.name} something."
    assert coop_talk.speaker_topic_line("clarify", maya, dev, False) == coop_talk.dyad_topic_line("clarify", dev, maya)
    assert coop_talk.dyad_topic_line("handover", dev, maya).endswith("are at the laser cutter at the shift change.")
    assert coop_talk.dyad_topic_line("catchup", dev, maya) is None
    cfg = ctx.cfg
    assert coop_talk.max_utterances(cfg, "clarify") == 4 and coop_talk.max_utterances(cfg, "handover") == 6
    assert coop_talk.max_utterances(cfg, "meeting") == 10 and coop_talk.max_utterances(cfg, "catchup") is None
    lines = coop_talk.register_group_topics({})
    assert lines["meeting"].format(names="x") == coop_talk.MEETING_LINE
    assert coop_talk.is_meeting_day(cfg, 2) and not coop_talk.is_meeting_day(cfg, 3)
    assert coop_talk.is_time(cfg, "meeting", dt.datetime(2026, 9, 15, 18, 30))
    assert coop_talk.meeting_participants(cfg, ["leo", "dev", "maya"]) == ["dev", "leo", "maya"]
    jobs = [dict(JOB, slot=s, shift="am" if s < 4 else "pm", start_tick=180 + k, operator=op)
            for s, (k, op) in enumerate(zip([6, 10, 14, 18, 24, 28, 32, 36],
                                            ["dev", "maya", "hana", "dev", "leo", "ethan", "jordan", "leo"]))]
    assert coop_talk.handover_pair(jobs) == ("dev", "leo")
    t = coop_talk.handover_talk(dev, ctx.agents["leo"])
    assert t[:3] == ("dev", "leo", None) and t[3]["topic"] == "handover"
    for line in (coop_talk.CLARIFY_LINE, coop_talk.HANDOVER_LINE, coop_talk.MEETING_LINE):
        assert not re.search(FORBIDDEN_TEMPLATE, line, re.I)
    rec = coop_talk.clarification_record("j04.2", 1, "dev", "maya", "q?", {"id": "c0", "participants": ["dev", "maya"],
                                                                          "utterances": [{"id": "c0.u0"}]})
    assert rec["utterance_ids"] == ["c0.u0"] and rec["n_utterances"] == 1
