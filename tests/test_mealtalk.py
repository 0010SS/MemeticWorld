"""`conversation.mealtalk` and `initial_memories_importance`: a reason to refer, and a retrievable seed.

The seeded naming experiment asks whose name for a place wins. For that to be measurable an agent has to
(a) have a reason to refer to the place at all, and (b) be able to recall what it calls it at that moment.
This module covers exactly those two, and the invariants that keep them honest:

  * the cue is an OCCASION, never a name -- the line it adds mentions no place and asks nobody to invent
    or adopt a word for one;
  * it fires only inside a configured meal window, only when someone still has a meal ahead of them in
    the plan the WORLD drew (not from any place name), and only with probability `prob`;
  * switching it on does not shift any other random draw (common random numbers across conditions), so a
    cell with the cue off is still comparable with one that has it on;
  * a seed memory's importance is configurable and keeps its v2 default, and the seed stays a MEMORY --
    it lands in the memory stream and in nothing that makes it a permanent linguistic habit;
  * the two shipped naming cells differ in the WORDING keys and in nothing else.

Mock LLM only; no live calls.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest

from backend import ga_compat
from backend.agents import conversation as CONV
from backend.agents.profile import FORBIDDEN_FIELDS, load_population
from backend.config import deep_merge, load_config
from backend.modules.base import build_modules
from backend.tracing import schema as TS

CELL1 = "configs/homewood100_naming.yaml"
CELL2 = "configs/homewood100_naming_wording.yaml"
# Exactly the WORDING block (plus the run's own name). Anything else appearing here means the two cells
# stopped being a single-mechanism comparison.
WORDING_KEYS = {"run_name", "memory.verbatim.enabled", "priming.enabled",
                "retrieval.source_weights", "conversation.catchup.enabled", "conversation.catchup.after"}
LUNCH = {"k": 14, "location": "Dining Hall", "arena": "Main Floor", "activity": "having lunch"}
AFTER = {"k": 20, "location": "Library", "arena": "Study Tables", "activity": "reading and preparing work"}


def _cfg(over=None):
    return load_config("configs/baseline.yaml", deep_merge(
        {"llm": {"backend": "mock"}, "day_start": "08:30", "day_end": "17:00"}, over or {}))


def _world(tmp_path, over=None, n=4, tick=12):
    """A minimal context: four agents in one room, a day plan with lunch ahead of them, mock LLM."""
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
        a.day_plan = [dict(LUNCH), dict(AFTER)]
        a.state.location, a.state.arena, a.state.activity = "Quad", "Lawn", "reading and preparing work"
        a.sync_scratch()
        add_simple_event(a, "The dryer in the dorm basement ate someone's laundry card again.", 5, [])
    return ctx


def _pair(ctx):
    ids = sorted(ctx.agents)[:2]
    return ctx.agents[ids[0]], ctx.agents[ids[1]]


def _trace(ctx):
    ctx.tracer.flush()
    return [json.loads(l) for l in open(ctx.tracer.run_dir / "trace.jsonl")]


ON = {"conversation": {"mealtalk": {"enabled": True, "prob": 1.0}}}


# --------------------------------------------------------------------------------- windows and eligibility
def test_meal_window_matches_inclusively_and_only_inside():
    cfg = _cfg()
    at = lambda h, m: dt.datetime(2026, 9, 14, h, m)  # noqa: E731
    assert CONV.meal_window(cfg, at(11, 30)) == "11:30-13:30"     # inclusive at both ends
    assert CONV.meal_window(cfg, at(13, 30)) == "11:30-13:30"
    assert CONV.meal_window(cfg, at(18, 0)) == "17:30-19:00"
    assert CONV.meal_window(cfg, at(11, 29)) is None
    assert CONV.meal_window(cfg, at(13, 31)) is None
    assert CONV.meal_window(cfg, at(15, 0)) is None
    assert CONV.meal_window(_cfg({"conversation": {"mealtalk": {"windows": []}}}), at(12, 0)) is None


def test_next_step_is_meal_reads_the_plan_not_the_place(tmp_path):
    """Eligibility is 'there is still a meal ahead in the plan the world drew'. Standing in the dining hall
    is not a meal, and sitting at lunch already is not either -- that agent has decided where to eat."""
    ctx = _world(tmp_path, tick=12)                       # 11:30, lunch at tick 14
    a, _ = _pair(ctx)
    assert CONV.next_step_is_meal(a)
    a.state.location, a.state.arena = "Dining Hall", "Main Floor"    # the place never enters into it
    assert CONV.next_step_is_meal(a)
    ctx.tracer.tick = 14                                  # at the table: the next step is the library
    assert not CONV.next_step_is_meal(a)
    ctx.tracer.tick = 12
    a.day_plan = [dict(AFTER)]                            # no meal in the plan at all
    assert not CONV.next_step_is_meal(a)
    a.day_plan = []
    assert not CONV.next_step_is_meal(a)


# ------------------------------------------------------------------------------------------ the cue itself
def test_cue_is_off_by_default_and_gated_on_window_plan_and_probability(tmp_path):
    assert CONV.mealtalk("c0", list(_pair(_world(tmp_path / "off")))) is None            # default: off
    on = _world(tmp_path / "on", ON, tick=12)                                            # 11:30, lunch ahead
    assert CONV.mealtalk("c0", list(_pair(on))) == {"focal": CONV.MEALTALK_FOCAL}
    late = _world(tmp_path / "late", ON, tick=26)                                        # 15:00: no window
    assert CONV.mealtalk("c0", list(_pair(late))) is None
    fed = _world(tmp_path / "fed", ON, tick=12)
    for a in fed.agents.values():
        a.day_plan = [dict(AFTER)]                                                       # nobody left to eat
    assert CONV.mealtalk("c0", list(_pair(fed))) is None
    never = _world(tmp_path / "never", deep_merge(ON, {"conversation": {"mealtalk": {"prob": 0.0}}}), tick=12)
    assert CONV.mealtalk("c0", list(_pair(never))) is None
    one = _world(tmp_path / "one", ON, tick=12)
    a, b = _pair(one)
    b.day_plan = [dict(AFTER)]                                # ONE participant with a meal ahead is enough
    assert CONV.mealtalk("c0", [a, b]) is not None


def test_cue_decision_is_stable_per_conversation_and_independent_of_other_draws(tmp_path):
    """Same tick and participants -> same decision, whatever else has drawn random numbers; and the
    decision depends on WHO is talking and WHEN, not on call order."""
    ctx = _world(tmp_path, deep_merge(ON, {"conversation": {"mealtalk": {"prob": 0.5}}}), n=4, tick=12)
    ids = sorted(ctx.agents)
    pair = [ctx.agents[ids[0]], ctx.agents[ids[1]]]
    first = CONV.mealtalk("c0", pair)
    for _ in range(3):
        ctx.agents[ids[2]].stream("talk").random()             # unrelated draws elsewhere
        assert CONV.mealtalk("c0", pair) == first
    assert CONV.mealtalk("c0", list(reversed(pair))) == first  # participant order does not matter
    outcomes = {tuple(sorted((x, y))): CONV.mealtalk("c0", [ctx.agents[x], ctx.agents[y]])
                for x in ids for y in ids if x < y}
    assert len(set(map(str, outcomes.values()))) == 2          # prob 0.5: some pairs fire, some do not


def test_traced_record_carries_its_registered_fields(tmp_path):
    ctx = _world(tmp_path, ON, tick=12)
    a, b = _pair(ctx)
    assert CONV.mealtalk("d1t0012c0", [a, b]) is not None
    recs = [r for r in _trace(ctx) if r["type"] == "mealtalk"]
    assert len(recs) == 1
    assert TS.validate(recs[0]) == []                          # registered, with every required field
    assert recs[0]["conversation_id"] == "d1t0012c0"
    assert recs[0]["participants"] == sorted([a.id, b.id]) and recs[0]["window"] == "11:30-13:30"
    assert TS.TRACE_TYPES["mealtalk"]["required"] == ["conversation_id", "participants", "window"]
    assert TS.strip(recs[0]) == recs[0]                        # nothing hidden: it is not ground truth


# ---------------------------------------------------------------------------- what reaches the agent's eyes
def test_added_line_gives_an_occasion_and_never_a_name(tmp_path):
    ctx = _world(tmp_path, ON, tick=12)
    a, b = _pair(ctx)
    plain = CONV._context(a, b, True, None)
    cued = CONV._context(a, b, True, None, meal={"focal": CONV.MEALTALK_FOCAL})
    added = [l for l in cued.splitlines() if l not in plain.splitlines()]
    assert added == [f"{a.name} and {b.name} have not eaten yet and are working out where to eat."]
    assert len(cued.splitlines()) == len(plain.splitlines()) + 1        # exactly ONE line, once
    text = added[0] + " " + CONV.MEALTALK_FOCAL
    from backend.simulation.world import ARENAS
    for place in ARENAS:                                                # no canonical place name anywhere
        assert not re.search(rf"\b{re.escape(place)}\b", text, re.I), place
    for word in ("FFC", "Hopkins", "cafe", "call it", "called", "name", "nickname", "invent", "refer to it as"):
        assert word.lower() not in text.lower(), word


def test_cue_adds_a_meal_focal_point_and_changes_nothing_else(tmp_path):
    """With `prob: 0`, mealtalk-on and mealtalk-off must produce the identical conversation: the mechanism
    draws from its own stream, so a condition that has it configured but never fires is byte-comparable."""
    from backend.llm.client import llm_scope
    off = _world(tmp_path / "off", tick=12)
    on = _world(tmp_path / "on", deep_merge(ON, {"conversation": {"mealtalk": {"prob": 0.0}}}), tick=12)
    outs = []
    for ctx, tag in ((off, "off"), (on, "on")):
        a, b = _pair(ctx)
        with llm_scope(tag):
            conv = CONV.run_conversation("c0", a, b, [], np.random.default_rng(5))
        outs.append([(u["speaker"], u["text"], tuple(u["retrieved"])) for u in conv["utterances"]])
    assert outs[0] and outs[0] == outs[1]
    fires = _world(tmp_path / "fires", ON, tick=12)
    a, b = _pair(fires)
    with llm_scope("fires"):
        CONV.run_conversation("c0", a, b, [], np.random.default_rng(5))
    utts = [r for r in _trace(fires) if r["type"] == "utterance"]
    assert utts and all("working out where to eat" in u["context"] for u in utts)


# ------------------------------------------------------------------------------------- seed retrievability
def _seed_records(tmp_path, over):
    from backend.simulation.engine import Simulation
    tmp_path.mkdir(parents=True, exist_ok=True)
    memories = tmp_path / "seeds.yaml"
    memories.write_text("maya:\n- I have called the dining hall beside AMR III FFC when arranging meals.\n")
    cfg = load_config("configs/baseline.yaml", deep_merge(
        {"simulation_days": 1, "day_end": "09:00", "population_size": 4, "latent_events": {"event_rate": 0},
         "llm": {"backend": "mock", "max_workers": 2}, "initial_memories_file": str(memories)}, over))
    sim = Simulation(cfg, tmp_path / "run", progress=False)
    sim.run()
    tr = [json.loads(l) for l in open(sim.run_dir / "trace.jsonl")]
    return sim, [r for r in tr if r["type"] == "memory_encoded" and r.get("initial_memory")]


def test_initial_memory_importance_is_configurable_and_defaults_to_v2(tmp_path):
    _, default = _seed_records(tmp_path / "default", {})
    assert len(default) == 1 and default[0]["importance"] == 5          # unchanged v2 behaviour
    sim, raised = _seed_records(tmp_path / "raised", {"initial_memories_importance": 7})
    assert len(raised) == 1 and raised[0]["importance"] == 7
    node = sim.agents["maya"].a_mem.id_to_node[raised[0]["node_id"]]
    assert node.poignancy == 7 and "FFC" in node.description
    assert sim.meta.get(node.node_id).source_type == "seed"


def test_a_seeded_name_stays_a_memory_and_never_becomes_a_habit(tmp_path):
    """Raising importance must not smuggle the name into anything permanent: it lives in the memory stream,
    not in the persona the identity prompt is built from."""
    sim, raised = _seed_records(tmp_path, {"initial_memories_importance": 9})
    maya = sim.agents["maya"]
    assert raised[0]["node_id"] in maya.a_mem.id_to_node                # it is a memory ...
    assert "FFC" not in maya.iss()                                      # ... and only a memory
    assert "FFC" not in maya.profile.ga_lifestyle() and "FFC" not in maya.profile.ga_daily_plan_req()
    assert not any("FFC" in str(h) for h in maya.profile.habits)
    assert not set(vars(maya.profile)) & FORBIDDEN_FIELDS


# ------------------------------------------------------------------------------------- the shipped configs
# open maps (backend.config.unknown_keys): compared whole, since their KEYS are part of the setting
OPEN_MAPS = {"retrieval.source_weights", "commons", "analysis.memetics", "module_params.prestige_bias.scores"}


def _flat(cfg, prefix=""):
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        out.update(_flat(v, key + ".") if isinstance(v, dict) and v and key not in OPEN_MAPS
                   else {key: json.dumps(v, sort_keys=True, default=str)})
    return out


def test_naming_cells_are_situated_cued_and_differ_only_in_wording():
    c1, c2 = load_config(CELL1), load_config(CELL2)
    for cfg in (c1, c2):
        assert cfg["world"]["reference_mode"] == "situated"       # the world describes, it does not name
        assert cfg["conversation"]["mealtalk"]["enabled"] is True  # and there is a reason to refer
        assert cfg["initial_memories_importance"] == 7             # and the seed can be recalled when it is
        assert cfg["initial_memories_file"].endswith("homewood100_initial_memories.yaml")
    f1, f2 = _flat(c1), _flat(c2)
    assert set(f1) == set(f2)
    assert {k for k in f1 if f1[k] != f2[k]} == WORDING_KEYS


def test_cue_fires_in_the_shipped_cells_lunch_window():
    """The default windows have to land on this scenario's clock: lunch does, dinner cannot (the day ends
    at 17:00), and the population still has people with a meal ahead of them inside the lunch window."""
    import yaml
    from pathlib import Path
    cfg = load_config(CELL1)
    windows = cfg["conversation"]["mealtalk"]["windows"]
    assert windows == ["11:30-13:30", "17:30-19:00"]
    at = lambda s: dt.datetime.strptime(s, "%H:%M")  # noqa: E731
    assert CONV.meal_window(cfg, at("12:00")) == "11:30-13:30"
    assert at(cfg["day_end"]) < at("17:30")                        # the dinner window never fires here
    pop = yaml.safe_load(Path(ga_compat.REPO_ROOT / cfg["population"]).read_text())
    ahead = sum(1 for a in pop["agents"]
                if any(CONV.MEAL_RE.search(e["activity"]) and at(str(e["time"])) > at("11:30")
                       for e in a["routine"]))
    assert ahead >= 50, ahead                                      # enough people to cue in the window


@pytest.mark.parametrize("place", ["Dining Hall", "Cafe", "Quad"])
def test_focal_point_and_line_template_carry_no_place(place):
    assert place.lower() not in (CONV.MEALTALK_LINE + CONV.MEALTALK_FOCAL).lower()
