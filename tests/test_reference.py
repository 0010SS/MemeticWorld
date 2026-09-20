"""The world reference layer (backend/simulation/reference.py, D75).

The principle under test: in `world.reference_mode: situated` the world hands agents percepts and relative
descriptions, never the canonical name of a place. Naming is the agents' business, so the only names that may
appear in an agent-facing prompt are the ones an agent was seeded with or heard from someone else.

Four things are checked:

  1. canonical mode still renders exactly the strings v2 rendered (every existing run reproduces);
  2. a 1-day mock run in situated mode leaks no canonical place key, arena name or real Homewood building
     into ANY prompt in llm_calls.jsonl -- every call, not a sample;
  3. personalisation resolves per agent, and is derived from the profile rather than written per place;
  4. the MOVE option list round-trips: what the agent is offered maps back to the canonical id, and the
     canonical id is still accepted.
"""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.agents.agent import CampusMaze, set_reference_cfg  # noqa: E402
from backend.agents.profile import load_population  # noqa: E402
from backend.config import deep_merge, load_config  # noqa: E402
from backend.simulation import reference as R  # noqa: E402
from backend.simulation.world import (ARENA_DESCRIPTIONS, ARENAS, HOMEWOOD_LABELS,  # noqa: E402
                                      PLACE_DESCRIPTIONS, WORLD_GRAPH)

SITUATED = {"world": {"reference_mode": "situated"}}
CANONICAL = {"world": {"reference_mode": "canonical"}}

PLACE_KEYS = sorted(WORLD_GRAPH)
ARENA_KEYS = sorted({a for v in ARENAS.values() for a in v})
BUILDINGS = sorted(set(HOMEWOOD_LABELS.values()))

# The seeded names are the experiment's dependent variable: they belong to the AGENTS, reach prompts through
# seed memories and through what other agents said, and are not world wording. They are removed before a
# prompt is scanned, so that "Hopkins Cafe" does not read as a leak of the canonical key "Cafe".
AGENT_ALIASES = re.compile(r"hopkins caf[eé]|fresh food caf[eé]|\bffc\b", re.I)

# The one canonical key that survives situated mode as a lower-case common noun. The seeded naming memories
# say "the dining hall beside AMR III", so the world has to use the same common noun for the seed to attach
# to the referent; the name analyzer scores it `no_target_alias`, not a name. The scan below is therefore
# case-sensitive for the canonical proper-name form, which is the label the experiment measured (72% of live
# chat prompts contained the literal string "Dining Hall").
COMMON_NOUNS_KEPT = ("dining hall",)


# ----------------------------------------------------------------------------- 1. canonical mode is unchanged

def test_canonical_mode_is_the_default():
    assert R.mode(None) == R.CANONICAL
    assert R.mode({}) == R.CANONICAL
    assert R.mode({"world": {"mode": "latent_events"}}) == R.CANONICAL
    assert R.mode({"world": {"reference_mode": "nonsense"}}) == R.CANONICAL
    assert R.mode(load_config("configs/default.yaml")) == R.CANONICAL
    assert R.mode(load_config("configs/baseline.yaml")) == R.CANONICAL
    # the seeded naming experiment is the scenario that has to opt in
    assert R.mode(load_config("configs/homewood100_naming.yaml")) == R.SITUATED
    assert R.mode(load_config("configs/homewood100_naming_wording.yaml")) == R.SITUATED


@pytest.mark.parametrize("cfg", [None, {}, CANONICAL])
def test_canonical_mode_reproduces_v2_strings(cfg):
    """Every string the call sites used to build, byte for byte."""
    profiles, _ = load_population("configs/population/homewood100.yaml", n=4)
    prof = next(iter(profiles.values()))
    for loc in PLACE_KEYS:
        assert R.place(loc, cfg=cfg) == loc
        assert R.place(loc, agent=prof, cfg=cfg) == loc                     # no personalisation either
        assert R.phrase(loc, cfg=cfg) == f"the {loc}"                       # engine / profile "the {loc}"
        for arena in ARENAS[loc]:
            assert R.room(loc, arena, cfg=cfg) == arena
            assert R.describe(loc, arena, cfg=cfg) == f"{loc} ({arena})"    # encoder / planner
            assert R.inside(loc, arena, cfg=cfg) == f"{arena} in {loc}"     # upstream chat prompt
    assert R.options(cfg=cfg) == ", ".join(sorted(WORLD_GRAPH))             # react MOVE list
    assert R.describe("Dining Hall", "Main Floor", cfg=cfg) == "Dining Hall (Main Floor)"
    # the two upstream-facing renderings the engine no longer writes by hand
    assert f"stopping by {R.phrase('Library', cfg=cfg)}" == "stopping by the Library"
    assert f"heading to {R.phrase('Gym', cfg=cfg)}" == "heading to the Gym"


def test_canonical_mode_leaves_the_daily_plan_and_the_maze_alone():
    profiles, _ = load_population("configs/population/homewood100.yaml", n=4)
    prof = profiles["p0001"]
    assert prof.ga_daily_plan_req() == prof.ga_daily_plan_req(CANONICAL)
    assert prof.ga_daily_plan_req() == ", ".join(
        f"{r.activity} at the {r.location} around {r.time}" for r in prof.routine[:6])
    set_reference_cfg(None)
    tile = CampusMaze().access_tile(("Dining Hall", "Main Floor"))
    assert (tile["sector"], tile["arena"]) == ("Dining Hall", "Main Floor")


def test_unknown_places_fall_back_to_their_own_name():
    """v3 uses an "Away" sentinel and the commons its own rooms; neither is on this map."""
    assert R.place("Away", cfg=SITUATED) == "Away"
    assert R.room("Away", "Away", cfg=SITUATED) == "Away"
    assert R.describe("Away", "Away", cfg=SITUATED) == "Away (Away)"


# ----------------------------------------------------------------------------- 2. the description table

def test_every_place_and_room_has_a_description():
    assert set(PLACE_DESCRIPTIONS) == set(WORLD_GRAPH)
    for loc, arenas in ARENAS.items():
        assert set(ARENA_DESCRIPTIONS[loc]) >= set(arenas), loc


def test_descriptions_are_descriptions_not_names():
    """No canonical key, no arena name, no real Homewood building, and short enough to read as a phrase."""
    entries = ([(loc, d) for loc, d in PLACE_DESCRIPTIONS.items()]
               + [(f"{loc}/{a}", d) for loc, m in ARENA_DESCRIPTIONS.items() for a, d in m.items()])
    for key, desc in entries:
        assert desc[:1].islower(), f"{key}: a description is a common noun, not a label: {desc!r}"
        assert len(desc.split()) <= 7, f"{key}: too long to be a description: {desc!r}"
        for name in PLACE_KEYS + ARENA_KEYS + BUILDINGS:
            if name.lower() in COMMON_NOUNS_KEPT:
                continue
            assert not re.search(rf"\b{re.escape(name)}\b", desc, re.I), f"{key} names {name!r}: {desc!r}"


def test_situated_mode_never_returns_a_canonical_name():
    profiles, _ = load_population("configs/population/homewood100.yaml", n=20)
    for loc in PLACE_KEYS:
        for arena in ARENAS[loc]:
            for prof in [None, *profiles.values()]:
                out = R.describe(loc, arena, agent=prof, cfg=SITUATED)
                assert loc not in out and arena not in out, out


# ----------------------------------------------------------------------------- 3. personalisation

def test_home_tile_is_personal_and_per_agent():
    profiles, _ = load_population("configs/population/homewood100.yaml")
    student = profiles["p0001"]                      # anchored in a dorm room
    worker = profiles["p0028"]                       # a dining worker, anchored at their own serving floor
    assert student.home["arena"].startswith("Room ")
    assert R.room(*student.home.values(), agent=student, cfg=SITUATED) == R.HOME_ARENA
    assert R.room(*worker.home.values(), agent=worker, cfg=SITUATED) == R.HOME_SPOT
    # ... and only for the agent whose home it is
    generic = R.room(*student.home.values(), cfg=SITUATED)
    assert generic == R.room(*student.home.values(), agent=worker, cfg=SITUATED) != R.HOME_ARENA


def test_routine_focus_is_derived_per_agent():
    profiles, _ = load_population("configs/population/homewood100.yaml")
    focus = {pid: R._focus(p) for pid, p in profiles.items()}
    # every agent gets a focus, and it is a place they actually go
    for pid, (loc, desc) in focus.items():
        assert loc in WORLD_GRAPH, pid
        assert loc in {r.location for r in profiles[pid].routine}, pid
        assert desc.startswith("the ") and "you" in desc, (pid, desc)
    # different people, different answers: a lab student, a lecturing faculty member, a dining worker
    assert focus["p0009"] == ("Research Lab", "the lab where you work")
    assert R.place("Research Lab", agent=profiles["p0009"], cfg=SITUATED) == "the lab where you work"
    assert {desc for _, desc in focus.values()} >= {
        "the lab where you work", "the building where you study", "the building where you work"}
    # a place the agent never visits keeps the generic description
    never = next(loc for loc in PLACE_KEYS if loc not in {r.location for r in profiles["p0009"].routine})
    assert R.place(never, agent=profiles["p0009"], cfg=SITUATED) == PLACE_DESCRIPTIONS[never]


def test_focus_phrase_is_read_off_the_routine_not_the_place():
    """The rule names no place: the same location yields different wordings for different routines."""
    assert R._focus_phrase("attending a class in Neuroscience", ["Lecture Hall"]) == \
        "the building where you have class"
    assert R._focus_phrase("teaching Neuroscience", ["Lecture Hall"]) == "the building where you teach"
    assert R._focus_phrase("working on coursework", ["Wet Lab"]) == "the lab where you work"
    assert R._focus_phrase("working as a cook and assisting people", ["Main Floor"]) == \
        "the building where you work"
    assert R._focus_phrase("reading and preparing work", ["Study Tables"]) == "the building where you study"
    assert R._focus_phrase("doing something unclassifiable", ["Lawn"]) == R._FOCUS_DEFAULT


def test_daily_plan_requirement_is_situated():
    profiles, _ = load_population("configs/population/homewood100.yaml", n=20)
    prof = profiles["p0009"]
    plan = prof.ga_daily_plan_req(SITUATED)
    assert "the lab where you work" in plan
    for name in PLACE_KEYS + ARENA_KEYS:
        assert not re.search(rf"\b{re.escape(name)}\b", plan), (name, plan)


# ----------------------------------------------------------------------------- 4. the MOVE round trip

def test_move_options_round_trip():
    profiles, _ = load_population("configs/population/homewood100.yaml", n=20)
    prof = profiles["p0009"]
    offered = R.options(agent=prof, cfg=SITUATED).split(", ")
    assert len(offered) == len(WORLD_GRAPH)
    for loc in PLACE_KEYS:
        for form in (loc,                                        # the canonical id is still accepted
                     R.place(loc, cfg=SITUATED),                  # the generic description
                     R.place(loc, agent=prof, cfg=SITUATED),      # this agent's personalised one
                     f'MOVE to {R.place(loc, agent=prof, cfg=SITUATED)}.'):   # wrapped in a sentence
            assert R.resolve(form, agent=prof, cfg=SITUATED) == loc, (loc, form)
    assert R.resolve("somewhere else entirely", agent=prof, cfg=SITUATED) is None
    assert R.resolve("", agent=prof, cfg=SITUATED) is None
    assert R.resolve(None, agent=prof, cfg=SITUATED) is None


def test_move_options_are_offered_as_descriptions():
    offered = R.options(cfg=SITUATED)
    for name in PLACE_KEYS + ARENA_KEYS:
        if name.lower() in COMMON_NOUNS_KEPT:
            continue
        assert not re.search(rf"\b{re.escape(name)}\b", offered, re.I), name


# ----------------------------------------------------------------------------- 5. the leak scan over a run

# `\b` cannot bound a name that ends in punctuation ("Hopkins Cafe (FFC)"), and a plain substring test
# reports "AMR II" inside the seed memories' "AMR III"; lookarounds get both right.
_BOUNDED = {n: re.compile(rf"(?<!\w){re.escape(n)}(?!\w)")
            for n in PLACE_KEYS + ARENA_KEYS + BUILDINGS}


def canonical_leaks(prompt: str) -> set:
    """Canonical labels in one agent-facing prompt.

    Case-sensitive: the canonical proper-name form is what the world writes and what the live run measured
    (72% of chat prompts contained the literal string "Dining Hall"), while a lower-case common noun ("the
    dining hall") is ordinary English the agent may equally have read in its own seed memory. The agents'
    own seeded aliases are stripped first, because they are the experiment's dependent variable, not world
    wording -- without that, "Hopkins Cafe" would read as a leak of the canonical key "Cafe".
    """
    text = AGENT_ALIASES.sub(" ", prompt or "")
    return {n for n, rx in _BOUNDED.items() if rx.search(text)}


def scan(run_dir: Path) -> dict:
    total, leaking, names = 0, 0, {}
    with (run_dir / "llm_calls.jsonl").open() as fh:
        for line in fh:
            call = json.loads(line)
            total += 1
            hits = canonical_leaks(call.get("prompt"))
            if hits:
                leaking += 1
                for n in hits:
                    names[n] = names.get(n, 0) + 1
    return {"prompts": total, "leaking": leaking, "names": names}


def _run(tmp_path, overrides, name):
    from backend.simulation.engine import Simulation
    cfg = load_config("configs/homewood100_naming.yaml", deep_merge(
        {"simulation_days": 1, "population_size": 24, "llm": {"backend": "mock", "max_workers": 8}},
        overrides))
    sim = Simulation(cfg, tmp_path / name, progress=False)
    sim.run()
    return tmp_path / name


def test_situated_run_leaks_no_canonical_place_name(tmp_path):
    """A whole mock day, every prompt scanned -- not a sample."""
    canonical = scan(_run(tmp_path, CANONICAL, "canonical"))
    situated = scan(_run(tmp_path, SITUATED, "situated"))
    assert canonical["prompts"] > 200 and situated["prompts"] > 200
    assert canonical["leaking"] > canonical["prompts"] * 0.5, \
        f"the canonical control should leak heavily, got {canonical}"
    assert situated["leaking"] == 0, \
        f"{situated['leaking']}/{situated['prompts']} prompts still name a place: {situated['names']}"
