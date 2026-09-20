"""The study CONFIG as an artefact: the two new blocks other builders implement against, the 2x2 the
registry is supposed to fill, the clock the day is supposed to cover, and the seed memories.

These are the failures a config file can have that no unit test of a mechanism would catch: a meal window
outside the day so a cue never fires, a phrase no verbatim filter can ever stick, a 2x2 with a cell missing,
a canonical campus name back in a seed memory. Config only -- no LLM calls and no simulation runs.
"""
import re
from pathlib import Path

import pytest
import yaml

from backend.agents.profile import load_population, meme_registry, plan_memes, seed_matching
from backend.config import load_config, read_config_yaml
from backend.memory import encoder as ENC
from backend.simulation import incidents as INC
from backend.simulation.lexicon import population_lexicon
from backend.simulation.world import ARENAS, HOMEWOOD_LABELS, WORLD_GRAPH, Clock

ROOT = Path(__file__).resolve().parents[1]
STUDY = "configs/homewood100_memes.yaml"
SEEDS = ROOT / "configs/population/homewood100_meme_memories.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(STUDY)


def _mins(s):
    h, m = map(int, str(s).split(":"))
    return h * 60 + m


# ------------------------------------------------------------------ the blocks other builders read
def test_default_yaml_declares_both_new_blocks_off():
    """Every new mechanism defaults off (R5), and the key names are the contract: the injector, the
    incident world and the repair module all read exactly these paths."""
    d = read_config_yaml(ROOT / "configs/default.yaml")
    assert d["memes"] == {"enabled": False, "registry": []}
    assert d["conversation"]["repair"] == {"enabled": False, "prob": 0.7, "max_followups": 1}


def test_default_yaml_lists_both_dining_halls_as_group_venues():
    d = read_config_yaml(ROOT / "configs/default.yaml")
    venues = [tuple(v) for v in d["conversation"]["group"]["venues"]]
    assert ("Dining Hall", "Main Floor") in venues and ("Nolans", "Servery") in venues
    for loc, arena in venues:                      # a venue that is not on the map can never be occupied
        assert loc in WORLD_GRAPH and arena in ARENAS[loc]


# ------------------------------------------------------------------ the 2x2
def test_the_registry_fills_every_cell_of_the_2x2_exactly_once(cfg):
    cells = {(e["grounding"], e["breadth"]) for e in meme_registry(cfg)}
    assert cells == {(g, b) for g in ("grounded", "ungrounded") for b in ("broad", "narrow")}
    assert len(meme_registry(cfg)) == 4


def test_grounding_is_carried_by_the_incident_and_nothing_else(cfg):
    for e in INC.registry(cfg):
        assert bool(e["incident"]) == (e["grounding"] == "grounded"), e["id"]


def test_the_two_grounded_incidents_are_matched_on_everything_but_their_site(cfg):
    """R6. If one origin runs for longer, or hits harder, or is more noticeable, a difference between the
    two grounded cells is a difference between the incidents rather than between broad and narrow."""
    inc = [e["incident"] for e in INC.registry(cfg) if e["incident"]]
    assert len(inc) == 2
    a, b = inc
    for key in ("days", "repaired_day", "repaired_days", "prob_per_meal", "salience", "max_per_tick"):
        assert a[key] == b[key], key
    assert len(a["windows"]) == len(b["windows"])
    assert len(a["surfaces"]) == len(b["surfaces"])
    assert len(a["repaired_surfaces"]) == len(b["repaired_surfaces"])


def test_every_minority_is_the_same_size_and_the_cohorts_are_matched(cfg):
    profiles, _ = load_population(cfg["population"], cfg["population_size"])
    plan = plan_memes(dict(profiles), cfg)
    report = seed_matching(profiles, plan, cfg)
    assert report["disjoint"], report["overlap"]
    assert len({len(m["seeds"]) for m in plan}) == 1
    assert report["population"]["matched"], report["balance"]


def test_habit_lines_are_comparable_in_length_and_register(cfg):
    """R6 again, on the one piece of text the agents actually get. A habit twice as long as another is a
    difference in how much the seed was told, which is not one of the two factors."""
    habits = [e["habit"] for e in meme_registry(cfg)]
    words = [len(h.split()) for h in habits]
    assert max(words) - min(words) <= 4, dict(zip([e["id"] for e in meme_registry(cfg)], words))
    assert max(len(h) for h in habits) <= 1.3 * min(len(h) for h in habits)
    for h in habits:                                # one shape: "<name> has a habit of calling X a "phrase"."
        assert h.startswith("has a habit of calling ")


# ------------------------------------------------------------------ can a phrase ever reach speech?
def test_every_phrase_can_actually_stick_under_this_run_s_own_verbatim_settings(cfg):
    """The mechanism the study stands on. A phrase with no word rare enough (or none outside the campus
    lexicon) yields no candidate span, is never stored word for word, is never primed back into speech,
    and cannot spread however well it is seeded -- silently, with no error anywhere."""
    profiles, _ = load_population(cfg["population"], cfg["population_size"])
    lex = population_lexicon(profiles)

    class _Ctx:
        pass

    ctx = _Ctx()
    ctx.lexicon = lex
    ctx.agents = {p.id: type("A", (), {"name": p.name})() for p in profiles.values()}
    agent = type("A", (), {})()
    agent.ctx, agent.cfg = ctx, cfg
    zmax = float(cfg["memory"]["verbatim"]["zipf_max"])
    for e in meme_registry(cfg):
        heard = f'Honestly the whole thing was {e["phrase"]} and everyone knew it.'
        spans = [p.lower() for p, _z in ENC.distinctive_phrases(agent, heard, zmax)]
        assert e["phrase"].lower() in spans, (e["id"], e["phrase"], spans)


# ------------------------------------------------------------------ the clock
def test_every_configured_window_falls_inside_the_day(cfg):
    """A window outside the day is a mechanism that is switched on and can never fire; the naming study
    shipped with one and said so in a comment. Machine-check it instead."""
    clock = Clock(cfg)
    lo = int(clock.day_start.total_seconds() // 60)
    hi = lo + (clock.ticks_per_day - 1) * clock.tick_minutes
    cc = cfg["conversation"]
    windows = [(w, "mealtalk") for w in cc["mealtalk"]["windows"]] + [(w, "group") for w in cc["group"]["windows"]]
    windows += [(f'{w[0]//60:02d}:{w[0]%60:02d}-{w[1]//60:02d}:{w[1]%60:02d}', "incident")
                for e in INC.registry(cfg) if e["incident"] for w in e["incident"]["windows"]]
    for w, which in windows:
        a, b = (_mins(x) for x in w.split("-"))
        assert a <= hi and b >= lo, f"{which} window {w} never overlaps the day {lo}-{hi}"
        assert a <= hi, f"{which} window {w} starts after the last tick"
    assert lo <= _mins(cc["catchup"]["after"]) <= hi, "catchup.after is outside the day"


def test_the_manipulation_has_days_on_both_sides_of_it(cfg):
    """Dying, calcifying and broadening can only be told apart after the basis is removed."""
    for e in INC.registry(cfg):
        inc = e["incident"]
        if not inc:
            continue
        assert max(inc["days"]) < inc["repaired_day"] <= cfg["simulation_days"]
        assert max(inc["repaired_days"]) <= cfg["simulation_days"], e["id"]


# ------------------------------------------------------------------ R1 / R2 / R3 on what this part owns
def _agent_facing_config_text() -> list[str]:
    """Everything in this study's own files that can end up in front of an agent: the seed memories and
    the population's own prose. NOT the registry's habit lines (R1 says the phrase lives there) and not
    probe_gradient or foils (R7: observer only, and they are checked separately for never being read)."""
    out = SEEDS.read_text(encoding="utf-8").splitlines()
    out += (ROOT / "configs/population/homewood100.yaml").read_text(encoding="utf-8").splitlines()
    return out


def test_no_registry_phrase_or_distinctive_word_reaches_the_seed_memories_or_the_population(cfg):
    assert INC.audit(INC.registry(cfg), _agent_facing_config_text()) == []


def test_the_seed_memories_carry_no_canonical_campus_name():
    """R3/D75. The seed files were the last place a canonical name survived into a prompt."""
    names = set(WORLD_GRAPH) | {a for v in ARENAS.values() for a in v} | set(HOMEWOOD_LABELS.values())
    names |= {"AMR III", "AMR II", "Homewood", "Hopkins"}
    text = yaml.safe_dump(yaml.safe_load(SEEDS.read_text(encoding="utf-8")))   # the memories, not the comments
    found = sorted(n for n in names if re.search(r"(?<![A-Za-z])" + re.escape(n) + r"(?![A-Za-z])", text))
    assert found == [], found


def test_the_seed_memories_cover_the_population_and_name_the_places_the_world_names(cfg):
    from backend.simulation.world import ARENA_DESCRIPTIONS, PLACE_DESCRIPTIONS
    data = yaml.safe_load(SEEDS.read_text(encoding="utf-8"))
    profiles, _ = load_population(cfg["population"], cfg["population_size"])
    assert set(data) == set(profiles), "every agent needs somewhere for an incident to attach"
    blob = " ".join(t for v in data.values() for t in v)
    for phrase in (PLACE_DESCRIPTIONS["Dining Hall"], PLACE_DESCRIPTIONS["Nolans"],
                   ARENA_DESCRIPTIONS["Research Lab"]["Wet Lab"]):
        assert phrase in blob, f"the seed memories do not describe a place the way the world does: {phrase!r}"
