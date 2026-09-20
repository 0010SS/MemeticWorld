"""Registry-driven origin incidents (`memes.registry[*].incident`, backend/simulation/incidents.py).

The load-bearing test here is the R2 machine check: for every phrase in the registry, no world string and no
perception may emit the words the phrase is built from. It reads the registry rather than naming a meme, so it
keeps working when the prior check swaps a phrase out - which is the point, since the candidate phrases in this
file are starting points, not decisions.

The other tests cover the mechanism itself: one code path for N incidents, an ungrounded entry producing no
incident without a special case, encounters through the ordinary memory path, the repaired state being
perceptible without being evaluated, determinism, and byte-identity with the mechanism off.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import pytest

from backend.config import deep_merge, load_config
from backend.simulation import incidents as INC
from backend.simulation.engine import Simulation, trace_digest

ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------------------------------ the cohort
# Surfaces are world text (D39): what a person could see, never what it means. No "the washer is broken",
# no "the problem is fixed" - the repaired surfaces state a clean tray and say nothing about why.
TRAY_SURFACES = [
    "{P} picked a tray off the stack and it was still slick under the thumb.",
    "{P} set a plate down and a smear stayed where the tray had touched it.",
    "{P}'s cutlery came out of the rack with a dull film on it.",
    "{P} had eaten half a meal before noticing the fork had a greasy coating.",
]
TRAY_REPAIRED = [
    "{P} took a tray off the stack and it squeaked dry under the thumb.",
    "{P}'s cutlery came out of the rack with nothing on it.",
    "{P} drew a thumb across the plate and it came away clean.",
]
HOOD_SURFACES = [
    "{P} pulled the hood sash down until it clicked and the tape at the opening still hung slack.",
    "{P} checked the gauge at the bench and the needle sat at zero with the sash shut.",
    "{P} caught solvent at the bench with the sash drawn all the way down.",
]
HOOD_REPAIRED = [
    "{P} drew the sash shut and the tape at the opening pulled flat against it.",
    "{P} checked the gauge at the bench and the needle held steady with the sash down.",
]


def meme(mid, phrase, grounding, breadth, habit, incident=None, **over):
    e = {"id": mid, "phrase": phrase, "grounding": grounding, "breadth": breadth,
         "habit": habit, "seeds": {"k": 3, "strategy": "spread"},
         "probe_gradient": {"literal": "L", "near": "N", "mid": "M", "far": "F"}, "foils": ["f"]}
    if incident is not None:
        e["incident"] = incident
    e.update(over)
    return e


def tray(days=(1, 2), repaired_day=3, **over):
    inc = {"location": "Dining Hall", "arena": "Main Floor", "days": list(days),
           "repaired_day": repaired_day, "prob_per_meal": 0.6,
           "surfaces": TRAY_SURFACES, "repaired_surfaces": TRAY_REPAIRED, **over.pop("incident", {})}
    # "tray" is the referent the world cannot avoid naming; "blue" is what the coinage adds. Narrowing
    # `distinctive_words` is the one sanctioned loosening of R2 and is reported as an exemption.
    return meme("tray_washer", "blue-tray", "grounded", "broad",
                '{name} has a habit of calling anything that looks fine but is not "blue-tray".',
                incident=inc, distinctive_words=["blue"], **over)


def hood(days=(1, 2), repaired_day=3, **over):
    inc = {"location": "Research Lab", "arena": "Wet Lab", "days": list(days),
           "repaired_day": repaired_day, "prob_per_meal": 0.6,
           "windows": ["09:00-12:30", "13:00-17:00"],
           "surfaces": HOOD_SURFACES, "repaired_surfaces": HOOD_REPAIRED, **over.pop("incident", {})}
    return meme("hood_sash", "hood-three", "grounded", "narrow",
                '{name} has a habit of calling the one fume hood that will not seal "hood-three".',
                incident=inc, distinctive_words=["three"], **over)


# The ungrounded arm. "running warm" and "the Tuesday list" - the phrases the design started from - are
# refused by the R2 pre-flight (test_r2_refuses_a_phrase_the_world_already_speaks); these two were picked with
# `incidents.preflight` instead and collide with nothing the world says.
CROSSWIRED = meme("crosswired", "crosswired", "ungrounded", "broad",
                  '{name} has a habit of calling anything overcommitted or behind "crosswired".')
SUMP = meme("sump", "the sump", "ungrounded", "narrow",
            '{name} has a habit of calling the same recurring chore "the sump".')


def cohort(**over):
    return [tray(**over), hood(**over), CROSSWIRED, SUMP]


# ------------------------------------------------------------------------------------------ run helpers
BASE = {"seed": 42, "simulation_days": 3, "day_end": "19:30",
        "llm": {"backend": "mock", "max_workers": 4},
        "latent_events": {"event_rate": 0.0},
        "world": {"reference_mode": "situated"}}


def make_cfg(memes=None, **over):
    base = deep_merge(BASE, over)
    if memes is not None:
        base["memes"] = memes
    return load_config("configs/baseline.yaml", base)


def run_sim(tmp, name, memes=None, **over):
    sim = Simulation(make_cfg(memes, **over), Path(tmp) / name, progress=False)
    sim.run()
    return sim


def trace_records(run_dir):
    out = defaultdict(list)
    for line in (Path(run_dir) / "trace.jsonl").read_text().splitlines():
        r = json.loads(line)
        out[r.get("type")].append(r)
    return out


def world_run_text(run_dir):
    """(where, text) for everything the WORLD said during the run. Agent-produced text (utterances, the
    agent's own reconstruction of what it saw) is excluded on purpose: R1 forbids the WORLD from saying a
    phrase, and a seeded agent saying its own habit phrase is the mechanism working, not contamination."""
    run_dir = Path(run_dir)
    out = []
    for line in (run_dir / "world_script.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out.append(("world_script.narrative", r.get("narrative") or ""))
        out += [("world_script.fact", f["text"]) for b in (r.get("beats") or []) for f in (b.get("facts") or [])]
        out += [("world_script.surfaces", s) for s in (r.get("surfaces") or [])]
        out += [("world_script.repaired_surfaces", s) for s in (r.get("repaired_surfaces") or [])]
    tr = trace_records(run_dir)
    out += [("trace.event_beat", f["text"]) for r in tr["event_beat"] for f in (r.get("facts") or [])]
    out += [("trace.incident_encounter", r.get("text") or "") for r in tr["incident_encounter"]]
    out += [("trace.observation", f.get("text") or "")
            for r in tr["observation"] if r.get("source_type") == "perception" for f in (r.get("facts") or [])]
    out += [("trace.day_plan", e.get("activity") or "") for r in tr["day_plan"] for e in (r.get("plan") or [])]
    out += [("trace.move.activity", r.get("activity") or "") for r in tr["move"]]
    for line in (run_dir / "frames.jsonl").read_text().splitlines():
        fr = json.loads(line)
        out += [("frame.beat", t) for b in (fr.get("beats") or []) for t in (b.get("facts") or [])]
        out += [("frame.activity", a.get("activity") or "") for a in fr.get("agents", {}).values()]
    return [(w, t) for w, t in out if t]


@pytest.fixture(scope="module")
def inc_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("memes")
    sim = run_sim(tmp, "on", {"enabled": True, "registry": cohort()})
    return sim, sim.run_dir


# =========================================================================== registry: one parser, N memes
def test_content_words_split_hyphens_and_match_inflections():
    assert INC.content_words("blue-tray") == ("blue", "tray")
    assert INC.content_words("the Tuesday list") == ("tuesday", "list")     # "the" carries no identity
    assert INC._phrase_re("blue-tray").search("they called it a Blue Tray again")   # hyphen or space
    assert INC._word_re("tray").search("stacked trays of food")                     # inflections count
    assert not INC._word_re("tray").search("a betrayal")                            # but not substrings


def test_registry_is_empty_unless_enabled():
    assert INC.registry({}) == []
    assert INC.registry({"memes": {"enabled": False, "registry": cohort()}}) == []
    assert len(INC.registry({"memes": {"enabled": True, "registry": cohort()}})) == 4


def test_cohort_fills_the_two_by_two():
    entries = INC.registry({"memes": {"enabled": True, "registry": cohort()}})
    assert INC.cohort_summary(entries) == {"grounded/broad": ["tray_washer"], "grounded/narrow": ["hood_sash"],
                                           "ungrounded/broad": ["crosswired"], "ungrounded/narrow": ["sump"]}


def test_grounding_and_the_incident_block_must_agree():
    bad = meme("x", "wobbly", "grounded", "broad", '{name} has a habit of saying "wobbly".')
    with pytest.raises(ValueError, match="needs an `incident` block"):
        INC.registry({"memes": {"enabled": True, "registry": [bad]}})
    bad2 = meme("y", "wobbly", "ungrounded", "broad", '{name} has a habit of saying "wobbly".',
                incident={"location": "Dining Hall", "arena": "Main Floor", "days": [1], "prob_per_meal": 0.5,
                          "surfaces": ["{P} saw it."]})
    with pytest.raises(ValueError, match="must not declare an `incident`"):
        INC.registry({"memes": {"enabled": True, "registry": [bad2]}})


@pytest.mark.parametrize("over, msg", [
    ({"location": "Atlantis"}, "unknown location"),
    ({"arena": "Broom Cupboard"}, "not a room of"),
    ({"days": []}, "`days` must be a non-empty list"),
    ({"prob_per_meal": 1.5}, "must be in \\[0, 1\\]"),
    ({"repaired_day": 2}, "must come after the last incident day"),
    ({"surfaces": ["a tray was slick."]}, "must contain"),          # no {P}: nobody it happened to
    ({"repaired_surfaces": []}, "non-empty list"),
    ({"windows": ["13:30-11:30"]}, "ends before it starts"),
])
def test_incident_block_is_validated(over, msg):
    with pytest.raises(ValueError, match=msg):
        INC.registry({"memes": {"enabled": True, "registry": [tray(incident=over)]}})


def test_duplicate_ids_are_refused():
    with pytest.raises(ValueError, match="unique id|duplicate id"):
        INC.registry({"memes": {"enabled": True, "registry": [tray(), tray()]}})


def test_repaired_days_match_the_fault_days_by_default():
    e = INC.registry({"memes": {"enabled": True, "registry": [tray(days=(1, 2, 3), repaired_day=5)]}})[0]
    # matched exposure: "nobody noticed the repair" must not be an artefact of a shorter window
    assert e["incident"]["repaired_days"] == [5, 6, 7]


# ================================================================================== R1 / R2 machine check
def test_r2_refuses_a_contaminated_surface(tmp_path):
    dirty = tray(incident={"surfaces": ["{P} picked up a blue tray and it was still slick."]})
    with pytest.raises(ValueError, match=r"(?s)contamination.*'blue'"):
        run_sim(tmp_path, "dirty", {"enabled": True, "registry": [dirty, hood(), CROSSWIRED, SUMP]})


def test_r2_refuses_a_surface_that_leaks_another_memes_words(tmp_path):
    # one meme's incident handing another meme's coinage to everyone who eats there
    leaky = tray(incident={"surfaces": ["{P} found the sash of the rack hung slack and the plate was greasy."]})
    slack = meme("slack_seal", "slack-seal", "ungrounded", "narrow",
                 '{name} has a habit of calling the same recurring chore "slack-seal".')
    with pytest.raises(ValueError, match=r"(?s)contamination.*'slack'"):
        run_sim(tmp_path, "leaky", {"enabled": True, "registry": [leaky, slack]})


def test_r2_refuses_a_phrase_the_world_already_speaks(tmp_path):
    """The two ungrounded candidates the design started from. The world hands agents "running" through the
    routines and the errands and "list" through a routine and a prompt, so either could be coined
    independently and neither can be used as it stands."""
    running = meme("running_warm", "running warm", "ungrounded", "broad",
                   '{name} has a habit of calling anything overcommitted or behind "running warm".')
    with pytest.raises(ValueError) as ei:
        run_sim(tmp_path, "rw", {"enabled": True, "registry": [running]})
    assert "'running'" in str(ei.value) and "profile[" in str(ei.value)

    tuesday = meme("tuesday_list", "the Tuesday list", "ungrounded", "narrow",
                   '{name} has a habit of calling the same recurring chore "the Tuesday list".')
    with pytest.raises(ValueError) as ei:
        run_sim(tmp_path, "tl", {"enabled": True, "registry": [tuesday]})
    assert "'list'" in str(ei.value)


def test_preflight_is_reachability_aware():
    """A phrase must not be refused over a string this config cannot emit. `referents.CATEGORIES` holds "the
    Blue Line shuttle", but with `latent_events.event_rate: 0` no agent can ever be shown it."""
    from backend.agents.profile import load_population
    cfg = make_cfg({"enabled": True, "registry": cohort()})
    profiles, _ = load_population(cfg["population"], cfg.get("population_size"))
    assert INC.preflight(cfg, profiles) == []
    words = {h["word"] for h in INC.audit(
        INC.registry(cfg), INC.world_vocabulary(cfg, profiles, all_sources=True))}
    assert "blue" in words          # unreachable here, but the report still sees it
    live = {w for w, _ in INC.world_vocabulary(cfg, profiles)}
    assert not any(w.startswith("referents.") or w.startswith("skins[") for w in live)


def test_no_world_string_or_perception_emits_a_registry_word(inc_run):
    """The R2 check over the run itself: the generated world script, every released beat, every perception,
    every planned and performed activity, and every frame."""
    sim, run_dir = inc_run
    entries = INC.registry(sim.cfg)
    assert [e["id"] for e in entries]                     # the check reads the registry, not a hard-coded list
    texts = world_run_text(run_dir)
    assert len(texts) > 1000, "the scan found almost nothing; it is not looking where the world speaks"
    hits = INC.audit(entries, texts)
    assert hits == [], "\n".join(f"{h['meme']} {h['kind']} {h['word']} in {h['where']}: {h['text']!r}"
                                 for h in hits[:10])


def test_exemptions_are_declared_and_reported(inc_run):
    sim, _ = inc_run
    mc = sim.incidents.manipulation_checks()
    assert mc["memes"]["tray_washer"]["exempt_words"] == ["tray"]
    assert mc["memes"]["tray_washer"]["distinctive_words"] == ["blue"]
    assert mc["memes"]["crosswired"]["exempt_words"] == []      # nothing narrowed, nothing to report


# ====================================================================== the mechanism: N incidents, one path
def test_ungrounded_entries_produce_no_incident(tmp_path):
    """The ungrounded arm has no basis to remove - that is what makes it the comparison - and it must fall out
    of the same loop rather than be special-cased."""
    sim = run_sim(tmp_path, "ungrounded", {"enabled": True, "registry": [CROSSWIRED, SUMP]},
                  simulation_days=1, day_end="14:30")
    assert len(sim.incidents.entries) == 2
    assert sim.incidents.conditions == []
    assert all(sim.incidents.world(t) == [] for t in range(sim.clock.total_ticks))
    assert not trace_records(sim.run_dir)["incident_encounter"]
    mc = sim.incidents.manipulation_checks()["memes"]
    assert [mc[m]["incident"] for m in ("crosswired", "sump")] == [False, False]
    assert [mc[m]["status"] for m in ("crosswired", "sump")] == ["inactive", "inactive"]


def test_conditions_go_into_the_world_script(inc_run):
    _, run_dir = inc_run
    recs = [json.loads(x) for x in (Path(run_dir) / "world_script.jsonl").read_text().splitlines() if x.strip()]
    inc = {r["meme"]: r for r in recs if r.get("kind") == "incident"}
    assert sorted(inc) == ["hood_sash", "tray_washer"]           # one per GROUNDED entry, none for the others
    assert inc["tray_washer"]["generator"] == INC.GENERATOR
    assert inc["tray_washer"]["days"] == [1, 2] and inc["tray_washer"]["repaired_day"] == 3
    assert inc["tray_washer"]["windows"] == list(INC.MEAL_WINDOWS)
    assert inc["hood_sash"]["windows"] == ["09:00-12:30", "13:00-17:00"]


def test_encounters_are_located_dated_and_witnessed(inc_run):
    sim, run_dir = inc_run
    enc = trace_records(run_dir)["incident_encounter"]
    assert enc, "no incident was ever run into: the condition is not reaching anybody"
    by = defaultdict(lambda: defaultdict(list))
    for r in enc:
        by[r["meme"]][r["state"]].append(r)
        assert (r["location"], r["arena"]) in (("Dining Hall", "Main Floor"), ("Research Lab", "Wet Lab"))
    for m, days in (("tray_washer", {1, 2}), ("hood_sash", {1, 2})):
        assert {r["day"] for r in by[m]["fault"]} <= days
        assert {r["day"] for r in by[m]["repaired"]} <= {3}      # repaired_day 3, run is 3 days
    mc = sim.incidents.manipulation_checks()["memes"]
    for m in ("tray_washer", "hood_sash"):
        assert mc[m]["encounters"]["fault"] == len(by[m]["fault"])
        assert mc[m]["witnesses"]["fault"] == len({r["agent"] for r in by[m]["fault"]})
        # the split the design needs: some people met the basis, some did not
        assert mc[m]["n_exposed"] + mc[m]["n_unexposed"] == len(sim.agents)
        assert mc[m]["n_exposed"] >= 1


def test_one_draw_per_person_per_meal(inc_run):
    """Not one per tick spent eating: an agent who sits through a whole lunch window meets the condition at
    most once in it."""
    _, run_dir = inc_run
    seen = defaultdict(int)
    for r in trace_records(run_dir)["incident_encounter"]:
        seen[(r["agent"], r["meme"], r["day"], r["occasion"])] += 1
    assert max(seen.values()) == 1


def test_witnesses_get_different_partial_views(inc_run):
    """Identical text for every witness would void the transmission question. Two channels give the variation:
    the surface each person draws from their own substream, and ordinary partial perception on top."""
    sim, run_dir = inc_run
    enc = [r for r in trace_records(run_dir)["incident_encounter"] if r["meme"] == "tray_washer"]
    templates = {re.sub(r"^\w+(?:'s)?", "{P}", r["text"]) for r in enc}
    assert len(templates) > 1, "every witness was handed the same sentence"
    # the same person does not always get the same one either
    per_agent = defaultdict(set)
    for r in enc:
        per_agent[r["agent"]].add(r["text"])
    assert max(len(v) for v in per_agent.values()) > 1


def test_encounters_run_through_the_ordinary_memory_path(inc_run):
    """Not protected memories, not injected text: a perception that is encoded, can decay and can be forgotten
    like anything else."""
    sim, run_dir = inc_run
    tr = trace_records(run_dir)
    inc_ids = {r["incident"] for r in tr["incident_encounter"]}
    linked = lambda r: any(str(e) in inc_ids for e in (r.get("originating_event_ids") or []))  # noqa: E731
    obs = [r for r in tr["observation"] if linked(r)]
    assert [r for r in obs if r["source_type"] == "perception"], "incident facts never became perceptions"
    enc = [r for r in tr["memory_encoded"] if linked(r)]
    assert [r for r in enc if r["source_type"] == "perception"], "no incident ever reached memory"
    # No injected text and no protected source: an incident memory competes and is forgotten like any other.
    protected = set(sim.cfg["memory"].get("protect_sources") or [])
    assert not {r["source_type"] for r in enc} & protected
    # that the same event ids also turn up on conversation/overheard memories is the mechanism working:
    # the incident is reaching speech, which is what the phrase needs in order to be said at all
    assert {r["source_type"] for r in enc} - {"perception"}


def test_bystanders_notice_less_than_the_person_it_happened_to(inc_run):
    """The fact `involves` the person it happened to (p_attend = 1) and is arena-visible to everyone else at
    ordinary attention, so who carries what away is not the same for the table as for the tray."""
    _, run_dir = inc_run
    tr = trace_records(run_dir)
    own, other = [], []
    whose = {r["fact"]: r["agent"] for r in tr["incident_encounter"]}   # one fact, one person it happened to
    for r in tr["observation"]:
        for f in r.get("facts") or []:
            if f.get("id") in whose:
                (own if r["agent"] == whose[f["id"]] else other).append(f.get("p_attend"))
    assert own and all(p == 1.0 for p in own if p is not None)
    assert other, "nobody at the table ever noticed anybody else's tray"
    assert all((p or 0) < 1.0 for p in other)


# ================================================================== the environmental change, and no gloss
NARRATOR = re.compile(r"\b(because|so that|which caused|fixed|repaired|solved|problem|issue|finally|"
                      r"no longer|back to normal|had been|the reason)\b", re.I)


def test_the_repair_is_perceptible(inc_run):
    sim, run_dir = inc_run
    enc = trace_records(run_dir)["incident_encounter"]
    rep = [r for r in enc if r["state"] == "repaired"]
    assert rep, "the basis was removed with nothing anybody could perceive; reinterpretation has no cue"
    assert {r["day"] for r in rep} == {3}
    assert len({r["agent"] for r in rep}) > 1


def test_the_repair_is_not_evaluated(inc_run):
    """D39: the world states what a person could see. No narrator, no causal gloss, no "the problem is fixed"."""
    _, run_dir = inc_run
    for r in trace_records(run_dir)["incident_encounter"]:
        assert not NARRATOR.search(r["text"]), r["text"]
    for e in INC.registry({"memes": {"enabled": True, "registry": cohort()}}):
        for key in ("surfaces", "repaired_surfaces"):
            for s in (e["incident"] or {}).get(key, []):
                assert not NARRATOR.search(s), s


def test_probe_gradient_never_reaches_an_agent(inc_run):
    """R7: the gradient is the observer's yardstick. It is carried through the registry and never rendered."""
    sim, run_dir = inc_run
    gradient = {v for e in INC.registry(sim.cfg) for v in e["probe_gradient"].values()} | \
               {f for e in INC.registry(sim.cfg) for f in e["foils"]}
    assert gradient
    blob = (Path(run_dir) / "trace.jsonl").read_text() + (Path(run_dir) / "frames.jsonl").read_text()
    for g in gradient:
        assert f'"{g}"' not in blob


# ============================================================================= determinism and default-off
def test_same_seed_same_incidents(tmp_path):
    a = run_sim(tmp_path, "d1", {"enabled": True, "registry": cohort()}, simulation_days=2, day_end="14:30")
    b = run_sim(tmp_path, "d2", {"enabled": True, "registry": cohort()}, simulation_days=2, day_end="14:30")
    key = lambda r: (r["tick"], r["meme"], r["agent"], r["state"], r["text"])  # noqa: E731
    ra = [key(r) for r in trace_records(a.run_dir)["incident_encounter"]]
    rb = [key(r) for r in trace_records(b.run_dir)["incident_encounter"]]
    assert ra and ra == rb
    assert trace_digest(a.run_dir) == trace_digest(b.run_dir)


def test_disabled_is_byte_identical(tmp_path):
    """R5: with the mechanism off the engine never reaches this module, so an existing run reproduces."""
    absent = run_sim(tmp_path, "absent", None, simulation_days=1, day_end="12:30")
    off = run_sim(tmp_path, "off", {"enabled": False, "registry": cohort()},
                  simulation_days=1, day_end="12:30")
    assert off.incidents is None and absent.incidents is None
    assert trace_digest(absent.run_dir) == trace_digest(off.run_dir)
    assert absent.world_sha == off.world_sha


def test_incidents_coexist_with_latent_events(tmp_path):
    """E1-E4 generation is untouched: incidents are appended to the world script and released after movement,
    so with `event_rate: 0` they may be the only thing happening, and with it non-zero both still fire."""
    sim = run_sim(tmp_path, "both", {"enabled": True, "registry": cohort()},
                  simulation_days=2, day_end="14:30", latent_events={"event_rate": 0.2})
    tr = trace_records(sim.run_dir)
    assert tr["world_event_start"], "the v2 latent events stopped being generated"
    assert tr["incident_encounter"]
    recs = [json.loads(x) for x in (Path(sim.run_dir) / "world_script.jsonl").read_text().splitlines() if x.strip()]
    assert [r for r in recs if r.get("kind") == "incident"] and [r for r in recs if "latent_type" in r]
