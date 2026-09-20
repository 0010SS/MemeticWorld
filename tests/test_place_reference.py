"""Who calls a place what: the observer must find rival names without being told them, must not invent a
name where agents only described the place, and must not break the dining-name outputs it sits beside.

Synthetic traces only -- no LLM calls and no simulation runs.
"""
import json
from pathlib import Path
import sys
import unittest

from backend.analysis import places, trends
from scripts.analyze_dining_names import analyze

WORLD = {
    "graph": {"Dining Hall": ["Dorm", "Library"], "Dorm": ["Dining Hall"], "Library": ["Dining Hall"]},
    "arenas": {"Dining Hall": ["Main Floor"], "Dorm": ["Room 1"], "Library": ["Study Tables"]},
}
TICKS_PER_DAY = 10
SEED = "In my recent everyday conversations, I have called the dining hall beside the dorms {name} when arranging meals."


def write_run(path: Path, agents: dict, rows: list) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.resolved.yaml").write_text("world: {}\n", encoding="utf-8")
    (path / "manifest.json").write_text(json.dumps({
        "agents": agents, "world": WORLD, "ticks_per_day": TICKS_PER_DAY, "tick_minutes": 15,
        "start": "2026-09-14T08:30:00",
    }), encoding="utf-8")
    (path / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def agent(aid: str, name: str) -> tuple:
    return aid, {"id": aid, "name": name, "demographics": {"category": "student", "year": "junior"},
                 "routine": [{"time": "12:00", "location": "Dining Hall", "activity": "having lunch"}]}


def seed_row(index: int, aid: str, name: str) -> dict:
    return {"id": f"seed#{index}", "type": "memory_encoded", "tick": 0, "agent": aid, "kind": "thought",
            "text": SEED.format(name=name), "source_type": "seed"}


def utterance(uid: str, tick: int, speaker: str, text: str, listeners: list, location="Dining Hall",
              conversation=None, idx=0) -> dict:
    return {"id": uid, "type": "utterance", "tick": tick, "speaker": speaker, "text": text,
            "listeners": listeners, "location": location, "arena": "Main Floor",
            "conversation_id": conversation or uid, "idx": idx}


def move(uid: str, tick: int, aid: str, to="Dining Hall") -> dict:
    return {"id": uid, "type": "move", "tick": tick, "agent": aid, "frm": ["Dorm", "Room 1"],
            "to": [to, "Main Floor"], "activity": "having lunch"}


def competing_run(path: Path) -> Path:
    """Two seeded names for one hall. Day 1 they are used equally; on day 2 one of them takes over."""
    ids = ["k1", "k2", "k3", "n1", "n2", "n3"]
    names = ["Avery Stone", "Rowan Pike", "Sage Marsh", "Quinn Vale", "Reese Orr", "Lane Birch"]
    agents = dict(agent(aid, name) for aid, name in zip(ids, names))
    rows = [seed_row(i, aid, "Kirby" if aid.startswith("k") else "Nolans") for i, aid in enumerate(ids)]
    rows += [move(f"m{i}", 1, aid) for i, aid in enumerate(ids)]
    others = lambda me: [a for a in ids if a != me]
    day1 = [("k1", "Kirby"), ("k2", "Kirby"), ("k3", "Kirby"),
            ("n1", "Nolans"), ("n2", "Nolans"), ("n3", "Nolans")]
    day2 = [("k1", "Kirby"), ("k2", "Kirby"), ("k3", "Kirby"),
            ("n1", "Kirby"), ("n2", "Kirby"), ("n3", "Kirby"), ("n3", "Nolans")]
    for i, (speaker, name) in enumerate(day1):
        rows.append(utterance(f"d1u{i}", 2 + i, speaker, f"Want to grab lunch at {name} later?", others(speaker)))
    for i, (speaker, name) in enumerate(day2):
        rows.append(utterance(f"d2u{i}", 12 + i, speaker, f"I'm heading to {name} for lunch.", others(speaker)))
    return write_run(path, agents, rows)


def described_run(path: Path) -> Path:
    """Nobody names the hall: they describe it, or point at it from inside."""
    ids = ["a1", "a2", "a3"]
    agents = dict(agent(aid, name) for aid, name in zip(ids, ["Avery Stone", "Rowan Pike", "Sage Marsh"]))
    rows = [move(f"m{i}", 1, aid) for i, aid in enumerate(ids)]
    lines = ["Let's meet at the hall for lunch.", "I'm eating here today.", "This place gets busy at noon.",
             "I'm in Computer Science this term.", "Lunch at the hall again?", "See you here after class."]
    for i, text in enumerate(lines):
        speaker = ids[i % len(ids)]
        rows.append(utterance(f"u{i}", 2 + i, speaker, text, [a for a in ids if a != speaker]))
    return write_run(path, agents, rows)


class PlaceReferenceTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    # ---------------------------------------------------------------- two names competing for one place
    def test_two_rival_names_are_both_found_and_entropy_falls_as_one_wins(self):
        run = competing_run(self.root / "competing")
        result = places.resolve_place_references(run)
        hall = next(p for p in result["places"] if p["place"] == "Dining Hall")
        found = {e["expression"]: e for e in hall["expressions"]}
        self.assertEqual(set(found), {"Kirby", "Nolans"}, "both rival names must be resolved to the hall")
        self.assertTrue(all(e["kind"] == "name" for e in found.values()))
        self.assertEqual(found["Kirby"]["uses"], 9)
        self.assertEqual(found["Nolans"]["uses"], 4)
        self.assertEqual(found["Kirby"]["first_tick"], 2)
        self.assertEqual(hall["leader"], "Kirby")
        self.assertEqual(hall["agent_named"], 13)
        self.assertEqual(hall["world_label_uses"], 0)

        # each name starts inside the group that was seeded with it, and Kirby then spreads beyond it
        self.assertEqual(found["Nolans"]["seeded_group_share"]["share"], 1.0)
        self.assertEqual(found["Nolans"]["seeded_group_share"]["seeded_agents"], 3)
        self.assertLess(found["Kirby"]["seeded_group_share"]["share"], 1.0)
        self.assertEqual(found["Kirby"]["speakers"], 6)

        # day 1: 3 Kirby vs 3 Nolans = 1 bit. day 2: 6 vs 1 = well under a bit, and falling.
        day1, day2 = hall["naming_entropy_by_day"]
        self.assertEqual(day1, 1.0)
        self.assertLess(day2, day1)
        self.assertAlmostEqual(day2, 0.5917, places=3)
        self.assertEqual(found["Kirby"]["uses_by_day"], [3, 6])
        self.assertEqual(found["Nolans"]["share_by_day"], [0.5, round(1 / 7, 4)])

    def test_competition_block_reaches_trends_with_per_window_entropy(self):
        run = competing_run(self.root / "competing")
        resolved = places.resolve_place_references(run)
        windows = [0, 10]
        block = places.competition_series(resolved, windows, 10)
        hall = next(b for b in block if b["referent"] == "Dining Hall")
        self.assertEqual([r["expression"] for r in hall["rivals"]], ["Kirby", "Nolans"])
        self.assertEqual(hall["entropy_by_window"][0], 1.0)
        self.assertLess(hall["entropy_by_window"][1], hall["entropy_by_window"][0])
        self.assertEqual(hall["rivals"][0]["uses_by_window"], [3, 6])
        self.assertEqual(hall["leader"], "Kirby")

        payload = trends.compute([{"tick": t} for t in range(20)], [], tick=19, window=10, tpd=TICKS_PER_DAY,
                                 n_active=6, competition=block)
        self.assertEqual(payload["competition"], block)
        self.assertNotIn("competition", payload["deferred"], "a populated block is no longer deferred")
        self.assertEqual(len(payload["competition"][0]["entropy_by_window"]), len(payload["windows"]))

    def test_no_competition_block_without_rivals_keeps_the_key_deferred(self):
        payload = trends.compute([{"tick": 0}], [], tick=0, window=4, tpd=TICKS_PER_DAY, n_active=1)
        self.assertEqual(payload["competition"], [])
        self.assertIn("competition", payload["deferred"])

    # ---------------------------------------------------------------- everyone only describes the place
    def test_description_only_run_is_all_unnamed_and_invents_no_names(self):
        run = described_run(self.root / "described")
        result = places.resolve_place_references(run)
        hall = next(p for p in result["places"] if p["place"] == "Dining Hall")
        self.assertEqual(hall["agent_named"], 0, "a described place must not acquire a name")
        self.assertEqual(hall["world_label_uses"], 0, "nobody said the world's label either")
        self.assertEqual(hall["unnamed"], hall["references"])
        self.assertEqual(hall["leader"], None)
        self.assertEqual(hall["agent_name_share"], 0.0)
        self.assertEqual(hall["expressions"], [])
        self.assertGreater(hall["unnamed_breakdown"]["description"], 0)
        self.assertGreater(hall["unnamed_breakdown"]["deictic"], 0)
        self.assertEqual(hall["naming_entropy_by_day"], [None])

        named = [e["expression"] for p in result["places"] for e in p["expressions"] if e["kind"] == "name"]
        self.assertEqual(named, [], f"no place may be named in a description-only run, got {named}")
        self.assertNotIn("Computer Science", [u["expression"] for u in result["unattributed"]])
        self.assertEqual(places.competition_series(result, [0], 10), [],
                         "describing a place is not a naming contest")

    def test_an_unsituated_run_reports_no_situated_descriptors(self):
        result = places.resolve_place_references(described_run(self.root / "described"))
        self.assertIs(result["situated"], False)
        self.assertTrue(all(p["situated_descriptor"] is None for p in result["places"]))
        self.assertEqual(sum(p["unnamed_breakdown"]["situated_descriptor"] for p in result["places"]), 0)

    def test_a_situated_run_counts_the_world_descriptor_as_unnamed_not_as_the_label(self):
        try:
            from backend.simulation import reference
        except ImportError:                             # a build without the situated machinery
            self.skipTest("backend.simulation.reference is not present")
        descriptor = reference.place("Dining Hall", cfg={"world": {"reference_mode": "situated"}})
        if descriptor == "Dining Hall":
            self.skipTest("this build has no situated description for the Dining Hall")
        run = described_run(self.root / "situated")
        (run / "config.resolved.yaml").write_text("world:\n  reference_mode: situated\n", encoding="utf-8")
        rows = [json.loads(l) for l in (run / "trace.jsonl").read_text().splitlines()]
        rows.append(utterance("s1", 7, "a1", f"I'll be at {descriptor} at noon.", ["a2"]))
        (run / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        result = places.resolve_place_references(run)
        hall = next(p for p in result["places"] if p["place"] == "Dining Hall")
        self.assertIs(result["situated"], True)
        self.assertEqual(hall["situated_descriptor"], descriptor)
        self.assertEqual(hall["unnamed_breakdown"]["situated_descriptor"], 1)
        self.assertEqual(hall["world_label_uses"], 0,
                         "the world's description must not be read as agents echoing its label")
        self.assertEqual(hall["agent_named"], 0)

    # ---------------------------------------------------------------- the dining-name outputs still hold
    def test_dining_name_outputs_stay_backward_compatible(self):
        run = competing_run(self.root / "competing")
        # the same trace, plus two utterances carrying the experiment's own planted aliases
        rows = [json.loads(l) for l in (run / "trace.jsonl").read_text().splitlines()]
        rows.append(utterance("x1", 15, "k1", "We all say FFC now.", ["n1"]))
        rows.append(utterance("x2", 16, "n1", "I grew up calling it Hopkins Cafe.", ["k1"]))
        (run / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        result = analyze(run)

        for key in ("analysis", "schema_version", "run_directory", "aliases", "definitions", "limitations",
                    "speakers_missing_from_manifest", "overall", "by_category", "by_year", "by_category_and_year"):
            self.assertIn(key, result, f"schema-1 key {key} must survive")
        overall = result["overall"]
        self.assertEqual(overall["lexical_mentions"], {"ffc": 1, "hopkins_cafe": 1})
        self.assertEqual(overall["utterances_by_alias"]["ffc_only"], 1)
        self.assertEqual(overall["alias_bearing_utterance_denominator"], 2)
        self.assertEqual(overall["unique_speakers"]["ffc"], 1)
        self.assertEqual(overall["total_utterances"], 15)
        self.assertEqual(result["by_category"]["student"]["total_utterances"], 15)
        self.assertEqual(result["observed_naming_opportunities"] if "observed_naming_opportunities" in result
                         else overall["observed_naming_opportunities"], None)

    def test_dining_name_cohorts_crossovers_and_convergence(self):
        run = competing_run(self.root / "cohorts")
        rows = [json.loads(l) for l in (run / "trace.jsonl").read_text().splitlines()]
        # k-agents are seeded with one name, n-agents with the other; re-seed them with the planted aliases
        for row in rows:
            if row.get("type") == "memory_encoded":
                row["text"] = SEED.format(name="FFC" if row["agent"].startswith("k") else "Hopkins Cafe")
        rows.append(utterance("x1", 3, "k1", "Lunch at FFC?", ["n1", "k2"]))
        rows.append(utterance("x2", 13, "n1", "Meeting everyone at FFC, apparently.", ["k1"]))
        rows.append(utterance("x3", 14, "n2", "I still say Hopkins Cafe.", ["k1"]))
        rows.append(utterance("x4", 15, "k3", "Meet you at the dining hall.", ["n2"]))
        rows.append(utterance("x5", 16, "n3", "I'm eating here again.", ["k3"]))
        (run / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        result = analyze(run)

        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(result["seeded_cohorts"]["sizes"], {"seeded_ffc": 3, "seeded_hopkins_cafe": 3})
        self.assertEqual(result["days"], [1, 2])

        crossings = result["crossovers"]
        self.assertEqual([(c["agent"], c["alias"]) for c in crossings], [("n1", "ffc")])
        self.assertEqual(crossings[0]["seeded_with"], ["hopkins_cafe"])
        self.assertEqual(crossings[0]["day"], 2)
        self.assertEqual(crossings[0]["exposed_by"]["utterance_id"], "x1",
                         "the crossover must point at the earlier utterance that could have exposed it")
        self.assertEqual(result["first_uses_by_unseeded_agents"], [])

        ffc = result["by_seeded_cohort"]["seeded_ffc"]["by_day"]
        hopkins = result["by_seeded_cohort"]["seeded_hopkins_cafe"]["by_day"]
        self.assertEqual([r["ffc"] for r in ffc], [1, 0])
        self.assertEqual([r["ffc"] for r in hopkins], [0, 1])
        self.assertEqual([r["hopkins_cafe"] for r in hopkins], [0, 1])
        self.assertEqual(result["convergence"]["entropy_bits_by_day"], [0.0, 1.0])
        self.assertEqual(result["convergence"]["by_cohort_by_day"]["seeded_ffc"], [0.0, None])

        refs = result["dining_references"]
        self.assertEqual(refs["with_seeded_name"], 3)
        self.assertEqual(refs["description_only"], 1)
        self.assertEqual(refs["deictic_in_a_dining_location_only"], 1)
        self.assertEqual(refs["total"], 5)
        self.assertEqual(refs["dining_locations"], ["Dining Hall"])
        self.assertEqual(refs["share_using_neither_seeded_name"], 0.4)


if __name__ == "__main__":
    sys.exit(unittest.main())
