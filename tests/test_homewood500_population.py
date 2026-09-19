"""Offline checks of the generated population; never execute simulation ticks."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from backend.config import load_config
from backend.simulation import engine
from backend.simulation.rngs import seed_rng
from backend.simulation.scheduler import plan_day, routine_position
from backend.simulation.world import ARENAS
from scripts.generate_homewood500 import build, YEARS

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/homewood500_naming.yaml"


def scheduled_occupancy(sim):
    """Evaluate native plans on clock positions; no movement, perception, or LLM phases."""
    peaks = {}
    dining_peak = None
    for day in range(1, sim.clock.days + 1):
        for aid, agent in sim.agents.items():
            agent.day_plan = plan_day(agent, sim.clock, seed_rng(sim.world_seed, "plan", aid, day))
        for k in range(sim.clock.ticks_per_day):
            counts = Counter()
            dining_roles = Counter()
            for agent in sim.agents.values():
                pos = routine_position(agent, k)
                counts[(pos["location"], pos["arena"])] += 1
                if pos["location"] == "Dining Hall":
                    dining_roles[agent.profile.demographics["category"]] += 1
            label = sim.clock.label((day - 1) * sim.clock.ticks_per_day + k)
            for place, count in counts.items():
                if count > peaks.get(place, {}).get("agents", -1):
                    peaks[place] = {"location": place[0], "arena": place[1], "agents": count,
                                    "first_peak": label}
            count = sum(dining_roles.values())
            if dining_peak is None or count > dining_peak["agents"]:
                dining_peak = {"agents": count, "first_peak": label,
                               "by_role": dict(sorted(dining_roles.items()))}
            if sum(counts.values()) != 500:
                raise AssertionError(f"Scheduled occupancy does not sum to 500 at {label}")
    return {"method": "native plan_day and routine_position; no simulation phases",
            "days": sim.clock.days, "clock_positions_checked": sim.clock.total_ticks,
            "world_seed": sim.world_seed,
            "max_arena_occupancy": max(peaks.values(), key=lambda p: p["agents"]),
            "peak_dining_hall": dining_peak,
            "max_per_arena": [peaks[p] for p in sorted(peaks)]}


class Homewood500PopulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.cfg = load_config(CONFIG)
        cls.cfg["llm"]["backend"] = "mock"
        cls.raw = yaml.safe_load((ROOT / cls.cfg["population"]).read_text())
        cls.memories = yaml.safe_load((ROOT / cls.cfg["initial_memories_file"]).read_text())
        cls.audit_file = ROOT / "configs/population/homewood500_audit.json"
        cls.audit = json.loads(cls.audit_file.read_text())
        with patch.object(engine, "_code_version", return_value={}):
            cls.sim = engine.Simulation(cls.cfg, Path(cls.temp.name) / "offline-validation", progress=False)
        cls.addClassCleanup(cls.sim.pool.shutdown)
        cls.addClassCleanup(cls.sim.tracer.close)
        cls.addClassCleanup(cls.sim.frames_fh.close)
        cls.addClassCleanup(cls.sim.events_fh.close)
        cls.addClassCleanup(cls.sim.llm.close)
        cls.sim._seed_memories()
        cls.trace = [json.loads(line) for line in (cls.sim.run_dir / "trace.jsonl").read_text().splitlines()]
        cls.initial = [r for r in cls.trace if r.get("initial_memory")]
        cls.occupancy = scheduled_occupancy(cls.sim)
        cls.summary = {
            "schema_version": 1,
            "validation_kind": "offline population loading, seeding, and schedule inspection",
            "configuration": CONFIG,
            "generation_seed": cls.audit["seed"],
            "profile_sha256": hashlib.sha256((ROOT / cls.cfg["population"]).read_bytes()).hexdigest(),
            "agent_count": len(cls.sim.agents),
            "unique_names": len({a.name for a in cls.sim.agents.values()}),
            "roles": dict(sorted(Counter(a.profile.demographics["category"] for a in cls.sim.agents.values()).items())),
            "student_years": dict(sorted(Counter(a.profile.demographics["year"] for a in cls.sim.agents.values()
                                                  if a.profile.demographics["category"] == "student").items())),
            "initial_name_memories": {"Hopkins Cafe": sum("Hopkins Cafe" in r["text"] for r in cls.initial),
                                      "FFC": sum("FFC" in r["text"] for r in cls.initial)},
            "relationship_edges": len(cls.raw["relationships"]),
            "relationship_seed_memories": len(cls.trace) - len(cls.initial),
            "ordinary_groups": len(cls.raw["groups"]),
            "traits": {key: dict(sorted(Counter(a[key] for a in cls.audit["agents"].values()).items()))
                       for key in ("sociability", "planning", "communication")},
            "student_club_members": sum(bool(a["interests"]["clubs"]) for a in cls.raw["agents"]
                                        if a["demographics"]["category"] == "student"),
            "on_campus_students_by_year": {year: sum(a["demographics"]["residence"] == "on campus"
                                                     for a in cls.raw["agents"] if a["demographics"]["year"] == year)
                                           for year in YEARS},
            "schedule": cls.occupancy,
            "llm_calls": cls.sim.llm.stats["calls"],
            "simulation_ticks_executed": 0,
            "limitations": ["Counts describe a synthetic population, not measured campus behavior.",
                            "Scheduled room occupancy does not establish scalable interaction cost.",
                            "No conversation, confusion, learning, or naming alignment outcomes were tested."]}

    def test_population_counts_and_identity_rendering(self):
        sim = self.sim
        self.assertEqual(len(sim.agents), 500)
        self.assertEqual(len({a.name for a in sim.agents.values()}), 500)
        self.assertEqual(self.summary["roles"], {"student": 400, "faculty": 44, "staff": 56})
        self.assertEqual(self.summary["student_years"], {y: 100 for y in YEARS})
        faculty_roles = Counter(a.profile.demographics["role"] for a in sim.agents.values()
                                if a.profile.demographics["category"] == "faculty")
        self.assertEqual(faculty_roles, {"assistant professor": 12, "associate professor": 14,
                                         "professor": 14, "lecturer": 4})
        dining_workers = [a for a in sim.agents.values()
                          if a.profile.demographics["department"] == "Hopkins dining service"]
        self.assertEqual(len(dining_workers), 8)
        for agent in sim.agents.values():
            with self.subTest(agent=agent.id):
                identity = agent.iss()
                self.assertNotIn("Hopkins Cafe", identity)
                self.assertNotIn("FFC", identity)
                if agent.profile.demographics["category"] in {"faculty", "staff"}:
                    self.assertNotIn("studying", agent.profile.ga_learned())
                    self.assertIn(agent.profile.demographics["role"], identity)

    def test_seed_count_isolation_and_idempotence(self):
        self.assertEqual(len(self.initial), 500)
        self.assertEqual(self.summary["initial_name_memories"], {"Hopkins Cafe": 100, "FFC": 400})
        for record in self.initial:
            aid = record["agent"]
            expected = "Hopkins Cafe" if self.sim.agents[aid].profile.demographics["year"] == "first-year" else "FFC"
            other = "FFC" if expected == "Hopkins Cafe" else "Hopkins Cafe"
            self.assertIn(expected, record["text"])
            self.assertNotIn(other, record["text"])
            node = self.sim.agents[aid].a_mem.id_to_node[record["node_id"]]
            self.assertEqual(self.sim.meta.get(node.node_id).source_type, "seed")
            self.assertEqual(node.description, self.memories[aid][0])
        counts = {aid: len(a.a_mem.all_nodes()) for aid, a in self.sim.agents.items()}
        self.sim._seed_memories()
        self.assertEqual(counts, {aid: len(a.a_mem.all_nodes()) for aid, a in self.sim.agents.items()})
        self.assertEqual(self.sim.llm.stats["calls"], 0)
        self.assertEqual((self.sim.run_dir / "frames.jsonl").stat().st_size, 0)

    def test_rooms_times_and_relationships_are_valid(self):
        ids = set(self.sim.agents)
        for aid, agent in self.sim.agents.items():
            profile = agent.profile
            self.assertIn(profile.home["arena"], ARENAS[profile.home["location"]])
            minutes = []
            for entry in profile.routine:
                self.assertIn(entry.arena, ARENAS[entry.location])
                h, m = map(int, entry.time.split(":"))
                self.assertTrue(0 <= h < 24 and 0 <= m < 60)
                minutes.append(h * 60 + m)
            self.assertEqual(minutes, sorted(set(minutes)))
            self.assertEqual(set(profile.relationships), ids - {aid})
            for oid, rel in profile.relationships.items():
                self.assertEqual(asdict(rel), asdict(self.sim.agents[oid].profile.rel(aid)))
                self.assertTrue(0 <= rel.familiarity <= 1 and 0 <= rel.affinity <= 1)
        edge_pairs = [tuple(sorted((r["a"], r["b"]))) for r in self.raw["relationships"]]
        self.assertEqual(len(edge_pairs), len(set(edge_pairs)))

    def test_saved_population_and_sidecars_match_regeneration(self):
        population, memories, both, audit = build(self.audit["seed"])
        self.assertEqual(population, self.raw)
        self.assertEqual(memories, self.memories)
        self.assertEqual(audit, self.audit["agents"])
        self.assertEqual(both, yaml.safe_load((ROOT / "configs/population/homewood500_both_names_memories.yaml").read_text()))
        self.assertEqual(self.summary["profile_sha256"], self.audit["profile_sha256"])

    def test_traits_and_basic_representation_are_balanced(self):
        expected_labels = {"sociability": ("reserved", "moderate", "outgoing"),
                           "planning": ("structured", "flexible", "spontaneous"),
                           "communication": ("brief", "balanced", "detailed")}
        strata = defaultdict(list)
        for row in self.audit["agents"].values():
            strata[row["year"]].append(row)
        for cohort, rows in strata.items():
            n = len(rows)
            for key, labels in expected_labels.items():
                expected = dict(zip(labels, [round(n * .2), n - 2 * round(n * .2), round(n * .2)]))
                self.assertEqual(Counter(row[key] for row in rows), expected, (cohort, key))
        self.assertEqual(self.summary["student_club_members"], 332)
        self.assertEqual(self.summary["on_campus_students_by_year"], dict(zip(YEARS, [93, 90, 40, 25])))
        for year in YEARS:
            students = [a for a in self.raw["agents"] if a["demographics"]["year"] == year]
            self.assertEqual(Counter(a["demographics"]["division"] for a in students),
                             {"Engineering": 36, "Arts and Sciences": 64})

    def test_schedule_inspection_is_not_a_simulation_run(self):
        self.assertEqual(self.occupancy["clock_positions_checked"], 102)
        self.assertGreater(self.occupancy["peak_dining_hall"]["agents"], 8)
        self.assertEqual(sum(self.occupancy["peak_dining_hall"]["by_role"].values()),
                         self.occupancy["peak_dining_hall"]["agents"])
        self.assertLessEqual(self.occupancy["max_arena_occupancy"]["agents"], 500)
        self.assertEqual(self.sim.llm.stats["calls"], 0)
        self.assertEqual(self.sim.stats["conversations"], 0)


if __name__ == "__main__":
    unittest.main()
