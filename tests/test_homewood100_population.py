"""Offline checks of the connected 100-agent subset; never execute simulation ticks.

Companion to tests/test_homewood500_population.py for the subset written by
scripts/subsample_population.py (see docs/HOMEWOOD_100_SUBSET.md).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

from backend.agents.profile import AgentProfile, load_population
from backend.config import load_config
from backend.simulation.circles import load_circles
from backend.simulation.world import ARENAS

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/homewood100_naming.yaml"
SOURCE = "configs/population/homewood500.yaml"
SOURCE_MEMORIES = "configs/population/homewood500_initial_memories.yaml"
# The exact command that produced the committed files; the determinism test reruns it.
COMMAND = ["scripts/subsample_population.py", "--n", "100", "--split", "Hopkins Cafe=20", "--seed", "42"]
AGENT_FIELDS = set(AgentProfile.__dataclass_fields__) | {"coop_role"}


def edge_count(edges, ids) -> int:
    inside = set(ids)
    return sum(1 for edge in edges if edge["a"] in inside and edge["b"] in inside)


def components(ids, adjacency) -> list[set]:
    """Connected components of the retained relationship graph, largest first."""
    seen, found = set(), []
    for start in sorted(ids):
        if start in seen:
            continue
        stack, component = [start], set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            seen.add(node)
            stack.extend(adjacency[node] - component)
        found.append(component)
    return sorted(found, key=len, reverse=True)


class Homewood100SubsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(CONFIG)
        cls.raw = yaml.safe_load((ROOT / cls.cfg["population"]).read_text())
        cls.memories = yaml.safe_load((ROOT / cls.cfg["initial_memories_file"]).read_text())
        cls.both = yaml.safe_load((ROOT / "configs/population/homewood100_both_names_memories.yaml").read_text())
        cls.source = yaml.safe_load((ROOT / SOURCE).read_text())
        cls.ids = [agent["id"] for agent in cls.raw["agents"]]
        cls.profiles, cls.groups = load_population(cls.cfg["population"], cls.cfg["population_size"])
        cls.adjacency = defaultdict(set)
        for edge in cls.raw["relationships"]:
            cls.adjacency[edge["a"]].add(edge["b"])
            cls.adjacency[edge["b"]].add(edge["a"])

    def test_naming_split_is_exactly_twenty_eighty(self):
        names = Counter()
        for aid in self.ids:
            text = " ".join(self.memories[aid])
            self.assertEqual(len(self.memories[aid]), 1, aid)
            names["Hopkins Cafe" if "Hopkins Cafe" in text else "FFC"] += 1
            # Neither primary seed presents the other name (the 500-agent design's invariant).
            self.assertNotEqual("Hopkins Cafe" in text, "FFC" in text, aid)
        self.assertEqual(names, {"Hopkins Cafe": 20, "FFC": 80})
        hopkins = [aid for aid in self.ids if "Hopkins Cafe" in " ".join(self.memories[aid])]
        cohorts = {self.profiles[aid].demographics["year"] for aid in hopkins}
        self.assertEqual(cohorts, {"first-year"}, "Hopkins Cafe must stay the first-year cohort")

    def test_seed_files_cover_exactly_the_subset(self):
        self.assertEqual(list(self.memories), self.ids)
        self.assertEqual(list(self.both), self.ids)
        source_memories = yaml.safe_load((ROOT / SOURCE_MEMORIES).read_text())
        for aid in self.ids:
            self.assertEqual(self.memories[aid], source_memories[aid], aid)
            self.assertIn("both FFC and Hopkins Cafe", " ".join(self.both[aid]), aid)

    def test_profiles_load_with_the_run_configuration(self):
        self.assertEqual(len(self.profiles), 100)
        self.assertEqual(len({p.name for p in self.profiles.values()}), 100)
        self.assertEqual(Counter(p.demographics["category"] for p in self.profiles.values()),
                         {"student": 80, "staff": 11, "faculty": 9})
        self.assertEqual(Counter(p.demographics["year"] for p in self.profiles.values()),
                         {"first-year": 20, "sophomore": 20, "junior": 20, "senior": 20,
                          "staff": 11, "faculty": 9})
        for agent in self.raw["agents"]:
            unknown = set(agent) - AGENT_FIELDS
            self.assertEqual(unknown, set(), f"{agent['id']} carries fields the loader ignores: {unknown}")
        self.assertEqual(load_circles(self.cfg["population"], self.profiles), self.raw["circles"])
        for profile in self.profiles.values():
            self.assertNotIn("Hopkins Cafe", profile.ga_learned())
            self.assertNotIn("FFC", profile.ga_learned())

    def test_relationships_and_groups_stay_inside_the_subset(self):
        inside = set(self.ids)
        for edge in self.raw["relationships"]:
            self.assertIn(edge["a"], inside)
            self.assertIn(edge["b"], inside)
            self.assertTrue(0 <= edge["familiarity"] <= 1 and 0 <= edge["affinity"] <= 1)
        pairs = [tuple(sorted((edge["a"], edge["b"]))) for edge in self.raw["relationships"]]
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertTrue(self.raw["groups"])
        for gid, members in self.raw["groups"].items():
            self.assertGreaterEqual(len(members), 2, gid)
            self.assertEqual(len(set(members)), len(members), gid)
            self.assertEqual(set(members) - inside, set(), gid)
        # Every retained edge and membership is one the source really contains.
        source_pairs = {tuple(sorted((edge["a"], edge["b"]))) for edge in self.source["relationships"]}
        self.assertEqual(set(pairs) - source_pairs, set())
        for gid, members in self.raw["groups"].items():
            self.assertEqual(set(members) - set(self.source["groups"][gid]), set(), gid)

    def test_homes_and_routines_name_real_rooms(self):
        for profile in self.profiles.values():
            with self.subTest(agent=profile.id):
                self.assertIn(profile.home["location"], ARENAS)
                self.assertIn(profile.home["arena"], ARENAS[profile.home["location"]])
                minutes = []
                for entry in profile.routine:
                    self.assertIn(entry.location, ARENAS)
                    self.assertIn(entry.arena, ARENAS[entry.location])
                    hour, minute = map(int, entry.time.split(":"))
                    self.assertTrue(0 <= hour < 24 and 0 <= minute < 60)
                    minutes.append(hour * 60 + minute)
                self.assertEqual(minutes, sorted(set(minutes)))

    def test_subset_is_socially_connected(self):
        parts = components(self.ids, self.adjacency)
        sizes = [len(part) for part in parts]
        # A fully connected cut is impossible here: in the source only Hopkins dining service staff have
        # any tie outside their own department, so the 11-staff share cannot all reach the students.
        self.assertGreaterEqual(sizes[0], 90, f"largest component {sizes[0]}, components {sizes}")
        self.assertEqual(sum(sizes), 100)
        stragglers = set(self.ids) - parts[0]
        self.assertTrue(all(self.profiles[aid].demographics["category"] == "staff" for aid in stragglers),
                        f"only staff may fall outside the main component, got {sorted(stragglers)}")
        for aid in stragglers:
            self.assertNotEqual(self.profiles[aid].demographics["department"], "Hopkins dining service")

    def test_ties_survive_far_better_than_the_naive_prefix(self):
        prefix = [agent["id"] for agent in self.source["agents"][:100]]
        baseline = edge_count(self.source["relationships"], prefix)
        kept = len(self.raw["relationships"])
        self.assertEqual(kept, edge_count(self.source["relationships"], self.ids))
        self.assertGreaterEqual(2 * kept / 100, 2 * (2 * baseline / 100),
                                f"mean degree {2 * kept / 100} vs first-100 baseline {2 * baseline / 100}")
        # The problem the subset exists to fix: the naive prefix misses the split as well as the ties.
        source_memories = yaml.safe_load((ROOT / SOURCE_MEMORIES).read_text())
        prefix_hopkins = sum("Hopkins Cafe" in " ".join(source_memories[aid]) for aid in prefix)
        self.assertNotEqual(prefix_hopkins, 20)
        self.assertLess(baseline, kept)

    def test_regenerating_the_subset_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "homewood100.yaml"
            result = subprocess.run([sys.executable, str(ROOT / COMMAND[0]), *COMMAND[1:], "--out", str(out)],
                                    cwd=ROOT, capture_output=True, text=True, timeout=600)
            self.assertEqual(result.returncode, 0, result.stderr)
            for name in ("homewood100.yaml", "homewood100_initial_memories.yaml",
                         "homewood100_both_names_memories.yaml"):
                with self.subTest(file=name):
                    self.assertEqual((Path(temp) / name).read_bytes(),
                                     (ROOT / "configs/population" / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
