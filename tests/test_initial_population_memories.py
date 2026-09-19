"""Optional population seeds remain ordinary memories, never fixed naming instructions."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from backend import ga_compat
from backend.agents.profile import load_population
from backend.config import load_config
from backend.memory.retrieval import retrieve
from backend.simulation import engine


class InitialPopulationMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.version_patch = patch.object(engine, "_code_version", return_value={})
        self.version_patch.start()
        self.addCleanup(self.version_patch.stop)
        self.sims = []

    def tearDown(self):
        for sim in self.sims:
            sim.pool.shutdown()
            sim.tracer.close()
            sim.frames_fh.close()
            sim.events_fh.close()
            sim.llm.close()

    def make_sim(self, initial=None, replay_from=None):
        """Build a real simulator without running ticks or making LLM calls."""
        cfg = load_config(overrides={"population_size": 2, "llm": {"backend": "mock"},
                                     "latent_events": {"event_rate": 0}})
        if initial is not None:
            path = self.root / f"initial-{len(self.sims)}.yaml"
            path.write_text(yaml.safe_dump(initial))
            cfg["initial_memories_file"] = str(path)
        sim = engine.Simulation(cfg, self.root / f"run-{len(self.sims)}", replay_from=replay_from,
                                progress=False)
        self.sims.append(sim)
        return sim

    def test_employees_have_role_aware_identity(self):
        profiles, _ = load_population("configs/population/homewood8.yaml", 1)
        for category, role, department in [("faculty", "professor", "Applied Mathematics and Statistics"),
                                           ("staff", "dining worker", "Dining Services")]:
            with self.subTest(category=category):
                profile = copy.deepcopy(profiles["maya"])
                profile.demographics = {"age": 42, "category": category, "role": role, "department": department}
                text = profile.ga_learned()
                self.assertIn(f"42-year-old {role} in {department}", text)
                self.assertNotIn("studying", text)
                self.assertNotIn("None", text)
                self.assertIn(profile.background, text)

    def test_legacy_student_identity_is_unchanged(self):
        profiles, _ = load_population("configs/population/homewood8.yaml", 1)
        profile = profiles["maya"]
        expected = profile.ga_learned()
        self.assertIn("junior studying Computer Science (undergraduate research assistant)", expected)
        profile.demographics["category"] = "student"
        self.assertEqual(profile.ga_learned(), expected)

    def test_seed_labels_are_private_retrievable_and_not_permanent(self):
        new_text = "During orientation I learned that the dining hall near AMR III is called Hopkins Cafe."
        old_text = "Since arriving at Hopkins I have called the dining hall near AMR III FFC."
        sim = self.make_sim({"maya": [new_text], "priya": [old_text], "unselected_student": [old_text]})
        sim._seed_memories()
        for aid, own_text, forbidden in [("maya", new_text, "FFC"), ("priya", old_text, "Hopkins Cafe")]:
            agent = sim.agents[aid]
            nodes = agent.a_mem.all_nodes()
            seed = next(n for n in nodes if n.description == own_text)
            self.assertFalse(any(forbidden in n.description for n in nodes))
            self.assertNotIn("Hopkins Cafe", agent.iss())
            self.assertNotIn("FFC", agent.iss())
            self.assertEqual(sim.meta.get(seed.node_id).source_type, "seed")
            found = retrieve(agent, [own_text], rng=np.random.default_rng(11), touch=False)[own_text].nodes
            self.assertIn(seed, found)
            # It can be forgotten through the same mechanism as other memory nodes.
            agent.a_mem.remove(seed.node_id)
            found = retrieve(agent, [own_text], rng=np.random.default_rng(11), touch=False)[own_text].nodes
            self.assertNotIn(seed, found)
        records = [json.loads(line) for line in (sim.run_dir / "trace.jsonl").read_text().splitlines()]
        seeded = [r for r in records if r.get("initial_memory")]
        self.assertEqual({r["agent"] for r in seeded}, {"maya", "priya"})
        self.assertEqual(len(seeded), 2)
        self.assertTrue(all(r["source_type"] == "seed" and r["initial_memory_index"] == 0 for r in seeded))
        self.assertEqual(sim.llm.stats["calls"], 0)

    def test_seed_once_and_replay_seed_trace_is_deterministic(self):
        data = {"maya": ["I call the dining hall Hopkins Cafe."], "priya": ["I call the dining hall FFC."]}
        original = self.make_sim(data)
        original._seed_memories()
        before = (original.run_dir / "trace.jsonl").read_bytes()
        counts = {aid: len(a.a_mem.all_nodes()) for aid, a in original.agents.items()}
        original._seed_memories()
        self.assertEqual(counts, {aid: len(a.a_mem.all_nodes()) for aid, a in original.agents.items()})
        self.assertEqual((original.run_dir / "trace.jsonl").read_bytes(), before)
        replay = self.make_sim(data, replay_from=original.run_dir / "llm_calls.jsonl")
        replay._seed_memories()
        self.assertEqual((replay.run_dir / "trace.jsonl").read_bytes(), before)

    def test_absent_and_empty_seed_files_preserve_baseline(self):
        baseline, empty = self.make_sim(), self.make_sim({})
        baseline._seed_memories()
        empty._seed_memories()
        self.assertEqual((baseline.run_dir / "trace.jsonl").read_bytes(),
                         (empty.run_dir / "trace.jsonl").read_bytes())
        self.assertEqual(self.texts(baseline), self.texts(empty))

    @staticmethod
    def texts(sim):
        return {aid: [n.description for n in a.a_mem.all_nodes()] for aid, a in sim.agents.items()}

    def test_invalid_initial_memory_content_fails(self):
        for bad in [[], None, {"maya": "not a list"}, {"maya": [""]},
                    {"maya": ["   "]}, {"maya": [7]}, {3: ["a memory"]}]:
            with self.subTest(bad=bad):
                path = self.root / "bad.yaml"
                path.write_text(yaml.safe_dump(bad))
                with self.assertRaisesRegex(ValueError, "initial_memories_file"):
                    engine._load_initial_memories(path)

    def test_initial_memory_paths_resolve_from_repo_root(self):
        (self.root / "seeds.yaml").write_text("maya: ['I remember lunch.']\n")
        with patch.object(ga_compat, "REPO_ROOT", self.root):
            self.assertEqual(engine._load_initial_memories("seeds.yaml"), {"maya": ["I remember lunch."]})
        self.assertEqual(engine._load_initial_memories(None), {})
