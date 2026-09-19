"""Bridge the pure commons world to GA memory, LLM intentions, and recorded runs."""
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict

from backend.agents.agent import Agent
from backend.agents.commons_planner import decide_commons
from backend.agents.perception import AgentObservation
from backend.llm.client import llm_scope
from backend.memory.encoder import encode
from backend.memory.reflection import reflect, should_reflect
from backend.simulation.commons import ROOMS
from backend.simulation.world import default_arena


class CommonsRuntime:
    def __init__(self, sim):
        self.sim, self.world = sim, sim.commons
        self.evidence: dict[str, list[str]] = {}
        self.events_fh = open(sim.run_dir / "commons_events.jsonl", "w", encoding="utf-8")

    def _arrive(self, event):
        sim = self.sim
        old = sim.agents[event["departed"]]
        profile = copy.deepcopy(old.profile)
        profile.id, profile.name = event["arrived"], event["name"]
        profile.background = "Recently joined this campus cooperative and has no experience of its earlier work."
        profile.relationships = {}
        profile.habits = ["takes time to learn an unfamiliar setting"]
        profile.routine = []
        new = Agent(profile, sim.cfg, sim.ctx.mods, sim.cfg["seed"])
        new.ctx = sim.ctx
        sim.agents[new.id] = new
        for members in sim.groups.values():
            if old.id in members:
                members.append(new.id)
        sim.tracer.log("newcomer_initialized", agent=new.id, predecessor=old.id,
                       n_memories=len(new.a_mem.id_to_node), inherited_relationships=0)

    def _sync(self):
        sim = self.sim
        for aid, r in self.world.residents.items():
            a = sim.agents[aid]
            old_location = a.state.location
            a.state.location, a.state.arena = r.location, ROOMS.get(r.location, default_arena(r.location))
            op = self.world.operations.get(aid)
            a.state.activity = (op.intention["action"].lower() if op else "considering the next activity") if r.active else "departed"
            a.state.path = op.intention.get("path", []) if op else ([old_location, r.location] if old_location != r.location else [])
            a.state.current_goal = "Help the campus cooperative complete useful projects and share experience."
            a.set_time(sim.clock.time_of(self.world.tick))
            a.sync_scratch()

    def _consume(self, decisions=None):
        sim, observations = self.sim, {}
        for e in self.world.drain_events():
            self.events_fh.write(json.dumps(e) + "\n")
            sim.tracer.log("commons_" + e["kind"], **e)
            if e["kind"] == "membership":
                self._arrive(e)
            if e["kind"] == "action_started" and decisions:
                self.evidence[e["event_key"]] = decisions[e["actor"]]["retrieved"]
            speech = e["kind"] == "speech"
            if speech:
                uid = "speech:" + e["event_key"]
                listeners = [a for a in e["visible_to"] if self.world.residents[a].active]
                utterance = {"id": uid, "conversation_id": None, "idx": 0, "speaker": e["actor"],
                             "text": e["text"], "listeners": listeners,
                             "location": e["location"], "arena": ROOMS.get(e["location"], default_arena(e["location"])),
                             "retrieved": self.evidence.get(e["operation"], []), "retrieved_event_ids": [],
                             "source": "commons_speech", "context": "local cooperative activity"}
                sim.tracer.log("utterance", **utterance)
                sim.tracer.log("exposure", utterance_id=uid, speaker_id=e["actor"], listener_ids=listeners,
                               utterance=e["text"], conversation_id=None, location=e["location"], arena=utterance["arena"])
                sim.tick_utts.append(utterance)
                sim.stats["utterances"] += 1
            if not e["text"]:
                continue
            for aid in e["visible_to"]:
                if not self.world.residents[aid].active:
                    continue
                r = self.world.residents[aid]
                source = ("conversation" if speech else "record" if e["kind"] == "record_read"
                          else "self" if e["kind"] == "action_completed" else "perception")
                speakers = [e["actor"]] if speech else [e["author"]] if source == "record" else []
                text = f'{sim.agents[e["actor"]].name} said: {e["text"]}' if speech else e["text"]
                o = AgentObservation(id=f"obs:{e['event_key']}:{aid}", agent_id=aid, tick=self.world.tick,
                                     location=r.location, arena=ROOMS.get(r.location, default_arena(r.location)),
                                     source_type=source,
                                     facts=[{"id": e["event_key"], "text": text, "salience": 0.65,
                                             "involves": speakers}], speakers=speakers,
                                     utterance_ids=[uid] if speech else [])
                observations.setdefault(aid, []).append(o)
        self.events_fh.flush()
        self._sync()
        return observations

    def _encode(self, observations, phase):
        sim = self.sim

        def job(aid):
            a = sim.agents[aid]
            for o in observations[aid]:
                sim.tracer.log("observation", agent=aid, observation_id=o.id, source_type=o.source_type,
                               facts=o.facts, originating_event_ids=[])
                encode(a, o, a.rng)
        sim._parallel([(f"t{self.world.tick:04d}:{phase}:encode:{aid}", lambda aid=aid: job(aid))
                       for aid in sorted(observations)])
        self._check_provider()

    def _check_provider(self):
        if self.sim.llm.stats["errors"]:
            raise RuntimeError("Provider failure invalidated this run; inspect llm_calls.jsonl")

    def _checkpoint(self):
        sim = self.sim
        root = sim.run_dir / "checkpoints" / f"tick-{self.world.tick:05d}"
        root.mkdir(parents=True, exist_ok=True)
        for aid in self.world.active_ids():
            sim.agents[aid].a_mem.save_ga(root / aid / "associative_memory")
        state = {"world": self.world.snapshot(), "active_agents": self.world.active_ids(),
                 "profiles": {aid: sim.agents[aid].profile.to_public_dict() for aid in self.world.active_ids()},
                 "memory_meta": {nid: asdict(m) for nid, m in sim.meta.meta.items()}}
        (root / "snapshot.json").write_text(json.dumps(state, indent=1), encoding="utf-8")

    def run(self):
        sim, world = self.sim, self.world
        started = time.time()
        try:
            sim.write_manifest("running")
            sim._seed_memories()
            for tick in range(sim.clock.total_ticks):
                sim.tracer.tick, sim.tracer.time = tick, sim.clock.time_of(tick).isoformat()
                sim.tick_utts = []
                with llm_scope(f"t{tick:04d}:00commons"):
                    world.advance(tick)
                    obs = self._consume()
                self._encode(obs, "01")
                ids = [aid for aid in world.active_ids() if aid not in world.operations]
                views = {aid: world.view(aid) for aid in ids}
                results = sim._parallel([
                    (f"t{tick:04d}:02commons:{aid}",
                     lambda aid=aid: decide_commons(sim.agents[aid], views[aid], [o.text() for o in obs.get(aid, [])]))
                    for aid in ids])
                self._check_provider()
                decisions = dict(zip(ids, results))
                with llm_scope(f"t{tick:04d}:03commons"):
                    world.submit({aid: d["intention"] for aid, d in decisions.items()})
                    feedback = self._consume(decisions)
                self._encode(feedback, "04")
                reflect_ids = [aid for aid in world.active_ids() if should_reflect(sim.agents[aid])]
                reflections = sim._parallel([
                    (f"t{tick:04d}:07reflect:{aid}", lambda aid=aid: reflect(sim.agents[aid], sim.agents[aid].rng))
                    for aid in reflect_ids])
                self._check_provider()
                sim.stats["reflections"] += sum(len(r) for r in reflections)
                with llm_scope(f"t{tick:04d}:99frame"):
                    sim._frame(tick, [])
                sim.tracer.flush()
                if (tick + 1) % sim.clock.ticks_per_day == 0:
                    self._checkpoint()
                    sim.write_manifest("running")
                    if sim.progress:
                        complete = sum(p.completed is not None for p in world.projects.values())
                        print(f"[{sim.clock.label(tick)}] projects={complete}/{len(world.projects)} "
                              f"records={len(world.records)} llm={sim.llm.stats['calls']}", flush=True)
            (sim.run_dir / "commons_final.json").write_text(json.dumps(world.snapshot(), indent=1), encoding="utf-8")
            sim.finish(time.time() - started)
        finally:
            self.events_fh.close()
