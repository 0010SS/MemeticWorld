"""Read-only access to a finished (or running) run directory for the OBSERVER layer.

Works on v1 runs (events without circle/referents/structure fields) and v2 runs alike: every v2
field is read with .get() and a neutral default.
"""
from __future__ import annotations

import json
import re
from functools import cached_property
from pathlib import Path

import yaml


def chron_key(u: dict) -> tuple:
    """A total display order of utterances: within a tick, reaction remarks (phase 03) before
    conversations (phase 05), and a conversation's turns by index. It is NOT a causal order: use
    `precedes` to decide whether one utterance could have been heard before another was produced."""
    cid = u.get("conversation_id")
    return (u["tick"], 1 if cid else 0, cid or "", int(u.get("idx") or 0), u.get("id") or u.get("utterance_id") or "")


def precedes(x: dict, y: dict) -> bool:
    """Could utterance x have been heard before utterance y was produced?

    Only when x's tick is strictly earlier, or x is an earlier turn of the SAME conversation at the same
    tick. Everything else at one tick is simultaneous: every reaction remark and conversation opening
    of tick t is decided in parallel (engine phase 02) before any is emitted; conversations of one tick
    run independently of each other; and the remarks overheard at tick t reach memory only after tick
    t's conversations (phase 06), so not even a tick-t conversation turn can have heard a tick-t remark."""
    if x["tick"] != y["tick"]:
        return x["tick"] < y["tick"]
    cx, cy = x.get("conversation_id"), y.get("conversation_id")
    return bool(cx) and cx == cy and int(x.get("idx") or 0) < int(y.get("idx") or 0)


def utterance_key(u: dict) -> str:
    """The exchange an utterance belongs to: its conversation, or the remark itself."""
    return u.get("conversation_id") or u.get("utterance_id") or u.get("id") or ""


_QUOTED = re.compile(r"[\"“]([^\"”]+)[\"”]")


def planted_spec(cfg: dict, manifest: dict | None = None, agents: dict | None = None) -> dict | None:
    """The planted-phrase positive control as the simulation applied it: {"agent", "habit" (the string
    appended to the agent's habits), "phrase" (the quoted wording, lower case)}, or None.

    Prefers the manifest's record of what `profile.apply_planted` returned; otherwise re-applies
    apply_planted's own normalisation (collapse whitespace, drop the final period and a leading first
    name) to the config sentence, so the observer and the engine cannot disagree about the string."""
    pp = (cfg.get("controls") or {}).get("planted_phrase")
    if not pp:
        return None
    raw = str(pp.get("habit") or "")
    rec = (manifest or {}).get("planted") if isinstance((manifest or {}).get("planted"), dict) else None
    aid = (rec or {}).get("agent") or pp.get("agent")
    habit = (rec or {}).get("habit")
    if not habit:
        from types import SimpleNamespace

        from backend.agents.profile import apply_planted
        name = ((agents or {}).get(aid) or {}).get("name") or str(aid or "").title()
        stub = SimpleNamespace(first_name=name.split()[0] if name.split() else "", habits=[])
        try:
            habit = (apply_planted({aid: stub}, cfg) or {}).get("habit") or ""
        except ValueError:
            habit = ""
    q = _QUOTED.findall(raw) or _QUOTED.findall(habit)
    return {"agent": aid, "habit": habit, "phrase": (q[0] if q else habit).strip().lower()}


def simulated_profiles(cfg: dict, manifest: dict | None = None, planted: bool = True) -> dict:
    """Agent profiles as the simulation built them: the population file, then generated topology
    (`topology.mode: generated`, same world_seed stream as the engine), then (planted=True) the
    planted-phrase habit. With a manifest, each profile's recorded habits, routine, relationships and
    known locations win (the manifest records the profiles as simulated)."""
    from backend.agents.profile import Relationship, RoutineEntry, apply_planted, load_population
    res = load_population(cfg["population"], cfg.get("population_size"))
    profiles = res[0] if isinstance(res, tuple) else res
    if (cfg.get("topology") or {}).get("mode") == "generated":
        from backend.agents import topology as TOPO
        from backend.simulation.rngs import seed_rng, world_seed
        TOPO.apply(profiles, TOPO.generate(profiles, cfg, seed_rng(world_seed(cfg), "topology")))
    if planted:
        apply_planted(profiles, cfg)
    for aid, rec in ((manifest or {}).get("agents") or {}).items():
        prof = profiles.get(aid)
        if prof is None or not isinstance(rec, dict):
            continue
        if isinstance(rec.get("habits"), list):
            prof.habits = [str(h) for h in rec["habits"]]
        if isinstance(rec.get("routine"), list) and rec["routine"]:
            prof.routine = [RoutineEntry(time=r["time"], location=r["location"], activity=r["activity"],
                                         arena=r.get("arena")) for r in rec["routine"]]
        if isinstance(rec.get("relationships"), dict) and rec["relationships"]:
            prof.relationships = {b: Relationship(**{k: v[k] for k in ("relation_type", "familiarity", "affinity") if k in v})
                                  for b, v in rec["relationships"].items() if isinstance(v, dict)}
        if isinstance(rec.get("known_locations"), list) and rec["known_locations"]:
            prof.known_locations = list(rec["known_locations"])
    return profiles


class RunData:
    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        self.manifest = json.load(open(self.dir / "manifest.json"))
        self.cfg = yaml.safe_load(open(self.dir / "config.resolved.yaml"))

    @cached_property
    def trace(self) -> list[dict]:
        with open(self.dir / "trace.jsonl") as f:
            return [json.loads(l) for l in f]

    @cached_property
    def _by_type(self) -> dict:
        out: dict[str, list] = {}
        for r in self.trace:
            out.setdefault(r["type"], []).append(r)
        return out

    def of(self, *types) -> list[dict]:
        if len(types) == 1:
            return list(self._by_type.get(types[0], []))
        t = set(types)
        return [r for r in self.trace if r["type"] in t]

    @cached_property
    def utterances(self) -> list[dict]:
        us = self.of("utterance")
        us.sort(key=lambda u: (u["tick"], u["id"]))
        return us

    @cached_property
    def utt_by_id(self) -> dict:
        return {u["id"]: u for u in self.utterances}

    @cached_property
    def conversations(self) -> dict:
        return {c["id"]: c for c in self.of("conversation")}

    @cached_property
    def events(self) -> dict:
        out = {}
        p = self.dir / "events.jsonl"
        if p.exists():
            for l in open(p):
                if l.strip():
                    e = json.loads(l)
                    out[e["id"]] = e
        return out

    @cached_property
    def agents(self) -> dict:
        return self.manifest["agents"]

    @cached_property
    def groups(self) -> dict:
        return self.manifest.get("groups", {})

    def groups_of(self, agent_id) -> set:
        return {g for g, m in self.groups.items() if agent_id in m}

    @cached_property
    def names(self) -> dict:
        return {aid: a["name"] for aid, a in self.agents.items()}

    @cached_property
    def name_tokens(self) -> set:
        return {t.lower() for a in self.agents.values() for t in a["name"].split()}

    @property
    def ticks_per_day(self) -> int:
        return int(self.manifest["ticks_per_day"])

    @property
    def condition(self):
        return self.manifest.get("condition")

    @cached_property
    def memory_meta(self) -> dict:
        p = self.dir / "memory_meta.json"
        return json.load(open(p)) if p.exists() else {}

    @cached_property
    def nodes(self) -> dict:
        """node_id -> {"source_type", "event_ids"} for every memory the run ever created (the trace keeps
        forgotten nodes; memory_meta.json fills in anything the trace lacks)."""
        out: dict[str, dict] = {}

        def put(nid, st, evs):
            n = out.setdefault(nid, {"source_type": st, "event_ids": []})
            n["source_type"] = n["source_type"] or st
            n["event_ids"] += [e for e in evs or [] if e not in n["event_ids"]]
        for r in self.trace:
            t = r["type"]
            if t == "memory_encoded":
                put(r["node_id"], r.get("source_type"), r.get("originating_event_ids"))
            elif t == "memory_merged":
                put(r["node_id"], None, r.get("originating_event_ids"))
            elif t == "reflection" and r.get("node_id"):
                put(r["node_id"], "reflection", r.get("originating_event_ids"))
            elif t == "reminding" and r.get("node_id"):
                put(r["node_id"], "reminding", r.get("originating_event_ids"))
        for nid, m in self.memory_meta.items():
            if isinstance(m, dict):
                put(nid, m.get("source_type"), m.get("originating_event_ids"))
        return out

    @cached_property
    def provenance(self) -> dict:
        """utterance_id -> typed event links (see provenance.py)."""
        from backend.analysis.provenance import compute
        return compute(self)

    @cached_property
    def planted(self) -> dict | None:
        """The planted-phrase positive control as applied (see `planted_spec`), or None."""
        return planted_spec(self.cfg, self.manifest, self.agents)

    @cached_property
    def lexicon_tokens(self) -> set:
        """Population lexicon tokens (emergence.vocabulary: manifest, else the population file), without
        planted-only tokens."""
        from backend.analysis.emergence import vocabulary
        return set(vocabulary(self)["tokens"])

    @cached_property
    def world_text(self) -> str:
        """All surface text the WORLD produced (event facts, referent names, viewpoint renderings,
        routines, profile text). The planted-phrase habit (positive control) is not world wording: it is
        left out of the planted agent's habits in the exact form the engine appended it."""
        pl = self.planted or {}
        parts = []
        for e in self.events.values():
            parts.append(e.get("narrative") or "")
            parts += [r.get("name", "") for r in e.get("referents") or []]
        for r in self.of("viewpoint"):          # observer-specific renderings are world-provided wording too
            parts += [f.get("perceived") or "" for f in r.get("facts", [])]
        for aid, a in self.agents.items():
            parts += [a.get("background", "")] + [r["activity"] for r in a.get("routine", [])]
            parts += [h for h in a.get("habits", []) if not (aid == pl.get("agent") and _is_planted(h, pl))]
        return " ".join(parts).lower()

    def conversation_context(self, u: dict, window: int = 1) -> str:
        c = self.conversations.get(u.get("conversation_id") or "")
        if not c:
            return u.get("context", "") + "\n" + u["text"]
        tr = c.get("transcript", [])
        i = u["idx"]
        lo, hi = max(0, i - window), min(len(tr), i + window + 1)
        return "\n".join(f"{s}: {t}" for s, t in tr[lo:hi])


def _is_planted(habit: str, pl: dict) -> bool:
    """The planted agent's habit entry that carries the planted phrase (exact applied string, or, for
    runs whose manifest predates the normalisation, any habit quoting the planted phrase)."""
    h = " ".join(str(habit).split())
    if pl.get("habit") and h == pl["habit"]:
        return True
    ph = pl.get("phrase")
    return bool(ph) and any(q.strip().lower() == ph for q in _QUOTED.findall(h))
