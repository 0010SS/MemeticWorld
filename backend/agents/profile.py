"""Agent profile ontology (close to Generative Agents' persona description).

Deliberately contains NO cultural state: no memes, meanings, event labels, etc.
`tests/test_invariants.py` enforces that.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from backend.ga_compat import REPO_ROOT

FORBIDDEN_FIELDS = {"known_memes", "meme_dictionary", "meme_meanings", "meme_preference",
                    "meme_fitness", "current_meme", "current_culture",
                    "latent_event_understanding", "latent_event_id", "latent_type"}


@dataclass
class Relationship:
    relation_type: str = "stranger"
    familiarity: float = 0.05
    affinity: float = 0.5


@dataclass
class RoutineEntry:
    time: str
    location: str
    activity: str
    arena: str | None = None


@dataclass
class AgentProfile:
    id: str
    name: str
    demographics: dict
    background: str
    personality: dict
    interests: dict
    habits: list[str]
    routine: list[RoutineEntry]
    home: dict
    sprite: str = "Abigail_Chen"
    relationships: dict[str, Relationship] = field(default_factory=dict)
    known_locations: list[str] = field(default_factory=list)

    @property
    def first_name(self) -> str:
        return self.name.split()[0]

    def rel(self, other_id: str) -> Relationship:
        return self.relationships.get(other_id, Relationship())

    # --- GA "identity stable set" fields -------------------------------------
    def ga_innate(self) -> str:
        return ", ".join(self.personality.get("traits", []))

    def ga_learned(self) -> str:
        d = self.demographics
        it = self.interests
        parts = [f"{self.first_name} is a {d.get('age')}-year-old {d.get('year')} studying {d.get('major')}"
                 f" ({d.get('role')}). {self.background}"]
        if it.get("topics"):
            parts.append(f"{self.first_name} is interested in {', '.join(it['topics'])}.")
        if it.get("hobbies"):
            parts.append(f"Hobbies: {', '.join(it['hobbies'])}.")
        if it.get("clubs"):
            parts.append(f"Member of: {', '.join(it['clubs'])}.")
        style = self.personality.get("communication_style")
        if style:
            parts.append(f"Communication style: {style}.")
        return " ".join(parts)

    def ga_lifestyle(self) -> str:
        return f"{self.first_name} " + "; ".join(self.habits) + "."

    def ga_daily_plan_req(self) -> str:
        return ", ".join(f"{r.activity} at the {r.location} around {r.time}" for r in self.routine[:6])

    def to_public_dict(self) -> dict:
        d = asdict(self)
        d["relationships"] = {k: asdict(v) for k, v in self.relationships.items()}
        return d


def load_population(path: str | Path, n: int | None = None):
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    data = yaml.safe_load(open(p))
    raw = data["agents"][: n or None]
    ids = {a["id"] for a in raw}
    from backend.simulation.world import WORLD_GRAPH
    profiles = {}
    for a in raw:
        routine = [RoutineEntry(**r) for r in a["routine"]]
        known = sorted({r.location for r in routine} | {"Quad", "Dining Hall", "Cafe", "Library"})
        known = [k for k in known if k in WORLD_GRAPH]
        prof = AgentProfile(id=a["id"], name=a["name"], demographics=a["demographics"],
                            background=a["background"], personality=a["personality"],
                            interests=a["interests"], habits=a["habits"], routine=routine,
                            home=a["home"], sprite=a.get("sprite", "Abigail_Chen"),
                            known_locations=known)
        profiles[prof.id] = prof
    for r in data.get("relationships", []):
        if r["a"] in ids and r["b"] in ids:
            rel = Relationship(r["type"], float(r["familiarity"]), float(r["affinity"]))
            profiles[r["a"]].relationships[r["b"]] = rel
            profiles[r["b"]].relationships[r["a"]] = Relationship(**asdict(rel))
    for a in profiles.values():
        for b in profiles:
            if b != a.id and b not in a.relationships:
                a.relationships[b] = Relationship()
    groups = {g: [m for m in members if m in ids] for g, members in (data.get("groups") or {}).items()}
    groups = {g: m for g, m in groups.items() if len(m) >= 2}
    return profiles, groups
