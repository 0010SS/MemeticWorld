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

PERSONAL_LABELS = {
    "goal": "Personal goal",
    "values": "Values",
    "strengths": "Strengths",
    "blind_spots": "Blind spots",
    "stress_response": "Response to stress",
    "coping_strategy": "Coping strategy",
    "social_energy": "Social energy",
    "trust_style": "How trust develops",
    "conflict_style": "Approach to conflict",
    "humor_style": "Sense of humor",
    "pet_peeves": "Pet peeves",
    "small_joys": "Small joys",
}
PERSONAL_LIST_FIELDS = {"values", "strengths", "blind_spots", "pet_peeves", "small_joys"}


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
    personal: dict = field(default_factory=dict)
    friend_groups: list[dict] = field(default_factory=list)
    # v3 roster (ontology v3 §3): the co-op role (am_crew | pm_crew | stores; None = not a member, or a reserve
    # before arrival: the roster sets it on arrival) and whether the persona is a reserve newcomer.
    role: str | None = None
    reserve: bool = False

    @property
    def coop_role(self) -> str | None:
        return self.role

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
        for key, label in PERSONAL_LABELS.items():
            value = self.personal.get(key)
            if value:
                value = ", ".join(value) if isinstance(value, list) else value
                parts.append(f"{label}: {value.rstrip('.')}.")
        for group in self.friend_groups:
            parts.append(f"{self.first_name}'s {group['name']}: {', '.join(group['members'])}.")
        return " ".join(parts)

    def ga_lifestyle(self) -> str:
        return f"{self.first_name} " + "; ".join(self.habits) + "."

    def ga_daily_plan_req(self) -> str:
        return ", ".join(f"{r.activity} at the {r.location} around {r.time}" for r in self.routine[:6])

    def to_public_dict(self) -> dict:
        d = asdict(self)
        d["relationships"] = {k: asdict(v) for k, v in self.relationships.items()}
        return d


def apply_planted(profiles: dict[str, AgentProfile], cfg: dict) -> dict | None:
    """Planted-convention positive control (ontology v2 §5): `controls.planted_phrase` gives one agent a
    verbal habit, appended to its habits (GA `lifestyle`). No-op when null. Must run before the agents
    are built. Returns what was applied (for the manifest), or None.

    The habit is written as a full sentence ("Maya has a habit of ..."); a leading first name and the final
    period are dropped so it reads naturally inside `ga_lifestyle()` ("Maya grabs coffee ...; has a habit of ...")."""
    pp = (cfg.get("controls") or {}).get("planted_phrase")
    if not pp:
        return None
    aid, habit = pp.get("agent"), " ".join(str(pp.get("habit") or "").split())
    if aid not in profiles:
        raise ValueError(f"controls.planted_phrase.agent {aid!r} is not in the loaded population")
    if not habit:
        raise ValueError("controls.planted_phrase.habit is empty")
    prof = profiles[aid]
    habit = habit.rstrip(".").strip()
    if habit.startswith(prof.first_name + " "):
        habit = habit[len(prof.first_name) + 1:]
    prof.habits = list(prof.habits) + [habit]
    return {"agent": aid, "habit": habit}


def _personal_properties(agent: dict) -> dict:
    personal = agent.get("personal", {})
    if not isinstance(personal, dict):
        raise ValueError(f"Agent {agent['id']!r}: personal must be a mapping")
    for key, value in personal.items():
        if key not in PERSONAL_LABELS:
            raise ValueError(f"Agent {agent['id']!r}: unknown personal property {key!r}")
        valid = (isinstance(value, list) and all(isinstance(v, str) and v.strip() for v in value)
                 if key in PERSONAL_LIST_FIELDS else isinstance(value, str) and bool(value.strip()))
        if not valid:
            expected = "a list of nonempty strings" if key in PERSONAL_LIST_FIELDS else "a nonempty string"
            raise ValueError(f"Agent {agent['id']!r}: personal.{key} must be {expected}")
    return personal


def _friend_group_definitions(data: dict) -> dict:
    """Validate against the complete file before limiting the run's population."""
    all_ids = [a["id"] for a in data["agents"]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Population contains duplicate agent IDs")
    definitions = data.get("friend_groups", {})
    if not isinstance(definitions, dict):
        raise ValueError("friend_groups must be a mapping")
    for gid, group in definitions.items():
        if not isinstance(gid, str) or not gid.strip() or not isinstance(group, dict):
            raise ValueError("Each friend group needs a string ID and a mapping")
        if not isinstance(group.get("name"), str) or not group["name"].strip():
            raise ValueError(f"Friend group {gid!r} needs a nonempty display name")
        members = group.get("members")
        if (not isinstance(members, list) or len(members) < 2
                or not all(isinstance(m, str) for m in members)):
            raise ValueError(f"Friend group {gid!r} needs at least two agent IDs")
        if len(set(members)) != len(members):
            raise ValueError(f"Friend group {gid!r} has duplicate members")
        unknown = set(members) - set(all_ids)
        if unknown:
            raise ValueError(f"Friend group {gid!r} has unknown members: {sorted(unknown)}")
    for agent in data["agents"]:
        declared = agent.get("friend_groups", [])
        if not isinstance(declared, list) or not all(isinstance(gid, str) for gid in declared):
            raise ValueError(f"Agent {agent['id']!r}: friend_groups must be a list of group IDs")
        if len(set(declared)) != len(declared):
            raise ValueError(f"Agent {agent['id']!r}: duplicate friend group IDs")
        unknown = set(declared) - set(definitions)
        if unknown:
            raise ValueError(f"Agent {agent['id']!r}: unknown friend groups: {sorted(unknown)}")
        expected = {gid for gid, group in definitions.items() if agent["id"] in group["members"]}
        if set(declared) != expected:
            raise ValueError(f"Agent {agent['id']!r}: friend_groups do not match group memberships")
    return definitions


def load_population(path: str | Path, n: int | None = None, include_reserves: bool = False):
    """-> (profiles, groups). `n` takes the first n of `agents:` (the founders). `include_reserves` also loads
    the file's `reserves:` personas (v3 §3.1: the full persona universe; `reserve=True`, no role until they
    arrive). Without reserves the result is exactly the v2 one (plus `role` from `coop_role`)."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with p.open() as source:
        data = yaml.safe_load(source)
    friend_definitions = _friend_group_definitions(data)
    raw = [dict(a, _reserve=False) for a in data["agents"][: n or None]]
    if include_reserves:
        raw += [dict(a, _reserve=True) for a in (data.get("reserves") or [])]
    ids = {a["id"] for a in raw}
    from backend.simulation.world import PUBLIC_PLACES, WORLD_GRAPH
    profiles = {}
    for a in raw:
        routine = [RoutineEntry(**r) for r in a["routine"]]
        known = sorted({r.location for r in routine} | {"Quad", "Dining Hall", "Cafe", "Library"} | set(PUBLIC_PLACES))
        known = [k for k in known if k in WORLD_GRAPH]
        prof = AgentProfile(id=a["id"], name=a["name"], demographics=a["demographics"],
                            background=a["background"], personality=a["personality"],
                            interests=a["interests"], habits=a["habits"], routine=routine,
                            home=a["home"], sprite=a.get("sprite", "Abigail_Chen"),
                            known_locations=known, personal=_personal_properties(a),
                            role=None if a["_reserve"] else a.get("coop_role", a.get("role")),
                            reserve=bool(a["_reserve"]))
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
    for gid, group in friend_definitions.items():
        members = [m for m in group["members"] if m in ids]
        if len(members) < 2:
            continue
        if gid in groups and set(groups[gid]) != set(members):
            raise ValueError(f"Friend group {gid!r} conflicts with the ordinary group of the same ID")
        groups[gid] = members
        for aid in members:
            profiles[aid].friend_groups.append({"id": gid, "name": group["name"],
                                                "members": [profiles[m].name for m in members if m != aid]})
    return profiles, groups
