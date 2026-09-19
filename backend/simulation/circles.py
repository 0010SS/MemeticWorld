"""Friend circles (ontology v2 §1.5): DISJOINT groups of >= 3 agents, SIMULATOR-ONLY.

Circles are where latent-event assignment concentrates recurrence (`latent_events.assignment`) and
where referent memory lives (§1.4). Agents in no circle form the free pool. Circles never reach an
agent: agents only know their own relationships.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from backend.ga_compat import REPO_ROOT

MIN_MEMBERS = 3


def validate_circles(circles: dict, known_ids) -> None:
    """Raise ValueError unless circles are disjoint, have >= 3 members each, and name known agents."""
    known, seen = set(known_ids), {}
    for cid, members in circles.items():
        if not isinstance(members, list) or not all(isinstance(m, str) for m in members):
            raise ValueError(f"circle {cid!r}: members must be a list of agent ids, got {members!r}")
        unknown = [m for m in members if m not in known]
        if unknown:
            raise ValueError(f"circle {cid!r}: unknown agent ids {unknown}")
        if len(set(members)) < MIN_MEMBERS:
            raise ValueError(f"circle {cid!r}: needs >= {MIN_MEMBERS} distinct members, got {members}")
        for m in members:
            if m in seen and seen[m] != cid:
                raise ValueError(f"circles must be disjoint: {m!r} is in {seen[m]!r} and {cid!r}")
            seen[m] = cid


def load_circles(population_path: str | Path, agent_ids) -> dict[str, list[str]]:
    """Read and validate the population file's `circles:` section.

    Membership is validated against every agent in the file (typos fail loudly). When the run uses
    only the first N agents (`population_size`), members outside the run are dropped, and a circle
    left with fewer than 3 members is dissolved into the free pool.
    """
    p = Path(population_path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    data = yaml.safe_load(open(p)) or {}
    raw = data.get("circles") or {}
    validate_circles(raw, [a["id"] for a in data.get("agents", [])])
    ids = set(agent_ids)
    out = {}
    for cid in sorted(raw):
        members = list(dict.fromkeys(m for m in raw[cid] if m in ids))
        if len(members) >= MIN_MEMBERS:
            out[cid] = members
    return out


def circle_of(circles: dict[str, list[str]]) -> dict[str, str]:
    """agent id -> circle id (agents in the free pool are absent)."""
    return {m: cid for cid, members in circles.items() for m in members}
