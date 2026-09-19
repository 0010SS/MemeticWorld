"""Workshop content packs (ontology v3 §1.2) and the world text shared by every pack. SIMULATOR-ONLY.

A pack module defines NAME, CAUSES, STABLE, FIX, MAPPINGS, ACTIONS, ACTION_PAST, SURFACES, ODDITIES,
SUCCESS and FAIL. Every string agents may see is world text (the observer flags it; never counted as
agent coinage). Pack names, class ids, causes and mapping ids never appear in any text.
"""
from __future__ import annotations

import importlib

PACKS = ("laser_alpha", "laser_beta")
ACTION_IDS_FIXED = ("rerun", "slow", "stop")      # present in every pack; plus the three fixes

# campus projects: (job line, short form for the tally)
PROJECTS = [
    ("24 door plates for the robotics team's open house", "the robotics team's door plates"),
    ("tube racks for the bio lab", "the bio lab's tube racks"),
    ("stencils for the a cappella group's posters", "the a cappella stencils"),
    ("coasters for the dining hall's trivia night", "the trivia night coasters"),
    ("a wall sign for the chess club", "the chess club's wall sign"),
    ("gear models for the physics outreach day", "the outreach gear models"),
    ("puzzle boxes for the game design class", "the puzzle boxes"),
    ("phone stands for the orientation fair", "the phone stands"),
    ("ornaments for the dorm's winter party", "the dorm ornaments"),
    ("a bridge kit for the civil engineering club", "the bridge kit"),
    ("keychains for the rowing team's fundraiser", "the rowing keychains"),
    ("display risers for the art department's show", "the display risers"),
    ("a trophy base for the debate tournament", "the debate trophy base"),
    ("cable organizers for the computer lab", "the cable organizers"),
    ("bookends for the writing center", "the writing center's bookends"),
    ("a campus map for the visitor center", "the campus map"),
]

JOB_START = "{P} started a job on the co-op's laser cutter: {project}."
PANEL = "The laser's panel showed {code}."
STOPPED = "{P} stopped and left the job for later."
OUTCOME = "{P} {did}; {result}."
CLEAN = "{P}'s job came out fine."

CUES = {
    "new_supplier": "A pallet of acrylic sheets from a different supplier was delivered to the stockroom; the "
                    "co-op starts cutting from it tomorrow.",
    "usual_supplier": "A pallet of acrylic sheets from the co-op's usual supplier was delivered to the "
                      "stockroom; the co-op cuts from it again starting today.",
}

TALLY_ALL = "End-of-day tally at the laser: {n} of {m} jobs delivered."
TALLY_SOME = "End-of-day tally at the laser: {n} of {m} jobs delivered; {missing} {verb} not."

REQUIRED = ("NAME", "CAUSES", "STABLE", "FIX", "MAPPINGS", "ACTIONS", "ACTION_PAST", "SURFACES", "ODDITIES",
            "SUCCESS", "FAIL")


def load(name: str):
    """The content-pack module `name` (laser_alpha | laser_beta); raises ValueError otherwise."""
    if name not in PACKS:
        raise ValueError(f"workshop.content must be one of {PACKS}, got {name!r}")
    mod = importlib.import_module(f"backend.simulation.content.{name}")
    missing = [k for k in REQUIRED if not hasattr(mod, k)]
    if missing:
        raise ValueError(f"content pack {name} lacks {missing}")
    return mod


def surfaces(pack, cls: str, mapping: str) -> list[str]:
    """The 8 world surfaces of class `cls` (K3's depend on the mapping)."""
    s = pack.SURFACES[cls]
    return s[mapping] if isinstance(s, dict) else s


def fail_text(pack, cls: str, mapping: str) -> str:
    f = pack.FAIL[cls]
    return f[mapping] if isinstance(f, dict) else f


def join_names(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"
