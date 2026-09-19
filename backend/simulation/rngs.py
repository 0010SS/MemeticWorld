"""Named random substreams.

`seed_rng(*parts)` seeds a numpy Generator from the crc32 of every part, the same scheme as
`engine._seed_rng`. A stream is identified by its parts (e.g. world_seed + "timing"), so each
purpose gets its own sequence and switching one mechanism on never shifts another's draws.
"""
from __future__ import annotations

import zlib

import numpy as np


def seed_rng(*parts) -> np.random.Generator:
    return np.random.default_rng([zlib.crc32(str(p).encode()) for p in parts])


def world_seed(cfg: dict) -> int:
    """The seed of the world (the world script AND the routine plans it resolves beat places from:
    `seed_rng(world_seed, "plan", aid, day)` in both world_script and the engine): `world_seed`, or
    `seed` when it is null."""
    ws = cfg.get("world_seed")
    return int(cfg["seed"] if ws is None else ws)
