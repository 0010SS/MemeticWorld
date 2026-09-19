"""Hidden regimes, mapping and the change schedule (ontology v3 §1.4). WORLD layer, SIMULATOR-ONLY.

Nothing here is ever shown to agents: regime letters, mapping ids and causes are hidden state (the
`regime_active` trace record is stripped by the API). Only cue *texts* (world text) reach agents,
through cue beats at the Stockroom.

- mapping (`regimes.mapping: auto`): odd world_seed -> M1, even -> M2 (or an explicit "M1"/"M2");
- schedule (`regimes.schedule: [{day, regime}]`): the active regime of a day is the last entry with
  entry.day <= day; the first entry must be day 1;
- cues (`regimes.cues: [{day, time, arena, text_key}]`): deterministic world facts (no randomness).
"""
from __future__ import annotations

from backend.simulation.content import CUES
from backend.simulation.rngs import world_seed
from backend.simulation.world import ARENAS, _minutes

REGIMES = ("A", "B")
K3_REGIMES = ("B",)                 # regimes in which the K3 class (the B-cause) occurs
MAPPINGS = ("M1", "M2")
CUE_LOCATION = "Research Lab"
DEFAULT_SCHEDULE = [{"day": 1, "regime": "A"}]


def _rc(cfg: dict) -> dict:
    return cfg.get("regimes") or {}


def mapping(cfg: dict) -> str:
    m = _rc(cfg).get("mapping", "auto") or "auto"
    if m == "auto":
        return "M1" if world_seed(cfg) % 2 == 1 else "M2"
    if m not in MAPPINGS:
        raise ValueError(f"regimes.mapping must be auto or one of {MAPPINGS}, got {m!r}")
    return m


def schedule(cfg: dict) -> list[dict]:
    """The validated schedule, sorted by day."""
    sch = sorted((_rc(cfg).get("schedule") or DEFAULT_SCHEDULE), key=lambda e: int(e["day"]))
    if int(sch[0]["day"]) != 1:
        raise ValueError(f"regimes.schedule must start on day 1, got {sch}")
    days = [int(e["day"]) for e in sch]
    if len(set(days)) != len(days):
        raise ValueError(f"regimes.schedule has two entries for one day: {sch}")
    for e in sch:
        if e["regime"] not in REGIMES:
            raise ValueError(f"regimes.schedule regime must be one of {REGIMES}, got {e['regime']!r}")
    if _rc(cfg).get("split"):
        raise NotImplementedError("regimes.split (crew-specific material, RQ3) is not built in v3 study 1")
    return [{"day": int(e["day"]), "regime": e["regime"]} for e in sch]


def active(day: int, cfg: dict) -> str:
    """The regime active on `day` (1-based)."""
    reg = "A"
    for e in schedule(cfg):
        if e["day"] <= day:
            reg = e["regime"]
    return reg


def changes(cfg: dict) -> list[dict]:
    """Days on which the active regime differs from the day before: [{day, from, to}]."""
    out, prev = [], None
    for e in schedule(cfg):
        if prev is not None and e["regime"] != prev:
            out.append({"day": e["day"], "from": prev, "to": e["regime"]})
        prev = e["regime"]
    return out


def has_k3(regime: str) -> bool:
    return regime in K3_REGIMES


def causes(map_id: str, pack) -> dict:
    """{regime: K1's cause} under a mapping of a content pack."""
    return dict(pack.MAPPINGS[map_id])


def tick_of(day: int, time: str, clock) -> int:
    """Global tick of HH:MM on `day` (clock: world.Clock)."""
    k = (_minutes(time) - int(clock.day_start.total_seconds() // 60)) // clock.tick_minutes
    if not 0 <= k < clock.ticks_per_day:
        raise ValueError(f"time {time} is outside the day ({clock.ticks_per_day} ticks)")
    return (int(day) - 1) * clock.ticks_per_day + int(k)


def cues(cfg: dict, clock) -> list[dict]:
    """Cue records (§1.4, §1.8 kind `cue`) for days inside the clock, sorted by tick."""
    out = []
    for i, c in enumerate(_rc(cfg).get("cues") or []):
        key, arena = c["text_key"], c.get("arena", "Stockroom")
        if key not in CUES:
            raise ValueError(f"regimes.cues text_key must be one of {sorted(CUES)}, got {key!r}")
        if arena not in ARENAS[CUE_LOCATION]:
            raise ValueError(f"regimes.cues arena {arena!r} is not an arena of {CUE_LOCATION}")
        day = int(c["day"])
        if not 1 <= day <= clock.days:
            continue
        out.append({"id": f"cue{day:02d}.{i}", "kind": "cue", "generator": "workshop_v1", "day": day,
                    "tick": tick_of(day, str(c["time"]), clock), "location": CUE_LOCATION, "arena": arena,
                    "text_key": key, "text": CUES[key]})
    return sorted(out, key=lambda r: (r["tick"], r["id"]))


class Regimes:
    """Convenience wrapper: Regimes(cfg).active(day), .mapping, .causes(pack)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.mapping = mapping(cfg)
        self.schedule = schedule(cfg)

    def active(self, day: int) -> str:
        return active(day, self.cfg)

    def changes(self) -> list[dict]:
        return changes(self.cfg)

    def causes(self, pack) -> dict:
        return causes(self.mapping, pack)

    def trace(self, day: int) -> dict:
        """Fields of the hidden `regime_active` trace record (stripped by the API)."""
        return {"day": day, "regime": self.active(day), "mapping": self.mapping,
                "changed": any(c["day"] == day for c in self.changes())}
