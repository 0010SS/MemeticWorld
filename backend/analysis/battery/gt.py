"""Ground truth for the LB1 battery (ONTOLOGY_V3 §1.3, §1.4, §5.3). OBSERVER ONLY.

gt(item, regime, mapping) = argmax_a p(a | cause_regime(class)). No LLM ever judges correctness.

The contract requires ONE function shared by the engine's outcomes and this scorer: `gt` delegates to
`backend.simulation.workshop.gt` and `outcome_p` to `workshop.OutcomeModel.p` (the observer may import the world,
never the reverse). `local_gt` / `_local_p` restate the §1.3 table and are used only if the world module is absent;
tests/test_v3_isolation.py::test_gt_parity_with_world checks they agree.
"""
from __future__ import annotations

# the action menu (§1.2): identical wording in the world, the decision prompt and the battery
MENU = {
    "laser_alpha": [
        ("rerun", "re-run the sheet as it is"),
        ("lens", "clean the focus lens, then re-run"),
        ("dry", "dry the sheets on the heated rack for 15 minutes, then re-run"),
        ("belt", "tighten the drive belt, then re-run"),
        ("slow", "slow the cutting speed down, then re-run"),
        ("stop", "stop and leave the job for later"),
    ],
}

# the fix that matches each cause (pack-level; beta = AIR/WARP, §1.2)
CAUSE_FIX = {"LENS": "lens", "DAMP": "dry", "BELT": "belt", "AIR": "air", "WARP": "pins"}

# mapping -> (A-cause, B-cause) (§1.4); beta's pairing is decided at calibration (stub)
MAPPINGS = {
    "laser_alpha": {"M1": ("LENS", "DAMP"), "M2": ("DAMP", "LENS")},
}

SCORED_TYPES = ("K1c", "K1a", "K3", "K2", "K0")


def menu(pack: str = "laser_alpha") -> list[tuple[str, str]]:
    if pack not in MENU:
        raise NotImplementedError(f"battery menu for content pack {pack!r} (laser_beta items are stubbed)")
    return list(MENU[pack])


def mapping_for(cfg: dict) -> str:
    """regimes.mapping: auto -> world-seed parity (odd = M1, even = M2)."""
    m = str(((cfg or {}).get("regimes") or {}).get("mapping") or "auto")
    if m in ("M1", "M2"):
        return m
    ws = cfg.get("world_seed")
    seed = int(cfg.get("seed", 0) if ws is None else ws)
    return "M1" if seed % 2 == 1 else "M2"


def regime_at(cfg: dict, day: int) -> str:
    """The regime active on `day` from regimes.schedule (last entry with entry.day <= day)."""
    sched = ((cfg or {}).get("regimes") or {}).get("schedule") or [{"day": 1, "regime": "A"}]
    reg = "A"
    for e in sorted(sched, key=lambda e: int(e["day"])):
        if int(e["day"]) <= int(day):
            reg = str(e["regime"])
    return reg


def cause_of(cls: str, regime: str, mapping: str, pack: str = "laser_alpha") -> str | None:
    a, b = MAPPINGS[pack][mapping]
    if cls in ("K1", "K1c", "K1a", "CUE"):
        return a if regime == "A" else b
    if cls == "K2":
        return "BELT"
    if cls == "K3":
        return b            # K3 occurs only under B; its cause is always the B-cause
    return None             # K0: no fault


def _local_p(action: str, cause: str | None, success: dict | None = None) -> float:
    s = {"match": 0.85, "other_fix": 0.10, "rerun": 0.10, "slow": 0.35, "slow_belt": 0.20}
    s.update(success or {})
    if cause is None:
        return 1.0 if action != "stop" else 0.0
    if action == "stop":
        return 0.0
    if action == "rerun":
        return float(s["rerun"])
    if action == "slow":
        return float(s["slow_belt"] if cause == "BELT" else s["slow"])
    return float(s["match"] if CAUSE_FIX.get(cause) == action else s["other_fix"])


def outcome_p(action: str, cause: str | None, success: dict | None = None) -> float:
    try:   # the world's OutcomeModel is the single source of truth when present
        from backend.simulation.workshop import OutcomeModel  # observer may import the world, never the reverse
    except ImportError:
        return _local_p(action, cause, success)
    return float(OutcomeModel({"workshop": {"success": success}} if success else {}).p(action, cause))


def base_class(cls: str) -> str:
    """Battery item types map onto world classes: K1c / K1a / CUE are K1 symptoms."""
    return "K1" if cls in ("K1", "K1c", "K1a", "CUE") else cls


def local_gt(cls: str, regime: str, mapping: str, pack: str = "laser_alpha", success: dict | None = None) -> str:
    """The contract table's argmax (fallback when the world module is absent; parity-tested against it)."""
    cause = cause_of(base_class(cls), regime, mapping, pack)
    if cause is None:
        return "rerun"
    acts = [a for a, _ in menu(pack) if a != "stop"]
    return max(acts, key=lambda a: (_local_p(a, cause, success), -acts.index(a)))


def gt(item, regime: str, mapping: str, pack: str = "laser_alpha", success: dict | None = None) -> str:
    """The correct first action for an item (dict with "type", or a class string) under regime/mapping.
    Delegates to the world's single ground-truth function (`workshop.gt`, §1.3) when it is importable.
    K0 items (clean cut) -> "rerun": no fix is needed; choosing a fix there is overextension."""
    cls = base_class(item["type"] if isinstance(item, dict) else str(item))
    try:
        from backend.simulation.workshop import gt as world_gt   # observer may import the world, never the reverse
    except ImportError:
        return local_gt(cls, regime, mapping, pack, success)
    return world_gt(cls, regime, mapping, pack, {"workshop": {"success": success}} if success else None)


def score(action: str | None, item: dict, regime: str, mapping: str, pack: str = "laser_alpha") -> dict:
    """Scoring of one choice: correct (vs GT at the checkpoint's regime) and the three-way post-change
    category old/new/other (+hedge flag). Unscored item types (CUE) return correct=None."""
    cls = item["type"]
    out = {"gt": gt(item, regime, mapping, pack), "correct": None, "category": None, "hedge": action == "slow"}
    if action is None or cls not in SCORED_TYPES:
        return out
    out["correct"] = action == out["gt"]
    if cls in ("K1c", "K1a", "K3"):
        old, new = gt({"type": "K1"}, "A", mapping, pack), gt({"type": "K1"}, "B", mapping, pack)
        out["category"] = "old" if action == old else "new" if action == new else "other"
    return out
