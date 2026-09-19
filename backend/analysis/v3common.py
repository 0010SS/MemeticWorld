"""Shared helpers for the v3 observer metrics (OBSERVER ONLY; ontology v3 §5).

Nothing here is imported by simulation code. Ground truth is computed from the contract tables
(§1.2-1.4) so scoring never depends on an LLM. Every trace field is read with .get() and a neutral
default, because the producing parts (world, records, roster, work, battery) are built in parallel.

Assumed input formats (documented for the integrator; see also the O3-metrics report):
- trace.jsonl (v2 tracer): records carry `type` and `tick`.
- probes/C{d}/responses.jsonl (battery runner, §5.2-5.3), one line per scored response:
  {"agent", "item" (id), "type": K1c|K1a|K3|K2|K0|CUE, "form": P|P-sit|A|N|P-abl, "action": menu id,
   "cue": X|W|Y|NONCE|NONE (A and N), "framing", "fit": fits|doesn't fit|not sure (A), "hit": bool}
  Synonyms are normalised (see `norm_response`). Pre-branch checkpoints are looked up in the parent
  run (manifest `branch.parent_run`) when the branch has no copy (§5.2 item 8).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ACTIONS = ("rerun", "lens", "dry", "belt", "slow", "stop", "nozzle", "pins")
FIXES_BY_PACK = {
    "laser_alpha": {"LENS": "lens", "DAMP": "dry", "BELT": "belt"},
    "laser_beta": {"AIR": "nozzle", "WARP": "pins", "BELT": "belt"},
}
# (A cause, B cause) per pack and mapping (§1.4; beta mirrors alpha's order: first cause = M1's A).
CAUSES = {
    "laser_alpha": {"M1": ("LENS", "DAMP"), "M2": ("DAMP", "LENS")},
    "laser_beta": {"M1": ("AIR", "WARP"), "M2": ("WARP", "AIR")},
}
HEDGE = "slow"
K1_TYPES = ("K1c", "K1a", "K1")


def load_jsonl(p) -> list[dict]:
    p = Path(p)
    if not p.exists():
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def load_json(p, default=None):
    p = Path(p)
    if not p.exists():
        return default
    try:
        return json.load(open(p))
    except (json.JSONDecodeError, OSError):
        return default


def pack_of(cfg: dict) -> str:
    return ((cfg.get("workshop") or {}).get("content")) or "laser_alpha"


def mapping_of(cfg: dict) -> str:
    m = (cfg.get("regimes") or {}).get("mapping") or "auto"
    if m in ("M1", "M2"):
        return m
    ws = cfg.get("world_seed")
    ws = cfg.get("seed", 0) if ws is None else ws
    return "M1" if int(ws) % 2 == 1 else "M2"


def fix_of(cause: str | None, pack: str = "laser_alpha") -> str | None:
    if cause is None:
        return None
    return FIXES_BY_PACK.get(pack, FIXES_BY_PACK["laser_alpha"]).get(str(cause).upper())


def regime_fixes(cfg: dict) -> dict:
    """{"A": A-fix, "B": B-fix} for K1 under this run's pack and mapping."""
    pack = pack_of(cfg)
    a, b = CAUSES.get(pack, CAUSES["laser_alpha"])[mapping_of(cfg)]
    return {"A": fix_of(a, pack), "B": fix_of(b, pack)}


def regime_on_day(cfg: dict, day: int) -> str:
    sched = (cfg.get("regimes") or {}).get("schedule") or [{"day": 1, "regime": "A"}]
    reg = "A"
    for s in sorted(sched, key=lambda s: int(s.get("day", 1))):
        if int(s.get("day", 1)) <= day:
            reg = s.get("regime", reg)
    return reg


def shift_day(cfg: dict) -> int | None:
    """First day whose regime differs from day 1's (None for a no-shift schedule)."""
    sched = sorted((cfg.get("regimes") or {}).get("schedule") or [], key=lambda s: int(s.get("day", 1)))
    first = regime_on_day(cfg, 1)
    for s in sched:
        if s.get("regime") != first:
            return int(s["day"])
    return None


def gt(item_type: str, regime: str, cfg: dict) -> str | None:
    """GT(item, regime, mapping) = argmax_a p(a | cause_regime(class)) (§1.3). Delegates to the battery's
    gt.py (one scorer for the whole observer) when it exists; K0 (clean cut) -> "rerun"; CUE -> None."""
    t = str(item_type or "")
    if t in ("CUE", "") or t.startswith("CUE"):
        return None
    try:
        from backend.analysis.battery import gt as BG
        return BG.gt({"type": t}, regime, mapping_of(cfg), pack_of(cfg))
    except Exception:  # noqa: BLE001 - battery absent or pack stubbed there: contract table below
        pass
    fx = regime_fixes(cfg)
    if t.startswith("K1"):
        return fx.get(regime)
    if t.startswith("K2"):
        return fix_of("BELT", pack_of(cfg))
    if t.startswith("K3"):
        return fx["B"]
    return "rerun" if t.startswith("K0") else None


def ckpt_day(ckpt) -> int:
    return int(str(ckpt).lstrip("Cc").split("_")[0])


def ticks_per_day(rd_or_manifest) -> int:
    m = getattr(rd_or_manifest, "manifest", rd_or_manifest) or {}
    return int(m.get("ticks_per_day") or 60)


def day_of_tick(tick: int, tpd: int = 60) -> int:
    return int(tick) // tpd + 1


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def share(xs, pred) -> float | None:
    xs = list(xs)
    return round(sum(1 for x in xs if pred(x)) / len(xs), 4) if xs else None


def sub(a, b):
    return None if a is None or b is None else round(a - b, 4)


# ---------------------------------------------------------------- recommended action (keyword map, §5.8)
_KEYWORDS = {
    "lens": [r"\blens\b", r"\bfocus\b", r"\boptic"],
    "dry": [r"\bdr(y|ied|ying)\b", r"\bheated rack\b", r"\brack\b", r"\bdamp\b", r"\bmoist", r"\bwet\b",
            r"\bhumid"],
    "belt": [r"\bbelt\b", r"\btighten"],
    "slow": [r"\bslow(er|ed|ing)?\b", r"\bspeed\b", r"\bpower\b"],
    "rerun": [r"\bre-?run (it|the sheet)? ?as (it )?is\b", r"\bjust re-?run\b"],
    "stop": [r"\bstop(ped)? and leave\b", r"\bleave the job\b", r"\bdefer"],
    "nozzle": [r"\bnozzle\b", r"\bair[- ]assist\b"],
    "pins": [r"\bflatten", r"\bhold-?down\b", r"\bpins\b", r"\bwarp"],
}
_KW = {a: [re.compile(p, re.I) for p in ps] for a, ps in _KEYWORDS.items()}


def recommended_actions(text: str | None) -> set:
    """Every menu action a text mentions (a front page may recommend several fixes for several symptoms)."""
    t = text or ""
    return {a for a, ps in _KW.items() if any(p.search(t) for p in ps)}


def recommended_action(text: str | None) -> tuple[str | None, bool]:
    """(single recommended action or None, ambiguous). Ambiguous texts go to the coder fallback (§5.10;
    stubbed), so they count as no recommendation here."""
    acts = recommended_actions(text)
    if len(acts) == 1:
        return next(iter(acts)), False
    return None, len(acts) > 1


# ---------------------------------------------------------------- probe responses
_FORM = {"p": "P", "memory": "P", "memory_only": "P", "p-sit": "P-sit", "psit": "P-sit", "situated": "P-sit",
         "p_sit": "P-sit", "a": "A", "apply": "A", "applicability": "A", "n": "N", "note": "N",
         "comprehension": "N", "p-abl": "P-abl", "ablated": "P-abl", "p_abl": "P-abl", "source_ablated": "P-abl"}
_FIT = {"fits": 1.0, "fit": 1.0, "yes": 1.0, "not sure": 0.5, "unsure": 0.5, "doesn't fit": 0.0,
        "does not fit": 0.0, "no": 0.0, "doesnt fit": 0.0}


def norm_response(r: dict) -> list[dict]:
    """One battery response -> normalised rows (an A call with a per-situation list expands to rows)."""
    form = _FORM.get(str(r.get("form") or r.get("mode") or "P").lower(), r.get("form") or "P")
    base = {"agent": r.get("agent") or r.get("agent_id"), "form": form,
            "item": r.get("item") or r.get("item_id"), "type": r.get("type") or r.get("item_type"),
            "cue": r.get("cue"), "framing": r.get("framing"), "order": r.get("order"),
            "hit": r.get("hit", r.get("retrieval_hit")), "heard": r.get("heard"),
            "valid": r.get("valid", True) is not False}
    act = r.get("action") or r.get("choice_action")
    if act is None and r.get("choice") in ACTIONS:
        act = r.get("choice")
    base["action"] = act
    fits = r.get("fits") or r.get("situations")
    if form == "A" and isinstance(fits, (list, dict)):
        rows = fits.items() if isinstance(fits, dict) else [(x.get("item"), x) for x in fits]
        out = []
        for item, v in rows:
            v = v if isinstance(v, dict) else {"fit": v}
            out.append(dict(base, item=item or v.get("item"), type=v.get("type") or base["type"],
                            fit=_FIT.get(str(v.get("fit")).lower().strip())))
        return out
    if r.get("fit") is not None:
        f = r.get("fit")
        base["fit"] = f if isinstance(f, (int, float)) else _FIT.get(str(f).lower().strip())
    return [base]


def parent_run(run_dir) -> Path | None:
    m = load_json(Path(run_dir) / "manifest.json", {}) or {}
    p = (m.get("branch") or {}).get("parent_run")
    if not p:
        return None
    p = Path(p)
    if not p.is_absolute() and not p.exists():
        # a run-relative name: runs/<design>/<node>/s<seed> -> sibling node dir
        cand = Path(run_dir).parent.parent / p / Path(run_dir).name
        p = cand if cand.exists() else Path(run_dir).parent.parent / p
    return p if p.exists() else None


def probe_responses(run_dir, ckpt: str, _depth: int = 0) -> list[dict]:
    """Normalised responses at checkpoint `ckpt` of this run, or of its parent chain (shared prefix)."""
    ckpt = ckpt if str(ckpt).startswith("C") else f"C{ckpt}"
    p = Path(run_dir) / "probes" / ckpt / "responses.jsonl"
    if p.exists():
        return [row for r in load_jsonl(p) for row in norm_response(r)]
    par = parent_run(run_dir)
    if par is not None and _depth < 4:
        return probe_responses(par, ckpt, _depth + 1)
    return []


def c0_responses(run_dir) -> list[dict]:
    """C0 (fresh prior, §5.6): the run's own probes/C0, else the design dir's probes/C0."""
    for d in (Path(run_dir), Path(run_dir).parent.parent, Path(run_dir).parent):
        p = d / "probes" / "C0" / "responses.jsonl"
        if p.exists():
            return [row for r in load_jsonl(p) for row in norm_response(r)]
    return []


def node_of(run_dir) -> str:
    m = load_json(Path(run_dir) / "manifest.json", {}) or {}
    return m.get("node") or (m.get("branch") or {}).get("node") or Path(run_dir).parent.name


def sibling(run_dir, node: str) -> Path | None:
    """runs/<design>/<node>/s<seed>/ -> the same seed's run dir of another node (§6.3)."""
    p = Path(run_dir).parent.parent / node / Path(run_dir).name
    return p if (p / "manifest.json").exists() else None
