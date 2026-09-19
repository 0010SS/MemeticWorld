"""The co-op's laser cutter: jobs, symptom classes, the action menu and the outcome model (ontology v3
§1.2-1.8). WORLD layer, SIMULATOR-ONLY.

`generate(cfg, profiles, clock, roster=None)` pre-generates every job of the run from `world_seed`
(common random numbers). Streams are day-local and keyed per job:
`seed_rng(world_seed, "ws3", purpose, day, slot)`, and every job draws a FIXED number of values from
every stream (N_DRAWS), whatever the config does with them. So extending the horizon leaves earlier
days byte-identical, and switching `panel_code` or `causal` shifts nothing else.

The script is regime-independent: each job stores its class and cause under EACH regime
(`hidden.class`, `hidden.cause`); the schedule (regimes.active) only selects between them at runtime.
K3 is carved out of the fault-free (K0) mass, so K1/K2 slots are identical across regimes.

Beats are built at runtime from the record and the active regime (outcomes depend on decisions):
`start_beat`, `outcome_beat`, `cue_beat`, `tally_beat`. Nothing agent-visible ever contains a class,
cause, regime, mapping, job id, probability or uniform. `hidden.*` and the `job_truth`/`regime_active`
trace records are stripped by the API in demo mode.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from backend.simulation import content as C
from backend.simulation import regimes as R
from backend.simulation.rngs import seed_rng, world_seed
from backend.simulation.world import _minutes

GENERATOR = "workshop_v1"
CLASSES = ("K0", "K1", "K2", "K3")
LOCATION, ARENA, STOCKROOM = "Research Lab", "Makerspace", "Stockroom"
PURPOSES = ("fault", "class", "k3", "code", "surface", "project", "oddity", "menu", "outcome", "operator",
            "cause_scr")
N_ATTEMPT_DRAWS = 3            # menu orders and outcome uniforms drawn per job (>= max_attempts)
N_DRAWS = {"fault": 1, "class": 1, "k3": 1, "code": 1, "surface": len(CLASSES), "project": 1, "oddity": 2,
           "outcome": N_ATTEMPT_DRAWS, "operator": 1, "cause_scr": 1}     # + menu: N_ATTEMPT_DRAWS permutations
SALIENCE = {"job_start": 0.35, "symptom": 0.6, "code": 0.5, "oddity": 0.4, "outcome": 0.55, "cue": 0.5,
            "tally": 0.5}
DEFAULT_CREWS = {"am": ["maya", "dev", "hana"], "pm": ["ethan", "leo", "jordan"], "stores": ["priya", "sofia"]}
DEFAULTS = {
    "enabled": False, "content": "laser_alpha", "jobs_ticks": [6, 10, 14, 18, 24, 28, 32, 36], "p_fault": 0.75,
    "class_weights": {"K1": 0.70, "K2": 0.30}, "k3_from_k0": 0.6,
    "success": {"match": 0.85, "other_fix": 0.10, "rerun": 0.10, "slow": 0.35, "slow_belt": 0.20},
    "max_attempts": 2, "panel_code": {"enabled": False, "text": "F4", "prob": 0.5}, "oddity_prob": 0.3,
    "symptom_visibility": "arena", "encode": "per_job", "encode_reason": True, "causal": "real",
    "tally_time": "17:30",
}


def wcfg(cfg: dict) -> dict:
    """workshop.* with contract defaults filled in (nested dicts merged one level)."""
    w = dict(DEFAULTS)
    for k, v in (cfg.get("workshop") or {}).items():
        w[k] = {**DEFAULTS[k], **(v or {})} if isinstance(DEFAULTS.get(k), dict) and k != "class_weights" \
            else v
    if w["causal"] not in ("real", "scrambled"):
        raise ValueError(f"workshop.causal must be real or scrambled, got {w['causal']!r}")
    if not 1 <= int(w["max_attempts"]) <= N_ATTEMPT_DRAWS:
        raise ValueError(f"workshop.max_attempts must be in 1..{N_ATTEMPT_DRAWS}, got {w['max_attempts']}")
    if w["symptom_visibility"] not in ("arena", "all"):
        raise ValueError(f"workshop.symptom_visibility must be arena or all, got {w['symptom_visibility']!r}")
    return w


def pack_of(cfg: dict):
    return C.load(wcfg(cfg)["content"])


def _pick(u: float, n: int) -> int:
    return min(int(u * n), n - 1)


# ----------------------------------------------------------------------------- records
@dataclass
class JobInstance:
    """One pre-generated job (§1.8). `hidden` is simulator-only (stripped by the API)."""
    id: str
    day: int
    slot: int
    shift: str
    start_tick: int
    operator: str | None
    project: str
    project_short: str
    menu_order: list
    surface: dict
    code: bool
    oddity: str | None           # world text of an irrelevant oddity (shown only while the job is K0)
    hidden: dict
    kind: str = "job"
    generator: str = GENERATOR
    location: str = LOCATION
    arena: str = ARENA

    @property
    def tick(self) -> int:
        return self.start_tick

    def to_dict(self) -> dict:
        return asdict(self)

    def public(self) -> dict:
        """The record without hidden fields (demo-mode API)."""
        return {k: v for k, v in asdict(self).items() if k != "hidden"}

    @classmethod
    def from_dict(cls, d: dict) -> "JobInstance":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def cls_at(self, regime: str) -> str:
        return self.hidden["class"][regime]

    def cause_at(self, regime: str) -> str | None:
        return self.hidden["cause"][regime]


@dataclass
class WorldRecord:
    """A non-job world-script record (kinds `cue`, `tally`); deterministic, no randomness."""
    id: str
    kind: str
    day: int
    tick: int
    location: str = LOCATION
    arena: str = ARENA
    text_key: str | None = None
    text: str | None = None
    audience: list = field(default_factory=list)
    generator: str = GENERATOR

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WorldRecord":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def record_from_dict(d: dict):
    return JobInstance.from_dict(d) if d.get("kind") == "job" else WorldRecord.from_dict(d)


def to_jsonl(records) -> str:
    """One JSON line per record (sorted keys): appended to world_script.jsonl by the integrator."""
    return "".join(json.dumps(r.to_dict(), sort_keys=True) + "\n" for r in records)


# ----------------------------------------------------------------------------- outcome model
class OutcomeModel:
    """p(success | action, cause) (§1.3), ground truth and outcome resolution with pre-drawn uniforms."""

    def __init__(self, cfg: dict | None = None, pack=None, map_id: str | None = None):
        cfg = cfg or {}
        if "workshop" not in cfg and any(k in cfg for k in DEFAULTS):     # a bare workshop block
            cfg = {"workshop": cfg}
        w = wcfg(cfg)
        self.pack = pack or C.load(w["content"])
        self.success = {**DEFAULTS["success"], **(w.get("success") or {})}
        self.max_attempts = int(w["max_attempts"])
        self._cfg, self._mapping = cfg, map_id
        self.actions = [a for a in self.pack.ACTIONS if a != "stop"]

    @property
    def mapping(self) -> str:
        if self._mapping is None:
            self._mapping = R.mapping(self._cfg)
        return self._mapping

    def p(self, action: str, cause: str | None) -> float:
        if action not in self.pack.ACTIONS:
            raise ValueError(f"unknown action {action!r}")
        if action == "stop":
            return 0.0                                  # no attempt: deferred
        if cause is None:
            return 1.0                                  # K0: the clean outcome is certain
        s = self.success
        if action == "rerun":
            return float(s["rerun"])
        if action == "slow":
            return float(s["slow_belt"] if cause == self.pack.STABLE else s["slow"])
        return float(s["match"] if self.pack.FIX[cause] == action else s["other_fix"])

    def cause_of(self, item_class: str, regime: str, map_id: str | None = None) -> str | None:
        """The real-structure cause of a class under a regime and mapping (None for K0)."""
        m = self.pack.MAPPINGS[map_id or self.mapping]
        return {"K0": None, "K1": m[regime], "K2": self.pack.STABLE, "K3": m["B"]}[item_class]

    def gt(self, item_class: str, regime: str, map_id: str | None = None) -> str | None:
        """argmax_a p(a | cause_regime(class)). K0 (no fault) -> "rerun": no fix is needed, choosing one is
        overextension (same convention as analysis/battery/gt.py)."""
        cause = self.cause_of(item_class, regime, map_id)
        if cause is None:
            return "rerun"
        return max(self.actions, key=lambda a: (self.p(a, cause), -self.actions.index(a)))

    def score(self, action: str, item_class: str, map_id: str | None = None) -> str:
        """Post-change scoring category of an action for a class: old (GT under A), new (GT under B), other."""
        if action == self.gt(item_class, "A", map_id):
            return "old" if action != self.gt(item_class, "B", map_id) else "both"
        return "new" if action == self.gt(item_class, "B", map_id) else "other"

    def resolve(self, job: JobInstance, attempt: int, action: str, regime: str) -> str:
        """'success' | 'fail' | 'defer' for attempt (1-based) of `job` under the active regime.
        Success iff u[job, attempt] < p(action, cause) (maximal coupling across arms)."""
        return self.detail(job, attempt, action, regime)["outcome"]

    def detail(self, job: JobInstance, attempt: int, action: str, regime: str) -> dict:
        """resolve() with its hidden inputs (for the `job_truth` trace record)."""
        if not 1 <= attempt <= self.max_attempts:
            raise ValueError(f"attempt {attempt} outside 1..{self.max_attempts}")
        cause = job.cause_at(regime)
        u = float(job.hidden["u_attempt"][attempt - 1])
        if action == "stop":
            out, p = "defer", 0.0
        else:
            p = self.p(action, cause)
            out = "success" if u < p else "fail"
        return {"outcome": out, "p": p, "u": u, "cause": cause, "class": job.cls_at(regime)}


def gt(item_class: str, regime: str, map_id: str, content: str = "laser_alpha", cfg: dict | None = None) -> str:
    """The single ground-truth function (§1.3), shared by the engine and analysis/battery/gt.py."""
    c = {**(cfg or {}), "workshop": {**((cfg or {}).get("workshop") or {}), "content": content}}
    return OutcomeModel(c, map_id=map_id).gt(item_class, regime, map_id)


# ----------------------------------------------------------------------------- generation
def _draws(ws: int, day: int, slot: int) -> dict:
    """Fixed draws of one job from its day-local streams."""
    u = {p: [float(x) for x in seed_rng(ws, "ws3", p, day, slot).random(n)] for p, n in N_DRAWS.items()}
    mrng = seed_rng(ws, "ws3", "menu", day, slot)
    u["menu"] = [[int(i) for i in mrng.permutation(6)] for _ in range(N_ATTEMPT_DRAWS)]
    return u


ROLE_KEYS = {"am": "am_crew", "pm": "pm_crew", "stores": "stores"}


def _crews(roster, day: int) -> dict:
    """{am, pm, stores, newcomers} of a day from a callable(day) -> dict, or a roster.Roster-like object
    with members(role, day) and is_newcomer(aid, day)."""
    if callable(roster) and not hasattr(roster, "members"):
        return roster(day)
    out = {k: list(roster.members(role, day)) for k, role in ROLE_KEYS.items()}
    out["newcomers"] = [a for k in ("am", "pm", "stores") for a in out[k] if roster.is_newcomer(a, day)]
    return out


def default_roster(profiles: dict):
    """Crews by §1.6 (maya/dev/hana AM, ethan/leo/jordan PM, priya/sofia stores) when those agents exist,
    else the sorted population split 3/3/2. Nobody is a newcomer."""
    ids = sorted(profiles)
    if all(a in profiles for crew in DEFAULT_CREWS.values() for a in crew):
        crews = {k: list(v) for k, v in DEFAULT_CREWS.items()}
    else:
        crews = {"am": ids[0:3], "pm": ids[3:6], "stores": ids[6:8]}
    return lambda day: {**crews, "newcomers": []}


def rota(crew: list[str], newcomers, u_offset: float, n_jobs: int) -> list[str | None]:
    """Operators of a shift's n_jobs: round-robin over the sorted crew from a per-day offset, with
    newcomers (first two days) moved to the front, so the first in order runs 2 jobs of 4."""
    crew = sorted(crew)
    if not crew:
        return [None] * n_jobs
    off = _pick(u_offset, len(crew))
    order = crew[off:] + crew[:off]
    new = set(newcomers or [])
    order = [a for a in order if a in new] + [a for a in order if a not in new]
    return [order[i % len(order)] for i in range(n_jobs)]


def pm_start_tick(cfg: dict, clock) -> int:
    """Within-day tick where the PM shift begins (roster.shifts.pm start, default 13:00)."""
    pm = ((cfg.get("roster") or {}).get("shifts") or {}).get("pm", "13:00-17:45")
    start = _minutes(str(pm).split("-")[0])
    return (start - int(clock.day_start.total_seconds() // 60)) // clock.tick_minutes


def generate(cfg: dict, profiles: dict, clock, roster=None) -> list:
    """All workshop world-script records of the run, sorted by (tick, id): JobInstance (8 a day), the
    daily tally WorldRecord and cue WorldRecords (regimes.cues). Independent of agents' decisions.

    `roster`: optional roster.Roster (members(role, day), is_newcomer(aid, day)) or a callable
    day -> {"am": [...], "pm": [...], "stores": [...], "newcomers": [...]}; default: default_roster(profiles).
    It only decides operators (deterministic across arms: the roster is identical in every arm)."""
    w = wcfg(cfg)
    pack = C.load(w["content"])
    ws = world_seed(cfg)
    map_id = R.mapping(cfg)
    roster = roster or default_roster(profiles)
    tpd = clock.ticks_per_day
    ticks = [int(k) for k in w["jobs_ticks"]]
    if any(not 0 <= k < tpd for k in ticks):
        raise ValueError(f"workshop.jobs_ticks {ticks} must be inside the day (0..{tpd - 1})")
    pm_k = pm_start_tick(cfg, clock)
    shift_of = ["am" if k < pm_k else "pm" for k in ticks]
    wts = w["class_weights"]
    p_k1 = float(wts["K1"]) / (float(wts["K1"]) + float(wts["K2"]))
    code_cfg = w["panel_code"]
    n_att = int(w["max_attempts"])
    action_ids = list(pack.ACTIONS)
    k3_regimes = {r: R.has_k3(r) for r in R.REGIMES}

    out: list = []
    for day in range(1, clock.days + 1):
        crews = _crews(roster, day)
        draws = [_draws(ws, day, s) for s in range(len(ticks))]
        ops: dict = {}
        for sh in ("am", "pm"):
            slots = [s for s in range(len(ticks)) if shift_of[s] == sh]
            if slots:
                names = rota(crews.get(sh) or [], crews.get("newcomers"), draws[slots[0]]["operator"][0],
                             len(slots))
                ops.update(zip(slots, names))
        for s, k in enumerate(ticks):
            u = draws[s]
            fault = u["fault"][0] < float(w["p_fault"])
            base = ("K1" if u["class"][0] < p_k1 else "K2") if fault else "K0"
            cls, cause = {}, {}
            scr = pack.CAUSES[_pick(u["cause_scr"][0], len(pack.CAUSES))]     # causal: scrambled (drawn always)
            for reg in R.REGIMES:
                c = base
                if base == "K0" and k3_regimes[reg] and u["k3"][0] < float(w["k3_from_k0"]):
                    c = "K3"
                cls[reg] = c
                if c == "K0":
                    cause[reg] = None
                elif w["causal"] == "scrambled":
                    cause[reg] = scr
                else:
                    m = pack.MAPPINGS[map_id]
                    cause[reg] = {"K1": m[reg], "K2": pack.STABLE, "K3": m["B"]}[c]
            proj = C.PROJECTS[_pick(u["project"][0], len(C.PROJECTS))]
            odd = pack.ODDITIES[_pick(u["oddity"][1], len(pack.ODDITIES))] \
                if base == "K0" and u["oddity"][0] < float(w["oddity_prob"]) else None
            out.append(JobInstance(
                id=f"j{day:02d}.{s}", day=day, slot=s, shift=shift_of[s], start_tick=(day - 1) * tpd + k,
                operator=ops.get(s), project=proj[0], project_short=proj[1],
                menu_order=[[action_ids[i] for i in perm] for perm in u["menu"][:n_att]],
                surface={c: _pick(u["surface"][i], 8) for i, c in enumerate(CLASSES)},
                code=bool(code_cfg.get("enabled")) and base == "K1" and u["code"][0] < float(code_cfg["prob"]),
                oddity=odd,
                hidden={"fault": bool(fault), "class": cls, "cause": cause, "cause_scrambled": scr,
                        "u_attempt": [round(x, 6) for x in u["outcome"][:n_att]],
                        "u": {p: round(u[p][0], 6) for p in ("fault", "class", "k3", "code", "oddity",
                                                             "cause_scr")}}))
        out.append(WorldRecord(id=f"tally{day:02d}", kind="tally", day=day,
                               tick=R.tick_of(day, str(w["tally_time"]), clock), audience=["stores", "pm"]))
    for c in R.cues(cfg, clock):
        out.append(WorldRecord(id=c["id"], kind="cue", day=c["day"], tick=c["tick"], location=c["location"],
                               arena=c["arena"], text_key=c["text_key"], text=c["text"], audience=["stores"]))
    return sorted(out, key=lambda r: (r.tick, r.id))


# ----------------------------------------------------------------------------- runtime beats
def _fact(eid: str, beat: int, i: int, text: str, kind: str, vis, involves) -> dict:
    return {"id": f"{eid}.b{beat}.f{i}", "text": text, "salience": SALIENCE[kind], "visibility": vis,
            "kind": kind, "involves": sorted(a for a in involves if a)}


def _beat(idx: int, tick: int, location: str, arena: str, movers) -> dict:
    return {"idx": idx, "tick": int(tick), "location": location, "arena": arena, "facts": [],
            "movers": sorted(a for a in movers if a)}


def _as_job(job) -> JobInstance:
    return job if isinstance(job, JobInstance) else JobInstance.from_dict(job)


def symptom_text(job, cls: str, mapping: str, cfg: dict | None = None) -> str:
    """World surface of `job` for class `cls` (the active regime's class) under `mapping`."""
    job = _as_job(job)
    pack = C.load(wcfg(cfg or {})["content"])
    return C.surfaces(pack, cls, mapping)[job.surface[cls]]


def start_beat(job: JobInstance, regime: str, cfg: dict, name: str) -> dict:
    """Beat 0 at job.start_tick (Makerspace): the job-start fact, the symptom (or clean) surface of the
    active class, the panel code (K1, if drawn) and a K0 oddity (if drawn). `name`: operator's first name."""
    w, pack, map_id = wcfg(cfg), pack_of(cfg), R.mapping(cfg)
    vis = w["symptom_visibility"]
    cls = job.cls_at(regime)
    b = _beat(0, job.start_tick, job.location, job.arena, [job.operator])
    inv = [job.operator]
    texts = [(C.JOB_START.format(P=name, project=job.project), "job_start"),
             (symptom_text(job, cls, map_id, cfg), "symptom")]
    if job.code and cls == "K1":
        texts.append((C.PANEL.format(code=w["panel_code"].get("text", "F4")), "code"))
    if cls == "K0" and job.oddity is not None:
        texts.append((job.oddity, "oddity"))
    b["facts"] = [_fact(job.id, 0, i, t, k, vis, inv) for i, (t, k) in enumerate(texts)]
    return b


def needs_decision(job: JobInstance, regime: str) -> bool:
    """K0 jobs need no decision (the clean outcome is certain)."""
    return job.cls_at(regime) != "K0"


def outcome_text(name: str, action: str, success: bool, cls: str, mapping: str, content: str = "laser_alpha") -> str:
    """'Dev cleaned the focus lens and re-ran the sheet; the cut went all the way through this time.'"""
    pack = C.load(content)
    if action == "stop" or success is None:
        return C.STOPPED.format(P=name)
    result = pack.SUCCESS if success else C.fail_text(pack, cls, mapping)
    return C.OUTCOME.format(P=name, did=pack.ACTION_PAST[action], result=result)


def outcome_beat(job: JobInstance, idx: int, tick: int, action: str, outcome: str, regime: str, cfg: dict,
                 name: str) -> dict:
    """The outcome fact (salience .55) of an attempt, released at `tick` (t0+1 or later) as beat `idx`."""
    b = _beat(idx, tick, job.location, job.arena, [job.operator])
    text = outcome_text(name, action, None if outcome == "defer" else outcome == "success", job.cls_at(regime),
                        R.mapping(cfg), wcfg(cfg)["content"])
    b["facts"] = [_fact(job.id, idx, 0, text, "outcome", wcfg(cfg)["symptom_visibility"], [job.operator])]
    return b


def tally_text(results: list[tuple[JobInstance, bool]]) -> str:
    """'End-of-day tally at the laser: 6 of 8 jobs delivered; A and B were not.'"""
    m = len(results)
    missing = [j.project_short for j, ok in sorted(results, key=lambda x: x[0].slot) if not ok]
    if not missing:
        return C.TALLY_ALL.format(n=m, m=m)
    return C.TALLY_SOME.format(n=m - len(missing), m=m, missing=C.join_names(missing),
                               verb="was" if len(missing) == 1 else "were")


def tally_beat(rec: WorldRecord, results: list[tuple[JobInstance, bool]], cfg: dict, movers=()) -> dict:
    """The 17:30 tally at the Makerspace; `movers`: the rostered stores member and the PM crew."""
    b = _beat(0, rec.tick, rec.location, rec.arena, movers)
    b["facts"] = [_fact(rec.id, 0, 0, tally_text(results), "tally", wcfg(cfg)["symptom_visibility"], [])]
    return b


def cue_beat(rec: WorldRecord, cfg: dict, movers=()) -> dict:
    """A cue at its arena (Stockroom): perceivable only by agents there (arena visibility)."""
    b = _beat(0, rec.tick, rec.location, rec.arena, movers)
    b["facts"] = [_fact(rec.id, 0, 0, rec.text, "cue", wcfg(cfg)["symptom_visibility"], [])]
    return b


def menu_text(job: JobInstance, attempt: int, cfg: dict) -> list[tuple[str, str, str]]:
    """[(letter, action id, world wording)] of the pre-drawn order for attempt (1-based)."""
    pack = pack_of(cfg)
    return [(chr(65 + i), a, pack.ACTIONS[a]) for i, a in enumerate(job.menu_order[attempt - 1])]


# ----------------------------------------------------------------------------- trace records
def trace_job_start(job: JobInstance) -> dict:
    return {"job": job.id, "day": job.day, "slot": job.slot, "shift": job.shift, "tick": job.start_tick,
            "operator": job.operator, "project": job.project, "menu_order": job.menu_order}


def trace_job_attempt(job: JobInstance, attempt: int, action: str, outcome: str, tick: int) -> dict:
    return {"job": job.id, "attempt": attempt, "operator": job.operator, "action": action, "outcome": outcome,
            "tick": tick}


def trace_job_end(job: JobInstance, attempts: list[dict], delivered: bool) -> dict:
    return {"job": job.id, "operator": job.operator, "delivered": bool(delivered), "attempts": attempts,
            "result": "delivered" if delivered else
            ("defer" if attempts and attempts[-1].get("action") == "stop" else "failed")}


def trace_tally(rec: WorldRecord, results: list[tuple[JobInstance, bool]]) -> dict:
    return {"day": rec.day, "tick": rec.tick, "delivered": sum(ok for _, ok in results), "total": len(results),
            "missing": [j.id for j, ok in results if not ok], "text": tally_text(results)}


def trace_cue(rec: WorldRecord) -> dict:
    return {"cue": rec.id, "day": rec.day, "tick": rec.tick, "arena": rec.arena, "text_key": rec.text_key,
            "text": rec.text}


def trace_job_truth(job: JobInstance, regime: str, cfg: dict) -> dict:
    """HIDDEN (stripped by the API): the job's class, cause and ground truth under the active regime."""
    om = OutcomeModel(cfg)
    cls = job.cls_at(regime)
    return {"job": job.id, "regime": regime, "mapping": om.mapping, "class": cls, "cause": job.cause_at(regime),
            "fault": job.hidden["fault"], "gt": om.gt(cls, regime), "u_attempt": job.hidden["u_attempt"],
            "code": job.code, "p": {a: om.p(a, job.cause_at(regime)) for a in om.actions}}
