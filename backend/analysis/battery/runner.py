"""LB1 probe runner: isolated checkpoint copies answer the battery (ONTOLOGY_V3 §5.2-5.3). OBSERVER ONLY.

Isolation protocol (§5.2):
  1. meant to run as a separate process after a segment (CLI: `python -m backend.analysis.battery.runner <run>
     --checkpoint C4 --plan full`), never concurrently with a simulation;
  2. copies `checkpoints/C{d}` (or, for runs without checkpoints, `agents_final` + `memory_meta.json`) to
     `<probes>/C{d}/work/` and verifies checksums of the source before and after; a mismatch invalidates;
  3. builds Agent copies with ModuleStack([]), no tracer, clock at the checkpoint tick, memory_meta loaded (so
     retrieval.source_weights apply exactly as in the simulation) and the persona universe (departed included);
  4. retrieval: touch=False, k = probe.k, tau = probe.tau, rng = seed_rng(seed, "probe", ckpt, agent, item,
     form, order); focal points [item text, "the co-op's laser cutter"];
  5. copies never encode, reflect or save;
  6. a separate observer LLMClient and cache `<probes>/C{d}/llm_calls.jsonl`, scope
     `probe:C{d}:{agent}:{item}:{form}:{cue}:{framing}:{order}`;
  7. the answering model is cfg.probe (Haiku);
  8. a checkpoint with the same digest is probed once (meta.json digest match -> cached result);
  9. every response logs its retrieval hit (a retrieved node linked to a job or a binder record).

Nothing here writes outside `<probes>/C{d}/`; nothing in the simulation reads it.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields
from pathlib import Path

from backend import ga_compat
from backend.analysis.battery import gt as GT
from backend.analysis.battery.items import load_items, select
from backend.analysis.battery.nonce import NONE_TEXT, carrier, make_nonce
from backend.llm.client import llm_purpose, llm_scope
from backend.simulation.rngs import seed_rng

PDIR = ga_compat.REPO_ROOT / "backend" / "prompts"
MACHINE_FOCAL = "the co-op's laser cutter"
LETTERS = "ABCDEF"
PERSONA_FREE = "the student"
PERSONA_FREE_ISS = "A student who volunteers at a campus makerspace co-op."
FRAMINGS = {
    "note": '{first} finds a sticky note on the laser cutter. It says: "{text}"',
    "text": 'On the way in, {first} gets a text message from another member: "{text}"',
    "overheard": 'At lunch, {first} overhears another member say: "{text}"',
}
ABLATE_SOURCES = {"conversation", "overheard", "record"}
JOB_ID = re.compile(r"^j\d+\.\d+")
ALL_MODES = ("memory_only", "situated", "apply", "note", "ablated", "record_only")
FORM = {"memory_only": "P", "situated": "P-sit", "ablated": "P-abl", "apply": "A", "note": "N", "record_only": "R"}

# §5.3 "battery by checkpoint": plan -> {mode: spec}
PLANS = {
    "full": {"memory_only": {"items": ["all"]}, "situated": {"items": ["K1c_h", "K2_h", "K3"]},
             "apply": {"cues": ["X", "W", "Y"]}, "note": {"cues": ["X", "W", "Y", "NONCE", "NONE"]},
             "ablated": {"items": ["K1c", "K2_h"]}, "record_only": {"items": ["K1c_h", "K2_h", "K3"],
                                                                    "cues": ["X", "W", "Y", "NONCE", "NONE"]}},
    "C3": {"memory_only": {"items": ["all"]}, "note": {"cues": ["W", "NONCE"]}},
    "wipe4": {"memory_only": {"items": ["K1c", "K2", "K1a", "K0"]}},
    "light": {"memory_only": {"items": ["K1c", "K3"]}, "note": {"cues": ["X", "W", "NONCE"], "framings": ["note"]}},
}
DEFAULT_SPEC = {m: s for m, s in PLANS["full"].items()}


class ProbeInvalid(RuntimeError):
    """The checkpoint changed while it was probed (or failed verification): the probe is invalid."""


# --------------------------------------------------------------------------------------------- utilities
def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tree_checksums(root: Path) -> dict[str, str]:
    root = Path(root)
    if root.is_file():
        return {root.name: sha_file(root)}
    return {p.relative_to(root).as_posix(): sha_file(p)
            for p in sorted(root.rglob("*")) if p.is_file()}


def run_checksums(run_dir: Path, exclude=("probes",)) -> dict[str, str]:
    """sha256 of every file of a run directory except the observer's probe outputs (isolation tests)."""
    run_dir = Path(run_dir)
    return {k: v for k, v in tree_checksums(run_dir).items() if k.split("/", 1)[0] not in exclude}


def parse_json(text: str) -> dict | None:
    if not text or text.startswith("LLM_ERROR"):
        return None
    t = text.strip()
    if "```" in t:
        t = t.split("```")[1]
        t = t[4:] if t.lower().startswith("json") else t
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        out = json.loads(t[i:j + 1])
    except json.JSONDecodeError:
        return None
    return out if isinstance(out, dict) else None


def _render(template: str, inputs: list[str]) -> str:
    ga = ga_compat.load()
    return ga.gs.generate_prompt([str(x) for x in inputs], str(PDIR / template))


def menu_lines(order: list[str], pack: str) -> str:
    text = dict(GT.menu(pack))
    return "\n".join(f"{LETTERS[i]}) {text[a]}" for i, a in enumerate(order))


def draw_menu(seed, *parts, pack: str = "laser_alpha") -> list[str]:
    acts = [a for a, _ in GT.menu(pack)]
    perm = seed_rng(seed, "probe", *parts, "menu").permutation(len(acts))
    return [acts[i] for i in perm]


def letter_to_action(letter, order: list[str]) -> str | None:
    if not isinstance(letter, str):
        return None
    m = re.search(r"[A-F]", letter.strip().upper()[:3])
    if not m:
        return None
    i = LETTERS.index(m.group(0))
    return order[i] if i < len(order) else None


# ------------------------------------------------------------------------------------------ checkpoint
@dataclass
class CheckpointData:
    ckpt: str
    src: Path                   # the run's checkpoint dir (or the run dir for the agents_final fallback)
    work: Path                  # the isolated copy
    layout: str                 # "checkpoint" | "agents_final"
    digest: str
    src_checksums: dict
    agent_ids: list[str]
    memory_meta: dict
    agent_state: dict = field(default_factory=dict)
    binder: dict | None = None
    roster: dict | None = None
    world_state: dict = field(default_factory=dict)

    def mem_dir(self, aid: str) -> Path:
        return self.work / "agents" / aid / "associative_memory"


def _read_json(p: Path, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def _copy(src: Path, dst: Path):
    if src.is_dir():
        shutil.copytree(src, dst, copy_function=shutil.copyfile, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def _src_files(run_dir: Path, ckpt: str) -> tuple[str, Path, list[Path]]:
    cdir = run_dir / "checkpoints" / ckpt
    if cdir.is_dir():
        return "checkpoint", cdir, [cdir]
    if (run_dir / "agents_final").is_dir():
        return "agents_final", run_dir, [run_dir / "agents_final", run_dir / "memory_meta.json"]
    raise FileNotFoundError(f"{run_dir}: no checkpoints/{ckpt} and no agents_final")


def _src_checksums(layout: str, src: Path, paths: list[Path]) -> dict:
    out = {}
    for p in paths:
        if p.exists():
            for k, v in tree_checksums(p).items():
                out[f"{p.name}/{k}" if p.is_dir() and layout == "agents_final" else k] = v
    return out


def prepare_checkpoint(run_dir: Path, ckpt: str, out_dir: Path) -> CheckpointData:
    """Verify, then copy the checkpoint into out_dir/work (read-only use from then on)."""
    run_dir = Path(run_dir)
    layout, src, paths = _src_files(run_dir, ckpt)
    if layout == "checkpoint":
        try:
            from backend.experiment.checkpoint import verify_checkpoint
            if (src / "CHECKSUMS").exists() and not verify_checkpoint(src):
                raise ProbeInvalid(f"{src}: CHECKSUMS do not match before probing")
        except ImportError:
            pass
    before = _src_checksums(layout, src, paths)
    digest = (sha_file(src / "CHECKSUMS") if (src / "CHECKSUMS").exists()
              else hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest())
    work = out_dir / "work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    if layout == "checkpoint":
        _copy(src, work)
    else:
        _copy(run_dir / "agents_final", work / "agents")      # agents_final/<id>/associative_memory
        if (run_dir / "memory_meta.json").exists():
            _copy(run_dir / "memory_meta.json", work / "memory_meta.json")
    ids = sorted(p.name for p in (work / "agents").iterdir() if (p / "associative_memory").is_dir()) \
        if (work / "agents").is_dir() else []
    return CheckpointData(ckpt=ckpt, src=src, work=work, layout=layout, digest=digest, src_checksums=before,
                          agent_ids=ids, memory_meta=_read_json(work / "memory_meta.json", {}) or {},
                          agent_state=_read_json(work / "agent_state.json", {}) or {},
                          binder=_read_json(work / "binder.json"), roster=_read_json(work / "roster.json"),
                          world_state=_read_json(work / "world_state.json", {}) or {})


def verify_unchanged(run_dir: Path, cd: CheckpointData) -> bool:
    layout, src, paths = _src_files(Path(run_dir), cd.ckpt)
    return _src_checksums(layout, src, paths) == cd.src_checksums


# --------------------------------------------------------------------------------------------- agents
class _Ctx:
    """The copies' context: agents and the memory sidecar (so D50 source weights apply); no tracer."""

    def __init__(self, agents: dict, meta):
        self.agents = agents
        self.meta = meta


def _meta_from_json(d: dict):
    from backend.memory.store import MemoryMeta, SimMemoryMeta
    names = {f.name for f in fields(MemoryMeta)}
    sm = SimMemoryMeta()
    for nid, m in sorted((d or {}).items()):
        if isinstance(m, dict) and "agent_id" in m and "source_type" in m:
            sm.set(nid, MemoryMeta(**{k: v for k, v in m.items() if k in names}))
    return sm


def _profiles(cfg: dict, manifest: dict, ids: list[str]) -> dict:
    from backend.analysis.rundata import simulated_profiles
    profs = simulated_profiles(cfg, manifest)
    missing = [i for i in ids if i not in profs]
    if missing:   # departed / arriving members: the persona universe (founders + reserves, §3.1)
        try:
            from backend.agents.profile import load_population
            res = load_population(cfg["population"], None, include_reserves=True)
            allp = res[0] if isinstance(res, tuple) else res
            for i in missing:
                if i in allp:
                    profs[i] = allp[i]
        except Exception:   # noqa: BLE001 - older population loaders
            pass
    return profs


def checkpoint_time(cfg: dict, manifest: dict, cd: CheckpointData) -> dt.datetime:
    t = (cd.world_state or {}).get("time")
    if t:
        return dt.datetime.fromisoformat(str(t))
    start = dt.datetime.fromisoformat(manifest["start"])
    day = int(cd.ckpt[1:]) if cd.ckpt[1:].isdigit() else int(cfg.get("simulation_days", 1))
    return start + dt.timedelta(days=day - 1, hours=15)


def build_agents(cfg: dict, manifest: dict, cd: CheckpointData, when: dt.datetime, *,
                 ids: list[str] | None = None) -> dict:
    """Isolated Agent copies: ModuleStack([]), no tracer, clock at the checkpoint, memory_meta loaded."""
    from backend.agents.agent import Agent
    from backend.memory.store import MemoryStream
    from backend.modules.base import ModuleStack
    ids = ids or cd.agent_ids
    profs = _profiles(cfg, manifest, ids)
    acfg = copy.deepcopy(cfg)
    pc = cfg.get("probe") or {}
    acfg.setdefault("retrieval", {})["temperature"] = float(pc.get("tau", 0.7))
    agents = {}
    for aid in ids:
        if aid not in profs:
            continue
        a = Agent(profs[aid], acfg, ModuleStack([]), cfg.get("seed", 0))
        a.a_mem = MemoryStream.load_ga(aid, cd.mem_dir(aid))
        a.set_time(when)
        a.state.activity = "thinking"
        a.sync_scratch()
        agents[aid] = a
    ctx = _Ctx(agents, _meta_from_json(cd.memory_meta))
    for a in agents.values():
        a.ctx = ctx
    return agents


# ------------------------------------------------------------------------------------------ responders
@dataclass
class Responder:
    """Who answers: an isolated agent copy (memory), or the persona-free responder (fresh / record-only)."""
    rid: str
    iss: str
    first: str
    agent: object | None = None
    exclude: frozenset = frozenset()

    def memories(self, focal: str, seed, parts: tuple, k: int, raw_meta: dict) -> tuple[str, list[str], bool]:
        if self.agent is None or not self.agent.a_mem.id_to_node:
            return "- (nothing comes to mind)", [], False
        from backend.memory.retrieval import merged_nodes, retrieve
        res = retrieve(self.agent, [focal, MACHINE_FOCAL], k=k, rng=seed_rng(seed, "probe", *parts), touch=False,
                       exclude_ids=self.exclude)
        nodes = merged_nodes(res, limit=k)
        ids = [n.node_id for n in nodes]
        hit = any(_linked(raw_meta.get(i) or {}) for i in ids)
        return ("".join(f"- {n.description}\n" for n in nodes).rstrip("\n") or "- (nothing comes to mind)"), ids, hit


def _linked(m: dict) -> bool:
    if m.get("source_type") == "record" or m.get("record_ids"):
        return True
    return any(JOB_ID.match(str(e)) for e in (m.get("originating_event_ids") or []) + (m.get("event_ids") or []))


def ablation_excludes(agent, raw_meta: dict) -> frozenset:
    """Source-ablated retrieval (§5.3): drop conversation/overheard/record nodes, and reflections and reminding
    thoughts whose evidence contains any of them."""
    drop = {nid for nid in agent.a_mem.id_to_node if (raw_meta.get(nid) or {}).get("source_type") in ABLATE_SOURCES}
    changed = True
    while changed:
        changed = False
        for nid, n in agent.a_mem.id_to_node.items():
            if nid in drop:
                continue
            m = raw_meta.get(nid) or {}
            ev = set(m.get("source_ids") or []) | {str(x) for x in (n.filling or []) if isinstance(x, str)}
            if (n.type == "thought" or m.get("source_type") in ("reflection", "reminding")) and ev & drop:
                drop.add(nid)
                changed = True
    return frozenset(drop)


# ------------------------------------------------------------------------------------------------ asks
@dataclass
class Session:
    llm: object
    seed: int
    ckpt: str
    prefix: str                 # scope prefix: "probe" | "fresh"
    weekday: str
    k: int
    pack: str
    mapping: str
    regime: str
    raw_meta: dict
    binder_view: str = ""

    def scope(self, rid, item, form, cue="-", framing="-", order=0) -> str:
        return f"{self.prefix}:{self.ckpt}:{rid}:{item}:{form}:{cue}:{framing}:{order}"

    def _call(self, purpose: str, rid: str, scope: str, prompt: str) -> tuple[dict | None, str | None]:
        try:
            with llm_scope(scope), llm_purpose(purpose, rid):
                text = self.llm.complete(prompt, system=ga_compat.CHAT_SYSTEM, max_tokens=300, temperature=0)
        except Exception as e:  # noqa: BLE001 - an error is never recorded as an answer
            return None, f"{type(e).__name__}: {e}"[:300]
        out = parse_json(text)
        return out, None if out is not None else ("llm_error" if str(text).startswith("LLM_ERROR") else "unparsed")

    def situated_block(self, situated: bool) -> str:
        if not situated:
            return ""
        return "The binder next to the laser currently reads:\n" + (self.binder_view or "(nothing)") + "\n"

    def act(self, r: Responder, item: dict, form: str, order: int, situated: bool) -> dict:
        parts = (self.ckpt, r.rid, item["id"], form, order)
        mem, ids, hit = r.memories(item["text"], self.seed, parts, self.k, self.raw_meta)
        menu = draw_menu(self.seed, *parts, pack=self.pack)
        prompt = _render("probe_act_v1.txt", [r.iss, self.weekday, r.first, mem, item["text"],
                                              menu_lines(menu, self.pack), self.situated_block(situated)])
        scope = self.scope(r.rid, item["id"], form, order=order)
        out, err = self._call("probe_act", r.rid, scope, prompt)
        action = letter_to_action((out or {}).get("choice"), menu)
        rec = {"form": form, "agent": r.rid, "item": item["id"], "type": item["type"], "item_type": item["type"],
               "wording": item.get("wording"), "cue": None, "cue_text": None, "framing": None, "order": order,
               "menu": menu, "choice": (out or {}).get("choice"), "action": action,
               "confidence": (out or {}).get("confidence"), "describe": (out or {}).get("describe"),
               "retrieved": ids, "hit": hit, "scope": scope, "error": err, "valid": err is None and action is not None}
        rec.update(GT.score(action, item, self.regime, self.mapping, self.pack))
        return rec

    def apply(self, r: Responder, cue_key: str, cue: str, situations: list[dict]) -> dict:
        parts = (self.ckpt, r.rid, f"A13.{cue_key}", "A", 0)
        mem, ids, hit = r.memories(cue, self.seed, parts, self.k, self.raw_meta)
        perm = seed_rng(self.seed, "probe", *parts, "order").permutation(len(situations))
        shown = [situations[i] for i in perm]
        sit = "\n".join(f"{j + 1}. {it['text']}" for j, it in enumerate(shown))
        prompt = _render("probe_apply_v1.txt", [r.iss, self.weekday, r.first, mem, cue, sit])
        scope = self.scope(r.rid, "A13", "A", cue=cue_key)
        out, err = self._call("probe_apply", r.rid, scope, prompt)
        fit = {}
        for j, it in enumerate(shown):
            v = str((out or {}).get(str(j + 1), "")).strip().lower().replace("’", "'")
            fit[it["id"]] = {"type": it["type"], "fit": v or None,
                             "fit_value": 1.0 if v == "fits" else 0.0 if v.startswith("doesn") else 0.5 if v else None}
        return {"form": "A", "agent": r.rid, "item": "A13", "type": "A", "item_type": "A", "cue": cue_key,
                "cue_text": cue, "framing": None, "order": 0, "shown": [it["id"] for it in shown], "fits": fit,
                "retrieved": ids, "hit": hit, "scope": scope, "error": err, "valid": err is None}

    def note(self, r: Responder, cue_key: str, cue: str | None, framing: str, situated: bool = False,
             form: str = "N") -> dict:
        text = carrier(cue) if cue_key != "NONE" else NONE_TEXT
        parts = (self.ckpt, r.rid, f"N.{cue_key}", form, framing)
        mem, ids, hit = r.memories(text, self.seed, parts, self.k, self.raw_meta)
        menu = draw_menu(self.seed, *parts, pack=self.pack)
        frame = FRAMINGS[framing].format(first=r.first[:1].upper() + r.first[1:], text=text)
        prompt = _render("probe_note_v1.txt", [r.iss, self.weekday, r.first, mem, frame, menu_lines(menu, self.pack),
                                               self.situated_block(situated)])
        scope = self.scope(r.rid, "N", form, cue=cue_key, framing=framing)
        out, err = self._call("probe_note", r.rid, scope, prompt)
        action = letter_to_action((out or {}).get("choice"), menu)
        heard = (out or {}).get("heard_before")
        return {"form": form, "agent": r.rid, "item": "N", "type": "N", "item_type": "N", "cue": cue_key,
                "cue_text": text, "framing": framing, "order": 0, "menu": menu, "choice": (out or {}).get("choice"),
                "action": action, "meaning": (out or {}).get("meaning"), "heard": heard, "heard_before": heard,
                "retrieved": ids, "hit": hit, "scope": scope, "error": err, "valid": err is None and action is not None,
                "a_fix": GT.gt({"type": "K1"}, "A", self.mapping, self.pack),
                "b_fix": GT.gt({"type": "K1"}, "B", self.mapping, self.pack)}


def items_for(items: list[dict], specs: list[str]) -> list[dict]:
    out, seen = [], set()
    for s in specs:
        for it in select(items, s):
            if it["id"] not in seen:
                seen.add(it["id"])
                out.append(it)
    return out


def run_items(sess: Session, r: Responder, items: list[dict], form: str, situated: bool,
              k1c_orders: int = 2) -> list[dict]:
    out = []
    for it in items:
        for order in range(k1c_orders if it["type"] == "K1c" and form == "P" else 1):
            out.append(sess.act(r, it, form, order, situated))
    return out


def cue_table(targets: dict, seed) -> dict:
    """{X, W, Y, NONCE, NONE} -> text (None when the target does not exist)."""
    x, w, y = targets.get("X"), targets.get("W") or "F4", targets.get("Y")
    return {"X": x, "W": w, "Y": y, "NONCE": targets.get("NONCE") or make_nonce(x or w, seed), "NONE": None}


def load_targets(run_dir: Path, probes_root: Path, targets: dict | None) -> dict:
    if targets is not None:
        return dict(targets)
    for p in (probes_root / "targets.json", Path(run_dir) / "probes" / "targets.json"):
        if p.exists():
            from backend.analysis.battery.targets import probe_targets
            return probe_targets(json.loads(p.read_text()))
    return {"W": "F4"}


def render_binder_view(binder: dict | None, cfg: dict, day: int, time_of) -> str:
    """The binder view at the checkpoint in this arm's display (records.render_for), or '' without a binder."""
    if not binder or binder.get("enabled") is False or "front" not in binder:
        return ""
    try:
        from backend.simulation.records import Binder, render_for
        return render_for(Binder.from_dict(binder, time_of=time_of), cfg, day).text
    except Exception:   # noqa: BLE001 - records module absent or changed: minimal rendering
        lines = []
        if binder.get("front"):
            lines.append(f'Front page: "{binder["front"][-1].get("text", "")}"')
        for e in list(reversed(binder.get("log") or []))[:5]:
            lines.append(f"  {e.get('text', '')}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- driver
def resolve_spec(modes, plan: str | None) -> dict:
    if plan:
        if plan in ("C4", "C6"):
            plan = "full"
        if plan == "C5":
            plan = "light"
        if plan not in PLANS:
            raise ValueError(f"unknown battery plan {plan!r}; one of {sorted(PLANS)}")
        return copy.deepcopy(PLANS[plan])
    bad = [m for m in modes if m not in ALL_MODES]
    if bad:
        raise ValueError(f"unknown probe modes {bad}; one of {ALL_MODES}")
    return {m: copy.deepcopy(DEFAULT_SPEC[m]) for m in modes}


def _newcomer(state: dict) -> bool:
    return str(state.get("cohort") or "").upper() in ("W1", "W2") or int(state.get("arrival_day") or 1) > 1


def run_battery(run_dir, ckpt: str = "final", modes=("memory_only", "situated"), *, plan: str | None = None,
                targets: dict | None = None, llm=None, backend=None, probes_root=None, agents: list[str] | None = None,
                force: bool = False) -> dict:
    """Probe the isolated copies of checkpoint `ckpt` of `run_dir`. Returns the summary also written to
    `<probes_root>/<ckpt>/meta.json`; responses go to `<probes_root>/<ckpt>/responses.jsonl`.

    modes: subset of ALL_MODES (ignored when `plan` is given: "full" | "C3" | "wipe4" | "light", §5.3 table).
    targets: {"X", "W", "Y", "NONCE"?}; default <probes_root>/targets.json, else W = the panel code only.
    llm: an LLMClient (tests); default backend.llm.client.observer_client("probe", <probes>/<ckpt>/llm_calls.jsonl).
    """
    from backend.analysis.rundata import RunData
    run_dir = Path(run_dir)
    rd = RunData(run_dir)
    cfg, manifest = rd.cfg, rd.manifest
    pc = cfg.get("probe") or {}
    probes_root = Path(probes_root) if probes_root else run_dir / "probes"
    out_dir = probes_root / ckpt
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = resolve_spec(modes, plan)
    spec_key = hashlib.sha256(json.dumps({"spec": spec, "targets": targets}, sort_keys=True).encode()).hexdigest()[:16]

    cd = prepare_checkpoint(run_dir, ckpt, out_dir)
    prev = _read_json(out_dir / "meta.json")
    if prev and not force and prev.get("digest") == cd.digest and prev.get("spec_key") == spec_key and prev.get("valid"):
        shutil.rmtree(cd.work, ignore_errors=True)
        return dict(prev, cached=True)          # §5.2.8: identical checkpoints are probed once

    own_llm = llm is None
    if own_llm:
        from backend.llm.client import observer_client
        llm = observer_client("probe", out_dir / "llm_calls.jsonl", cfg, backend=backend)
    from backend.llm.embeddings import make_embedder
    ga_compat.load(None, make_embedder(cfg.get("embedding")))

    when = checkpoint_time(cfg, manifest, cd)
    day = int(cd.world_state.get("day") or (int(ckpt[1:]) if ckpt[1:].isdigit() else cfg.get("simulation_days", 1)))
    pack = (cfg.get("workshop") or {}).get("content", "laser_alpha")
    mapping = cd.world_state.get("mapping") or GT.mapping_for(cfg)
    regime = cd.world_state.get("regime") or GT.regime_at(cfg, day)
    items = load_items(pack, mapping)
    tg = load_targets(run_dir, probes_root, targets)
    cues = cue_table(tg, cfg.get("seed", 0))
    start = dt.datetime.fromisoformat(manifest["start"])
    tick_min = int(manifest.get("tick_minutes", 15))
    view = render_binder_view(cd.binder, cfg, day, lambda t: start + dt.timedelta(minutes=tick_min * t))
    framings_all = list(pc.get("framings") or ["note", "text", "overheard"])

    sess = Session(llm=llm, seed=int(cfg.get("seed", 0)), ckpt=ckpt, prefix="probe", weekday=when.strftime("%A"),
                   k=int(pc.get("k", 6)), pack=pack, mapping=mapping, regime=regime, raw_meta=cd.memory_meta,
                   binder_view=view)
    copies = build_agents(cfg, manifest, cd, when, ids=agents)
    skipped = sorted(set(agents or cd.agent_ids) - set(copies))

    def one(aid: str) -> list[dict]:
        a = copies[aid]
        r = Responder(aid, a.iss(), a.profile.first_name, a)
        recs = []
        for mode, s in spec.items():
            if mode == "record_only":
                continue
            if mode == "memory_only":
                recs += run_items(sess, r, items_for(items, s["items"]), "P", False)
            elif mode == "situated":
                if view:
                    recs += run_items(sess, r, items_for(items, s["items"]), "P-sit", True)
            elif mode == "ablated":
                if _newcomer(cd.agent_state.get(aid) or {}):
                    ra = Responder(aid, r.iss, r.first, a, ablation_excludes(a, cd.memory_meta))
                    recs += run_items(sess, ra, items_for(items, s["items"]), "P-abl", False)
            elif mode == "apply":
                sit = select(items, "A13")
                recs += [sess.apply(r, ck, cues[ck], sit) for ck in s["cues"] if cues.get(ck)]
            elif mode == "note":
                for ck in s["cues"]:
                    if ck != "NONE" and not cues.get(ck):
                        continue
                    for fr in s.get("framings") or framings_all:
                        recs.append(sess.note(r, ck, cues[ck], fr))
        return recs

    workers = max(1, int(pc.get("workers", 8)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(one, sorted(copies)))
    records = [x for rs in results for x in rs]
    if "record_only" in spec and view:
        from backend.analysis.battery.record_only import run_record_only
        records += run_record_only(sess, items_for(items, spec["record_only"]["items"]), cues,
                                   spec["record_only"]["cues"], framings_all)

    valid = verify_unchanged(run_dir, cd) and tree_checksums(cd.work / "agents") == _work_agents_sums(cd)
    for rec in records:
        rec.update({"ckpt": ckpt, "digest": cd.digest, "regime": regime, "mapping": mapping,
                    "cohort": (cd.agent_state.get(rec["agent"]) or {}).get("cohort")})
    records.sort(key=lambda x: x["scope"])
    with open(out_dir / "responses.jsonl", "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    summary = {"ckpt": ckpt, "run": str(run_dir), "layout": cd.layout, "digest": cd.digest, "spec_key": spec_key,
               "spec": spec, "plan": plan, "valid": bool(valid), "n_calls": len(records),
               "n_errors": sum(1 for x in records if x.get("error")), "agents": sorted(copies),
               "skipped_agents": skipped, "targets": {k: v for k, v in cues.items()},
               "missing_cues": sorted(k for k, v in cues.items() if k != "NONE" and not v),
               "situated_available": bool(view), "hit_rate": hit_rates(records), "regime": regime,
               "mapping": mapping, "time": when.isoformat()}
    (out_dir / "meta.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    shutil.rmtree(cd.work, ignore_errors=True)
    if own_llm:
        llm.close()
    if not valid:
        raise ProbeInvalid(f"{run_dir}/{ckpt}: source files changed during probing")
    return summary


def _work_agents_sums(cd: CheckpointData) -> dict:
    """The agents' memory files as copied (source side), for the 'copies never save' check."""
    src = cd.src / "agents" if cd.layout == "checkpoint" else cd.src / "agents_final"
    return tree_checksums(src)


def hit_rates(records: list[dict]) -> dict:
    """§5.2.9: share of responses with a retrieved node linked to a job or a binder record, per form x item type."""
    acc: dict[str, list[int]] = {}
    for r in records:
        if r["agent"] == "record_only":
            continue
        key = f"{r['form']}:{r.get('item_type')}:{r.get('wording') or '-'}"
        acc.setdefault(key, []).append(int(bool(r.get("hit"))))
    return {k: round(sum(v) / len(v), 3) for k, v in sorted(acc.items())}


def read_responses(run_dir, ckpt: str, probes_root=None) -> list[dict]:
    p = (Path(probes_root) if probes_root else Path(run_dir) / "probes") / ckpt / "responses.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def main(argv=None):
    ap = argparse.ArgumentParser(description="Probe an isolated checkpoint copy with the LB1 battery (observer).")
    ap.add_argument("run")
    ap.add_argument("--checkpoint", default="final")
    ap.add_argument("--plan", default=None, help="full | C3 | wipe4 | light (default: --modes)")
    ap.add_argument("--modes", default="memory_only,situated")
    ap.add_argument("--backend", default=None, help="override probe.backend (e.g. mock)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    backend = None
    if a.backend:
        from backend.llm.client import make_backend
        backend = make_backend({"backend": a.backend, "model": "haiku"})
    s = run_battery(a.run, a.checkpoint, a.modes.split(","), plan=a.plan, backend=backend, force=a.force)
    print(json.dumps({k: s[k] for k in ("ckpt", "valid", "n_calls", "n_errors", "agents")}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
