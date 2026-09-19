"""FastAPI server: run browser, replay data, agent inspection, causal trace
chains, analysis results, and launching new runs. Serves the frontend at '/'.

Runs live anywhere under the runs root (env MEMEWORLD_RUNS_ROOT, else runs/): a
run id is the run directory's path relative to that root, e.g. `20260919-..._s1`
or, for design runs, `<design>/<cell>/s<seed>`. Routes take it as a path
(`/api/runs/<design>/<cell>/s1/manifest`).

Normal (demo) mode strips hidden ground truth from every response: event
families and structure (latent types, schemas, skins, composition, regime),
casting (circles, assignment), event provenance ids, and the observer's
grounding and family-matching output. v3 adds job truth, regime, mapping,
symptom-class and cause ids, uniforms, the binder manipulation, rotation and
tree/branch names (HIDDEN_* below). `?debug=1` (Research Debug Mode) returns
everything.

Trace records are also stripped by the trace registry (backend/tracing/schema.py, docs/TRACE_SCHEMA.md): a type
registered as hidden is dropped and a public type loses its registered hidden fields, before the generic
HIDDEN_KEYS / HIDDEN_VALUE / HIDDEN_PATHS filters run. GET /api/schema serves the registry (public part in demo
mode), GET /api/runs/<id>/coop the co-op view of a v3 run (jobs, binder timeline, roster, tallies).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.ga_compat import GA_ASSETS, REPO_ROOT
from backend.tracing import schema as TS

RUNS = Path(os.environ.get("MEMEWORLD_RUNS_ROOT") or REPO_ROOT / "runs").resolve()
FRONTEND = REPO_ROOT / "frontend"
MAX_DEPTH = 4            # <design>/<cell>/s<seed> is depth 3 below the runs root
app = FastAPI(title="MemeWorld")

# Dropped wherever they occur (any depth) in demo mode.
HIDDEN_KEYS = {
    # world ground truth: families, structure, skins, composition, regime, holdout
    "latent_type", "latent_types", "latent_type_base_rates", "latent_distribution", "family", "family_home_groups",
    "schema", "skin", "skins", "composed_from", "structure_mode", "scenario", "narrative", "holdout", "roles",
    "generator", "world_script_sha256", "world_event",
    # casting: circles and how instances were assigned to them
    "circle", "circles", "cast_from_home", "assignment",
    # event provenance (which world event a record came from; links and remindings across events)
    "event_id", "event_ids", "originating_event_ids", "retrieved_event_ids", "referent_event_ids",
    "direct_event_ids", "carried_event_ids", "trigger_event_ids", "from_event_ids", "to_event_ids",
    "reminded_of_event_ids", "latent_event_id", "forced_by_event", "forced_by",
    # observer output that is scored against the hidden families
    "grounding", "funnel", "evaluation", "family_distribution", "best_family", "best_type", "match_family",
    "latent_alignment", "n_grounded", "n_grounding_tested", "grounded", "n_aligned_p05", "mean_alignment",
    "mean_lift",
    # v3 (ontology v3 §1.8, §6.5): job truth, regime, mapping, causes, symptom classes, uniforms, arms/branches
    "hidden", "regime", "regimes", "mapping", "world_state", "job_truth", "regime_active", "cause_by_regime",
    "cause", "causes", "class", "klass", "classes", "gt", "u_attempt", "fault", "surface", "code", "p_success",
    "branch", "branches", "arm", "arms", "salt", "at_day", "replay_until_tick",
}
# whole trace record types that exist only for the observer (hidden ground truth): the registry's hidden types
HIDDEN_TRACE_TYPES = {"world_event_start", "event_beat", "job_truth", "regime_active"} | set(TS.HIDDEN_TYPES)
# v3: hidden ids as keys or scalar values (K0-K3, causes, mappings) are dropped wherever they occur
HIDDEN_VALUE = re.compile(r"^(K[0-3]|LENS|DAMP|BELT|AIR|WARP|M[12])$")
# Dropped only at these paths (the key names are too generic to drop everywhere).
HIDDEN_PATHS = {
    ("config", "latent_events", "structure"),        # regime of the run (real / scrambled / none)
    ("config", "latent_events", "script_from"),
    ("topology", "free"), ("topology", "bridges"),   # circle structure (free pool, cross-circle ties)
    ("condition", "cell"), ("condition", "levels"), ("condition", "control"),
    ("config", "_condition", "cell"), ("config", "_condition", "levels"), ("config", "_condition", "control"),
    # v3: the manipulation (transitions, schedules), the world's hidden structure, rotation, tree nodes
    ("config", "records", "transitions"), ("config", "records", "history_from_day"),
    ("config", "records", "consult_from_day"), ("config", "workshop", "causal"), ("config", "workshop", "content"),
    ("config", "workshop", "class_weights"), ("config", "workshop", "p_fault"), ("config", "workshop", "k3_from_k0"),
    ("config", "workshop", "success"), ("config", "turnover", "rotate"), ("config", "turnover", "waves"),
    ("roster",), ("workshop", "content"), ("condition", "node"), ("config", "_condition", "node"),
    ("config", "design"), ("config", "_design"), ("config", "run_name"),
}


def _hidden_value(v) -> bool:
    return isinstance(v, str) and bool(HIDDEN_VALUE.match(v))


def _strip(obj, path: tuple = ()):
    if isinstance(obj, dict):
        if obj.get("type") in HIDDEN_TRACE_TYPES and "tick" in obj:
            return None
        if TS.is_trace_record(obj):              # registry: the type's hidden fields go first
            obj = TS.strip(obj)
            if obj is None:
                return None
        return {k: _strip(v, path + (k,)) for k, v in obj.items()
                if k not in HIDDEN_KEYS and path + (k,) not in HIDDEN_PATHS
                and not _hidden_value(k) and not _hidden_value(v)}
    if isinstance(obj, list):
        return [y for y in (_strip(x, path) for x in obj) if y is not None and not _hidden_value(y)]
    return obj


def _strip_manifest(man: dict) -> dict:
    out = _strip(man)
    # generated topology: the engine's `groups` ARE the circles (D62), so they go with them
    if man.get("circles") and man.get("groups") == man.get("circles"):
        out.pop("groups", None)
    return out


def _run_dir(run_id: str) -> Path:
    d = (RUNS / run_id).resolve()
    if RUNS not in d.parents or not (d / "manifest.json").exists():
        raise HTTPException(404, f"no run {run_id}")
    return d


def discover_runs(root: Path | None = None, need: str = "manifest.json") -> list[Path]:
    """Run directories under `root` (default: the runs root), up to MAX_DEPTH levels down, sorted by path.
    A directory with a manifest.json is a run and is not searched further; `need` filters by another file
    (e.g. analysis.json)."""
    root = Path(root or RUNS)
    out: list[Path] = []

    def walk(d: Path, depth: int):
        try:
            subs = sorted(x for x in d.iterdir() if x.is_dir() and not x.name.startswith("."))
        except OSError:
            return
        for x in subs:
            if (x / "manifest.json").exists():
                if (x / need).exists():
                    out.append(x)
            elif depth < MAX_DEPTH:
                walk(x, depth + 1)
    if root.exists():
        walk(root, 1)
    return out


def run_id_of(d: Path) -> str:
    """Run id = path relative to the runs root, e.g. `bottleneck_factorial/need-on__link-off/s1`."""
    try:
        return Path(d).relative_to(RUNS).as_posix()
    except ValueError:
        return Path(d).resolve().relative_to(RUNS).as_posix()


def _importance(D, nid: str | None, depth: int = 0):
    """Importance of a memory node from its trace record. A reminding thought has none of its own in the
    trace: it is the larger of its two linked memories' (memory/reminding.py)."""
    m = D["node"].get(nid) if nid else None
    if not m or depth > 50:
        return None
    if m["type"] != "reminding":
        return m.get("importance")
    imps = [x for x in (_importance(D, m.get("new_node"), depth + 1), _importance(D, m.get("reminded_of"), depth + 1))
            if x is not None]
    return max(imps) if imps else None


def _reminding_mem(D, r: dict) -> dict:
    """A reminding thought (LINK) as an inspector memory: it links the new experience to an earlier one."""
    ev = list(dict.fromkeys((r.get("originating_event_ids") or []) + (r.get("reminded_of_event_ids") or [])))
    return {"node_id": r["node_id"], "tick": r["tick"], "time": r["time"], "kind": "thought", "text": r["text"],
            "importance": _importance(D, r["node_id"]), "source_type": "reminding",
            "evidence": [x for x in (r.get("new_node"), r.get("reminded_of")) if x],
            "what_felt_alike": r.get("what_felt_alike"), "originating_event_ids": ev}


def _mtime(d: Path) -> float:
    return max((d / f).stat().st_mtime for f in ("trace.jsonl", "manifest.json") if (d / f).exists())


@lru_cache(maxsize=16)
def _load(run_id: str, mtime: float):
    d = _run_dir(run_id)
    trace = [json.loads(l) for l in open(d / "trace.jsonl")] if (d / "trace.jsonl").exists() else []
    frames = [json.loads(l) for l in open(d / "frames.jsonl")] if (d / "frames.jsonl").exists() else []
    events = [json.loads(l) for l in open(d / "events.jsonl")] if (d / "events.jsonl").exists() else []
    by_id = {r["id"]: r for r in trace}
    idx = defaultdict(list)
    for r in trace:
        idx[r["type"]].append(r)
    node_rec = {}
    for r in idx["memory_encoded"] + idx["reflection"] + [r for r in idx["reminding"] if r.get("node_id")]:
        node_rec[r["node_id"]] = r
    utt = {r["id"]: r for r in idx["utterance"]}
    obs = {r["observation_id"]: r for r in idx["observation"]}
    return {"trace": trace, "frames": frames, "events": events, "by_id": by_id, "idx": idx,
            "node": node_rec, "utt": utt, "obs": obs}


def data(run_id: str):
    d = _run_dir(run_id)
    return _load(run_id, _mtime(d))


@app.get("/api/runs")
def list_runs():
    out = []
    for d in sorted(discover_runs(), key=run_id_of, reverse=True):
        try:
            man = json.load(open(d / "manifest.json"))
        except (OSError, ValueError):
            continue                                   # a run that is just starting to write its manifest
        out.append({"run_id": run_id_of(d), "status": man.get("status"), "run_name": man["config"].get("run_name"),
                    "seed": man["config"].get("seed"), "days": man["config"].get("simulation_days"),
                    "modules": man.get("modules"), "stats": man.get("stats"),
                    "llm_backend": man["config"]["llm"]["backend"],
                    "has_analysis": (d / "analysis.json").exists()})
    return out


@app.get("/api/runs/{run_id:path}/manifest")
def manifest(run_id: str, debug: int = 0):
    from backend.simulation.world import MAP_POS
    man = json.load(open(_run_dir(run_id) / "manifest.json"))
    man["world"]["map_pos"] = MAP_POS  # display-only layout; always use the current one
    return man if debug else _strip_manifest(man)


@app.get("/api/runs/{run_id:path}/frames")
def frames(run_id: str, debug: int = 0):
    fr = data(run_id)["frames"]
    return fr if debug else _strip(fr)


@app.get("/api/runs/{run_id:path}/events")
def events(run_id: str, debug: int = 0):
    if not debug:
        raise HTTPException(403, "ground-truth events are only available in Research Debug Mode (?debug=1)")
    return data(run_id)["events"]


# fields naming one agent / lists of agents: a record "involves" an agent through any of them (trace ?agent=)
AGENT_FIELDS = ("agent", "speaker", "speaker_id", "operator", "target", "ask_target", "stores", "outgoing",
                "incoming", "heard_from")
AGENT_LIST_FIELDS = ("listeners", "listener_ids", "participants", "invited", "movers")


def _involves(r: dict, agent: str) -> bool:
    return any(r.get(f) == agent for f in AGENT_FIELDS) or \
        any(agent in (r.get(f) or []) for f in AGENT_LIST_FIELDS if isinstance(r.get(f), list))


@app.get("/api/runs/{run_id:path}/trace")
def trace(run_id: str, types: str = "", start: int = 0, end: int = 10 ** 9, agent: str = "", debug: int = 0,
          limit: int = 5000, type_: str = Query("", alias="type"), tick_from: int | None = None,
          tick_to: int | None = None, conversation: str = ""):
    """Trace records, oldest first. Filters: `type` (or the older `types`; comma-separated), `agent` (any record
    involving the agent: agent/speaker/operator/target/... or listeners/participants/invited/...), `tick_from` /
    `tick_to` (inclusive; the older `start` / `end`), `conversation` (the conversation record and everything
    carrying its conversation_id), `limit` (after filtering; hidden types never count)."""
    D = data(run_id)
    ts = {t for t in (types + "," + type_).split(",") if t} or None
    lo = start if tick_from is None else tick_from
    hi = end if tick_to is None else tick_to
    out = []
    for r in D["trace"]:
        if ts and r["type"] not in ts:
            continue
        if not (lo <= r["tick"] <= hi):
            continue
        if agent and not _involves(r, agent):
            continue
        if conversation and conversation not in (r.get("conversation_id"),
                                                 r["id"] if r["type"] == "conversation" else None):
            continue
        if not debug and r["type"] in HIDDEN_TRACE_TYPES:
            continue
        out.append(r)
        if len(out) >= limit:
            break
    return out if debug else _strip(out)


def _rec(r: dict, keys=None) -> dict:
    """A trace record without its envelope id/type (tick and time kept); only `keys` if given."""
    if keys is None:
        return {k: v for k, v in r.items() if k not in ("id", "type")}
    return {"tick": r.get("tick"), "time": r.get("time"), **{k: r.get(k) for k in keys}}


@app.get("/api/runs/{run_id:path}/coop")
def coop(run_id: str, debug: int = 0):
    """The co-op (ontology v3) view of a run, built from its trace: jobs (job_* records, the job's world text from
    coop_fact), the binder timeline and its reconstructed state (record_*), roster changes, onboarding, farewells,
    tallies, cues, handovers, meetings, day digests and checkpoints. Demo mode strips hidden fields; `?debug=1`
    adds each job's ground truth (`truth`, from job_truth) and the day regimes (`regimes`). A v2 run (or one from
    before the co-op existed) returns the same shape with empty lists and enabled=false."""
    D = data(run_id)
    idx = D["idx"]
    man = json.load(open(_run_dir(run_id) / "manifest.json"))
    cfg = man.get("config") or {}
    names = {aid: (p.get("name") or aid).split(" ")[0] for aid, p in (man.get("agents") or {}).items()}

    # the jobs' released world text: coop_fact (by ref and beat), else (older runs) perceived fact ids
    facts = {(r.get("ref"), r.get("beat")): r.get("facts") or [] for r in idx.get("coop_fact", [])}
    seen = {}
    if not facts:
        for o in idx.get("observation", []):
            if o.get("source_type") == "perception":
                for f in o.get("facts") or []:
                    if isinstance(f, dict) and f.get("id"):
                        seen.setdefault(f["id"], f.get("text"))

    def fact_text(job: str, beat: int, role: str):
        fs = facts.get((job, beat))
        if fs is not None:
            return next((f.get("text") for f in fs if f.get("role") == role), None)
        return seen.get(f"{job}.b{beat}.f{1 if role == 'symptom' else 0}")

    jobs = {}
    for r in idx.get("job_start", []):
        jobs[r["job"]] = {"job": r["job"], "day": r.get("day"), "slot": r.get("slot"), "shift": r.get("shift"),
                          "operator": r.get("operator"), "project": r.get("project"), "start_tick": r.get("tick"),
                          "symptom": fact_text(r["job"], 0, "symptom"), "decisions": [], "clarifications": [],
                          "attempts": [], "end": None, "writes": []}
    for r in idx.get("job_decision", []):
        if r.get("job") in jobs:
            jobs[r["job"]]["decisions"].append(_rec(r, ("attempt", "choice", "action", "question", "ask_target",
                                                         "says_aloud", "reason", "valid", "consult", "binder_shown",
                                                         "binder_chosen", "binder_entry_ids")))
    for r in idx.get("clarification", []):
        if r.get("job") in jobs:
            jobs[r["job"]]["clarifications"].append(_rec(r, ("attempt", "agent", "target", "question")))
    for r in idx.get("job_attempt", []):
        if r.get("job") in jobs:
            jobs[r["job"]]["attempts"].append({**_rec(r, ("attempt", "action", "outcome")),
                                               "text": fact_text(r["job"], r.get("attempt"), "outcome")})
    # scheduled / repair talk: the conversation each clarification, handover and meeting became (by its trigger)
    tpd = int(man.get("ticks_per_day") or 0)
    talk = defaultdict(list)
    for r in idx.get("conversation", []):
        tr = r.get("trigger") or {}
        if tr.get("topic") in ("clarify", "handover", "meeting"):
            talk[(tr["topic"], tr.get("job"), (r["tick"] // tpd + 1) if tpd else None)].append(r)

    def conv_of(topic: str, day, job=None, who=()):
        for c in talk.get((topic, job, day), []):
            if set(who) <= set(c.get("participants") or []):
                return c["id"]
        return None

    for j in jobs.values():
        for q in j["clarifications"]:
            q["conversation_id"] = conv_of("clarify", j["day"], j["job"], (q["agent"], q["target"]))
    ended = {}
    for r in idx.get("job_end", []):
        if r.get("job") in jobs:
            jobs[r["job"]]["end"] = _rec(r, ("delivered", "result"))
            ended[(r.get("operator"), r.get("tick"))] = r["job"]

    # binder: timeline of reads, writes and transitions in trace order, and the state they add up to
    timeline = []
    state = {"binder_id": "binder-1", "front": [], "log": [], "archived": []}
    n_binder = 1
    for r in D["trace"]:
        t = r["type"]
        if t == "record_write":
            job = ended.get((r.get("agent"), r.get("tick"))) if r.get("offer") == "job" else None
            ev = {**_rec(r, ("agent", "offer", "choice", "entry_id", "rev_id", "text", "truncated")), "kind": "write",
                  "author_name": names.get(r.get("agent"), r.get("agent")), "job": job}
            timeline.append(ev)
            item = {k: ev[k] for k in ("tick", "time", "author_name", "text", "offer")}
            if ev["choice"] == "log" and ev["entry_id"]:
                state["log"].append({"entry_id": ev["entry_id"], "author": ev["agent"], **item})
            elif ev["choice"] == "front" and ev["rev_id"]:
                state["front"].append({"rev_id": ev["rev_id"], "author": ev["agent"], **item})
            if job and (ev["entry_id"] or ev["rev_id"]):
                jobs[job]["writes"].append(ev["entry_id"] or ev["rev_id"])
        elif t == "record_transition":
            if r.get("mode") == "wipe":
                state["archived"].append({"binder_id": state["binder_id"], "archived_tick": r.get("tick"),
                                          "front": state["front"], "log": state["log"]})
                n_binder += 1
                state.update(binder_id=f"binder-{n_binder}", front=[], log=[])
            timeline.append({**_rec(r, ("day", "mode", "archived_binder_id")), "kind": "transition",
                             "binder_id": state["binder_id"]})
        elif t == "record_read":
            timeline.append({**_rec(r, ("agent", "context", "view_mode", "new_ids")), "kind": "read",
                             "shown_ids": list(r.get("rev_ids") or []) + list(r.get("entry_ids") or [])})

    mech = {k: bool((cfg.get(k) or {}).get("enabled")) for k in ("workshop", "records", "roster", "turnover")}
    out = {"run_id": run_id, "mechanisms": mech,
           "jobs": sorted(jobs.values(), key=lambda j: (j["start_tick"] or 0, j["job"])),
           "binder": {"timeline": timeline, "state": state},
           "roster_changes": [_rec(r) for r in idx.get("roster_change", [])],
           "onboarding": [_rec(r) for r in idx.get("onboarding", [])],
           "farewells": [_rec(r) for r in idx.get("farewell", [])],
           "tallies": [_rec(r) for r in idx.get("tally", [])],
           "cues": [_rec(r) for r in idx.get("cue_event", [])],
           "handovers": [{**_rec(r), "conversation_id": conv_of("handover", r.get("day"), None,
                                                                (r.get("outgoing"), r.get("incoming")))}
                         for r in idx.get("handover", [])],
           "meetings": [{**_rec(r), "conversation_id": conv_of("meeting", r.get("day"))}
                        for r in idx.get("meeting", [])],
           "days": [_rec(r) for r in idx.get("coop_day", [])],
           "checkpoints": [{"id": r.get("id"), "tick": r.get("tick"), "time": r.get("time")}
                           for r in idx.get("checkpoint", [])]}
    out["enabled"] = any(mech.values()) or bool(jobs or timeline or out["roster_changes"])
    if not debug:
        return _strip(out)
    truth = {r.get("job"): _rec(r) for r in idx.get("job_truth", [])}
    for j in out["jobs"]:
        j["truth"] = truth.get(j["job"])
    out["regimes"] = [_rec(r) for r in idx.get("regime_active", [])]
    return out


@app.get("/api/runs/{run_id:path}/agent/{agent_id}")
def agent_state(run_id: str, agent_id: str, tick: int = Query(...), debug: int = 0):
    """Agent memory stream as of `tick`, reconstructed from the trace."""
    D = data(run_id)
    man = json.load(open(_run_dir(run_id) / "manifest.json"))
    if agent_id not in man["agents"]:
        raise HTTPException(404)
    mems, forgotten = {}, set()
    for r in D["idx"]["memory_encoded"]:
        if r["agent"] == agent_id and r["tick"] <= tick:
            mems[r["node_id"]] = {"node_id": r["node_id"], "tick": r["tick"], "time": r["time"], "kind": r["kind"],
                                  "text": r["text"], "importance": r["importance"], "source_type": r["source_type"],
                                  "observation": r.get("observation"), "encoding_ops": r.get("encoding_ops"),
                                  "originating_event_ids": r.get("originating_event_ids")}
    for r in D["idx"]["reflection"]:
        if r["agent"] == agent_id and r["tick"] <= tick:
            mems[r["node_id"]] = {"node_id": r["node_id"], "tick": r["tick"], "time": r["time"], "kind": "thought",
                                  "text": r["text"], "importance": r["importance"], "source_type": "reflection",
                                  "focal_point": r["focal_point"], "evidence": r["evidence"],
                                  "originating_event_ids": r.get("originating_event_ids")}
    for r in D["idx"]["reminding"]:
        if r["agent"] == agent_id and r["tick"] <= tick and r.get("node_id"):
            mems[r["node_id"]] = _reminding_mem(D, r)
    for r in D["idx"]["memory_forgotten"]:
        if r["agent"] == agent_id and r["tick"] <= tick:
            forgotten.add(r["node_id"])
    alive = [m for k, m in mems.items() if k not in forgotten]
    alive.sort(key=lambda m: (m["tick"], m["node_id"]), reverse=True)
    last_retrieved = None
    for r in D["idx"]["utterance"] + D["idx"]["decision"]:
        who = r.get("speaker") or r.get("agent")
        if who == agent_id and r["tick"] <= tick and r.get("retrieved"):
            if last_retrieved is None or (r["tick"], r["id"]) > (last_retrieved["tick"], last_retrieved["id"]):
                last_retrieved = r
    retrieved = []
    if last_retrieved:
        for nid in last_retrieved["retrieved"]:
            m = mems.get(nid)
            if m:
                sc = (last_retrieved.get("retrieval_scores") or {}).get(nid)
                retrieved.append({**m, "score": sc})
    convs = [c for c in D["idx"]["conversation"] if agent_id in c["participants"] and c["tick"] <= tick][-6:]
    heard = [u for u in D["idx"]["utterance"] if agent_id in u.get("listeners", []) and u["tick"] <= tick][-10:]
    fr = D["frames"][min(tick, len(D["frames"]) - 1)] if D["frames"] else {}
    out = {"profile": man["agents"][agent_id], "state": (fr.get("agents") or {}).get(agent_id),
           # experiences, plus reminding thoughts (an experience linked to an earlier one); not seed/reflection
           "memories": [m for m in alive if m["kind"] != "thought" or m["source_type"] == "reminding"][:60],
           "reflections": [m for m in alive if m["kind"] == "thought" and m["source_type"] == "reflection"][:30],
           "remindings": [m for m in alive if m["source_type"] == "reminding"][:30],
           "n_memories": len(alive),
           "retrieved": {"context_id": last_retrieved["id"] if last_retrieved else None,
                         "tick": last_retrieved["tick"] if last_retrieved else None, "memories": retrieved},
           "conversations": convs[::-1], "heard": heard[::-1]}
    return out if debug else _strip(out)


@app.get("/api/runs/{run_id:path}/chain/{utterance_id:path}")
def chain(run_id: str, utterance_id: str, depth: int = 3, debug: int = 0):
    """Causal chain for an utterance: world event -> observation -> memory -> retrieval -> utterance
    -> listeners -> listener memories. Recurses through memories that came from earlier utterances."""
    D = data(run_id)
    u = D["utt"].get(utterance_id)
    if not u:
        raise HTTPException(404, "utterance not found")

    def mem_origin(nid, d):
        m = D["node"].get(nid)
        if not m:
            return {"node_id": nid, "missing": True}
        node = {"node_id": nid, "agent": m["agent"], "tick": m["tick"], "text": m["text"],
                "kind": m.get("kind", "thought"), "source_type": m.get("source_type", "reflection"),
                "originating_event_ids": m.get("originating_event_ids")}
        if m["type"] == "reflection":
            node["from_memories"] = [mem_origin(e, d - 1) for e in (m.get("evidence") or [])[:4]] if d > 0 else []
            return node
        if m["type"] == "reminding":        # a LINK thought: its origin is the two memories it connects
            rm = _reminding_mem(D, m)
            node.update(kind="thought", source_type="reminding", importance=rm["importance"],
                        what_felt_alike=rm["what_felt_alike"], originating_event_ids=rm["originating_event_ids"])
            node["from_memories"] = [mem_origin(x, d - 1) for x in rm["evidence"]] if d > 0 else []
            return node
        oid = m.get("observation_id")
        o = D["obs"].get(oid) if oid else None
        if o:
            node["observation"] = {"id": oid, "source_type": o["source_type"], "facts": o.get("facts"),
                                   "event_id": o.get("event_id")}
            if o.get("event_id"):
                ev = next((e for e in D["events"] if e["id"] == o["event_id"]), None)
                if ev:
                    node["world_event"] = {"id": ev["id"], "latent_type": ev.get("latent_type"),
                                           "scenario": ev.get("scenario") or ev.get("skin") or ev.get("schema"),
                                           "narrative": ev.get("narrative")}
        if d > 0 and m.get("utterance_ids"):
            node["from_utterances"] = [utt_node(x, d - 1) for x in m["utterance_ids"][:4] if x in D["utt"]]
        return node

    def utt_node(uid, d):
        x = D["utt"][uid]
        return {"id": uid, "tick": x["tick"], "time": x["time"], "speaker": x["speaker"], "text": x["text"],
                "listeners": x["listeners"], "conversation_id": x.get("conversation_id"),
                "retrieved": [mem_origin(n, d) for n in x.get("retrieved", [])] if d >= 0 else []}

    root = utt_node(utterance_id, depth)
    root["context"] = u.get("context")
    listener_mems = []
    for r in D["idx"]["memory_encoded"]:
        if utterance_id in (r.get("utterance_ids") or []):
            listener_mems.append({"agent": r["agent"], "node_id": r["node_id"], "text": r["text"],
                                  "tick": r["tick"], "importance": r["importance"], "source_type": r["source_type"]})
    root["listener_memories"] = listener_mems
    later = [x for x in D["idx"]["utterance"] if any(n in {m["node_id"] for m in listener_mems}
                                                     for n in x.get("retrieved", []))]
    root["later_utterances_using_these_memories"] = [{"id": x["id"], "speaker": x["speaker"], "text": x["text"],
                                                      "tick": x["tick"]} for x in later[:10]]
    return root if debug else _strip(root)


@app.get("/api/runs/{run_id:path}/analysis")
def analysis(run_id: str, debug: int = 0):
    p = _run_dir(run_id) / "analysis.json"
    if not p.exists():
        raise HTTPException(404, "not analyzed yet")
    a = json.load(open(p))
    if debug:
        return a
    a = _strip(a)
    man = json.load(open(p.parent / "manifest.json"))
    if man.get("circles") and man.get("groups") == man.get("circles"):
        for sem in (a.get("semantics") or {}).values():   # generated topology: groups are the circles
            if isinstance(sem, dict):
                sem.pop("within_group", None)
    return a


@app.get("/api/runs/{run_id:path}/llm_calls")
def llm_calls(run_id: str, agent: str = "", purpose: str = "", start: int = 0, limit: int = 200):
    out = []
    with open(_run_dir(run_id) / "llm_calls.jsonl") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            r = json.loads(line)
            if (agent and r.get("agent") != agent) or (purpose and r.get("purpose") != purpose):
                continue
            out.append(r)
            if len(out) >= limit:
                break
    return out


@app.get("/api/compare")
def compare_runs(debug: int = 0):
    from backend.analysis.compare import compare
    dirs = discover_runs(need="analysis.json")
    rows = compare(dirs, allow_mixed_observers=True)
    if len(rows) == len(dirs):                  # one row per analysed run: use the full (nested) run id
        for r, d in zip(rows, dirs):
            r["run_id"] = run_id_of(d)
    if not debug:
        rows = [{k: v for k, v in _strip(r).items() if k not in ("cell", "levels")} for r in rows]
    return rows


@app.get("/api/schema")
def trace_schema(debug: int = 0):
    """The trace record registry (backend/tracing/schema.py) and the frame fields. Demo mode lists only public
    record types and their public fields; `?debug=1` adds hidden types, hidden fields and the hidden flags."""
    reg = TS.public_registry(debug=bool(debug))
    return reg if debug else _strip(reg)


@app.get("/api/configs")
def configs():
    return sorted(p.name for p in (REPO_ROOT / "configs").glob("*.yaml") if p.name != "default.yaml")


@app.post("/api/runs")
def launch(config: str, days: int | None = None, backend: str | None = None):
    cfg = REPO_ROOT / "configs" / Path(config).name
    if not cfg.exists():
        raise HTTPException(404, "unknown config")
    cmd = [sys.executable, "-m", "backend.cli", "run", "--config", str(cfg), "--analyze"]
    sets = []
    if days:
        sets.append(f"simulation_days={int(days)}")
    if backend:
        sets.append(f"llm.backend={backend}")
    if sets:
        cmd += ["--set", *sets]
    RUNS.mkdir(parents=True, exist_ok=True)
    log = open(RUNS / f"launch_{cfg.stem}.log", "a")
    # the new run goes where this server looks for runs
    subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                     env={**os.environ, "MEMEWORLD_RUNS_ROOT": str(RUNS)})
    return JSONResponse({"launched": " ".join(cmd)})


@app.post("/api/runs/{run_id:path}/analyze")
def launch_analysis(run_id: str):
    d = _run_dir(run_id)
    log = open(d / "analysis.log", "a")
    subprocess.Popen([sys.executable, "-m", "backend.cli", "analyze", str(d)], cwd=REPO_ROOT, stdout=log,
                     stderr=subprocess.STDOUT)
    return {"launched": True}


app.mount("/ga_assets", StaticFiles(directory=str(GA_ASSETS)), name="ga_assets")
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")
