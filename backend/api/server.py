"""FastAPI server: run browser, replay data, agent inspection, causal trace
chains, analysis results, and launching new runs. Serves the frontend at '/'.

Runs live anywhere under the runs root (env MEMEWORLD_RUNS_ROOT, else runs/): a
run id is the run directory's path relative to that root, e.g. `20260919-..._s1`
or, for design runs, `<design>/<cell>/s<seed>`. Routes take it as a path
(`/api/runs/<design>/<cell>/s1/manifest`).

Normal (demo) mode strips hidden ground truth from every response: event
families and structure (latent types, schemas, skins, composition, regime),
casting (circles, assignment), event provenance ids, and the observer's
grounding and family-matching output. `?debug=1` (Research Debug Mode) returns
everything.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.ga_compat import GA_ASSETS, REPO_ROOT

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
}
# Dropped only at these paths (the key names are too generic to drop everywhere).
HIDDEN_PATHS = {
    ("config", "latent_events", "structure"),        # regime of the run (real / scrambled / none)
    ("config", "latent_events", "script_from"),
    ("topology", "free"), ("topology", "bridges"),   # circle structure (free pool, cross-circle ties)
    ("condition", "cell"), ("condition", "levels"), ("condition", "control"),
    ("config", "_condition", "cell"), ("config", "_condition", "levels"), ("config", "_condition", "control"),
}


def _strip(obj, path: tuple = ()):
    if isinstance(obj, dict):
        return {k: _strip(v, path + (k,)) for k, v in obj.items()
                if k not in HIDDEN_KEYS and path + (k,) not in HIDDEN_PATHS}
    if isinstance(obj, list):
        return [_strip(x, path) for x in obj]
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


@app.get("/api/runs/{run_id:path}/trace")
def trace(run_id: str, types: str = "", start: int = 0, end: int = 10 ** 9, agent: str = "", debug: int = 0,
          limit: int = 5000):
    D = data(run_id)
    ts = set(types.split(",")) if types else None
    out = []
    for r in D["trace"]:
        if ts and r["type"] not in ts:
            continue
        if not (start <= r["tick"] <= end):
            continue
        if agent and agent not in (r.get("agent"), r.get("speaker"), r.get("speaker_id")) and \
                agent not in (r.get("listeners") or []) and agent not in (r.get("participants") or []):
            continue
        if not debug and r["type"] in ("world_event_start", "event_beat"):
            continue
        out.append(r)
        if len(out) >= limit:
            break
    return out if debug else _strip(out)


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
