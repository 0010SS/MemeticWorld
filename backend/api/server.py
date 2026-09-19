"""FastAPI server: run browser, replay data, agent inspection, causal trace
chains, analysis results, and launching new runs. Serves the frontend at '/'.

Normal (demo) mode strips hidden ground-truth fields (latent types, scenario
names, event narratives) from responses; `?debug=1` (Research Debug Mode)
returns them.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.ga_compat import GA_ASSETS, REPO_ROOT

RUNS = REPO_ROOT / "runs"
FRONTEND = REPO_ROOT / "frontend"
app = FastAPI(title="MemeWorld")

HIDDEN_KEYS = {"latent_type", "latent_types", "scenario", "narrative", "holdout", "latent_distribution",
               "best_type", "originating_event_ids", "retrieved_event_ids", "event_id", "latent_type_base_rates",
               "evaluation", "roles", "generator", "research", "bench_mode"}


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in HIDDEN_KEYS}
    if isinstance(obj, list):
        return [_strip(x) for x in obj]
    return obj


def _run_dir(run_id: str) -> Path:
    d = (RUNS / run_id).resolve()
    if RUNS.resolve() not in d.parents or not (d / "manifest.json").exists():
        raise HTTPException(404, f"no run {run_id}")
    return d


def _mtime(d: Path) -> float:
    return max((d / f).stat().st_mtime for f in ("trace.jsonl", "manifest.json") if (d / f).exists())


@lru_cache(maxsize=16)
def _load(run_id: str, mtime: float):
    d = RUNS / run_id
    trace = [json.loads(l) for l in open(d / "trace.jsonl")] if (d / "trace.jsonl").exists() else []
    frames = [json.loads(l) for l in open(d / "frames.jsonl")] if (d / "frames.jsonl").exists() else []
    events = [json.loads(l) for l in open(d / "events.jsonl")] if (d / "events.jsonl").exists() else []
    by_id = {r["id"]: r for r in trace}
    idx = defaultdict(list)
    for r in trace:
        idx[r["type"]].append(r)
    node_rec = {}
    for r in idx["memory_encoded"] + idx["reflection"]:
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
    for d in sorted(RUNS.iterdir(), reverse=True) if RUNS.exists() else []:
        m = d / "manifest.json"
        if not m.exists():
            continue
        man = json.load(open(m))
        out.append({"run_id": d.name, "status": man.get("status"), "run_name": man["config"].get("run_name"),
                    "seed": man["config"].get("seed"), "days": man["config"].get("simulation_days"),
                    "modules": man.get("modules"), "stats": man.get("stats"),
                    "llm_backend": man["config"]["llm"]["backend"],
                    "has_analysis": (d / "analysis.json").exists()})
    return out


@app.get("/api/runs/{run_id}/manifest")
def manifest(run_id: str, debug: int = 0):
    from backend.simulation.world import MAP_POS
    man = json.load(open(_run_dir(run_id) / "manifest.json"))
    man["world"]["map_pos"] = MAP_POS  # display-only layout; always use the current one
    return man if debug else _strip(man)


@app.get("/api/runs/{run_id}/frames")
def frames(run_id: str, debug: int = 0):
    fr = data(run_id)["frames"]
    return fr if debug else _strip(fr)


@app.get("/api/runs/{run_id}/events")
def events(run_id: str, debug: int = 0):
    if not debug:
        raise HTTPException(403, "ground-truth events are only available in Research Debug Mode (?debug=1)")
    return data(run_id)["events"]


@app.get("/api/runs/{run_id}/trace")
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


@app.get("/api/runs/{run_id}/agent/{agent_id}")
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
           "memories": [m for m in alive if m["kind"] != "thought"][:60],
           "reflections": [m for m in alive if m["kind"] == "thought" and m["source_type"] == "reflection"][:30],
           "n_memories": len(alive),
           "retrieved": {"context_id": last_retrieved["id"] if last_retrieved else None,
                         "tick": last_retrieved["tick"] if last_retrieved else None, "memories": retrieved},
           "conversations": convs[::-1], "heard": heard[::-1]}
    return out if debug else _strip(out)


@app.get("/api/runs/{run_id}/chain/{utterance_id:path}")
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
        oid = m.get("observation_id")
        o = D["obs"].get(oid) if oid else None
        if o:
            node["observation"] = {"id": oid, "source_type": o["source_type"], "facts": o.get("facts"),
                                   "event_id": o.get("event_id")}
            if o.get("event_id"):
                ev = next((e for e in D["events"] if e["id"] == o["event_id"]), None)
                if ev:
                    node["world_event"] = {"id": ev["id"], "latent_type": ev["latent_type"],
                                           "scenario": ev["scenario"], "narrative": ev["narrative"]}
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


@app.get("/api/runs/{run_id}/analysis")
def analysis(run_id: str, debug: int = 0):
    p = _run_dir(run_id) / "analysis.json"
    if not p.exists():
        raise HTTPException(404, "not analyzed yet")
    a = json.load(open(p))
    return a if debug else _strip(a)


@app.get("/api/runs/{run_id}/llm_calls")
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
    rows = compare([d for d in sorted(RUNS.iterdir()) if (d / "analysis.json").exists()]) if RUNS.exists() else []
    if not debug:
        for r in rows:
            r.pop("mean_alignment", None)
            r.pop("mean_lift", None)
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
    log = open(RUNS / f"launch_{cfg.stem}.log", "a")
    subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
    return JSONResponse({"launched": " ".join(cmd)})


@app.post("/api/runs/{run_id}/analyze")
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
