"""REST + Server-Sent Events API. Everything is under /api."""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.analysis import meme_detector, pipeline, propagation, semantics
from app.config import settings
from app.db import database as db
from app.llm.client import LLMClient
from app.simulation import runner
from app.simulation.campus import world_layout
from app.simulation.personas import PERSONAS
from app.simulation.world import CONDITIONS, SimConfig

router = APIRouter(prefix="/api")

# Strong references to background run tasks (asyncio only keeps weak ones).
_tasks: set[asyncio.Task] = set()
# run_id -> (cache key, corpus, memes, control id). Recomputed only when a log grows.
_analysis_cache: dict[int, tuple[tuple, meme_detector.Corpus, list[dict], int | None]] = {}


def conn():
    return db.get_conn()


def _run_or_404(run_id: int, with_config: bool = True) -> dict:
    run = db.get_run(conn(), run_id, with_config=with_config)
    if run is None:
        raise HTTPException(404, f"run {run_id} not found")
    run["live"] = runner.is_live(run)
    if run["status"] in ("running", "pending") and not run["live"]:
        run["status"] = "interrupted"
    return run


def _analysis(run_id: int) -> tuple[meme_detector.Corpus, list[dict], int | None]:
    """Detected memes for a run, baselined against its same-seed control run when one exists."""
    control_id = pipeline.control_for(conn(), run_id)
    key = (pipeline.last_event_id(conn(), run_id), control_id, pipeline.last_event_id(conn(), control_id))
    cached = _analysis_cache.get(run_id)
    if cached and cached[0] == key:
        return cached[1], cached[2], cached[3]
    corpus, memes, _ = pipeline.analyze_run(conn(), run_id, control_id)
    _analysis_cache[run_id] = (key, corpus, memes, control_id)
    return corpus, memes, control_id


# --- meta -------------------------------------------------------------------------------

@router.get("/health")
async def health():
    return {"ok": True, "llm_provider": settings.llm_provider, "model": settings.model_label,
            "embed_provider": settings.embed_provider}


@router.get("/world")
async def world():
    return {"world": world_layout(), "agents": PERSONAS, "conditions": CONDITIONS,
            "model": settings.model_label, "llm_provider": settings.llm_provider}


# --- runs -------------------------------------------------------------------------------

class RunRequest(BaseModel):
    name: str | None = None
    days: int = Field(2, ge=1, le=14)
    condition: str = "full"
    seed: int = 7
    tick_delay: float = Field(0.0, ge=0, le=10)
    agent_ids: list[str] | None = None
    with_control: bool = False  # also queue a no_speech_memory run with the same seed


@router.get("/runs")
async def list_runs():
    runs = db.list_runs(conn())
    for run in runs:
        run["live"] = runner.is_live(run)
        if run["status"] in ("running", "pending") and not run["live"]:
            run["status"] = "interrupted"
    return runs


@router.post("/runs")
async def create_run(req: RunRequest):
    try:
        cfg = SimConfig(days=req.days, condition=req.condition, seed=req.seed, tick_delay=req.tick_delay,
                        agent_ids=req.agent_ids)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    run_ids = [runner.create_run(conn(), cfg, req.name)]
    if req.with_control and cfg.condition == "full":
        control = SimConfig(days=req.days, condition="no_speech_memory", seed=req.seed, tick_delay=req.tick_delay,
                            agent_ids=req.agent_ids)
        run_ids.append(runner.create_run(conn(), control, f"{req.name} (control)" if req.name else None))
    task = asyncio.create_task(runner.execute_runs(conn(), run_ids))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return {"run_ids": run_ids, "runs": [_run_or_404(i, with_config=False) for i in run_ids]}


@router.get("/runs/{run_id}")
async def get_run(run_id: int):
    return _run_or_404(run_id)


@router.post("/runs/{run_id}/stop")
async def stop_run(run_id: int):
    _run_or_404(run_id, with_config=False)
    if not runner.request_stop(conn(), run_id):
        raise HTTPException(409, "run is not running in this server process")
    return {"ok": True}


@router.delete("/runs/{run_id}")
async def delete_run(run_id: int):
    run = _run_or_404(run_id, with_config=False)
    if run["live"]:
        raise HTTPException(409, "stop the run before deleting it")
    db.delete_run(conn(), run_id)
    _analysis_cache.pop(run_id, None)
    return {"ok": True}


# --- replay log -------------------------------------------------------------------------

@router.get("/runs/{run_id}/events")
async def events(run_id: int, after_id: int = 0, limit: int = 20000):
    return db.fetch_events(conn(), run_id, after_id=after_id, limit=min(limit, 50000))


@router.get("/runs/{run_id}/stream")
async def stream(run_id: int, request: Request, after_id: int = 0):
    """Live event feed. Sends everything after `after_id`, then polls the log until the run ends."""
    _run_or_404(run_id, with_config=False)

    async def generate():
        last, quiet = after_id, 0
        while not await request.is_disconnected():
            batch = db.fetch_events(conn(), run_id, after_id=last, limit=1000)
            for event in batch:
                last = event["id"]
                yield f"id: {event['id']}\ndata: {json.dumps(event)}\n\n"
            if batch:
                quiet = 0
                continue
            run = db.get_run(conn(), run_id, with_config=False)
            if run is None or not runner.is_live(run):
                yield "event: end\ndata: {}\n\n"
                return
            quiet += 1
            if quiet % 30 == 0:
                yield ": keepalive\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --- agent inspector --------------------------------------------------------------------

@router.get("/runs/{run_id}/agents/{agent_id}/memories")
async def memories(run_id: int, agent_id: str, max_tick: int | None = None, limit: int = 60):
    return db.fetch_memories(conn(), run_id, agent_id=agent_id, max_tick=max_tick, limit=min(limit, 1000))


@router.get("/runs/{run_id}/agents/{agent_id}/decisions")
async def decisions(run_id: int, agent_id: str, max_tick: int | None = None, limit: int = 5):
    rows = db.fetch_decisions(conn(), run_id, agent_id, max_tick=max_tick, limit=min(limit, 100))
    for row in rows:
        row["retrieved_memories"] = db.fetch_memories(conn(), run_id, ids=row["retrieved_memory_ids"])
    return rows


# --- analysis ---------------------------------------------------------------------------

@router.get("/runs/{run_id}/memes")
async def memes(run_id: int):
    _run_or_404(run_id, with_config=False)
    corpus, found, control_id = _analysis(run_id)
    return {
        "memes": found,
        "control_run_id": control_id,
        "summary": meme_detector.summarize(found),
        "groups": semantics.referent_groups(corpus, found),
        "curve": propagation.adoption_curve(found),
        "social": propagation.social_graph(corpus),
        "n_utterances": len(corpus.utterances),
    }


@router.get("/runs/{run_id}/memes/{meme_id}")
async def meme_detail(run_id: int, meme_id: int):
    corpus, found, _ = _analysis(run_id)
    meme = next((m for m in found if m["id"] == meme_id), None)
    if meme is None:
        raise HTTPException(404, "meme not found")
    return {**meme, **propagation.cascade(corpus, meme)}


@router.get("/runs/{run_id}/phrase")
async def phrase(run_id: int, q: str):
    """Trace any phrase, detected or not (useful for checking a hunch)."""
    corpus, _, _ = _analysis(run_id)
    key = " ".join(meme_detector.tokenize(q))
    stats = meme_detector.trace_phrase(corpus, key)
    return {**stats, **propagation.cascade(corpus, stats)} if stats["n_uses"] else stats


@router.post("/runs/{run_id}/memes/gloss")
async def gloss(run_id: int, top: int = 8):
    corpus, found, _ = _analysis(run_id)
    llm = LLMClient(conn(), cache_salt="gloss")
    try:
        results = await asyncio.gather(*(semantics.gloss_meme(llm, corpus, m) for m in found[:top]))
    finally:
        await llm.aclose()
    return {m["phrase"]: g for m, g in zip(found[:top], results)}


@router.get("/compare")
async def compare(a: int, b: int):
    _run_or_404(a, with_config=False)
    _run_or_404(b, with_config=False)
    return propagation.compare(_analysis(a)[0], _analysis(b)[0])
