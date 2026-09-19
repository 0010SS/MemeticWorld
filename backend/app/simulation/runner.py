"""Create and execute runs. Shared by the API (background tasks) and the CLI."""
from __future__ import annotations

import logging
import traceback
from datetime import datetime, timezone

from app.config import settings
from app.db import database as db
from app.llm.client import LLMClient, LLMError
from app.simulation.scheduler import Clock
from app.simulation.world import SimConfig, Simulation, run_config_snapshot

log = logging.getLogger("memeticworld")

# Simulations running (or queued to run) in *this* process, so the API can stop them.
ACTIVE: dict[int, Simulation] = {}
QUEUED: set[int] = set()


def create_run(conn, cfg: SimConfig, name: str | None = None) -> int:
    label = "control" if cfg.condition == "no_speech_memory" else "full"
    return db.create_run(
        conn, name=name or f"{label} · seed {cfg.seed} · {cfg.days}d", condition=cfg.condition, seed=cfg.seed,
        days=cfg.days, total_ticks=Clock(cfg.tick_minutes).total_ticks(cfg.days), model=settings.model_label,
        config=run_config_snapshot(cfg))


async def execute_run(conn, run_id: int) -> str:
    run = db.get_run(conn, run_id)
    if run is None:
        raise ValueError(f"run {run_id} not found")
    cfg = SimConfig(**run["config"]["sim"])
    llm = LLMClient(conn, cache_salt=f"seed={cfg.seed}")
    sim = Simulation(conn, run_id, cfg, llm)
    ACTIVE[run_id] = sim
    try:
        try:
            await llm.embed(["connectivity check"])
        except LLMError as e:
            raise LLMError(f"Embedding API failed ({e}). If your provider has no embeddings endpoint, "
                           "set EMBED_PROVIDER=hash.") from e
        return await sim.run()
    except Exception as e:  # keep the server alive; surface the error on the run
        log.error("run %s failed: %s", run_id, traceback.format_exc())
        db.update_run(conn, run_id, status="failed", error=f"{type(e).__name__}: {e}", stats=llm.stats)
        return "failed"
    finally:
        ACTIVE.pop(run_id, None)
        await llm.aclose()


async def execute_runs(conn, run_ids: list[int]) -> None:
    """Run several runs back to back (e.g. a full run and its control) to avoid doubling API load."""
    QUEUED.update(run_ids)
    try:
        for run_id in run_ids:
            if run_id not in QUEUED:  # stopped while waiting
                continue
            QUEUED.discard(run_id)
            await execute_run(conn, run_id)
    finally:
        QUEUED.difference_update(run_ids)


def request_stop(conn, run_id: int) -> bool:
    if run_id in QUEUED:
        QUEUED.discard(run_id)
        db.update_run(conn, run_id, status="stopped")
        return True
    sim = ACTIVE.get(run_id)
    if sim is None:
        return False
    sim.stop_requested = True
    return True


def is_live(run: dict) -> bool:
    """Running here, queued here, or heartbeating from another process (e.g. the CLI)."""
    if run["id"] in ACTIVE or run["id"] in QUEUED:
        return True
    if run["status"] not in ("running", "pending"):
        return False
    age = datetime.now(timezone.utc) - datetime.fromisoformat(run["updated_at"])
    return age.total_seconds() < 180
