"""Research interface: experiments, saved interpretations, evidence and background jobs."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.config import REPO_ROOT
from backend.research import experiments as E, jobs
from backend.research.common import digest, read_json
from backend.research.evidence import Evidence

router = APIRouter(prefix="/api/research")


def runs_root():
    from backend.api.server import RUNS
    return RUNS


def run_path(run):
    from backend.api.server import _run_dir
    return _run_dir(run)


def experiment(name):
    choices = {e["id"] for e in E.catalog()}
    if name not in choices:
        p = (REPO_ROOT / name).resolve()
        if (REPO_ROOT / "configs").resolve() not in p.parents or p.suffix != ".yaml" or not p.exists():
            raise HTTPException(400, "Use a catalog ID or a design YAML inside configs")
        name = str(p)
    return E.resolve(name, str(runs_root()))


@router.get("/experiments")
def experiments():
    result = []
    for entry in E.catalog():
        design = E.resolve(entry["id"], str(runs_root()))
        cells = E.D.expand(design)
        result.append({**entry, "runs": len(cells), "seeds": design.seeds, "replicates": design.replicates,
                       "factors": design.factors, "days": design.days or cells[0].config()["simulation_days"],
                       "observer": design.observer})
    return result


@router.get("/experiment")
def experiment_state(name: str, backend: str | None = None):
    d = experiment(name)
    root = E.D.design_dir(d, backend)
    return {"name": d.name, "runs": E.D.status(d, backend), "questions": d.questions,
            "pipeline": read_json(root / "pipeline.json"), "report": read_json(root / "report.json"),
            "directory": root.relative_to(runs_root()).as_posix()}


@router.get("/analysis")
def analysis(run: str, analysis_id: str | None = None):
    root = run_path(run)
    if analysis_id and (not analysis_id.startswith("a_") or not analysis_id.replace("_", "").isalnum()):
        raise HTTPException(400, "Invalid analysis ID")
    path = root / "analyses" / analysis_id / "analysis.json" if analysis_id else root / "analysis.json"
    result = read_json(path)
    if not result or result.get("kind") != "memetics":
        raise HTTPException(404, "No memetics observation yet; select Observe in Research")
    return result


@router.get("/snapshots")
def snapshots(run: str):
    return [read_json(p) for p in sorted((run_path(run) / "analyses").glob("*/manifest.json"))]


@router.get("/evidence")
def evidence(run: str, event_id: str | None = None, q: str = "", start: int | None = None,
             end: int | None = None, agent: str | None = None, limit: int = Query(100, ge=1, le=1000)):
    index = Evidence(run_path(run))
    try:
        index.build()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    scope = {"start": start, "end": end, "actor": agent, "channels": ["speech", "record", "private", "environment", "initial"]}
    rows = index.events(ids=[event_id], **scope) if event_id else index.search(q, limit, **scope)
    return [{k: v for k, v in e.items() if k != "raw"} for e in rows[:limit]]


@router.get("/inquiries")
def inquiries(run: str):
    return [read_json(p) for p in sorted((run_path(run) / "inquiries").glob("*/results.json"))]


@router.get("/audits")
def audits(run: str):
    return [read_json(p) for p in sorted((run_path(run) / "audits").glob("*/results.json"))]


@router.get("/artifact")
@router.get("/files/{path:path}")
def artifact(path: str, download: bool = False):
    root = runs_root().resolve()
    target = (root / path).resolve()
    allowed = {"report.html", "report.md", "analysis.json", "results.json", "evidence.json", "occurrences.csv", "runs.csv"}
    if (root not in target.parents or not target.is_file() or
            (target.name not in allowed and target.suffix != ".svg")):
        raise HTTPException(404, "No research artifact at this path")
    return FileResponse(target, filename=target.name if download else None)


class JobRequest(BaseModel):
    action: Literal["experiment", "observe", "inquire", "continue", "audit", "report", "synthesize"]
    run: str | None = None
    experiment: str | None = None
    backend: Literal["mock", "openai", "codex_cli", "claude_cli", "anthropic"] | None = None
    model: str | None = None
    question: str | None = None
    parallel: int = Field(default=1, ge=1, le=64)
    days: int | None = Field(default=None, ge=1, le=10000)
    out: str | None = None
    start: int | None = None
    end: int | None = None
    agent: str | None = None
    public_only: bool = False
    limit: int = Field(default=100, ge=1, le=1000)
    no_questions: bool = False
    fresh: bool = False
    population: str | None = None
    population_size: int | None = Field(default=None, ge=1, le=10000)
    background_markdown: str | None = Field(default=None, max_length=100000)
    agent_model: str | None = None
    observer_model: str | None = None


def configured_experiment(request):
    if not request.experiment:
        raise HTTPException(400, "Choose an experiment")
    population = request.population
    if population:
        p = (REPO_ROOT / population).resolve()
        if (REPO_ROOT / "configs" / "population").resolve() not in p.parents or p.suffix != ".yaml" or not p.is_file():
            raise HTTPException(400, "Choose a population YAML inside configs/population")
    background = None
    if request.background_markdown is not None:
        if not request.background_markdown.strip():
            raise HTTPException(400, "Shared background must contain nonempty Markdown")
        p = runs_root() / ".backgrounds" / (digest(request.background_markdown) + ".md")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(request.background_markdown, encoding="utf-8")
        background = str(p)
    try:
        return E.configure(experiment(request.experiment), backend=request.backend, population=population,
                           population_size=request.population_size, days=request.days, background=background,
                           agent_model=request.agent_model, observer_model=request.observer_model)
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/experiment-preview")
def preview(request: JobRequest):
    d = configured_experiment(request)
    root = E.D.design_dir(d, request.backend)
    cells = E.D.expand(d, request.backend)
    cfg = cells[0].config()
    return {"name": d.name, "runs": E.D.status(d, cells=cells), "questions": d.questions,
            "pipeline": read_json(root / "pipeline.json"), "report": read_json(root / "report.json"),
            "directory": root.relative_to(runs_root()).as_posix(),
            "population": cfg["population"], "population_size": cfg["population_size"],
            "active_hours": f"{cfg['day_start']}–{cfg['day_end']}",
            "initial_memories_file": cfg.get("initial_memories_file"),
            "shared_background": any(bool(c.config().get("shared_background", {}).get("markdown")) for c in cells),
            "days": sorted({c.config()["simulation_days"] for c in cells}),
            "agent_model": cfg["llm"]["model"], "observer": cfg["analysis"]["observer"]}


@router.post("/jobs")
def launch(request: JobRequest):
    args = []
    if request.action in ("experiment", "report", "synthesize"):
        if not request.experiment:
            raise HTTPException(400, "Choose an experiment")
        d = configured_experiment(request)
        action = "run" if request.action == "experiment" else request.action
        args = ["experiment", action, str(d.path), "--runs-root", str(runs_root())]
        if action == "run":
            args += ["--parallel", str(request.parallel)]
            if request.no_questions:
                args.append("--no-questions")
        if action == "synthesize":
            if not request.question or not request.question.strip():
                raise HTTPException(400, "Enter a research question")
            args.append(request.question)
    else:
        if not request.run:
            raise HTTPException(400, "Choose a run")
        root = run_path(request.run)
        args = [request.action, str(root)]
        if request.action == "observe" and request.fresh:
            args.append("--fresh")
        if request.action == "inquire":
            if not request.question or not request.question.strip():
                raise HTTPException(400, "Enter a research question")
            args += [request.question, "--limit", str(request.limit)]
            for key in ("start", "end", "agent"):
                value = getattr(request, key)
                if value is not None:
                    args += ["--" + key, str(value)]
            if request.public_only:
                args.append("--public-only")
        if request.action == "continue":
            if not request.out:
                raise HTTPException(400, "Name a new output directory")
            out = (runs_root() / request.out).resolve()
            if runs_root() not in out.parents or out.exists():
                raise HTTPException(400, "Use a new directory inside runs")
            args += ["--out", str(out), "--analyze"]
            if request.days is not None:
                args += ["--days", str(request.days)]
    if request.backend:
        args += ["--backend", request.backend]
    if request.model and request.action not in ("experiment", "report", "synthesize"):
        args += ["--model", request.model]
    return jobs.launch(runs_root(), args, request.action)


@router.get("/jobs")
def job_list():
    paths = sorted((runs_root() / ".research_jobs").glob("*/job.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [jobs.status(p.parent) for p in paths[:50]]


@router.post("/pause")
def pause(run: str):
    from backend.experiment.execution import request_pause
    try:
        return request_pause(run_path(run))
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
