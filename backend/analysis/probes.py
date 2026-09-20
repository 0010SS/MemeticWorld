"""Private meaning / generalization probes (OBSERVER ONLY).

Each probe loads a *copy* of the agent's final memory stream (upstream GA
format, runs/<id>/agents_final/<agent>/associative_memory), retrieves memories
relevant to the expression without updating access times, and asks the agent
what the expression means. Answers are written only to analysis.json: they
never enter any agent memory and are never shown to other agents, and the
simulation that produced the memories has already finished.

SCOPE, so that nobody builds a fourth probe path. This module is the FINAL-MEMORY probe over the mined
candidate pool: one meaning question and one four-way match, on `agents_final`, at the end of a run. The
memo is explicit that final-memory probes "give weak evidence about intermediate meanings", so for a
declared meme cohort use `backend.analysis.battery.graded` instead: it probes checksummed copies of
several checkpoints, asks a graded applicability question with foils, and is driven by the config registry
rather than by whatever the miner happened to surface. `battery.runner` remains the v3 workshop battery.
`graded_probe_ready` says which of the two a given run supports.
"""
from __future__ import annotations

import datetime as dt
import zlib

import numpy as np

from backend import ga_compat
from backend.agents.agent import Agent
from backend.analysis.rundata import simulated_profiles
from backend.llm.client import llm_purpose, llm_scope
from backend.memory.retrieval import merged_nodes, retrieve
from backend.memory.store import MemoryStream
from backend.modules.base import ModuleStack
from backend.simulation import latent_events as LE

ga = ga_compat.load()
PDIR = ga_compat.REPO_ROOT / "backend" / "prompts"


class _Ctx:
    def __init__(self, agents):
        self.agents = agents


def probe_profiles(rd) -> dict:
    """The profiles as simulated (rundata.simulated_profiles: population file, generated topology, planted
    habit, then the manifest's record of each profile), so the probed persona's ISS (lifestyle,
    daily-plan line) is the one the agent had in every prompt."""
    return simulated_profiles(rd.cfg, rd.manifest)


def load_probe_agents(rd, embed) -> dict:
    profiles = probe_profiles(rd)
    end = dt.datetime.fromisoformat(rd.manifest["start"]) + dt.timedelta(
        days=rd.cfg["simulation_days"] - 1, hours=15)
    agents = {}
    for aid, prof in profiles.items():
        a = Agent(prof, rd.cfg, ModuleStack([]), rd.cfg["seed"])
        a.a_mem = MemoryStream.load_ga(aid, rd.dir / "agents_final" / aid / "associative_memory")
        a.set_time(end)
        a.state.activity = "thinking"
        a.sync_scratch()
        agents[aid] = a
    ctx = _Ctx(agents)
    for a in agents.values():
        a.ctx = ctx
    return agents


NODE_ORDER = ("n0", "n0_private", "n1", "n2")
ROLE_FILL = {"P": "a student", "S": "a friend", "Q": "another student"}


def _fill(text: str, subst: dict) -> str:
    out = text
    for k, v in subst.items():
        out = out.replace("{" + k + "}", v)
    return out[:1].upper() + out[1:]


def _v2_options(rng, families: list[str]) -> list[tuple[str, str]]:
    """One held-out skin per family (v2 skins: held-out topics and wording), NPC roles, first slot values."""
    from backend.simulation.referents import CATEGORIES
    from backend.simulation.skins import load_skins
    skins = [s for s in load_skins() if s.get("holdout")]
    opts = []
    for fam in families:
        pool = sorted((s for s in skins if s["family"] == fam), key=lambda s: s["key"])
        if not pool:
            return []
        sk = pool[int(rng.integers(len(pool)))]
        subst = dict(ROLE_FILL, R=CATEGORIES[sk["domain"]][0])
        subst.update({k: v[0] for k, v in (sk.get("slots") or {}).items()})
        opts.append((fam, " ".join(_fill(sk["facts"][n], subst) for n in NODE_ORDER if n in sk["facts"])))
    return opts


def _v1_options(rng, families: list[str]) -> list[tuple[str, str]]:
    opts = []
    for fam in families:
        pool = LE.scenarios_for(fam, True)
        scn = pool[int(rng.integers(len(pool)))]
        subst = dict(ROLE_FILL)
        for slot, vals in (scn.get("slots") or {}).items():
            subst[slot] = vals[0]
        opts.append((fam, " ".join(LE._fill(f["text"], subst) for b in scn["beats"] for f in b["facts"])))
    return opts


def holdout_options(rng, families: list[str] | None = None, v2: bool = True) -> list[tuple[str, str]]:
    """One surface-novel situation per latent family (held-out skins / templates, NPC roles), shuffled.
    v2 runs are probed with v2 skins, v1 runs with the v1 held-out scenario templates."""
    families = sorted(families or ["E1", "E2", "E3", "E4"])
    opts = []
    for build in ((_v2_options, _v1_options) if v2 else (_v1_options,)):
        try:
            opts = build(rng, families)
        except Exception:        # skins or legacy scenarios unavailable
            opts = []
        if opts:
            break
    order = rng.permutation(len(opts))
    return [opts[i] for i in order]


def probe_candidate(cand: dict, agents: dict, llm, heard: dict, seed: int, families: list[str] | None = None,
                    v2: bool = True) -> dict:
    expr = cand["canonical_form"]
    out = {}
    rng = np.random.default_rng(seed)
    options = holdout_options(rng, families, v2)
    letters = "ABCDEFGH"[:len(options)]
    opt_text = "\n".join(f"{letters[i]}. {t}" for i, (_, t) in enumerate(options))
    for aid in sorted(agents):
        a = agents[aid]
        res = retrieve(a, [expr], k=6, rng=np.random.default_rng([seed, zlib.crc32(aid.encode())]), touch=False)
        mem = "".join(f"- {n.description}\n" for n in merged_nodes(res)) or "- (nothing relevant)\n"
        with llm_scope(f"probe:{cand['id']}:{aid}"):
            p1 = ga.gs.generate_prompt([a.iss(), a.name, mem, expr], str(PDIR / "probe_meaning_v1.txt"))
            with llm_purpose("probe_meaning", aid):
                meaning = llm.complete(p1, max_tokens=120, temperature=0).strip()
            ans = ""
            if options:
                p2 = ga.gs.generate_prompt([a.iss(), a.name, mem, expr, opt_text], str(PDIR / "probe_match_v1.txt"))
                with llm_purpose("probe_match", aid):
                    ans = llm.complete(p2, max_tokens=5, temperature=0).strip().upper()
        letter = next((ch for ch in ans if ch in letters), None)
        fam = options[letters.index(letter)][0] if letter else None
        out[aid] = {"meaning": meaning, "heard_before": aid in heard.get("heard", set()),
                    "used": aid in heard.get("used", set()), "match_choice": letter, "match_family": fam}
    return {"agents": out, "options": [{"letter": letters[i], "family": f, "text": t} for i, (f, t) in enumerate(options)]}


def graded_probe_ready(run_dir) -> dict:
    """Whether a run can be probed with the graded meme battery, and what it is missing if not.

    Temporal checkpoints are the part most often absent: without `checkpoints.enabled` a run has only
    `agents_final`, so the battery can measure where a boundary ENDED but not that it MOVED - and movement
    is the finding."""
    from pathlib import Path as _P

    from backend.analysis.battery import registry as REG
    from backend.analysis.battery.graded import available_checkpoints
    run_dir = _P(run_dir)
    specs = REG.load_registry(run_dir, only_enabled=False)
    cks = sorted(p.name for p in (run_dir / "checkpoints").glob("C*")) if (run_dir / "checkpoints").is_dir() else []
    missing = []
    if not specs:
        missing.append("no memes.registry in the run config")
    if not cks:
        missing.append("no checkpoints/: only the final memory state can be probed, so boundary MOVEMENT "
                       "is unmeasurable (set checkpoints.enabled)")
    return {"registry": [s.id for s in specs], "registry_errors": REG.validate(specs),
            "checkpoints_on_disk": cks, "final_available": (run_dir / "agents_final").is_dir(),
            "probed": available_checkpoints(run_dir), "missing": missing, "ready": not missing}
