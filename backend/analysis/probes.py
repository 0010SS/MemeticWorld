"""Private meaning / generalization probes (OBSERVER ONLY).

Each probe loads a *copy* of the agent's final memory stream (upstream GA
format, runs/<id>/agents_final/<agent>/associative_memory), retrieves memories
relevant to the expression without updating access times, and asks the agent
what the expression means. Answers are written only to analysis.json: they
never enter any agent memory and are never shown to other agents, and the
simulation that produced the memories has already finished.
"""
from __future__ import annotations

import datetime as dt
import zlib

import numpy as np

from backend import ga_compat
from backend.agents.agent import Agent
from backend.agents.profile import load_population
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


def load_probe_agents(rd, embed) -> dict:
    profiles, _ = load_population(rd.cfg["population"], rd.cfg.get("population_size"))
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


def holdout_options(rng) -> list[tuple[str, str]]:
    """One surface-novel situation per latent family (held-out templates, NPC roles)."""
    opts = []
    for fam in sorted(LE.LATENT_TYPES):
        pool = LE.scenarios_for(fam, True)
        scn = pool[int(rng.integers(len(pool)))]
        subst = {"P": "a student", "S": "a friend", "Q": "another student"}
        for slot, vals in (scn.get("slots") or {}).items():
            subst[slot] = vals[0]
        text = " ".join(LE._fill(f["text"], subst) for b in scn["beats"] for f in b["facts"])
        opts.append((fam, text))
    order = rng.permutation(len(opts))
    return [opts[i] for i in order]


def probe_candidate(cand: dict, agents: dict, llm, heard: dict, seed: int) -> dict:
    expr = cand["canonical_form"]
    out = {}
    rng = np.random.default_rng(seed)
    options = holdout_options(rng)
    letters = "ABCD"
    opt_text = "\n".join(f"{letters[i]}. {t}" for i, (_, t) in enumerate(options))
    for aid in sorted(agents):
        a = agents[aid]
        res = retrieve(a, [expr], k=6, rng=np.random.default_rng([seed, zlib.crc32(aid.encode())]), touch=False)
        mem = "".join(f"- {n.description}\n" for n in merged_nodes(res)) or "- (nothing relevant)\n"
        with llm_scope(f"probe:{cand['id']}:{aid}"):
            p1 = ga.gs.generate_prompt([a.iss(), a.name, mem, expr], str(PDIR / "probe_meaning_v1.txt"))
            with llm_purpose("probe_meaning", aid):
                meaning = llm.complete(p1, max_tokens=120, temperature=0).strip()
            p2 = ga.gs.generate_prompt([a.iss(), a.name, mem, expr, opt_text], str(PDIR / "probe_match_v1.txt"))
            with llm_purpose("probe_match", aid):
                ans = llm.complete(p2, max_tokens=5, temperature=0).strip().upper()
        letter = next((ch for ch in ans if ch in letters), None)
        fam = options[letters.index(letter)][0] if letter else None
        out[aid] = {"meaning": meaning, "heard_before": aid in heard.get("heard", set()),
                    "used": aid in heard.get("used", set()), "match_choice": letter, "match_family": fam}
    return {"agents": out, "options": [{"letter": letters[i], "family": f, "text": t} for i, (f, t) in enumerate(options)]}
