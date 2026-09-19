"""Work cognition at the co-op's laser cutter (ontology v3 §4.2-4.4).

* ``decide_job``: one operator decision for one attempt of one job (prompt ``job_decision_v1.txt``). The menu
  shows the six actions in the job's pre-drawn order for this attempt, then "ask {first} something first"
  (choice G, at most ``comm.clarify.max_per_job`` per job, only with someone present) and, only under
  ``records.consult: on_choice``, "look through the binder first".
* ``job_episode`` / ``encode_job``: one perception episode per perceiver per job (``workshop.encode: per_job``),
  with the operator's own reasons as ``inner`` facts when ``workshop.encode_reason``.

Agents get no new knowledge fields: ``JobView`` is a transient prompt bundle, discarded after the decision.
Nothing rendered here names a class, cause, regime, mapping, job id or arm; job ids live only in trace fields
and in ``event_ids`` (SIM-ONLY). Nothing asks for names or labels.

Randomness: ``decide_job`` draws only from the rng it is given (retrieval takes exactly one integer per call);
the integrator passes ``agent.stream("work")``.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field

from backend import ga_compat
from backend.agents.ga_prompts import as_json
from backend.agents.perception import AgentObservation
from backend.llm.client import llm_purpose
from backend.memory.retrieval import merged_nodes, retrieve

ga = ga_compat.load()
PROMPT = str(ga_compat.REPO_ROOT / "backend" / "prompts" / "job_decision_v1.txt")
LETTERS = "ABCDEFGHIJKL"
K_MEMORIES = 5
MACHINE_FOCAL = "the co-op's laser cutter"
SHIFT_WORD = {"am": "morning", "pm": "afternoon", "morning": "morning", "afternoon": "afternoon"}

# laser_alpha action texts (ontology v3 §1.2). The content pack (W3) is the owner; pass ``menu_text`` (or
# JobView.menu_text) to use another pack's wording (laser_beta).
DEFAULT_MENU_TEXT = {
    "rerun": "re-run the sheet as it is",
    "lens": "clean the focus lens, then re-run",
    "dry": "dry the sheets on the heated rack for 15 minutes, then re-run",
    "belt": "tighten the drive belt, then re-run",
    "slow": "slow the cutting speed down, then re-run",
    "stop": "stop and leave the job for later",
}
BINDER_OPTION = "look through the binder first"
ASK_OPTION = "ask {first} something first"
NULLISH = {"", "null", "none", "n/a", "nothing"}


@dataclass
class JobView:
    """Transient prompt bundle for one decision (ontology v3 §4.1). Never stored on the agent."""
    job_id: str                                   # SIM-ONLY: traced, never rendered
    attempt: int                                  # 1-based
    menu_order: list                              # action ids, the job's pre-drawn order for this attempt
    project: str                                  # "24 name plates for the robotics team's open house"
    facts: list = field(default_factory=list)     # own perceived job facts (str or {"text": ...}), oldest first
    present: list = field(default_factory=list)   # Agent objects in the Makerspace besides the operator
    shift: str = "am"                             # am | pm
    questions_asked: int = 0                      # clarifications already asked on this job
    focal: str | None = None                      # perceived symptom text (retrieval focal point)
    day: int | None = None                        # simulation day (for records.consult_from_day)
    menu_text: dict | None = None                 # action id -> wording (default: laser_alpha)


def make_job_view(job: dict, attempt: int, facts: list, present: list, *, questions_asked: int = 0,
                  focal: str | None = None, menu_text: dict | None = None) -> JobView:
    """Build a JobView from a workshop JobInstance record (ontology v3 §1.8)."""
    orders = job.get("menu_order") or [list(DEFAULT_MENU_TEXT)]
    order = orders[min(attempt, len(orders)) - 1]
    return JobView(job_id=job["id"], attempt=attempt, menu_order=list(order), project=job.get("project", "a job"),
                   facts=list(facts), present=list(present), shift=job.get("shift", "am"),
                   questions_asked=questions_asked, focal=focal, day=job.get("day"), menu_text=menu_text)


def _as_view(jv) -> JobView:
    return jv if isinstance(jv, JobView) else JobView(**jv)


def _fact_text(f) -> str:
    return f if isinstance(f, str) else str(f.get("text", ""))


def binder_parts(binder_view) -> tuple[str | None, list]:
    """(text, entry/revision ids) from a rendered binder view: a str, a dict with ``text`` and ``entry_ids`` /
    ``rev_ids``, or an object with those attributes (records.py's view)."""
    if binder_view is None:
        return None, []
    if isinstance(binder_view, str):
        return binder_view, []
    get = binder_view.get if isinstance(binder_view, dict) else (lambda k, d=None: getattr(binder_view, k, d))
    ids = list(get("entry_ids", None) or []) + list(get("rev_ids", None) or [])
    return get("text", None), ids


def consult_mode(cfg: dict, day: int | None) -> str:
    """records.consult in force on ``day``: never when records are off; the configured mode from
    records.consult_from_day on, "always" (study-1 default) before it."""
    rc = cfg.get("records") or {}
    if not rc.get("enabled", False):
        return "never"
    mode = rc.get("consult", "always")
    frm = rc.get("consult_from_day")
    if frm is not None and day is not None and day < int(frm):
        return "always"
    return mode


def ask_allowed(cfg: dict, jv: JobView) -> bool:
    cc = (cfg.get("comm") or {}).get("clarify") or {}
    return bool(cc.get("enabled", False)) and bool(jv.present) and \
        jv.questions_asked < int(cc.get("max_per_job", 1))


def build_options(jv: JobView, *, ask: bool, binder_option: bool, menu_text: dict | None = None) -> list[dict]:
    """Lettered options: the six actions in the pre-drawn order, then one ask option per person present,
    then (on_choice only) the binder option."""
    texts = {**DEFAULT_MENU_TEXT, **(jv.menu_text or {}), **(menu_text or {})}
    opts = [{"kind": "action", "action": a, "text": texts.get(a, a)} for a in jv.menu_order]
    if ask:
        for p in sorted(jv.present, key=lambda x: x.id):
            opts.append({"kind": "ask", "action": "ask", "target": p.id,
                         "text": ASK_OPTION.format(first=p.profile.first_name)})
    if binder_option:
        opts.append({"kind": "binder", "action": "binder", "text": BINDER_OPTION})
    for i, o in enumerate(opts):
        o["letter"] = LETTERS[i]
    return opts


def _clean(v) -> str | None:
    if not isinstance(v, str):
        return None
    v = " ".join(v.split()).strip().strip('"').strip()
    return None if v.lower() in NULLISH else v


def parse_choice(raw: str, options: list[dict]) -> dict | None:
    """The chosen option (+ question/says_aloud/reason), or None when unparseable / invalid."""
    if not raw or raw.startswith("LLM_ERROR"):
        return None
    d = as_json(raw)
    if not isinstance(d, dict):
        return None
    c = str(d.get("choice") or "").strip()
    by_letter = {o["letter"]: o for o in options}
    opt = None
    m = re.match(r"^[\(\[]?([A-La-l])(?:[\)\]\.:,]|\s|$)", c)
    if m:
        opt = by_letter.get(m.group(1).upper())
    if opt is None:
        low = c.lower().rstrip(".")
        opt = next((o for o in options if low and (low == o["text"].lower() or
                                                    (o["kind"] == "action" and low == o["action"]))), None)
    if opt is None:
        return None
    question = _clean(d.get("question"))
    if opt["kind"] == "ask" and not question:
        return None                    # an ask with nothing asked is not a usable answer
    return {"option": opt, "question": question if opt["kind"] == "ask" else None,
            "says_aloud": _clean(d.get("says_aloud")), "reason": _clean(d.get("reason"))}


def _render_prompt(agent, jv: JobView, options, binder_text, memories, present) -> str:
    first = agent.profile.first_name
    facts = "".join(f"- {_fact_text(f)}\n" for f in jv.facts if _fact_text(f)) or "- nothing out of the ordinary yet\n"
    binder = f"{binder_text.rstrip()}\n" if binder_text else ""
    mem = "".join(f"- {n.description}\n" for n in memories) or "- (nothing in particular)\n"
    people = ", ".join(p.name for p in present) or "nobody"
    lines = agent.mods.modify_prompt(agent, "work", [agent.relationship_line(p) for p in present])
    menu = "\n".join(f"{o['letter']}. {o['text']}" for o in options)
    ask_letters = [o["letter"] for o in options if o["kind"] == "ask"]
    if ask_letters:
        qhint = f'"<for {" or ".join(ask_letters)} only: exactly what {first} asks, else null>"'
    else:
        qhint = "null"
    return ga.gs.generate_prompt(
        [agent.iss(), agent.scratch.curr_time.strftime("%A %H:%M"), first, SHIFT_WORD.get(jv.shift, jv.shift),
         jv.project, facts.rstrip("\n"), binder, mem.rstrip("\n"), people, "\n".join(lines), menu, qhint], PROMPT)


def _ask_llm(agent, prompt: str, options: list[dict]) -> tuple[dict | None, list[str]]:
    """One call plus one retry on an unparseable reply (ontology v3 §4.3)."""
    raws = []
    for _ in range(2):
        with llm_purpose("job_decision", agent.id):
            raw = agent.ctx.llm.complete(prompt, max_tokens=200, temperature=0.8)
        raws.append(raw)
        parsed = parse_choice(raw, options)
        if parsed is not None:
            return parsed, raws
    return None, raws


def decide_job(agent, job_view, binder_view=None, rng=None) -> dict:
    """One job decision. Returns
    ``{action, choice, reason, ask, says_aloud, valid, retrieved, binder_shown, binder_view_sha,
       binder_entry_ids, prompt, response}`` where ``action`` is one of the six action ids, or ``"ask"`` (then
    ``ask = {"target": agent_id, "question": str}``: run it as a forced TALK, coop_talk.clarify_talk, and call
    decide_job again at t0+1 with ``questions_asked`` incremented). An unparseable reply is retried once, then
    counts as ``stop`` with ``valid = False`` (traced; excluded from accuracy)."""
    jv = _as_view(job_view)
    cfg = agent.cfg
    rng = rng if rng is not None else agent.stream("work")
    present = [p for p in jv.present if p.id != agent.id]
    jv.present = present
    mode = consult_mode(cfg, jv.day)
    btext, bids = binder_parts(binder_view)
    show_binder = btext is not None and mode == "always"
    offer_binder = btext is not None and mode == "on_choice"

    focal_text = jv.focal or " ".join(_fact_text(f) for f in jv.facts[:2]) or jv.project
    focal = [focal_text, MACHINE_FOCAL]
    per = max(1, math.ceil(K_MEMORIES / len(focal)))
    res = retrieve(agent, focal, k=per, rng=rng)
    nodes = merged_nodes(res, limit=K_MEMORIES)

    ask = ask_allowed(cfg, jv)
    options = build_options(jv, ask=ask, binder_option=offer_binder)
    prompt = _render_prompt(agent, jv, options, btext if show_binder else None, nodes, present)
    parsed, raws = _ask_llm(agent, prompt, options)
    prompts = [prompt]
    binder_chosen = False
    if parsed is not None and parsed["option"]["kind"] == "binder":
        # on_choice: re-ask with the view shown and the binder option gone
        binder_chosen, show_binder = True, True
        options = build_options(jv, ask=ask, binder_option=False)
        prompt = _render_prompt(agent, jv, options, btext, nodes, present)
        prompts.append(prompt)
        parsed, more = _ask_llm(agent, prompt, options)
        raws += more

    valid = parsed is not None
    if valid:
        opt = parsed["option"]
        action = opt["action"]
        choice = opt["letter"]
        askd = {"target": opt["target"], "question": parsed["question"]} if opt["kind"] == "ask" else None
        says, reason = parsed["says_aloud"], parsed["reason"]
    else:
        action, choice, askd, says, reason = "stop", None, None, None, None
    sha = hashlib.sha256(btext.encode()).hexdigest()[:16] if (show_binder and btext is not None) else None
    out = {"action": action, "choice": choice, "reason": reason, "ask": askd, "says_aloud": says, "valid": valid,
           "retrieved": [n.node_id for n in nodes], "binder_shown": show_binder, "binder_chosen": binder_chosen,
           "binder_view_sha": sha, "binder_entry_ids": bids if show_binder else [],
           "prompt": prompt, "response": raws[-1] if raws else None}
    agent.ctx.tracer.log(
        "job_decision", agent=agent.id, job=jv.job_id, attempt=jv.attempt, menu_order=list(jv.menu_order),
        choice=choice, action=action, question=askd["question"] if askd else None,
        ask_target=askd["target"] if askd else None, says_aloud=says, reason=reason, valid=valid,
        n_calls=len(raws), consult=mode, binder_shown=show_binder, binder_chosen=binder_chosen,
        options=[{"letter": o["letter"], "kind": o["kind"], "action": o["action"], "target": o.get("target")}
                 for o in options],
        retrieved=out["retrieved"], retrieval_scores={k: v for r in res.values() for k, v in r.scores.items()},
        retrieved_event_ids=agent.ctx.meta.events_of(out["retrieved"]),
        binder_view_sha=sha, binder_entry_ids=out["binder_entry_ids"], prompt=prompts if len(prompts) > 1 else prompt,
        response=raws if len(raws) > 1 else (raws[0] if raws else None))
    return out


def remark_of(decision: dict) -> dict | None:
    """A says_aloud line as a v2 REACT decision, for the engine's remark path (same listener sampling and
    exposures as REACT remarks)."""
    if not decision.get("says_aloud"):
        return None
    return {"action": "REACT", "target": None, "utterance": decision["says_aloud"], "reason": decision.get("reason"),
            "retrieved": list(decision.get("retrieved") or [])}


# --------------------------------------------------------------------------- job episodes (§4.2)
def inner_fact(agent, job_id: str, reason: str, i: int = 0) -> dict:
    text = f"{agent.profile.first_name} had thought: {reason}"
    return {"id": f"{job_id}.inner.{agent.id}.{i}", "text": text, "world_text": text, "salience": 0.5,
            "involves": [agent.id], "visibility": "all", "kind": "inner", "vantage": "participant",
            "recognised": [], "rendered": True}


def job_episode(agent, job_id: str, facts: list, *, tick: int, reasons=(), obs_id: str | None = None):
    """This perceiver's noticed facts for one job as ONE perception observation (``event_ids=[job_id]``), plus
    the operator's reasons as inner facts when ``workshop.encode_reason`` (default true). Facts keep their
    fields; ones already viewpoint-rendered should carry ``rendered: True``. None if nothing was noticed."""
    fs = [dict(f) for f in facts if f and f.get("text")]
    if (agent.cfg.get("workshop") or {}).get("encode_reason", True):
        fs += [inner_fact(agent, job_id, r, i) for i, r in enumerate(x for x in reasons if _clean(x))]
    if not fs:
        return None
    return AgentObservation(id=obs_id or f"{job_id}.ep.{agent.id}", agent_id=agent.id, tick=tick,
                            location=agent.state.location, arena=agent.state.arena, source_type="perception",
                            facts=fs, event_ids=[job_id])


def render_pending(agent, obs) -> None:
    """Viewpoint-render the facts not rendered yet (in place, order kept), then mark all as rendered."""
    from backend.agents import viewpoint
    todo = [f for f in obs.facts if not f.get("rendered")]
    if todo and agent.cfg["perception"].get("viewpoints", True):
        tmp = AgentObservation(id=obs.id, agent_id=obs.agent_id, tick=obs.tick, location=obs.location,
                               arena=obs.arena, source_type=obs.source_type,
                               facts=[dict(f, world_text=f.get("world_text", f["text"])) for f in todo],
                               event_ids=list(obs.event_ids))
        viewpoint.render(agent, tmp)
        done = {f["id"]: f for f in tmp.facts}
        obs.facts[:] = [done[f["id"]] if f["id"] in done else f for f in obs.facts
                        if f.get("rendered") or f["id"] in done]
    for f in obs.facts:
        f["rendered"] = True


def encode_job(agent, job_record: dict, outcome: dict, rng=None, *, remind: bool = True):
    """Encode one job episode now (helper; the engine may instead queue ``job_episode(...)`` into pending_obs
    for next tick's phase 4). ``outcome = {"facts": [this agent's noticed job facts], "reasons": [the operator's
    reasons] (operator only), "tick": int}``. Renders unrendered facts, encodes with the v2 encoder and runs
    reminding. Returns (observation, node) or (None, None)."""
    from backend.memory.encoder import encode
    from backend.memory.reminding import maybe_remind
    tick = outcome.get("tick", getattr(agent.ctx.tracer, "tick", 0))
    obs = job_episode(agent, job_record["id"], outcome.get("facts") or [], tick=tick,
                      reasons=outcome.get("reasons") or ())
    if obs is None:
        return None, None
    render_pending(agent, obs)
    if not obs.facts:
        return obs, None
    agent.ctx.tracer.log("observation", agent=agent.id, observation_id=obs.id, source_type=obs.source_type,
                         facts=obs.facts, originating_event_ids=obs.event_ids, episode="job")
    agent.mods.on_observation(agent, obs.facts)
    node = encode(agent, obs, rng if rng is not None else agent.stream("encode"))
    if remind:
        maybe_remind(agent, obs, node, agent.stream("remind"))
    return obs, node
