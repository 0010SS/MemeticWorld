"""RecordWrite (ontology v3 §2.4): does a member write in the co-op binder, and what?

Offers (the engine decides when; `records.offer_enabled`):
  job      - to the operator after every faulted job's final outcome;
  tally    - to the rostered stores member at the tally;
  farewell - to a departing member at the end of their last shift.
Writing is never forced. `decide_write` only asks (one LLM call, prompt record_write_v1.txt) and returns a
decision; nothing is written until the engine applies all of a tick's decisions in sorted (tick, agent) order
with `records.apply_writes` (phase 4c). After that, `remember_write` gives the author a no-LLM self memory.

No wording asks for tips, rules, procedures or names; lengths are given only in characters.
"""
from __future__ import annotations

from backend import ga_compat
from backend.agents.ga_prompts import as_json
from backend.llm.client import llm_purpose
from backend.memory.store import MemoryMeta
from backend.simulation import records as R

ga = ga_compat.load()
PROMPT = str(ga_compat.REPO_ROOT / "backend" / "prompts" / "record_write_v1.txt")
PURPOSE = "record_write"
CHOICES = ("none", "log", "front")
SELF_IMPORTANCE = 4
FAREWELL_FOCAL = "the co-op's laser cutter"


def situation(offer: str, first: str) -> str:
    return {"job": f"{first} just finished a job on the co-op's laser cutter.",
            "tally": f"{first} just went over today's jobs at the laser.",
            "farewell": f"It is {first}'s last shift at the co-op."}[offer]


def options(cfg: dict, first: str) -> tuple[str, str]:
    """(options sentence, allowed choice values) for the prompt; the front page is offered only if revisable."""
    rc = R.rcfg(cfg)
    log_n, front_n = int(rc["log_max_chars"]), int(rc["front_max_chars"])
    if rc["revisable"]:
        return (f"{first} can leave the binder as it is, add a note to the log (up to {log_n} characters), or "
                f"rewrite the whole front page (up to {front_n} characters).", '"none" | "log" | "front"')
    return (f"{first} can leave the binder as it is or add a note to the log (up to {log_n} characters).",
            '"none" | "log"')


def remembered_lines(agent, remembered, rng=None, k: int = 3) -> tuple[str, list[str]]:
    """What the author remembers of it. `remembered`: list of lines supplied by the caller (the operator's own
    perceived job facts, attempts and outcomes; the tally facts), or None -> k retrieved memories about the
    laser (farewell). -> (text block, retrieved node ids)"""
    if remembered is not None:
        lines = [str(x) for x in remembered if str(x).strip()]
        return ("\n".join(f"- {x}" for x in lines) or "- (nothing in particular)"), []
    from backend.memory.retrieval import merged_nodes, retrieve
    res = retrieve(agent, [FAREWELL_FOCAL], k=k, rng=rng if rng is not None else agent.stream("record"))
    nodes = merged_nodes(res)[:k]
    return ("\n".join(f"- {n.description}" for n in nodes) or "- (nothing in particular)"), [n.node_id for n in nodes]


def build_prompt(agent, offer: str, view_text: str, remembered_text: str) -> str:
    first = agent.profile.first_name
    opts, choices = options(agent.cfg, first)
    return ga.gs.generate_prompt(
        [agent.iss(), agent.scratch.curr_time.strftime("%A %H:%M"), situation(offer, first), first,
         remembered_text, view_text or R.EMPTY_TEXT, opts, choices], PROMPT)


def parse(raw: str) -> tuple[str, str | None, bool]:
    """-> (choice, text, parsed_ok). LLM errors and unparseable replies are 'none' (never recorded as writing)."""
    if not raw or raw.startswith("LLM_ERROR"):
        return "none", None, False
    d = as_json(raw)
    if not isinstance(d, dict):
        return "none", None, False
    choice = str(d.get("choice") or "none").strip().strip('"').lower()
    choice = {"note": "log", "add": "log", "rewrite": "front", "front page": "front"}.get(choice, choice)
    if choice not in CHOICES:
        return "none", None, False
    text = d.get("text")
    text = " ".join(str(text).split()).strip().strip('"').strip() if text is not None else None
    if not text or text.lower() in ("null", "none"):
        return "none", None, True
    return (choice, text, True) if choice != "none" else ("none", None, True)


def decide_write(agent, offer: str, view_text: str, remembered=None, *, tick: int, job_id: str | None = None,
                 rng=None) -> dict:
    """Ask `agent` whether to write (offer: job | tally | farewell). `view_text` is the binder as rendered for
    this agent now (records.render_for(...).text). Pure w.r.t. world state: returns a decision for
    records.apply_writes. Run it inside the engine's per-agent scope (e.g. t{tick:04d}:03record:{agent})."""
    if offer not in R.OFFER_KEYS:
        raise ValueError(f"unknown record-write offer {offer!r}")
    mem_text, retrieved = remembered_lines(agent, remembered, rng)
    prompt = build_prompt(agent, offer, view_text, mem_text)
    with llm_purpose(PURPOSE, agent.id):
        raw = agent.ctx.llm.complete(prompt, max_tokens=300, temperature=1.0)
    choice, text, ok = parse(raw)
    return {"agent": agent.id, "author_name": agent.profile.first_name, "tick": tick, "offer": offer,
            "choice": choice, "text": text, "job_id": job_id, "parsed": ok, "retrieved": retrieved,
            "prompt": prompt, "response": raw}


def remember_write(agent, applied: dict):
    """The author's no-LLM self memory of what they wrote (source type `self`, record id in record_ids)."""
    if applied.get("choice") not in ("log", "front") or not applied.get("text"):
        return None
    ctx = agent.ctx
    from backend.memory.encoder import _keywords, forget
    where = "the log of the co-op binder" if applied["choice"] == "log" else "the front page of the co-op binder"
    verb = "wrote in" if applied["choice"] == "log" else "rewrote"
    text = f'{agent.name} {verb} {where}: "{applied["text"]}"'
    rid = applied.get("entry_id") or applied.get("rev_id")
    emb = ctx.embed(text)
    node = agent.a_mem.add("event", agent.scratch.curr_time, agent.name, verb, "the co-op binder", text,
                           _keywords(agent, text, []), SELF_IMPORTANCE, emb, [])
    ctx.meta.set(node.node_id, MemoryMeta(agent_id=agent.id, source_type="self", salience=R.RECORD_SALIENCE,
                                          source_ids=[rid], record_ids=[rid]))
    agent.scratch.importance_trigger_curr -= SELF_IMPORTANCE
    agent.scratch.importance_ele_n += 1
    ctx.tracer.log("memory_encoded", agent=agent.id, node_id=node.node_id, kind="event", text=node.description,
                   importance=SELF_IMPORTANCE, salience=R.RECORD_SALIENCE, source_type="self", observation_id=None,
                   observation=text, encoding_ops=[], prompt=None, originating_event_ids=[], speakers=[],
                   utterance_ids=[], record_ids=[rid])
    forget(agent)
    return node
