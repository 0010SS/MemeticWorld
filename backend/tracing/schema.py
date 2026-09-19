"""Trace record registry: every record type the simulation writes to `trace.jsonl` (v1, v2 and v3).

The registry is the contract between the simulation (which writes records), the observer (which reads them)
and the API (which serves them). `scripts/trace_schema_doc.py` renders it to docs/TRACE_SCHEMA.md, and
tests/test_trace_schema.py checks that every type a mock v2 all-on day or v3 co-op day emits is registered
and carries its required fields.

Every record has the envelope written by `TraceLogger.log`: `id` ("<scope>#<n>", unless the record sets its own
id, e.g. utterances, conversations, checkpoints), `type`, `tick` (a record may set its own tick, e.g. job_start)
and `time` (ISO wall-clock of the tick; "" for seed records). Each registry entry says:

    layer          world | agent | controller | observer (who produces it; the observer never writes the trace)
    description    one line
    required       fields every record of this type carries, beyond the envelope
    optional       fields some records carry
    required_if    {field: (other_field, value)}: required when other_field == value
    hidden         True: the whole record is ground truth; demo mode drops it
    hidden_fields  fields (dotted paths reach into nested dicts) demo mode removes from a public record
    section        where the ontology / decision log defines it
    since          v1 | v2 | v3

Demo mode (the API without `?debug=1`) applies `strip(rec)` to every trace record, and then the server's generic
HIDDEN_KEYS / HIDDEN_VALUE filters as a second line of defence. This module imports nothing from the simulation,
the agents or the observer.
"""
from __future__ import annotations

import copy

ENVELOPE = ("id", "type", "tick", "time")
LAYERS = ("world", "agent", "controller", "observer")

# event provenance: simulator-side ids of the hidden world events a record came from (v2 §1.7, D26, D73)
_PROV = ("originating_event_ids",)


def _t(layer: str, since: str, section: str, description: str, required=(), optional=(), hidden_fields=(),
       hidden: bool = False, required_if: dict | None = None) -> dict:
    assert layer in LAYERS, layer
    return {"layer": layer, "since": since, "section": section, "description": description,
            "required": list(required), "optional": list(optional), "required_if": dict(required_if or {}),
            "hidden": bool(hidden), "hidden_fields": list(hidden_fields)}


TRACE_TYPES: dict[str, dict] = {
    "shared_background": _t(
        "controller", "v3", "EXPERIMENT_PIPELINE.md", "Shared Markdown supplied in identity prompts; initial material, not agent invention.",
        ["agent", "text", "sha256", "presentation"]),
    # ------------------------------------------------------------------------------------------------ WORLD
    "world_event_start": _t(
        "world", "v1", "v2 §1.1, §1.7; D51, D73",
        "A pre-generated latent-event instance (E1-E4) is released: family, schema, skin, cast, narrative.",
        ["event_id", "latent_type", "structure_mode", "schema", "skin", "composed_from", "holdout", "roles",
         "circle", "cast_from_home", "referents", "narrative"], hidden=True),
    "event_beat": _t(
        "world", "v1", "v2 §1.7; D24, D25; v3 §4.7 phase 1",
        "One beat of a world event (v2 latent events and every v3 co-op beat) with its facts, before perception.",
        ["event_id", "latent_type", "beat", "location", "arena", "facts"], hidden=True),
    "coop_fact": _t(
        "world", "v3", "v3 §1.5, §3.3-3.4, §4.7 phase 1",
        "Public world text of a co-op beat (job start / symptom / outcome, tally, cue, farewell, orientation, "
        "binder transition): the facts as released, before perception. Its hidden twin is event_beat.",
        ["ref", "kind", "beat", "location", "arena", "facts"], ["movers"]),
    "day_plan": _t(
        "world", "v1", "D11; v2 §1.1 (routine plans are world, agreement a)",
        "An agent's routine plan for a day (seeded by world_seed).", ["agent", "day", "plan"]),
    "move": _t(
        "world", "v1", "engine phase 2; D25",
        "An agent changed place; forced_by_event names the hidden event whose beat moved them.",
        ["agent", "frm", "to", "path", "activity", "forced_by_event"], hidden_fields=["forced_by_event"]),
    "exposure": _t(
        "world", "v1", "D30; v2 §4 (emergence: exposure-conditioned adoption)",
        "Who heard an utterance (listener sampling is world-side).",
        ["utterance_id", "speaker_id", "listener_ids", "utterance", "conversation_id", "location", "arena"]),
    "job_start": _t(
        "world", "v3", "v3 §1.5, §1.8",
        "A laser-cutter job starts at its slot: operator, project and the pre-drawn menu orders.",
        ["job", "day", "slot", "shift", "operator", "project", "menu_order"]),
    "job_truth": _t(
        "world", "v3", "v3 §1.3, §1.8",
        "Ground truth of a job under the active regime (class, cause, best action, uniforms).",
        ["job", "regime", "mapping", "class", "cause", "fault", "gt", "u_attempt", "code", "p"], hidden=True),
    "job_attempt": _t(
        "world", "v3", "v3 §1.5",
        "The outcome of one attempt is released (success | fail | defer).",
        ["job", "attempt", "operator", "action", "outcome"]),
    "job_end": _t(
        "world", "v3", "v3 §1.5",
        "A job ends: delivered or not, with its attempts.", ["job", "operator", "delivered", "attempts", "result"]),
    "tally": _t(
        "world", "v3", "v3 §1.5",
        "The 17:30 end-of-day tally at the laser (templated text) and the rostered stores member.",
        ["day", "delivered", "total", "missing", "text", "stores"]),
    "cue_event": _t(
        "world", "v3", "v3 §1.4",
        "A scheduled world cue (e.g. a new supplier's sheets arrive), identical in every arm.",
        ["cue", "day", "arena", "text_key", "text"]),
    "regime_active": _t(
        "world", "v3", "v3 §1.4",
        "The hidden regime and mapping of the day (day start).", ["day", "regime", "mapping", "changed"],
        hidden=True),
    "record_read": _t(
        "world", "v3", "v3 §2.3, §2.8",
        "An agent reads the co-op binder (decision | tally); world-side receipts say which items were new.",
        ["agent", "context", "view_mode", "entry_ids", "rev_ids", "new_ids", "view_sha"]),
    "record_write": _t(
        "world", "v3", "v3 §2.4, §2.8",
        "A binder write offer and its outcome (none | log | front), applied in sorted (tick, agent) order.",
        ["agent", "offer", "choice", "entry_id", "rev_id", "text", "truncated", "prompt", "response"]),
    "record_transition": _t(
        "world", "v3", "v3 §2.5, §2.8",
        "The binder manipulation at a day start: keep, or wipe (the old binder is archived).",
        ["day", "mode", "archived_binder_id"]),
    "roster_change": _t(
        "world", "v3", "v3 §3.3, §3.4",
        "A member departs or a newcomer arrives at a day start.", ["day", "agent", "kind", "role"], ["replaces"],
        required_if={"replaces": ("kind", "arrive")}),
    "onboarding": _t(
        "world", "v3", "v3 §3.4",
        "A newcomer's symmetric onboarding ties to crewmates and other members.", ["agent", "day", "ties"]),
    "farewell": _t(
        "world", "v3", "v3 §3.3; §8.1 step 4",
        "A departing member's farewell fact is released in their shift arena (the day before they leave).",
        ["agent", "day", "role", "location", "arena", "text"]),
    "handover": _t(
        "world", "v3", "v3 §4.5",
        "The day's scheduled shift-change handover (outgoing 12:00 operator, incoming 13:30 operator).",
        ["day", "outgoing", "incoming"], ["conversation_id", "participants", "utterance_ids", "n_utterances"]),
    "meeting": _t(
        "world", "v3", "v3 §4.5",
        "A scheduled members' meeting at the Makerspace (who is invited).", ["day", "invited"],
        ["conversation_id", "participants", "utterance_ids", "n_utterances"]),
    "coop_day": _t(
        "world", "v3", "v3 §4.7 phase 8b; §5.11 item 14",
        "End-of-day digest of the co-op's mechanism counts (manipulation checks).", ["day", "counts"]),
    # ------------------------------------------------------------------------------------------------ AGENT
    "observation": _t(
        "agent", "v1", "D13; D42; v3 §2.3 (binder reads), §4.2 (job episodes)",
        "What an agent noticed (perception | conversation | overheard | record), as world text.",
        ["agent", "observation_id", "source_type", "facts", *_PROV], ["event_id", "episode"],
        hidden_fields=["event_id", *_PROV]),
    "viewpoint": _t(
        "agent", "v1", "D42",
        "The agent's own rendering of what it noticed (participant, near, distracted, far).",
        ["agent", "observation_id", "prompt", "response", "facts", "dropped", "fallback", *_PROV],
        hidden_fields=list(_PROV)),
    "memory_encoded": _t(
        "agent", "v1", "D15, D44; v3 §2.3-2.4",
        "A memory node was written (lossy encoding, seed, self or binder-read memories).",
        ["agent", "node_id", "kind", "text", "importance", "salience", "source_type", "observation_id",
         "observation", "encoding_ops", "prompt", *_PROV, "speakers", "utterance_ids"],
        ["lens", "self_experience", "record_ids"], hidden_fields=list(_PROV)),
    "memory_merged": _t(
        "agent", "v1", "D15",
        "A new experience was merged into a near-duplicate memory.",
        ["agent", "node_id", "into_text", "dropped_text", "similarity", "observation_id", *_PROV],
        hidden_fields=list(_PROV)),
    "memory_forgotten": _t(
        "agent", "v1", "D17", "A weak memory node was forgotten.", ["agent", "node_id", "text"]),
    "memory_link": _t(
        "agent", "v2", "v2 §2.3; D59",
        "LINK: the agent connected two of its memories (reminding, association, merge, reflection evidence).",
        ["agent", "from", "to", "mechanism", "reason", "from_event_ids", "to_event_ids"], ["source_type", "via"],
        hidden_fields=["from_event_ids", "to_event_ids"]),
    "reminding": _t(
        "agent", "v2", "v2 §2.3; D48, D59",
        "Spontaneous reminding after an encoding (may create a reminding thought node).",
        ["agent", "observation_id", "source_type", "new_node", "candidates", "prompt", "response", "ordinary",
         "reminded_of", "what_felt_alike", "node_id", "text", *_PROV, "reminded_of_event_ids", "chained"],
        hidden_fields=[*_PROV, "reminded_of_event_ids"]),
    "wording": _t(
        "agent", "v2", "v2 §2.4; D58",
        "WORDING: a distinctive phrase stuck verbatim in a memory.",
        ["agent", "node_id", "observation_id", "source_type", "phrase", "heard_from", "utterance_id",
         "self_produced"]),
    "priming": _t(
        "agent", "v2", "v2 §2.4; D61",
        "Production priming: recently heard wordings put in mind for a conversation.",
        ["agent", "conversation_id", "phrases"]),
    "open_matter": _t(
        "agent", "v2", "v2 §2.5; D60",
        "NEED: an open matter was opened, refreshed or discussed.", ["agent", "action", "node_id", "text", "strength"],
        ["reason", "importance", "observation_id", "n_open"]),
    "open_matter_focal": _t(
        "agent", "v2", "v2 §2.5; D60",
        "NEED: the open matters used as conversation focal points.",
        ["agent", "conversation_id", "node_ids", "strengths", "texts"]),
    "reflection": _t(
        "agent", "v1", "D21", "A reflection thought and the memories it cites.",
        ["agent", "node_id", "text", "focal_point", "evidence", "retrieved", "retrieval_scores", "importance", *_PROV],
        hidden_fields=list(_PROV)),
    "decision": _t(
        "agent", "v1", "D12 (react), D22 (decide_to_talk)",
        "A react or decide_to_talk decision with its retrieved memories.",
        ["agent", "kind", "prompt", "decision", "retrieved"],
        ["observation_id", "response", "retrieval_scores", "retrieved_event_ids", "target"],
        hidden_fields=["retrieved_event_ids"]),
    "utterance": _t(
        "agent", "v1", "D23; v3 §4.3 (says_aloud remarks)",
        "One line said aloud (chat, group chat, reaction remark) with its retrieved memories.",
        ["conversation_id", "idx", "speaker", "text", "listeners", "location", "arena", "retrieved",
         "retrieved_event_ids", "source", "context"], ["relationship_summary", "retrieval_scores"],
        hidden_fields=["retrieved_event_ids"]),
    "conversation": _t(
        "agent", "v1", "D23; v2 §3 (group talk); v3 §4.4-4.5 (trigger.topic: clarify | handover | meeting)",
        "A whole conversation (dyadic or group); `trigger` says what started it.",
        ["participants", "location", "arena", "trigger", "relationship_summaries", "utterance_ids", "transcript"],
        hidden_fields=["trigger.event_ids"]),
    "invitation": _t(
        "agent", "v1", "engine phase 6 (reactive re-planning)",
        "After a warm conversation one participant tags along with the other for a while.",
        ["agent", "target", "until", "conversation_id"]),
    "replan": _t(
        "agent", "v1", "engine phase 5 (MOVE reaction)", "A reaction decision re-plans the agent to another place.",
        ["agent", "reason", "to", "until"]),
    "job_decision": _t(
        "agent", "v3", "v3 §4.3",
        "The operator's job decision (an action, or G = ask someone first) with retrieval and binder view.",
        ["agent", "job", "attempt", "menu_order", "choice", "action", "question", "says_aloud", "reason",
         "retrieved", "binder_view_sha", "binder_entry_ids", "prompt", "response"],
        ["ask_target", "valid", "n_calls", "consult", "binder_shown", "binder_chosen", "options",
         "retrieval_scores", "retrieved_event_ids"], hidden_fields=["retrieved_event_ids"]),
    "clarification": _t(
        "agent", "v3", "v3 §4.4",
        "Choice G: the operator asks someone present; runs as a forced dyadic talk (topic clarify).",
        ["job", "attempt", "agent", "target", "question"],
        ["conversation_id", "participants", "utterance_ids", "n_utterances"]),
    # ------------------------------------------------------------------------------------------- CONTROLLER
    "checkpoint": _t(
        "controller", "v3", "v3 §5.1",
        "A day-end checkpoint was written (`id` is the checkpoint id, e.g. C4).", ["files"]),
}

HIDDEN_TYPES = frozenset(t for t, s in TRACE_TYPES.items() if s["hidden"])


# ------------------------------------------------------------------------------------------------ queries
def spec(type_: str) -> dict | None:
    return TRACE_TYPES.get(type_)


def is_registered(type_: str) -> bool:
    return type_ in TRACE_TYPES


def is_trace_record(obj) -> bool:
    """A dict shaped like a trace record of a registered type (the API's detector)."""
    return isinstance(obj, dict) and obj.get("type") in TRACE_TYPES and "tick" in obj


def validate(rec) -> list[str]:
    """Problems with one record ([] = valid): envelope, unregistered type, missing required fields."""
    if not isinstance(rec, dict):
        return ["record is not an object"]
    errs = [f"missing envelope field {k!r}" for k in ENVELOPE if k not in rec]
    s = TRACE_TYPES.get(rec.get("type"))
    if s is None:
        return errs + [f"unregistered trace type {rec.get('type')!r}"]
    errs += [f"{rec['type']}: missing required field {f!r}" for f in s["required"] if f not in rec]
    for f, (other, value) in s["required_if"].items():
        if rec.get(other) == value and f not in rec:
            errs.append(f"{rec['type']}: missing field {f!r} (required when {other} == {value!r})")
    return errs


def _drop(obj, path: list[str]):
    if isinstance(obj, list):
        for x in obj:
            _drop(x, path)
        return
    if not isinstance(obj, dict):
        return
    if len(path) == 1:
        obj.pop(path[0], None)
    elif path[0] in obj:
        _drop(obj[path[0]], path[1:])


def strip(rec: dict) -> dict | None:
    """Demo-mode copy of one trace record: None for a hidden type, else the record without its hidden fields.
    Unregistered types pass through unchanged (the API's generic filters still apply)."""
    s = TRACE_TYPES.get(rec.get("type")) if isinstance(rec, dict) else None
    if s is None:
        return rec
    if s["hidden"]:
        return None
    if not s["hidden_fields"]:
        return dict(rec)
    nested = any("." in p for p in s["hidden_fields"])
    out = copy.deepcopy(rec) if nested else dict(rec)
    for p in s["hidden_fields"]:
        _drop(out, p.split("."))
    return out


def public_registry(debug: bool = False) -> dict:
    """The registry as served by GET /api/schema. Demo mode lists only public types and public fields (no
    hidden flags, no hidden field names); debug mode returns everything."""
    types = {}
    for t, s in sorted(TRACE_TYPES.items()):
        if debug:
            types[t] = copy.deepcopy(s)
            continue
        if s["hidden"]:
            continue
        hid = {p.split(".")[0] for p in s["hidden_fields"] if "." not in p}
        types[t] = {"layer": s["layer"], "since": s["since"], "section": s["section"],
                    "description": s["description"],
                    "required": [f for f in s["required"] if f not in hid],
                    "optional": [f for f in s["optional"] if f not in hid]}
    return {"envelope": list(ENVELOPE), "layers": list(LAYERS), "types": types,
            "frame": copy.deepcopy(FRAME_SCHEMA)}


# ------------------------------------------------------------------------------------------------ frames
# frames.jsonl: one line per tick, written by engine._frame (phase 8). Documented here so the frontend, the API
# and docs/TRACE_SCHEMA.md share one description. Fields marked (v3) appear only in co-op runs; old runs may
# lack every field added after v2 (`away`, the snapshot's active/role/cohort/open_matters/wordings, coop.jobs).
FRAME_SCHEMA = {
    "frame": {
        "tick": "int", "time": "ISO datetime of the tick", "day": "int (1-based)", "label": "'Day 2 10:15'",
        "agents": "{agent_id: agent snapshot} for every persona in the run so far (a newcomer appears from "
                  "its arrival day; departed members stay, with active=false)",
        "away": "[agent ids with active=false] (departed / not on the co-op roster today); [] in v2 runs",
        "utterances": "[{id, speaker, text, listeners, conversation_id}] said this tick",
        "beats": "[{location, arena, facts: [text]}] world beats released this tick (debug adds event_id, "
                 "latent_type)",
        "coop": "(v3 co-op runs only) {binder (records on), jobs}",
    },
    "agent": {
        "id": "agent id", "location": "place ('Away' when inactive)", "arena": "arena ('Away' when inactive)",
        "activity": "text", "goal": "text", "path": "[places traversed this tick]",
        "conversation": "conversation id or null", "n_memories": "int", "n_reflections": "int",
        "modules": "{module: state} of switched-on pressure modules",
        "active": "bool (false: departed or not on the roster)",
        "role": "(v3) am_crew | pm_crew | stores (a departed member keeps its last role), else null",
        "cohort": "(v3, roster on) founder | newcomer, else null",
        "open_matters": "int: open matters still on the agent's mind (NEED; 0 when off)",
        "wordings": "int: stuck wordings heard from others held in live memories (WORDING; 0 when off)",
    },
    "coop.binder": {
        "binder_id": "binder-N (a wipe starts a new one)",
        "front": "{author (first name), when ('Wed 11:20'), text, revisions} or null",
        "log": "[{author, when, text}] newest first, at most 5",
        "archived": "int: archived binders",
    },
    "coop.jobs[]": {
        "id": "job id (j04.2)", "day": "int", "slot": "int", "shift": "am | pm", "operator": "agent id",
        "project": "text", "symptom": "the symptom fact as released (world text)", "attempt": "int: attempts made",
        "status": "deciding | asking | running | delivered | defer | failed",
        "tried": "[{attempt, action, outcome}] attempts whose outcome has been released",
    },
}
