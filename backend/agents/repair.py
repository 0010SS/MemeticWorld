"""Conversational repair: how a phrase carries its MEANING, not just its form.

When A says a coined expression to B who has never heard it, copying the token transmits nothing. What
transmits a convention is the repair sequence -- "the what?" / "oh, it's when something looks fine and
isn't" -- because that exchange is what attaches the expression to a class of situations in someone
else's head. Clark's collaborative account of reference, conceptual pacts and least collaborative effort
(Clark & Wilkes-Gibbs 1986; Brennan & Clark 1996; Clark & Brennan 1991) is the principle; the audit entry
is docs/COGNITIVE_GROUNDING.md §6.3, which records that our overhearing channel currently models acoustic
access and nothing else. The research memo draws from GlossoGen the explicit lesson that a world must
allow ordinary clarification and negotiation and treat "without explicit agreement" as a SEPARATE
experiment -- hence `conversation.repair.enabled: false`. This is an experimental variable, not scenery.

Three commitments, none of which may be relaxed:

  * **What an agent knows is what is in its memory stream.** Whether a phrase is new to the hearer is
    decided by scanning that hearer's own live memories and their stuck wordings (`recall`). No registry,
    no roster of "planted" phrases, no config list is ever consulted to decide what an agent knows.
  * **No designer-supplied gloss.** The only thing that can answer "what does that mean" is what the
    answering agent retrieves from its own memory (`explain`). If retrieval brings back little, the
    answer is thin; if it brings back the wrong occasion, the answer drifts. That decay IS the mechanism
    of semantic change this build is trying to observe, so it is never patched around.
  * **The question never contains the phrase.** A hearer who does not know an expression must not be
    recorded as having used it. Keeping the trouble source out of the question ("Sorry -- the what?")
    keeps the dependent variable clean: every occurrence of a phrase in the transcript is somebody
    actually using it.

A hearer who DOES have the phrase but remembers it for something else can say so (`collisions`), which is
where meaning gets negotiated. That sub-channel ships OFF, on a measurement: the run's default embedder
cannot tell agreement from disagreement about the same phrase (see `DEFAULTS["collide_below"]`). The
comparison and its raw score are logged on every repair record so the channel can be switched on with a
semantic embedder, or re-thresholded by the observer afterwards.

Only the ADDRESSEE repairs. Overhearers hear the whole exchange (it reaches exactly the ears that heard
the line that triggered it) but cannot ask, which is Schober & Clark's (1989) asymmetry and the one place
in this build where an overhearer is treated differently from an addressee.

Randomness (ontology v2 §2.1): every draw comes from the hearer's own "repair" substream, addressed by
the utterance and the phrase -- `seed_rng(seed, agent_id, "repair", utterance_id, phrase)`, which is
`Agent.stream("repair")`'s seeding scheme with the event appended. Addressing it that way means a draw
never depends on how many repairs happened earlier in the run, so a condition with repair on is still
comparable with one that has it off. The exchange reaches the trigger line's listeners, so switching
repair on takes no overhearing draw either.
"""
from __future__ import annotations

import functools
import re

from backend import ga_compat
from backend.llm.client import llm_purpose
from backend.llm.embeddings import cos
from backend.memory import encoder as ENC
from backend.simulation import reference
from backend.simulation.rngs import seed_rng

ga = ga_compat.load()
PROMPT = str(ga_compat.REPO_ROOT / "backend" / "prompts" / "repair_explain_v1.txt")
PURPOSE = "repair_explain"

# `conversation.repair`. The three keys in the block written by the config builder are the experimental
# dial; the rest are mechanism constants with defaults here so a config that carries only the three
# behaves exactly as specified.
DEFAULTS = {
    "enabled": False,
    "prob": 0.7,                # p(the addressee asks), on the one best candidate phrase per line
    "max_followups": 1,         # repair exchanges per conversation (an ask is one, a collision is one)
    "phrase_source": "both",    # rare | repeated | both -- which channels may find a candidate
    "zipf_max": None,           # null -> memory.verbatim.zipf_max (one rarity dial for the whole build)
    "lexicon_filter": True,     # a phrase made only of campus/routine vocabulary is not a coinage
    "min_speaker_memories": 1,  # the speaker must be able to find the phrase in their OWN memory
    # <= 0 turns the collision channel OFF, which is the default, on a measurement rather than a
    # preference: under the run's default hash embedder (backend/llm/embeddings.py) the cosine between
    # the clause a speaker used a phrase in and a hearer's memory of that phrase is 0.26-0.35 when the
    # two agree and 0.26-0.36 when they do not. The two distributions coincide, because the embedder is
    # lexical and the shared phrase dominates both vectors, so any threshold here would fire on wording
    # rather than on meaning. `collisions()` and the raw `agreement` it logs are kept so the channel can
    # be switched on with `embedding.backend: sentence_transformers`, or re-thresholded after the fact.
    "collide_below": 0.0,
    "explain_k": 3,             # memories retrieved for the explanation
}

# Fixed openers. They carry no place name, no phrase and no content word rare enough to be picked up as a
# candidate expression by anything (agent-side or observer-side), so a repair question can never be
# mistaken for the coining or the using of one. `tests/test_repair.py` machine-checks both properties.
QUESTIONS = ("Sorry -- the what?",
             "Hold on, what do you mean by that?",
             "What's that, when you say that?",
             "Sorry, I don't know that one. What is it?")

ASK_LINE = '{other} has just asked {me} what "{phrase}" means.'
COLLIDE_LINE = ('{other} has just used the words "{phrase}" for something that does not match what {me} '
                'has been using them for.')
EXPLAIN_FOCAL = 'what {first} was talking about when {first} said "{phrase}"'

MIN_REPLY = 8          # shorter than this is a failed generation, not a thin answer
MAX_REPLY = 400
MAX_CHAT_ROWS = 4      # transcript rows shown to the explainer (conversation.py shows the same 4)

# Prefixes the encoder writes onto a memory ("Maya Chen remembers that ...", "Maya Chen overheard: ...").
# Stripped only in the no-LLM fallback, which speaks the agent's own remembered words back; it is never
# used to manufacture content.
_MEM_PREFIX = re.compile(
    r"^[A-Z][\w'-]+(?: [A-Z][\w'-]+)? (?:remembers that |remembers |saw:? |experienced(?: first-hand)?:? "
    r"|heard in a conversation:? |overheard:? |read in the co-op binder:? |did:? )", re.I)


# ----------------------------------------------------------------------------------------------- config
def rcfg(cfg: dict) -> dict:
    """`conversation.repair` with the defaults above filled in."""
    return {**DEFAULTS, **(((cfg or {}).get("conversation") or {}).get("repair") or {})}


def enabled(cfg: dict) -> bool:
    return bool(rcfg(cfg).get("enabled"))


def _zipf_max(agent) -> float:
    rc = rcfg(agent.cfg)
    if rc.get("zipf_max") is not None:
        return float(rc["zipf_max"])
    return float((agent.cfg["memory"].get("verbatim") or {}).get("zipf_max", 3.6))


# ------------------------------------------------------------------------------- what an agent knows
@functools.lru_cache(maxsize=8192)
def _rx(phrase: str):
    """Word-bounded, case-insensitive. Lookarounds rather than \\b so a phrase that ends in punctuation
    or contains an internal hyphen still matches on both edges."""
    return re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.I)


def mentions(text, phrase: str) -> bool:
    return bool(_rx(phrase).search(str(text or "")))


def memory_texts(agent) -> list[tuple[str, str]]:
    """(node_id, searchable text) for every LIVE memory of this agent: its description plus any wording
    that stuck verbatim on it. Rebuilt per call -- a forgotten memory must stop counting as knowledge."""
    meta = getattr(getattr(agent, "ctx", None), "meta", None)
    out = []
    for nid, node in sorted(agent.a_mem.id_to_node.items()):
        m = meta.get(nid) if meta is not None else None
        extra = " ".join(str(w.get("phrase") or "") for w in m.wordings) if m is not None else ""
        out.append((nid, f"{node.description} {extra}" if extra else node.description))
    return out


def _recall_in(texts: list[tuple[str, str]], phrase: str) -> list[str]:
    """Node ids among `texts` that carry `phrase`. The cheap substring test runs first because this is
    the innermost loop of the mechanism (every candidate span x every memory of two agents per line)."""
    low = phrase.lower()
    return [nid for nid, t in texts if low in t.lower() and mentions(t, phrase)]


def recall(agent, phrase: str) -> list[str]:
    """Node ids of the agent's OWN live memories that carry `phrase`: in the memory text, or as a stuck
    wording on the node's sidecar (`MemoryMeta.wordings`, WORDING v2 §2.4).

    This is the whole definition of "does this agent know this expression". It reads the memory stream
    and nothing else, so a phrase the agent once heard and has since forgotten is unknown again, and a
    phrase nobody has said to them is unknown however central it is to the experiment.
    """
    return _recall_in(memory_texts(agent), phrase)


# --------------------------------------------------------------------------------- candidate phrases
def _clauses(text) -> list[str]:
    return ENC._BREAK.split(str(text or "").replace("’", "'"))


def candidate_spans(agent, text: str, zipf_max: float, *, lexicon: bool = True) -> list[dict]:
    """Every phrase in `text` that could be somebody's coined expression, judged with the encoder's own
    rules (`is_function_word`, the population lexicon, the roster of person names) and never against any
    list of memes. -> [{"phrase", "channel", "zipf", "n"}]

    Two channels, because no single test finds both kinds of coinage:

      rare      one content word rare enough to stand out on its own (wordfreq Zipf <= `zipf_max`).
                Finds a true neologism and little else.
      repeated  a phrase made of ordinary words: a 2-3 word content span, or a hyphenated compound
                written as one token, at least part of which is not campus/routine vocabulary. This
                channel exists because rarity CANNOT see a coinage built from common words, and
                measurement says that is most of them. wordfreq scores a compound at the frequency of
                its rarest PART, never below it, so the two hyphenated compounds of the cohort this
                was built for score Zipf 3.73 and 4.20 against parts of 3.75 and 4.21 -- both above
                the 3.6 rarity cutoff -- and its two-word phrases score 4.63 and 4.71, far above it.
                So this channel only PROPOSES the phrase, and the caller decides with MEMORY: it is a
                candidate when the speaker can find it in their own memories and the hearer cannot
                find it in theirs. That is a common-ground test rather than a lexical one, and it is
                what keeps a cohort of memes matched instead of silently split by which of them happen
                to use a rare word (R6). Nothing here knows any phrase; run `detectable` per registry
                entry to check one.

    `shape` (word | compound | span) records which form was matched, for the observer.
    """
    names = ENC._person_names(agent)
    lex = ENC._lexicon_tokens(agent) if lexicon else frozenset()
    out: dict[str, dict] = {}
    for seg in _clauses(text):
        toks = [t for t in (x.strip("'-") for x in ENC._TOK.findall(seg)) if t]
        base = [ENC._base(t) for t in toks]
        name = [b in names for b in base]
        content = [not ENC.is_function_word(t) and not nm for t, nm in zip(toks, name)]
        for i, b in enumerate(base):
            if content[i] and len(b) >= 4 and b not in lex and ENC._zipf(b) <= zipf_max:
                out.setdefault(toks[i].lower(), {"phrase": toks[i], "channel": "rare", "shape": "word",
                                                 "zipf": ENC._zipf(b), "n": 1})
            # a hyphenated compound is a multi-word coinage written as one token, so the function-word
            # edge rule does not apply to it -- the hyphen already marks the unit
            parts = [p for p in b.split("-") if p]
            if (len(parts) >= 2 and len(b) >= 5 and not name[i] and all(len(p) >= 2 for p in parts)
                    and not any(p in names for p in parts)
                    and b not in lex and not all(p in lex for p in parts)):
                out.setdefault(toks[i].lower(), {"phrase": toks[i], "channel": "repeated",
                                                 "shape": "compound", "zipf": ENC._zipf(b), "n": 1})
        for n in (2, 3):
            for j in range(0, len(toks) - n + 1):
                if not all(content[j:j + n]) or any(name[j:j + n]):
                    continue
                if lex and all(b in lex for b in base[j:j + n]):
                    continue                       # ordinary campus talk, not somebody's coinage
                ph = " ".join(toks[j:j + n])
                z = min(ENC._zipf(b) for b in base[j:j + n])
                prev = out.get(ph.lower())
                if prev is None or z < prev["zipf"]:
                    out[ph.lower()] = {"phrase": ph, "channel": "repeated", "shape": "span",
                                       "zipf": z, "n": n}
    return [out[k] for k in sorted(out)]


def _wanted(channel: str, source: str) -> bool:
    return source == "both" or source == channel


def _rank(c: dict) -> tuple:
    """Most-used first, then rarest, then shortest: a phrase the speaker keeps reaching for is the one a
    hearer notices as a stock expression, and between overlapping spans the bare coinage wins."""
    return (-c["speaker_memories"], c["zipf"], len(c["phrase"]), c["phrase"].lower())


def candidates(speaker, hearer, text: str) -> list[dict]:
    """Phrases in `text` that the SPEAKER has in memory and the HEARER has not, best first.

    The speaker-side requirement is not bookkeeping: repair hands over remembered meaning, and an agent
    who has just improvised a phrase has nothing to hand over. Applying it to both channels also keeps
    them matched, so a difference between two memes cannot come from one of them having a repair channel
    and the other not.
    """
    rc = rcfg(speaker.cfg)
    need = int(rc["min_speaker_memories"])
    mine_texts, theirs_texts = memory_texts(speaker), memory_texts(hearer)
    out = []
    for c in candidate_spans(speaker, text, _zipf_max(speaker), lexicon=bool(rc["lexicon_filter"])):
        if not _wanted(c["channel"], str(rc["phrase_source"])):
            continue
        if _recall_in(theirs_texts, c["phrase"]):          # not new to them: nothing to repair
            continue
        mine = _recall_in(mine_texts, c["phrase"])
        if len(mine) < need:
            continue
        out.append({**c, "speaker_memories": len(mine), "speaker_nodes": mine})
    return sorted(out, key=_rank)


def collisions(speaker, hearer, text: str) -> list[dict]:
    """Phrases in `text` the hearer DOES have in memory, but remembers for something else.

    `agreement` is the cosine between the clause the speaker used the phrase in and the hearer's own
    memory of it, under the run's embedder. It is logged raw next to the verdict so the observer can
    re-threshold after the fact: `collide_below` decides whether the hearer says anything, not what the
    analysis is allowed to conclude.

    OFF unless `collide_below > 0`; see DEFAULTS for the measurement that puts it there. Cosines are
    signed, so a threshold of 0 must mean "never", not "everything unrelated".
    """
    rc = rcfg(speaker.cfg)
    below = float(rc["collide_below"])
    if below <= 0:
        return []
    embed = hearer.ctx.embed
    mine_texts, theirs_texts = memory_texts(speaker), memory_texts(hearer)
    out = []
    for c in candidate_spans(speaker, text, _zipf_max(speaker), lexicon=bool(rc["lexicon_filter"])):
        if not _wanted(c["channel"], str(rc["phrase_source"])):
            continue
        theirs = _recall_in(theirs_texts, c["phrase"])
        mine = _recall_in(mine_texts, c["phrase"])
        if not theirs or len(mine) < int(rc["min_speaker_memories"]):
            continue
        v = embed(_clause_with(text, c["phrase"]))
        best, best_node = -1.0, None
        for nid in theirs:
            s = cos(v, embed(hearer.a_mem.id_to_node[nid].description))
            if s > best:
                best, best_node = s, nid
        if best < below:
            out.append({**c, "speaker_memories": len(mine), "speaker_nodes": mine,
                        "hearer_nodes": theirs, "agreement": round(float(best), 3),
                        "hearer_node": best_node})
    return sorted(out, key=_rank)


def _clause_with(text: str, phrase: str) -> str:
    """The clause the phrase was used in -- the speaker's applied context, which is what a hearer
    compares against their own memory. Falls back to the whole line."""
    for seg in _clauses(text):
        if mentions(seg, phrase):
            return seg.strip()
    return str(text or "").strip()


def detectable(agent, phrase: str, cfg: dict | None = None) -> dict:
    """Could `phrase` ever trigger repair in this world? -> {"channel": rare|repeated|None, "reason"}.

    For the cohort builder and its tests, not for the simulation: a meme whose phrase no channel can see
    has no repair channel at all and would lose on the mechanism rather than on grounding or breadth
    (R6). Call it once per registry entry before a run and fix the phrase, not the code.
    """
    rc = rcfg(cfg if cfg is not None else agent.cfg)
    zmax = float(rc["zipf_max"]) if rc.get("zipf_max") is not None else _zipf_max(agent)
    found = [c for c in candidate_spans(agent, phrase, zmax, lexicon=bool(rc["lexicon_filter"]))
             if c["phrase"].lower() == phrase.lower() and _wanted(c["channel"], str(rc["phrase_source"]))]
    if found:
        c = found[0]
        return {"channel": c["channel"], "shape": c["shape"], "zipf": c["zipf"],
                "reason": f"found as a {c['channel']} candidate ({c['shape']})"}
    return {"channel": None, "shape": None, "zipf": None,
            "reason": ("no channel sees it: it is longer than three words, starts or ends with a function "
                       "word, contains a person's name, or is built only from campus vocabulary")}


# --------------------------------------------------------------------------------------- the exchange
def draw(hearer, utterance_id: str, phrase: str):
    """The hearer's own "repair" substream, addressed by the line and the phrase (see the module note)."""
    return seed_rng(hearer.seed, hearer.id, "repair", utterance_id, phrase)


def _listeners(utt: dict, talker_id: str) -> list[str]:
    """Exactly the ears that heard the line being repaired, minus whoever is now talking. No new
    overhearing draw, so switching repair on shifts no other mechanism's stream."""
    out = []
    for x in [utt.get("speaker"), *(utt.get("listeners") or [])]:
        if x and x != talker_id and x not in out:
            out.append(x)
    return out


def _transcript(chat: list) -> str:
    rows = list(chat or [])[-MAX_CHAT_ROWS:]
    return "\n".join(f"{who}: {what}" for who, what in rows) or "[The conversation has just started.]"


def _clean(raw: str) -> str | None:
    text = " ".join(str(raw or "").split()).strip()
    if not text or text.startswith("LLM_ERROR"):
        return None
    text = re.sub(r'^"(.*)"$', r"\1", text).strip()
    text = re.sub(r"^[A-Z][\w'-]+(?: [A-Z][\w'-]+)?\s*:\s*", "", text).strip()
    return text[:MAX_REPLY] if len(text) >= MIN_REPLY else None


def _fallback(talker, nodes) -> str | None:
    """No usable generation: the agent says their own remembered words back, with the encoder's
    third-person prefix taken off. It adds nothing the agent does not already hold."""
    if not nodes:
        return None
    text = _MEM_PREFIX.sub("", nodes[0].description).strip()
    if not text:
        return None
    return (text[:1].upper() + text[1:])[:MAX_REPLY]


def explain(talker, other, phrase: str, *, kind: str, chat: list, rng) -> dict:
    """What `talker` says the phrase applies to, retrieved from their OWN memory. One LLM call.

    Retrieval is the ordinary stochastic retrieval, and it is NOT forced to return a memory that carries
    the phrase. When it does not, the agent explains from whatever came to mind instead -- a decayed,
    partly wrong account handed on as if it were the thing. That is the drift mechanism; it is left
    alone deliberately (R7).
    """
    from backend.memory.retrieval import merged_nodes, retrieve
    rc = rcfg(talker.cfg)
    k = int(rc["explain_k"])
    first = talker.profile.first_name
    focal = [phrase, EXPLAIN_FOCAL.format(first=first, phrase=phrase)]
    res = retrieve(talker, focal, k=max(1, -(-k // len(focal))), rng=rng)
    nodes = merged_nodes(res, limit=k)
    remembered = "\n".join(f"- {n.description}" for n in nodes) or "- (nothing comes to mind)"
    line = (ASK_LINE if kind == "ask" else COLLIDE_LINE).format(
        me=talker.name, other=other.name, phrase=phrase)
    prompt = ga.gs.generate_prompt(
        [talker.iss(), talker.name, talker.scratch.curr_time.strftime("%A %H:%M"),
         reference.describe(talker.state.location, talker.state.arena, agent=talker, cfg=talker.cfg),
         line, _transcript(chat), remembered, phrase], PROMPT)
    with llm_purpose(PURPOSE, talker.id):
        raw = talker.ctx.llm.complete(prompt, max_tokens=120, temperature=1.0)
    text = _clean(raw)
    fell_back = text is None
    if fell_back:
        text = _fallback(talker, nodes)
    return {"text": text, "retrieved": [n.node_id for n in nodes], "prompt": prompt, "response": raw,
            "fallback": fell_back, "grounded_in_phrase": any(mentions(n.description, phrase) for n in nodes)}


def _turn(conv_id: str, utt: dict, talker, text: str, source: str, seq: int,
          *, retrieved=(), context: str = "") -> dict:
    """One repair turn, shaped exactly like a conversation utterance and logged the same way, so every
    downstream reader (exposure, the transcript, the participants' end-of-conversation memory, the
    observer) treats it as what it is: a line somebody said out loud.

    It carries the TRIGGER line's `idx`, not a position of its own. The observer orders usages by
    (tick, conversation, idx, id) (`analysis/rundata.chron_key`), and sharing the trigger's idx makes
    the id break the tie in the right direction: u0 < u0.r0 < u0.r1 < u1. Numbering a repair turn by its
    own position would instead sort the NEXT ordinary line before the answer to the previous one, which
    would corrupt the transmission chains this whole build is measuring.
    """
    ctx = talker.ctx
    u = {"id": f"{utt['id']}.r{seq}", "conversation_id": conv_id, "idx": utt.get("idx", 0),
         "speaker": talker.id,
         "text": text, "listeners": _listeners(utt, talker.id), "location": talker.state.location,
         "arena": talker.state.arena, "retrieved": list(retrieved),
         "retrieved_event_ids": ctx.meta.events_of(list(retrieved)), "source": source}
    ctx.tracer.log("utterance", **u, context=context)
    ctx.tracer.log("exposure", utterance_id=u["id"], speaker_id=talker.id, listener_ids=u["listeners"],
                   utterance=text, conversation_id=conv_id, location=u["location"], arena=u["arena"])
    return u


def maybe_repair(conv_id: str, utt: dict, speaker, hearer, utterances: list, chat: list) -> list[dict]:
    """Does the addressee stop the speaker over something they just said? -> extra utterance records.

    Called by `conversation.run_conversation` immediately after a line has been said and logged. The
    returned turns are appended to the conversation's utterances and transcript by the caller, which is
    what puts them through the normal memory path: at the end of the conversation both participants
    encode the whole exchange, so the hearer's link from PHRASE to what it applies to is a lossy memory
    that decays and is retrieved like any other, with the speaker in its provenance.

    Off by default, and when off this returns before touching an rng, an embedder or the LLM.
    """
    rc = rcfg(speaker.cfg)
    if not rc.get("enabled") or speaker.id == hearer.id:
        return []
    # one repair exchange writes one of these turns, so this counts exchanges, not lines
    done = sum(1 for u in utterances if u.get("source") in ("repair_question", "repair_collision"))
    if done >= int(rc["max_followups"]):
        return []

    kind, cand = "ask", None
    news = candidates(speaker, hearer, utt.get("text") or "")
    if news:
        cand = news[0]
    else:
        clashes = collisions(speaker, hearer, utt.get("text") or "")
        if clashes:
            kind, cand = "collide", clashes[0]
    if cand is None:
        return []

    phrase = cand["phrase"]
    rng = draw(hearer, utt["id"], phrase)
    if rng.random() >= float(rc["prob"]):
        return []

    turns: list[dict] = []
    if kind == "ask":
        q = QUESTIONS[int(rng.integers(len(QUESTIONS)))]
        turns.append(_turn(conv_id, utt, hearer, q, "repair_question", 0,
                           context=f'{speaker.name}: "{utt.get("text")}"'))
        answerer, listener = speaker, hearer
    else:
        answerer, listener = hearer, speaker
    said = explain(answerer, listener, phrase, kind=kind, chat=chat, rng=rng)
    if said["text"]:
        turns.append(_turn(conv_id, utt, answerer, said["text"],
                           "repair_answer" if kind == "ask" else "repair_collision", len(turns),
                           retrieved=said["retrieved"],
                           context=(ASK_LINE if kind == "ask" else COLLIDE_LINE).format(
                               me=answerer.name, other=listener.name, phrase=phrase)))
    # a failed generation with nothing in memory to fall back on still cost a retrieval and a call, so
    # it is recorded: "asked and got nothing" is an outcome, not an absence
    speaker.ctx.tracer.log(
        "repair", conversation_id=conv_id, utterance_id=utt["id"], kind=kind, phrase=phrase,
        channel=cand["channel"], shape=cand["shape"], asker=hearer.id, answerer=answerer.id,
        explanation=said["text"], explained=bool(said["text"]),
        competing=(kind == "collide"), agreement=cand.get("agreement"),
        asker_memories=list(cand.get("hearer_nodes") or []),
        answerer_memories=list(cand.get("speaker_nodes") or []),
        retrieved=said["retrieved"], grounded_in_phrase=said["grounded_in_phrase"],
        fallback=said["fallback"], turn_ids=[t["id"] for t in turns],
        prompt=said["prompt"], response=said["response"])
    return turns
