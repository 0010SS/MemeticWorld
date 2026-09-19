"""Lossy, reconstructive memory encoding: observation -> symbolic memory node.

Pipeline (world event never touches this code; only an AgentObservation does):
  1. semantic pre-filter (deterministic given the seed):
       - drop low-salience facts with p = noise * (1 - salience)
       - generalise unfamiliar people ("Leo" -> "a student") with
         p = entity_generalization * noise
  2. reconstruction by the LLM in the agent's own framing, with a fidelity
     instruction set by `memory.encoding_noise` (0 = verbatim, no LLM rewrite)
     and a randomly sampled memory lens (focus, length, fidelity jitter,
     uncertainty, occasional distortion; `memory.encoding_variability`, D44).
     Facts the agent took part in are framed as its own experience (D43).
     Distinctive wording heard from OTHER people may stick word for word
     (`memory.verbatim`, D49 fixed in v2); stuck phrases become Wording records.
     The prompt spells out the agent's relationship only to people it can name: for perception, the
     people it recognised (D42) or who are still named in what it perceived; a stranger the viewpoint
     renderer or the pre-filter turned into "someone" / "a student" is not named again here.
  3. importance = upstream GA poignancy prompt (same for every condition)
  4. experimental modules may adjust the draft (emotion, prestige, social reward)
  5. merge into a near-duplicate recent memory (if noise > 0), else store
  6. forget the weakest memory while over capacity

Randomness (ontology v2 §2.1): the pre-filter uses the rng passed in; the lens draws only from
agent.stream("lens"), its association from agent.stream("assoc") and stickiness from
agent.stream("verbatim"), so switching one mechanism on never changes another one's draws.
"""
from __future__ import annotations

import functools
import re

from backend import ga_compat
from backend.agents import ga_prompts
from backend.agents.perception import RECOGNISE_FAMILIARITY
from backend.llm.client import llm_purpose
from backend.llm.embeddings import cos
from backend.memory import wording
from backend.memory.store import MemoryMeta, recency_score, record_link

ga = ga_compat.load()
PROMPT_DIR = ga_compat.REPO_ROOT / "backend" / "prompts"
ENCODE_PROMPT = "encode_memory_v2.txt"

VERB = {"perception": "saw", "conversation": "heard in a conversation",
        "overheard": "overheard", "self": "did",
        "record": "read in the co-op binder"}         # v3 §2.3: binder reads (backend/simulation/records.py)
HEARD = ("conversation", "overheard")      # default sources whose wording can stick verbatim


def verbatim_sources(agent) -> tuple:
    """Sources whose wording may stick (`memory.verbatim.on_sources`, default HEARD; v3_base adds record)."""
    return tuple((agent.cfg["memory"].get("verbatim") or {}).get("on_sources") or HEARD)


def fidelity_instruction(noise: float, first: str) -> str:
    if noise < 0.15:
        return "Keep essentially every detail, including the exact wording people used."
    if noise < 0.45:
        return (f"Keep the main points and anything that stood out to {first}; small details fade. "
                f"Paraphrase, except for any wording that particularly stuck with {first}.")
    if noise < 0.75:
        return (f"Keep only the gist and what mattered to {first}, in one or two sentences in {first}'s own "
                f"words; minor details and exact wording fade.")
    return ("Only a vague gist survives: one short sentence, no quotes; details, times and secondary "
            "people blur (\"someone\", \"something went wrong\").")


FOCI = ["the people involved", "what was said or done", "the place and the scene", "how it made {first} feel",
        "the oddest detail", "what it means for {first}'s own plans", "who else was around"]
LENGTHS = ["one short sentence", "one or two sentences", "two or three sentences"]
STYLES = ["as a plain note to self", "in {first}'s own casual voice", "as a quick mental snapshot",
          "the way {first} would later tell it to a friend"]


def sample_lens(noise: float, var: float, first: str, rng) -> dict:
    """Random, seeded shape of one act of remembering (D44). Recorded LLM calls replay it exactly."""
    if var <= 0:
        return {"noise": noise, "text": ""}
    eff = float(min(1.0, max(0.05, noise + rng.normal(0, 0.2 * var))))
    focus = FOCI[int(rng.integers(len(FOCI)))].format(first=first)
    length = LENGTHS[int(rng.integers(len(LENGTHS)))]
    unsure = bool(rng.random() < 0.3 * var)
    distort = bool(rng.random() < 0.5 * var * noise)
    style = STYLES[int(rng.integers(len(STYLES)))].format(first=first)
    lines = [f"What stuck with {first} most was {focus}. Write {length}, {style}."]
    if unsure:
        lines.append(f"{first} is not completely sure about some of the details.")
    if distort:
        lines.append(f"{first} misremembers one minor detail (for example the time, the exact place, "
                     f"an object, or who else was there).")
    return {"noise": round(eff, 3), "focus": focus, "length": length, "style": style, "unsure": unsure,
            "distort": distort, "associate": bool(rng.random() < 0.35 * var), "text": " ".join(lines)}


def _association(agent, raw: str, rng):
    """Reconstruction borrows from what the new experience brings to mind: one related older memory,
    drawn by the (stochastic) retrieval, may colour the new one. Returns that node or None."""
    from backend.memory.retrieval import retrieve
    nodes = retrieve(agent, [raw], rng=rng, touch=False)[raw].nodes[:5]
    if not nodes:
        return None
    return nodes[int(rng.integers(len(nodes)))]


_TOK = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_BREAK = re.compile(r"[.,!?;:\"()\[\]“”—–…]|\s-\s|--")   # spans never cross these
# function words, light verbs, discourse markers and adverbs: none may sit at the edge of a stuck span
FUNCTION_WORDS = frozenset("""
a an the this that these those some any each every either neither no all both half several many much more most
few fewer less least other another such what which whatever whichever whose own same whole entire enough
i me my mine myself you your yours yourself yourselves he him his himself she her hers herself it its itself
we us our ours ourselves they them their theirs themselves one ones someone somebody something anyone anybody
anything everyone everybody everything nobody nothing none who whom y'all
about above across after against along amid among around as at before behind below beneath beside besides
between beyond by despite down during except for from in inside into like near of off on onto out outside over
past per since than through throughout till to toward towards under underneath until up upon via with within
without and but or nor so yet because cause although though while whereas if unless whether once
am is are was were be been being have has had having do does did doing done can could may might must shall
should will would ought get gets got getting gotten go goes going went gone say says said saying tell told
know knew known think thought mean meant guess let make made take took come came see saw look looked
want wanted need needed keep keeps kept keeping start started try tried trying seem seems seemed
feel feels felt give gave put use used gonna wanna gotta kinda sorta lemme dunno
not yes yeah yep yup nope oh ah uh um hmm huh hey hi okay ok well just very really quite rather pretty too
also even still already again ever never always often sometimes usually maybe perhaps probably definitely
basically literally actually honestly seriously totally completely absolutely clearly obviously apparently
anyway anyways somehow somewhat almost nearly only simply exactly especially mostly mainly kind sort lot lots
bit here there where when why how then now today tonight tomorrow yesterday later soon earlier early once twice
back away else instead ago right sure super
two three four five six seven eight nine ten
""".split())
_CONTRACTION_SUFFIX = ("t", "ll", "re", "ve", "d", "m")


@functools.lru_cache(maxsize=65536)
def _zipf(w: str) -> float:
    try:
        from wordfreq import zipf_frequency
        return zipf_frequency(w, "en")
    except ImportError:
        return 3.0


def _is_ly_adverb(w: str) -> bool:
    """'weirdly', 'basically', 'probably', 'happily': an -ly word whose stem is itself a word. Some -ly
    adjectives ('friendly', 'costly') count too; they are common words that only lose edge positions."""
    if len(w) < 6 or not w.endswith("ly"):
        return False
    stems = {w[:-2], w[:-2] + "e", w[:-1] + "e" if w.endswith("bly") else "",
             w[:-3] + "y" if w.endswith("ily") else "", w[:-4] if w.endswith("ically") else ""}
    return any(len(st) >= 3 and _zipf(st) >= 3.1 for st in stems)


def is_function_word(tok: str) -> bool:
    """Function words, light verbs, adverbs and contractions ("mine's", "don't"); a possessive of a
    content word ("printer's") is content."""
    w = tok.lower().strip("'-")
    if w in FUNCTION_WORDS:
        return True
    if "'" in w:
        stem, suf = w.split("'", 1)
        return stem in FUNCTION_WORDS or suf in _CONTRACTION_SUFFIX
    return _is_ly_adverb(w)


def _base(tok: str) -> str:
    return tok.lower().strip("'-").split("'")[0]


def _person_names(agent) -> frozenset:
    return frozenset(t.lower() for o in agent.ctx.agents.values() for t in o.name.split())


def _lexicon_tokens(agent) -> frozenset:
    """Population lexicon (campus/routine/profile vocabulary): the engine's `ctx.lexicon` if set, else
    computed once from the population's profiles; the token set is cached on ctx."""
    ctx = agent.ctx
    toks = getattr(ctx, "_lexicon_tokens", None)
    if toks is None:
        lex = getattr(ctx, "lexicon", None)
        if lex is None:
            from backend.simulation.lexicon import population_lexicon
            lex = population_lexicon({a.id: a.profile for a in ctx.agents.values()})
        toks = frozenset(lex["tokens"])
        try:
            ctx._lexicon_tokens = toks
        except AttributeError:
            pass
    return toks


def distinctive_phrases(agent, text: str, zipf_max: float, mode: str | None = None) -> list[tuple[str, float]]:
    """Unusual wording in something heard: 2-3 word spans around a rare content word whose first and
    last tokens are content words (not function words, adverbs or contractions), within one clause, with
    no names of people. Distinctive wording is what people tend to remember verbatim (von Restorff); this
    is agent-side memory salience, independent of the observer's analysis (D49).
    mode "lexicon" (default): the rare word must also not be campus/routine vocabulary every agent is
    handed by its world and profile; "zipf": rarity only. -> [(phrase, zipf of its rarest word)]"""
    mode = mode or (agent.cfg["memory"].get("verbatim") or {}).get("distinctiveness", "lexicon")
    names = _person_names(agent)
    lex = _lexicon_tokens(agent) if mode == "lexicon" else frozenset()
    out: dict[str, tuple[str, float]] = {}
    for seg in _BREAK.split(text.replace("’", "'")):
        toks = [t.strip("'-") for t in _TOK.findall(seg)]
        toks = [t for t in toks if t]
        base = [_base(t) for t in toks]
        name = [b in names for b in base]
        content = [not is_function_word(t) and not nm for t, nm in zip(toks, name)]
        for i, b in enumerate(base):
            if not content[i] or len(b) < 4 or b in lex:
                continue
            z = _zipf(b)
            if z > zipf_max:
                continue
            for n in (2, 3):
                for j in range(max(0, i - n + 1), min(i, len(toks) - n) + 1):
                    if not (content[j] and content[j + n - 1]) or any(name[j:j + n]):
                        continue
                    ph = " ".join(toks[j:j + n])
                    if ph.lower() not in out or z < out[ph.lower()][1]:
                        out[ph.lower()] = (ph, z)
    return sorted(out.values(), key=lambda kv: (kv[1], kv[0].lower()))


def _speaker(agent, fact: dict) -> str | None:
    """Who said a heard line: fact["speaker"] if given, else the person it involves; conversation facts
    for the agent's own lines involve nobody."""
    if fact.get("speaker"):
        return fact["speaker"]
    inv = fact.get("involves")
    if inv:
        return inv[0]
    return agent.id if inv is not None else None


def sticky_phrases(agent, facts, rng) -> list[dict]:
    """Which exact wordings survive into memory -> [{"phrase", "heard_from", "utterance_id"}].
    p = base + gain * (#existing memories already containing the phrase), so wording the agent has met
    before sticks more readily (familiarity; no extra state). With `exclude_self`, only other people's
    lines are candidates. `rng` should be agent.stream("verbatim")."""
    vc = agent.cfg["memory"].get("verbatim") or {}
    if not vc.get("enabled"):
        return []
    zmax = float(vc.get("zipf_max", 3.6))
    cands = []
    for f in facts:
        who = _speaker(agent, f)
        if vc.get("exclude_self", True) and who == agent.id:
            continue
        for ph, _z in distinctive_phrases(agent, f["text"].split(":", 1)[-1], zmax):
            cands.append({"phrase": ph, "heard_from": who, "utterance_id": f.get("id")})
    if not cands:
        return []
    mem = [n.description.lower() for n in agent.a_mem.all_nodes()]
    chosen, used_words = [], set()
    for c in cands:
        if len(chosen) >= int(vc.get("max_phrases", 2)):
            break
        words = set(c["phrase"].lower().split())
        if words & used_words:
            continue
        seen = sum(1 for m in mem if c["phrase"].lower() in m)
        p = min(float(vc.get("max", 0.9)), float(vc.get("base", 0.3)) + float(vc.get("familiarity_gain", 0.2)) * seen)
        if rng.random() < p:
            chosen.append(c)
            used_words |= words
    return chosen


def _prefilter(agent, obs, noise: float, gen: float, rng):
    facts = list(obs.facts)
    ops = []
    if obs.source_type == "perception" and len(facts) > 1 and noise > 0:
        best = max(range(len(facts)), key=lambda i: facts[i]["salience"])
        keep = []
        for i, f in enumerate(facts):
            if i != best and rng.random() < noise * (1 - f["salience"]):
                ops.append({"op": "drop_fact", "fact": f["id"]})
            else:
                keep.append(f)
        facts = keep
    if noise > 0 and gen > 0:
        out = []
        for f in facts:
            text = f["text"]
            for other_id in f.get("involves", []):
                if other_id == agent.id:
                    continue
                if agent.profile.rel(other_id).familiarity < 0.3 and rng.random() < gen * noise:
                    first = agent.ctx.agents[other_id].profile.first_name
                    text = re.sub(rf"\b{re.escape(first)}\b", "a student", text, count=1)
                    text = re.sub(rf"\b{re.escape(first)}\b", "they", text)
                    text = text[0].upper() + text[1:]
                    ops.append({"op": "generalize_entity", "who": other_id})
            out.append(dict(f, text=text))
        facts = out
    return facts, ops


def context_people(agent, obs, facts, involves) -> list[str]:
    """Ids of the other people the agent can put a name to, in `involves` order. Heard sources: everyone
    involved (speakers are named in what was heard). Perception: people the agent recognised (the facts'
    `recognised`, from perception.observe; familiarity >= RECOGNISE_FAMILIARITY for facts without it) and
    people still named in the perceived text (viewpoints off). Never a stranger whose name the viewpoint
    guard (D42) or the pre-filter's generalisation took out."""
    others = [a for a in involves if a != agent.id]
    if obs.source_type != "perception":
        return others
    known = set()
    for f in facts:
        if "recognised" in f:
            known.update(f["recognised"])
        else:
            known.update(a for a in f.get("involves", ())
                         if a != agent.id and agent.profile.rel(a).familiarity >= RECOGNISE_FAMILIARITY)
    text = "\n".join(f["text"] for f in facts)
    out = []
    for a in others:
        o = agent.ctx.agents[a]
        if a in known or any(re.search(rf"\b{re.escape(n)}\b", text) for n in (o.name, o.profile.first_name)):
            out.append(a)
    return out


def _keywords(agent, text: str, involves: list[str]) -> set:
    kws = {agent.name.lower()}
    for a in involves:
        kws.add(agent.ctx.agents[a].name.lower())
    for w in re.findall(r"[A-Za-z][a-z]{4,}", text):
        if len(kws) >= 8:
            break
        kws.add(w.lower())
    return kws


def encode(agent, obs, rng) -> object | None:
    """Encode an AgentObservation into agent memory. Returns the node (new or merged)."""
    ctx = agent.ctx
    mc = agent.cfg["memory"]
    noise = float(mc["encoding_noise"])
    facts, ops = _prefilter(agent, obs, noise, float(mc.get("entity_generalization", 0.5)), rng)
    if not facts:
        return None
    raw = "\n".join(f["text"] for f in facts)
    involves = sorted({a for f in facts for a in f.get("involves", [])} | set(obs.speakers))
    named = context_people(agent, obs, facts, involves)    # who the agent can name (D42 name guard)
    first = agent.profile.first_name
    prompt = None
    lens = None
    # facts the agent took part in are its own experience, not something it watched someone else do
    self_part = obs.source_type == "perception" and any(agent.id in f.get("involves", []) for f in facts)
    verb = "experienced first-hand" if self_part else VERB[obs.source_type]
    stuck, assoc = [], None
    if noise <= 0:
        text = f"{agent.name} {'experienced' if self_part else VERB[obs.source_type]}: " + " ".join(f["text"] for f in facts)
    else:
        vc = mc.get("verbatim") or {}
        people = "\n".join(agent.relationship_line(ctx.agents[a]) for a in named)
        lens = sample_lens(noise, float(mc.get("encoding_variability", 1.0)), first, agent.stream("lens"))
        if obs.source_type in verbatim_sources(agent):
            stuck = sticky_phrases(agent, facts, agent.stream("verbatim"))
        render = bool(stuck) and vc.get("render_in_text", True)
        if stuck:
            lens["verbatim"] = [w["phrase"] for w in stuck]
        if render:
            lens["text"] += " " + " ".join(f'The exact words "{ph}" stuck with {first}; keep them word for word, in quotes.'
                                          for ph in lens["verbatim"])
        if lens.get("associate"):
            assoc = _association(agent, raw, agent.stream("assoc"))
            if assoc is not None:
                lens["association"] = assoc.description
                lens["association_node"] = assoc.node_id
                lens["text"] += f" It brought to mind something {first} already remembered: \"{assoc.description}\""
        note = (f"(Where these mention {first}, they describe what {first} did or what happened to {first}; "
                f"{first} remembers them as their own experience.)") if self_part else ""
        prompt = ga.gs.generate_prompt(
            [agent.name, agent.iss(), agent.scratch.curr_time.strftime("%A %H:%M"),
             f"{obs.location} ({obs.arena})", verb, raw,
             fidelity_instruction(lens["noise"], first), people, lens["text"], note],
            str(PROMPT_DIR / ENCODE_PROMPT))
        with llm_purpose("encode_memory", agent.id):
            text = ctx.llm.complete(prompt, max_tokens=160, temperature=1.0)
        text = text.strip().strip('"').strip()
        if not text or text.startswith("LLM_ERROR"):
            text = f"{agent.name} {verb}: " + " ".join(f["text"] for f in facts)
        text = " ".join(text.split())[:600]
        for ph in (lens["verbatim"] if render else []):
            if ph.lower() not in text.lower():            # the model dropped it: the wording still stuck
                text += f' {first} remembers the words "{ph}".'

    kind = "event" if obs.source_type in ("perception", "record") else "chat"
    record_ids = [f["record_id"] for f in facts if f.get("record_id")]     # v3 binder items (§2.3)
    importance = ga_prompts.poignancy(agent, text, "chat" if kind == "chat" else "event")
    salience = max(f["salience"] for f in facts)
    draft = {"text": text, "importance": importance, "salience": salience, "source_type": obs.source_type,
             "speakers": list(obs.speakers), "involves": involves}
    draft = agent.mods.modify_memory(agent, draft)
    emb = ctx.embed(draft["text"])
    now = agent.scratch.curr_time

    # merge into a near-duplicate recent memory
    if noise > 0:
        recent = [n for n in agent.a_mem.all_nodes() if n.type == kind][:40]
        best, best_sim = None, 0.0
        for n in recent:
            s = cos(emb, agent.a_mem.embeddings[n.embedding_key])
            if s > best_sim:
                best, best_sim = n, s
        if best is not None and best_sim >= float(mc["merge_threshold"]):
            best.poignancy = max(best.poignancy, draft["importance"])
            best.last_accessed = now
            m = ctx.meta.get(best.node_id)
            if m:
                for e in obs.event_ids:
                    if e not in m.originating_event_ids:
                        m.originating_event_ids.append(e)
                m.source_ids.append(obs.id)
                m.record_ids.extend(r for r in record_ids if r not in m.record_ids)
            ctx.tracer.log("memory_merged", agent=agent.id, node_id=best.node_id, into_text=best.description,
                           dropped_text=draft["text"], similarity=round(best_sim, 3), observation_id=obs.id,
                           originating_event_ids=list(obs.event_ids))
            # the new experience was taken for the old one: a link, and what stuck now lives on the old memory
            record_link(agent, obs.id, best.node_id, "merge", f"similarity {best_sim:.3f}", holder=best.node_id,
                        from_event_ids=list(obs.event_ids))
            if stuck:
                wording.record(agent, best.node_id, stuck, obs)
            return best

    s_name = agent.name
    o_name = obs.partner or (ctx.agents[named[0]].name if named else obs.location)
    pred = {"perception": "saw", "conversation": "chat with", "overheard": "overheard",
            "record": "read"}.get(obs.source_type, "noticed")
    node = agent.a_mem.add(kind, now, s_name, pred, o_name, draft["text"],
                           _keywords(agent, draft["text"], named), draft["importance"], emb, [])
    ctx.meta.set(node.node_id, MemoryMeta(agent_id=agent.id, source_type=obs.source_type, salience=salience,
                                          originating_event_ids=list(obs.event_ids),
                                          speakers=list(obs.speakers),
                                          source_ids=[obs.id] + list(obs.utterance_ids),
                                          record_ids=record_ids))
    agent.scratch.importance_trigger_curr -= draft["importance"]
    agent.scratch.importance_ele_n += 1
    ctx.tracer.log("memory_encoded", agent=agent.id, node_id=node.node_id, kind=kind, text=node.description,
                   importance=node.poignancy, salience=salience, source_type=obs.source_type,
                   observation_id=obs.id, observation=raw, encoding_ops=ops, prompt=prompt, lens=lens,
                   self_experience=self_part,
                   originating_event_ids=list(obs.event_ids), speakers=list(obs.speakers),
                   utterance_ids=list(obs.utterance_ids), **({"record_ids": record_ids} if record_ids else {}))
    if stuck:
        wording.record(agent, node.node_id, stuck, obs)
    if assoc is not None:
        record_link(agent, node.node_id, assoc.node_id, "association", "brought to mind while remembering")
    forget(agent)
    return node


def add_simple_event(agent, text: str, importance: int, involves: list[str], source_ids=()):
    """Low-importance perception of someone's routine activity (no LLM calls)."""
    ctx = agent.ctx
    emb = ctx.embed(text)
    o = ctx.agents[involves[0]].name if involves else agent.state.location
    node = agent.a_mem.add("event", agent.scratch.curr_time, agent.name, "saw", o, text,
                           _keywords(agent, text, involves), importance, emb, [])
    ctx.meta.set(node.node_id, MemoryMeta(agent_id=agent.id, source_type="ambient", salience=0.1,
                                          speakers=[], source_ids=list(source_ids)))
    agent.scratch.importance_trigger_curr -= importance
    agent.scratch.importance_ele_n += 1
    ctx.tracer.log("memory_encoded", agent=agent.id, node_id=node.node_id, kind="event", text=text,
                   importance=importance, salience=0.1, source_type="ambient", observation_id=None,
                   observation=text, encoding_ops=[], prompt=None, originating_event_ids=[], speakers=[],
                   utterance_ids=[])
    forget(agent)
    return node


def forget(agent):
    cap = int(agent.cfg["memory"]["memory_capacity"])
    nodes = agent.a_mem.all_nodes()
    if len(nodes) <= cap:
        return
    now = agent.scratch.curr_time
    dr = float(agent.cfg["memory"]["decay_rate"])
    ranked = sorted(nodes, key=lambda n: ((n.poignancy / 10.0) * recency_score(n, now, dr), n.node_id))
    for n in ranked[: len(nodes) - cap]:
        agent.a_mem.remove(n.node_id)
        agent.ctx.tracer.log("memory_forgotten", agent=agent.id, node_id=n.node_id, text=n.description)
