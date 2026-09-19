"""Find phrases that were *transmitted* between agents, not just repeated.

A candidate phrase is a 1-4 token n-gram from agent speech. For each phrase we replay the run
in event order and classify every speaker:

  exposure      agent X heard the phrase spoken by someone else (addressee, audience, or overhearer)
  independent   X's first use came with no prior exposure. The originator is the first
                independent user. Many independent users means ordinary language or a shared
                model prior, not a meme.
  echo          X used it in the same conversation where X just heard it (not adoption)
  adopter       X used it in a *different* conversation after being exposed. This is transmission.
  confirmed     the adopting utterance's prompt contained a retrieved memory that quotes the
                phrase from someone else: direct evidence it traveled through memory

Exposure is not causation: two agents who both saw the tray drop may both say "tray incident".
So when a control run (same seed, condition=no_speech_memory) exists, every phrase is also
traced there. A phrase that spreads just as well without speech memory is a shared model prior
or an obvious description, and is demoted to tier "baseline".

Tiers: strong = one originator + at least one memory-confirmed adoption + no spread in control;
       suggestive = passes the filters but lacks that evidence; baseline = also spreads in control.

Filters drop phrases built only from vocabulary the world hands the agents (names, places,
event descriptions, schedule activities, prompt wording): quoting the environment isn't
invention. Everything is heuristic and meant to be read alongside the control run.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from app.db import database as db
from app.llm import prompts

TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")
MAX_N = 4
MIN_USERS = 2
MIN_USES = 3
MIN_ADOPTERS = 1
MAX_INDEPENDENT = 2
NESTING_RATIO = 0.8

STOPWORDS = set("""
a about above after again against all almost also am an and any anyone anything are aren't around as at away back
be because been before being below between both but by can can't cannot could couldn't did didn't do does doesn't
doing don't down during each else enough even ever every few for from further get gets getting go goes going gone
got gotta had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his
how how's i i'd i'll i'm i've if in into is isn't it it'd it'll it's its itself just let let's like lot lots made
make makes making many me might more most much must mustn't my myself need needs never no nor not now of off oh ok
okay on once one only or other ought our ours ourselves out over own pretty probably quite rather really right
same say says said see seems shall shan't she she'd she'll she's should shouldn't so some something still such sure
than that that's the their theirs them themselves then there there's these they they'd they'll they're they've
thing things think this those though through to too totally under until up us very want wanna was wasn't way we
we'd we'll we're we've well were weren't what what's whatever when when's where where's whether which while who
who's whole whom why why's will with won't would wouldn't yeah yep yes yet you you'd you'll you're you've your
yours yourself yourselves
ha haha hah lol omg wow ugh hmm hm um uh yo hey hi hello bye thanks thank please sorry
actually basically definitely honestly literally maybe seriously kind sort bit gonna know knew mean means guess
feel feels felt look looks looking looked tell told come came coming take took day days time times today
tonight tomorrow yesterday morning afternoon evening week good great nice cool fine bad best better big little
new old last next first two three
""".split())


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower().replace("’", "'"))


def stem(token: str) -> str:
    """Crude suffix stripping so 'jams'/'jammed'/'jamming' all hit the world vocabulary."""
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    if len(token) >= 4 and token[-1] == token[-2]:
        token = token[:-1]
    return token


def normalized(text: str) -> str:
    return " " + " ".join(tokenize(text)) + " "


@dataclass
class Utterance:
    event_id: int
    tick: int
    sim_time: str
    speaker: str
    location: str
    text: str
    conversation_id: str
    kind: str
    audience: list[str]
    overheard_by: list[str]
    memory_ids: list[int]
    tokens: list[str] = field(default_factory=list)


@dataclass
class Corpus:
    run_id: int
    run: dict
    utterances: list[Utterance]
    world_events: list[dict]
    world_vocab: set[str]
    names: dict[str, str]                 # agent id -> display name
    memories: dict[int, dict]             # memory id -> {text, source_type, source_event_id}
    by_event: dict[int, Utterance]
    index: dict[str, list[int]]           # phrase -> utterance indices (event order)


def world_vocabulary(config: dict, world_event_texts: list[str]) -> set[str]:
    """Every word the environment puts in front of the agents. Stemmed."""
    texts: list[str] = list(world_event_texts)
    for agent in config.get("agents", []):
        texts += [agent["name"], agent["role"], agent["personality"], *agent["interests"]]
        texts += [label for label, _ in agent["relationships"].values()]
        texts += [f"{loc} {activity}" for _, loc, activity in agent["schedule"]]
    for loc in config.get("world", {}).get("locations", []):
        texts += [loc["name"], loc["description"]]
    texts += [t["text"] for t in config.get("event_templates", [])]
    texts += [prompts.SYSTEM, prompts.DECISION, prompts.REPLY, "talking with walking to hanging around"]
    vocab: set[str] = set()
    for text in texts:
        for token in tokenize(text.replace("{", " ").replace("}", " ")):
            vocab.add(stem(token))
            vocab.update(stem(part) for part in token.split("-"))
    return vocab


def _is_content(token: str) -> bool:
    return token not in STOPWORDS and not token.isdigit() and len(token) > 1


def _novel(token: str, world_vocab: set[str]) -> bool:
    return _is_content(token) and stem(token) not in world_vocab


def candidate_ngrams(tokens: list[str], world_vocab: set[str]) -> set[str]:
    grams: set[str] = set()
    for n in range(1, MAX_N + 1):
        for i in range(len(tokens) - n + 1):
            gram = tokens[i:i + n]
            if not (_is_content(gram[0]) and _is_content(gram[-1])):
                continue
            if n == 1 and len(gram[0]) < 3:
                continue
            if not any(_novel(t, world_vocab) for t in gram):
                continue
            grams.add(" ".join(gram))
    return grams


def load_corpus(conn, run_id: int) -> Corpus:
    run = db.get_run(conn, run_id)
    if run is None:
        raise ValueError(f"run {run_id} not found")
    rows = db.fetch_events(conn, run_id, types=["utterance", "world_event"], limit=10**9)
    world_events = [r for r in rows if r["type"] == "world_event"]
    utterances = []
    for r in rows:
        if r["type"] != "utterance" or not r["text"]:
            continue
        d = r["data"]
        utterances.append(Utterance(
            event_id=r["id"], tick=r["tick"], sim_time=r["sim_time"], speaker=r["agent_id"], location=r["location"],
            text=r["text"], conversation_id=d.get("conversation_id", str(r["id"])), kind=d.get("kind", "talk"),
            audience=d.get("audience", []), overheard_by=d.get("overheard_by", []),
            memory_ids=d.get("memory_ids", []), tokens=tokenize(r["text"])))

    vocab = world_vocabulary(run["config"], [e["text"] for e in world_events])
    index: dict[str, list[int]] = defaultdict(list)
    for i, u in enumerate(utterances):
        for gram in candidate_ngrams(u.tokens, vocab):
            index[gram].append(i)

    memory_ids = sorted({m for u in utterances for m in u.memory_ids})
    memories: dict[int, dict] = {}
    for start in range(0, len(memory_ids), 900):
        for m in db.fetch_memories(conn, run_id, ids=memory_ids[start:start + 900], limit=900):
            memories[m["id"]] = {"text": m["text"], "source_type": m["source_type"],
                                 "source_event_id": m["source_event_id"], "norm": normalized(m["text"])}

    names = {a["id"]: a["name"] for a in run["config"].get("agents", [])}
    return Corpus(run_id, run, utterances, world_events, vocab, names, memories,
                  {u.event_id: u for u in utterances}, dict(index))


def trace_phrase(corpus: Corpus, phrase: str) -> dict:
    """Replay one phrase through the run and classify every use and user."""
    uses = [corpus.utterances[i] for i in corpus.index.get(phrase, [])]
    padded = f" {phrase} "
    exposures: dict[str, list[Utterance]] = defaultdict(list)
    first_use: dict[str, Utterance] = {}
    independent: list[str] = []
    adoption: dict[str, tuple[Utterance, dict]] = {}
    use_rows = []

    for u in uses:
        prior = exposures.get(u.speaker, [])
        if u.speaker not in first_use:
            first_use[u.speaker] = u
            if not prior:
                independent.append(u.speaker)
        heard_here = any(e.conversation_id == u.conversation_id for e in prior)
        if not prior:
            role = "origin" if u.speaker == independent[0] and first_use[u.speaker] is u else (
                "independent" if first_use[u.speaker] is u else "reuse")
        elif heard_here:
            role = "echo"
        elif u.speaker not in adoption and u.speaker not in independent:
            role = "adoption"
            adoption[u.speaker] = (u, _attribute(corpus, u, prior, padded))
        else:
            role = "reuse"
        use_rows.append({"event_id": u.event_id, "tick": u.tick, "sim_time": u.sim_time, "speaker": u.speaker,
                         "location": u.location, "text": u.text, "conversation_id": u.conversation_id,
                         "role": role})
        for listener in [*u.audience, *u.overheard_by]:
            if listener != u.speaker:
                exposures[listener].append(u)

    edges = [{"source": src["source"], "target": agent, "tick": u.tick, "evidence": src["evidence"],
              "memory_id": src.get("memory_id"), "exposure_event_id": src["event_id"], "adoption_event_id": u.event_id}
             for agent, (u, src) in adoption.items()]
    users = list(first_use)
    return {
        "phrase": phrase,
        "uses": use_rows,
        "n_uses": len(uses),
        "users": users,
        "n_users": len(users),
        "originator": independent[0] if independent else None,
        "independent": independent,
        "adopters": list(adoption),
        "n_adopters": len(adoption),
        "n_confirmed": sum(1 for e in edges if e["evidence"] == "memory"),
        "edges": edges,
        "first_use": {a: {"tick": u.tick, "event_id": u.event_id} for a, u in first_use.items()},
        "first_tick": uses[0].tick if uses else None,
        "last_tick": uses[-1].tick if uses else None,
        "exposed": sorted(exposures),
        "depth": _depth(edges),
    }


def _attribute(corpus: Corpus, use: Utterance, prior: list[Utterance], padded: str) -> dict:
    """Who did the adopter get it from? Prefer a retrieved memory quoting the phrase."""
    for memory_id in use.memory_ids:
        memory = corpus.memories.get(memory_id)
        if not memory or memory["source_type"] not in ("heard", "overheard") or padded not in memory["norm"]:
            continue
        source = corpus.by_event.get(memory["source_event_id"])
        if source and source.speaker != use.speaker:
            return {"source": source.speaker, "evidence": "memory", "memory_id": memory_id, "event_id": source.event_id}
    last = prior[-1]
    return {"source": last.speaker, "evidence": "exposure", "event_id": last.event_id}


def _depth(edges: list[dict]) -> int:
    parent = {e["target"]: e["source"] for e in edges}
    best = 0
    for node in parent:
        depth, seen = 0, set()
        while node in parent and node not in seen:
            seen.add(node)
            node = parent[node]
            depth += 1
        best = max(best, depth)
    return best


def spread(stats: dict | None) -> dict:
    """Compact numbers for tables and control comparisons."""
    if not stats or not stats["n_uses"]:
        return {"n_uses": 0, "n_users": 0, "n_adopters": 0, "n_independent": 0, "n_confirmed": 0}
    return {"n_uses": stats["n_uses"], "n_users": stats["n_users"], "n_adopters": stats["n_adopters"],
            "n_independent": len(stats["independent"]), "n_confirmed": stats["n_confirmed"]}


def score(stats: dict) -> float:
    base = (stats["n_adopters"] + 0.5 * stats["n_confirmed"] + 0.1 * stats["n_uses"]) / max(1, len(stats["independent"]))
    control = stats.get("control")
    return base / (1 + control["n_adopters"]) if control else base


def tier(stats: dict) -> str:
    control = stats.get("control")
    if control and (control["n_adopters"] > 0 or control["n_users"] >= 3):
        return "baseline"
    if stats["n_confirmed"] > 0 and len(stats["independent"]) == 1:
        return "strong"
    return "suggestive"


def variants(corpus: Corpus, meme: dict, limit: int = 6) -> list[dict]:
    """Longer phrasings that contain this phrase ("tray-gate" -> "classic tray-gate", ...)."""
    tokens = meme["phrase"].split()
    counts: dict[str, int] = defaultdict(int)
    for use in meme["uses"]:
        for gram in candidate_ngrams(corpus.by_event[use["event_id"]].tokens, corpus.world_vocab):
            gram_tokens = gram.split()
            if len(gram_tokens) > len(tokens) and _contains(gram_tokens, tokens):
                counts[gram] += 1
    best = sorted(counts.items(), key=lambda kv: (-kv[1], len(kv[0])))
    return [{"phrase": p, "n_uses": n} for p, n in best if n >= 2][:limit]


def _contains(longer: list[str], shorter: list[str]) -> bool:
    n = len(shorter)
    return any(longer[i:i + n] == shorter for i in range(len(longer) - n + 1))


def detect_memes(corpus: Corpus, control: Corpus | None = None, limit: int = 40) -> list[dict]:
    candidates = []
    for phrase, idx in corpus.index.items():
        if len(idx) < MIN_USES or len({corpus.utterances[i].speaker for i in idx}) < MIN_USERS:
            continue
        stats = trace_phrase(corpus, phrase)
        if stats["n_adopters"] < MIN_ADOPTERS or len(stats["independent"]) > MAX_INDEPENDENT:
            continue
        stats["control"] = spread(trace_phrase(control, phrase)) if control else None
        stats["score"] = round(score(stats), 3)
        stats["tier"] = tier(stats)
        candidates.append(stats)

    # For nested phrases keep one: the longer form if it covers most uses of the shorter, else the shorter.
    tokens = {c["phrase"]: c["phrase"].split() for c in candidates}
    dropped: set[str] = set()
    for short in candidates:
        for long in candidates:
            s, l = short["phrase"], long["phrase"]
            if len(tokens[l]) <= len(tokens[s]) or not _contains(tokens[l], tokens[s]):
                continue
            if long["n_uses"] >= NESTING_RATIO * short["n_uses"]:
                dropped.add(s)
            else:
                dropped.add(l)
    memes = [c for c in candidates if c["phrase"] not in dropped]
    tier_rank = {"strong": 0, "suggestive": 1, "baseline": 2}
    memes.sort(key=lambda c: (tier_rank[c["tier"]], -c["score"], c["first_tick"]))
    memes = memes[:limit]
    for rank, meme in enumerate(memes):
        meme["id"] = rank
        meme["variants"] = variants(corpus, meme)
    return memes


def find_control(conn, run: dict) -> int | None:
    """Latest no_speech_memory run with the same seed and length as a full run."""
    if run["condition"] != "full":
        return None
    row = conn.execute(
        "SELECT id FROM runs WHERE condition = 'no_speech_memory' AND seed = ? AND days = ? AND current_tick > 0"
        " ORDER BY id DESC LIMIT 1", (run["seed"], run["days"])).fetchone()
    return int(row[0]) if row else None


def summarize(memes: list[dict]) -> dict:
    return {
        "n_memes": len(memes),
        "n_strong": sum(1 for m in memes if m.get("tier") == "strong"),
        "n_baseline": sum(1 for m in memes if m.get("tier") == "baseline"),
        "total_adopters": sum(m["n_adopters"] for m in memes),
        "total_confirmed": sum(m["n_confirmed"] for m in memes),
        "max_users": max((m["n_users"] for m in memes), default=0),
        "max_depth": max((m["depth"] for m in memes), default=0),
    }
