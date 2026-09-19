"""Views over detected memes: per-meme cascades, run-level adoption curves, the social graph,
and full-vs-control comparisons."""
from __future__ import annotations

from collections import Counter

from app.analysis.meme_detector import Corpus, detect_memes, spread, summarize, trace_phrase


def cascade(corpus: Corpus, meme: dict) -> dict:
    """Per-agent roles and a tick-by-tick timeline for one meme."""
    adoption_tick = {e["target"]: e["tick"] for e in meme["edges"]}
    exposed_tick: dict[str, int] = {}
    for use in meme["uses"]:
        u = corpus.by_event[use["event_id"]]
        for listener in [*u.audience, *u.overheard_by]:
            if listener != u.speaker:
                exposed_tick.setdefault(listener, u.tick)

    nodes = []
    for agent_id, name in corpus.names.items():
        if agent_id == meme["originator"]:
            role = "originator"
        elif agent_id in meme["independent"]:
            role = "independent"
        elif agent_id in adoption_tick:
            role = "adopter"
        elif agent_id in meme["first_use"]:
            role = "echo"
        elif agent_id in exposed_tick:
            role = "exposed"
        else:
            role = "untouched"
        nodes.append({"id": agent_id, "name": name, "role": role,
                      "first_use_tick": meme["first_use"].get(agent_id, {}).get("tick"),
                      "adoption_tick": adoption_tick.get(agent_id), "exposed_tick": exposed_tick.get(agent_id)})

    per_tick = Counter(u["tick"] for u in meme["uses"])
    timeline, users, adopters = [], set(), set()
    for use in meme["uses"]:
        users.add(use["speaker"])
        if use["role"] == "adoption":
            adopters.add(use["speaker"])
        if timeline and timeline[-1]["tick"] == use["tick"]:
            timeline[-1].update(cumulative_users=len(users), cumulative_adopters=len(adopters))
        else:
            timeline.append({"tick": use["tick"], "uses": per_tick[use["tick"]],
                             "cumulative_users": len(users), "cumulative_adopters": len(adopters)})
    return {"nodes": nodes, "timeline": timeline}


def adoption_curve(memes: list[dict]) -> list[dict]:
    """Cumulative number of (agent, meme) adoptions over time, across all detected memes."""
    ticks = sorted(e["tick"] for m in memes for e in m["edges"])
    return [{"tick": t, "adoptions": i + 1} for i, t in enumerate(ticks)]


def social_graph(corpus: Corpus) -> list[dict]:
    """How many utterances passed between each pair of agents (undirected)."""
    counts: Counter = Counter()
    for u in corpus.utterances:
        for listener in u.audience:
            if listener != u.speaker:
                counts[tuple(sorted((u.speaker, listener)))] += 1
    return [{"a": a, "b": b, "utterances": n} for (a, b), n in counts.most_common()]


def compare(corpus_a: Corpus, corpus_b: Corpus, top: int = 15) -> dict:
    """Side-by-side: every top meme of either run, traced in both runs."""
    memes_a, memes_b = detect_memes(corpus_a), detect_memes(corpus_b)
    rows, seen = [], set()
    for source, memes in (("a", memes_a), ("b", memes_b)):
        for meme in memes[:top]:
            if meme["phrase"] in seen:
                continue
            seen.add(meme["phrase"])
            rows.append({"phrase": meme["phrase"], "found_in": source,
                         "a": spread(trace_phrase(corpus_a, meme["phrase"])),
                         "b": spread(trace_phrase(corpus_b, meme["phrase"]))})
    return {
        "a": {"run_id": corpus_a.run_id, "condition": corpus_a.run["condition"], "summary": summarize(memes_a),
              "curve": adoption_curve(memes_a), "total_ticks": corpus_a.run["total_ticks"],
              "n_utterances": len(corpus_a.utterances)},
        "b": {"run_id": corpus_b.run_id, "condition": corpus_b.run["condition"], "summary": summarize(memes_b),
              "curve": adoption_curve(memes_b), "total_ticks": corpus_b.run["total_ticks"],
              "n_utterances": len(corpus_b.utterances)},
        "rows": rows,
    }
