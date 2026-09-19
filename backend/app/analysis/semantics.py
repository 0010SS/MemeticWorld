"""Meaning-level grouping of detected memes.

- referents: which world incident a phrase is about (the tray drop, the printer...). Phrases
  with the same referent are *competing variants*; convergence = one variant taking over.
- variant keys: 'tray gate' / 'tray-gate' / 'traygate' collapse to one spelling-insensitive key.
- gloss: optional analyst-LLM explanation of a phrase from its uses. This prompt is for the
  *analysis* model and may talk about memes freely; it is never shown to agents.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from pydantic import BaseModel, field_validator

from app.analysis.meme_detector import STOPWORDS, Corpus, stem, tokenize
from app.llm.client import LLMClient


def variant_key(phrase: str) -> str:
    return "".join(ch for ch in phrase.lower() if ch.isalnum())


def _content_stems(text: str, exclude: set[str]) -> set[str]:
    stems: set[str] = set()
    for token in tokenize(text):
        for part in {token, *token.split("-")}:
            if part not in STOPWORDS and len(part) > 2 and stem(part) not in exclude:
                stems.add(stem(part))
    return stems


def assign_referents(corpus: Corpus, memes: list[dict]) -> None:
    """Sets meme["referent"] = world event key (or None) using words specific to each event."""
    exclude = {stem(t) for name in corpus.names.values() for t in tokenize(name)}
    for loc in corpus.run["config"].get("world", {}).get("locations", []):
        exclude.update(stem(t) for t in tokenize(loc["name"]))

    tokens_by_key: dict[str, set[str]] = defaultdict(set)
    first_tick: dict[str, int] = {}
    for event in corpus.world_events:
        key = event["data"].get("key", "event")
        tokens_by_key[key] |= _content_stems(event["text"], exclude)
        first_tick.setdefault(key, event["tick"])
    doc_freq = Counter(t for toks in tokens_by_key.values() for t in toks)
    specific = {k: {t for t in toks if doc_freq[t] == 1} for k, toks in tokens_by_key.items()}

    for meme in memes:
        meme["referent"] = None
        phrase_stems = _content_stems(meme["phrase"], exclude)
        started = meme["first_tick"] if meme["first_tick"] is not None else 0
        plausible = [k for k in specific if first_tick[k] <= started]
        direct = [k for k in plausible if phrase_stems & specific[k]]
        if direct:
            meme["referent"] = direct[0]
            continue
        # Otherwise: a specific event word shows up in at least half of the phrase's uses.
        best, best_share = None, 0.0
        for key in plausible:
            hits = sum(1 for use in meme["uses"] if _content_stems(use["text"], exclude) & specific[key])
            share = hits / max(1, len(meme["uses"]))
            if share > best_share:
                best, best_share = key, share
        if best_share >= 0.5:
            meme["referent"] = best


def referent_groups(corpus: Corpus, memes: list[dict]) -> list[dict]:
    """Competing phrases per incident, with how dominant the leading phrase is."""
    labels = {}
    for event in corpus.world_events:
        labels.setdefault(event["data"].get("key"), {"text": event["text"], "topic": event["data"].get("topic")})
    groups: dict[str, list[dict]] = defaultdict(list)
    for meme in memes:
        if meme.get("referent"):
            groups[meme["referent"]].append(meme)
    out = []
    for key, members in groups.items():
        members.sort(key=lambda m: -m["n_uses"])
        total = sum(m["n_uses"] for m in members)
        out.append({
            "key": key, "topic": labels.get(key, {}).get("topic"), "example_text": labels.get(key, {}).get("text"),
            "meme_ids": [m["id"] for m in members], "leader": members[0]["phrase"],
            "leader_share": round(members[0]["n_uses"] / total, 3) if total else 0.0,
        })
    out.sort(key=lambda g: -len(g["meme_ids"]))
    return out


class Gloss(BaseModel):
    gloss: str
    refers_to: str | None = None
    ordinary_language: bool = False

    @field_validator("ordinary_language", mode="before")
    @classmethod
    def _parse_bool(cls, value):
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1"}
        return bool(value)


GLOSS_PROMPT = """You are analyzing a multi-agent simulation of students on a small campus.
These lines were spoken by different students, in order, and all contain the phrase "{phrase}":
{lines}

Things that happened on campus before or around then:
{events}

Reply with only a JSON object with keys "gloss", "refers_to", "ordinary_language":
- "gloss": one sentence on what "{phrase}" means to this group of students
- "refers_to": the incident, person or thing it points to, or null
- "ordinary_language": true if this is everyday English anyone would say without shared history, false if it depends on this group's shared experience"""


async def gloss_meme(llm: LLMClient, corpus: Corpus, meme: dict) -> dict:
    lines = "\n".join(f'- {corpus.names.get(u["speaker"], u["speaker"])} ({u["sim_time"]}): "{u["text"]}"'
                      for u in meme["uses"][:12])
    first = meme["first_tick"] or 0
    events = [e for e in corpus.world_events if e["tick"] <= first + 4][-8:]
    event_lines = "\n".join(f"- ({e['sim_time']}, {e['location']}) {e['text']}" for e in events) or "- (none)"
    prompt = GLOSS_PROMPT.format(phrase=meme["phrase"], lines=lines, events=event_lines)
    mock_ctx = {"kind": "gloss", "originator": corpus.names.get(meme["originator"], "someone"),
                "hint": meme.get("referent")}
    gloss, _, error = await llm.chat_json([{"role": "user", "content": prompt}], Gloss, mock_ctx)
    if gloss is None:
        return {"gloss": None, "refers_to": None, "ordinary_language": None, "error": error}
    return gloss.model_dump()
