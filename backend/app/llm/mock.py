"""Offline stand-in for the agent LLM (LLM_PROVIDER=mock).

This is a test double, not a model of anything. It exists so the whole pipeline
(simulation -> event log -> meme detector -> UI) runs without an API key or cost.

It picks actions from simple trait-weighted rules, sometimes "coins" a phrase about a salient
event, and reuses phrases it finds quoted in its retrieved memories, so transmission through
memory is exercised end to end. Results from mock runs say nothing about real LLM agents.
"""
from __future__ import annotations

import hashlib
import json
import random
import re

COIN_PATTERNS = ["{t}-gate", "the {t} curse", "{t} energy", "{t}-pocalypse"]
COINED_RE = re.compile(r"\b[a-z][a-z-]*-(?:gate|pocalypse)\b|\bthe [a-z-]+ curse\b|\b[a-z-]+ energy\b")

OPENERS = ["Hey {n}.", "{n}!", "Oh hi {n}.", "What's up, {n}?"]
REUSE_LINES = ["Total {x} again.", "Honestly, {x}.", "This is so {x}.", "Remember {x}? Still funny.", "Classic {x}."]
COIN_LINES = ["That was {x}, no question.", "Pure {x}.", "Okay, that was {x}.", "Peak {x} today."]
SMALL_TALK = ["How's your day going?", "I'm just {activity}.", "Long day, huh?",
              "Are you going to the lecture tomorrow?", "I need more coffee."]
REPLY_REUSE = ["Ha, {x} for real.", "Classic {x}.", "Yeah, {x} strikes again."]
REPLIES = ["Yeah, totally.", "Ha, same.", "Not much, honestly.", "Right? Anyway, see you around.",
           "I know, it's been a week.", "Makes sense."]


def _rng(messages: list[dict]) -> random.Random:
    digest = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def known_phrases(texts: list[str]) -> list[str]:
    """Coined phrases that appear inside quotes in memory texts."""
    found: list[str] = []
    for text in texts:
        for quote in re.findall(r'"([^"]+)"', text):
            found += COINED_RE.findall(quote.lower())
    return list(dict.fromkeys(found))


def coin(topic: str, rng: random.Random) -> str:
    return rng.choice(COIN_PATTERNS).format(t=topic)


class MockLLM:
    def complete(self, messages: list[dict], ctx: dict) -> str:
        rng = _rng(messages)
        kind = ctx.get("kind")
        if kind == "decide":
            out = self._decide(ctx, rng)
        elif kind == "reply":
            out = self._reply(ctx, rng)
        elif kind == "gloss":
            out = {"gloss": f"A phrase this group uses after hearing it from {ctx.get('originator', 'someone')}.",
                   "refers_to": ctx.get("hint") or "unclear", "ordinary_language": False}
        else:
            out = {}
        return json.dumps(out)

    def _decide(self, ctx: dict, rng: random.Random) -> dict:
        traits = ctx.get("traits", {})
        extraversion, humor = traits.get("extraversion", 0.5), traits.get("humor", 0.5)
        nearby, topics = ctx.get("nearby", []), ctx.get("topics", [])
        phrases = known_phrases(ctx.get("memories", []))
        activity = ctx.get("activity") or "hanging around"
        base = {"target": None, "utterance": None, "activity": activity, "importance": 3}

        if ctx.get("scheduled_location") and rng.random() < 0.75:
            return {**base, "action": "MOVE", "target": ctx["scheduled_location"],
                    "activity": "walking over", "reason": "I'm supposed to be there now."}
        if nearby and rng.random() < 0.2 + 0.5 * extraversion:
            target = rng.choice(nearby)
            return {**base, "action": "TALK", "target": target, "activity": f"talking with {target}",
                    "utterance": self._line(target, topics, phrases, humor, activity, rng),
                    "reason": "Felt like chatting.", "importance": 4}
        if topics and rng.random() < 0.25 + 0.5 * humor:
            return {**base, "action": "REACT", "utterance": self._reaction(topics, phrases, humor, rng),
                    "activity": "watching what's going on", "reason": "That was something.", "importance": 5}
        return {**base, "action": "CONTINUE_ACTIVITY", "reason": "Carrying on."}

    def _line(self, target: str, topics: list[str], phrases: list[str], humor: float,
              activity: str, rng: random.Random) -> str:
        opener = rng.choice(OPENERS).format(n=target)
        roll = rng.random()
        if phrases and roll < 0.35 + 0.35 * humor:
            body = rng.choice(REUSE_LINES).format(x=rng.choice(phrases))
        elif topics and roll < 0.6 + 0.3 * humor:
            topic = rng.choice(topics)
            body = (rng.choice(COIN_LINES).format(x=coin(topic, rng)) if rng.random() < humor
                    else f"Did you see the {topic.replace('-', ' ')} thing?")
        else:
            body = rng.choice(SMALL_TALK).format(activity=activity)
        return f"{opener} {body}"

    def _reaction(self, topics: list[str], phrases: list[str], humor: float, rng: random.Random) -> str:
        if phrases and rng.random() < 0.5:
            return rng.choice(REUSE_LINES).format(x=rng.choice(phrases))
        topic = rng.choice(topics)
        if rng.random() < humor:
            return rng.choice(COIN_LINES).format(x=coin(topic, rng))
        return f"Wow, the {topic.replace('-', ' ')}. Again."

    def _reply(self, ctx: dict, rng: random.Random) -> dict:
        humor = ctx.get("traits", {}).get("humor", 0.5)
        history = ctx.get("history", [])
        last = history[-1][1] if history else ""
        phrases = list(dict.fromkeys(known_phrases(ctx.get("memories", [])) + COINED_RE.findall(last.lower())))
        if phrases and rng.random() < 0.3 + 0.4 * humor:
            utterance = rng.choice(REPLY_REUSE).format(x=rng.choice(phrases))
        else:
            utterance = rng.choice(REPLIES)
        return {"utterance": utterance, "end_conversation": len(history) >= 2 and rng.random() < 0.55}
