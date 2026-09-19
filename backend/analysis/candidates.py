"""Meme candidate detection (OBSERVER ONLY -- never fed back to agents).

Hybrid method:
  1. n-gram statistics over all utterances (n = 1..4), with speaker counts,
     first occurrence, novelty (out-of-dictionary tokens), nickname patterns,
     and a penalty for plain factual repetition of world-provided wording;
  2. grouping of lexical variants (token containment / Jaccard / char-trigram
     embedding similarity);
  3. an LLM classifier (and an LLM discovery pass) on the grouped candidates.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

from backend.llm.client import llm_purpose
from backend.llm.embeddings import HashEmbedder, cos

STOP = set("""a an the and or but if then so of to in on at for with by from as is are was were be been being it its
this that these those i you he she they we me him her them my your his their our us do does did done have has had
not no yes just very really about into over after before up down out there here what which who whom when where why
how all any some can could would should will shall may might must also too than im i'm it's that's don't didn't
can't won't isn't wasn't you're they're we're i've you've i'll you'll he's she's there's what's let's oh ok okay
yeah yep hey hi well like so um uh wow totally honestly actually literally kind sort lot bit maybe right sure
get got getting go going gone gonna went come came know knew think thought mean said say says tell told make made
see saw look looked want need feel felt one two thing things something anything nothing everything someone anyone
today day time now still even again always never ever much many more most other another same own way back good
great nice cool fine bad new old last next first little big whole kinda pretty guess""".split())

_TOK = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")


def tokens(text: str) -> list[str]:
    return [t.lower().strip("'-") for t in _TOK.findall(text)]


try:
    from wordfreq import zipf_frequency as _zipf
except ImportError:  # optional: without it, commonness is not modelled
    _zipf = None


def zipf(tok: str) -> float:
    return _zipf(tok, "en") if _zipf else 3.0


def _dictionary() -> set:
    p = Path("/usr/share/dict/words")
    if p.exists():
        return {w.strip().lower() for w in open(p)}
    return set()


class CandidateExtractor:
    def __init__(self, rd, cfg: dict):
        self.rd = rd
        self.cfg = cfg
        self.dict = _dictionary()
        names = set()
        for a in rd.agents.values():
            names |= {t.lower() for t in a["name"].split()}
        self.names = names
        locs = set()
        for l in rd.manifest["world"]["graph"]:
            locs |= set(tokens(l))
        for arenas in rd.manifest["world"]["arenas"].values():
            for ar in arenas:
                locs |= set(tokens(ar))
        self.locs = locs
        self.emb = HashEmbedder(256)
        # world wording as a token stream, so factual repetition respects word boundaries ("tray" is not in "portrayed")
        self.world_tokens = " " + " ".join(tokens(rd.world_text)) + " "

    def ngrams(self):
        stats = defaultdict(lambda: {"uses": [], "speakers": set(), "surface": defaultdict(int)})
        for u in self.rd.utterances:
            raw = _TOK.findall(u["text"])
            toks = [t.lower().strip("'-") for t in raw]
            seen = set()
            for n in range(1, 5):
                for i in range(len(toks) - n + 1):
                    g = tuple(toks[i:i + n])
                    if g in seen:
                        continue
                    if g[0] in STOP or g[-1] in STOP:
                        continue
                    if all(t in STOP or t.isdigit() for t in g):
                        continue
                    if n == 1 and (len(g[0]) < 4 or g[0] in self.names or g[0] in self.locs):
                        continue
                    if n == 1 and zipf(g[0]) >= 3.6:  # ordinary English words are not candidates on their own
                        continue
                    if all(t in self.names or t in self.locs or t in STOP for t in g):
                        continue
                    seen.add(g)
                    st = stats[g]
                    st["uses"].append(u["id"])
                    st["speakers"].add(u["speaker"])
                    st["surface"][" ".join(raw[i:i + n])] += 1
        return stats

    def score(self, g: tuple, st: dict) -> dict:
        phrase = " ".join(g)
        uses, spk = len(st["uses"]), len(st["speakers"])
        novel = any(t not in self.dict and t not in self.names and not t.isdigit() and len(t) > 3 for t in g) if self.dict else False
        nickname = any(t in self.names for t in g) and any(t not in self.names and t not in STOP for t in g) and len(g) >= 2
        factual = f" {phrase} " in self.world_tokens
        quoted = sum(1 for uid in st["uses"] if re.search(r"[\"'“‘][^\"'”’]*" + re.escape(phrase), self.rd.utt_by_id[uid]["text"].lower()))
        s = math.log1p(uses) * (1 + math.log(spk)) * (1 + 0.8 * novel + 0.6 * nickname + 0.3 * (quoted > 0))
        s *= (0.35 if factual else 1.0) * (1 + 0.15 * (len(g) - 1))
        # commonness prior: phrases made only of frequent English words ("thanks for asking") are
        # usually ordinary language; rare or novel words are more likely to be local coinages
        content = [t for t in g if t not in STOP] or list(g)
        mz = sum(zipf(t) for t in content) / len(content)
        s *= min(1.5, max(0.15, (5.5 - mz) / 1.5))
        return {"phrase": phrase, "uses": uses, "mean_zipf": round(mz, 2), "speakers": spk, "novel": novel, "nickname": nickname,
                "factual_repetition": factual, "quoted": quoted, "score": round(s, 3)}

    def extract(self, max_candidates: int = 40) -> list[dict]:
        min_spk = int(self.cfg.get("min_speakers", 2))
        min_uses = int(self.cfg.get("min_uses", 3))
        stats = self.ngrams()
        rows = []
        for g, st in stats.items():
            if len(st["speakers"]) < min_spk or len(st["uses"]) < min_uses:
                continue
            rows.append((g, st, self.score(g, st)))
        # subsumption: drop n-grams whose uses are (almost) all covered by a longer kept n-gram
        rows.sort(key=lambda r: (-len(r[0]), -r[2]["score"]))
        kept = []
        for g, st, sc in rows:
            us = set(st["uses"])
            covered = False
            for g2, st2, _ in kept:
                if len(g2) > len(g) and " ".join(g) in " ".join(g2) and len(us - set(st2["uses"])) <= max(1, 0.2 * len(us)):
                    covered = True
                    break
            if not covered:
                kept.append((g, st, sc))
        kept.sort(key=lambda r: -r[2]["score"])
        return self.group(kept[: max_candidates * 2])[:max_candidates]

    def _sim(self, a: str, b: str) -> float:
        ta, tb = set(a.split()), set(b.split())
        jac = len(ta & tb) / max(1, len(ta | tb))
        contain = 1.0 if (a in b or b in a) else 0.0
        stem = 0.0
        for x in ta:
            for y in tb:
                if len(x) >= 5 and len(y) >= 5 and x[:5] == y[:5] and x not in STOP:
                    stem = max(stem, 0.7)
        return max(jac, 0.85 * contain, stem, 0.6 * cos(self.emb(a), self.emb(b)))

    def group(self, rows) -> list[dict]:
        groups: list[dict] = []
        for g, st, sc in rows:
            phrase = sc["phrase"]
            target = None
            for grp in groups:
                if max(self._sim(phrase, v) for v in grp["variants"]) >= 0.6 and \
                        len(set(st["uses"]) & grp["_uses"]) >= 0.3 * min(len(st["uses"]), len(grp["_uses"])):
                    target = grp
                    break
            if target is None:
                target = {"variants": [], "_uses": set(), "_speakers": set(), "_scores": [], "_surface": defaultdict(int)}
                groups.append(target)
            target["variants"].append(phrase)
            target["_uses"] |= set(st["uses"])
            target["_speakers"] |= st["speakers"]
            target["_scores"].append(sc)
            for k, v in st["surface"].items():
                target["_surface"][k] += v
        out = []
        for i, grp in enumerate(groups):
            best = max(grp["_scores"], key=lambda s: s["score"])
            surface = max(grp["_surface"].items(), key=lambda kv: (kv[1], -len(kv[0])))[0]
            canon = max((s["phrase"] for s in grp["_scores"]), key=lambda p: (sum(1 for s in grp["_scores"] if s["phrase"] == p and s["uses"] >= best["uses"] * 0.5), len(p.split())))
            usages = []
            for uid in sorted(grp["_uses"], key=lambda x: (self.rd.utt_by_id[x]["tick"], x)):
                u = self.rd.utt_by_id[uid]
                low = u["text"].lower()
                variant = max((v for v in grp["variants"] if v in " ".join(tokens(low))), key=len, default=canon)
                usages.append({"utterance_id": uid, "tick": u["tick"], "time": u["time"], "speaker": u["speaker"],
                               "listeners": u["listeners"], "text": u["text"], "variant": variant,
                               "conversation_id": u.get("conversation_id"),
                               "retrieved_event_ids": u.get("retrieved_event_ids", []),
                               "context": self.rd.conversation_context(u)})
            first = usages[0]
            out.append({"id": f"m{i:02d}", "canonical_form": canon, "display_form": surface,
                        "variants": grp["variants"], "usage_count": len(usages),
                        "speakers": sorted(grp["_speakers"]),
                        "first_occurrence": {k: first[k] for k in ("tick", "time", "speaker", "utterance_id")},
                        "contexts": [u["context"] for u in usages[:12]], "usages": usages,
                        "features": best, "score": round(sum(s["score"] for s in grp["_scores"]) / len(grp["_scores"]) + 0.3 * math.log1p(len(usages)), 3)})
        out.sort(key=lambda c: -c["score"])
        for i, c in enumerate(out):
            c["id"] = f"m{i:02d}"
        return out


CLASSIFY = """Here are uses of the expression "{expr}" in conversations among students on a simulated college campus (each with its surrounding lines):
{uses}

Does this expression appear to have acquired a locally specific meaning shared across multiple speakers (e.g. a local nickname, in-joke, coined phrase, or ordinary word used in a special local sense)? Plain factual repetition of what happened, generic small talk, or ordinary descriptive wording does NOT count.

Answer with a JSON object only: {{"is_convention": true or false, "gloss": "<what it seems to mean locally, one sentence>", "confidence": <0..1>}}"""

DISCOVER = """Below are conversations among students on a simulated college campus.

{convs}

Identify expressions (words, phrases, nicknames, numbers, metaphors) that appear to have acquired a locally specific meaning across multiple speakers. Do not treat ordinary factual repetition as a cultural convention.

Answer with a JSON object only: {{"expressions": [{{"expression": "<exact wording as used>", "reason": "<short>"}}]}}  (an empty list is fine)"""


def llm_classify(cands: list[dict], llm, top: int = 30, judge=None) -> None:
    """Classify the top candidates through a judge (backend/analysis/judge.py). Keeps the legacy
    c["llm"] = {is_convention, gloss, confidence, raw} and adds c["judgements"][judge_id] = Verdict.
    judge=None: the legacy classifier prompt (judge prompt v0) on `llm`, i.e. the pre-judge behaviour."""
    from backend.analysis.judge import LLMJudge, judge_input
    judge = judge or LLMJudge.legacy(llm)
    for c in cands[:top]:
        expr, contexts = judge_input(c)
        v = judge.judge(expr, contexts, {"n_uses": c.get("usage_count"), "n_speakers": len(c.get("speakers") or [])})
        c["llm"] = {"is_convention": v["is_convention"], "gloss": v["gloss"] or None,
                    "confidence": v["confidence"], "raw": v["raw"][:500]}
        c.setdefault("judgements", {})[v["judge_id"]] = v


def llm_discover(rd, llm, max_chars: int = 14000) -> list[str]:
    from backend.agents.ga_prompts import as_json
    blocks = []
    for c in rd.conversations.values():
        tr = "\n".join(f"{s}: {t}" for s, t in c.get("transcript", []))
        blocks.append(tr)
    text = ""
    for b in blocks[::-1]:            # most recent conversations first (conventions have had time to form)
        if len(text) + len(b) > max_chars:
            break
        text = b + "\n---\n" + text
    if not text:
        return []
    with llm_purpose("analysis_discovery"):
        raw = llm.complete(DISCOVER.format(convs=text), max_tokens=500, temperature=0)
    d = as_json(raw) or {}
    return [str(x.get("expression", "")).strip() for x in d.get("expressions", []) if isinstance(x, dict)]
