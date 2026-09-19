"""Meme candidate detection (OBSERVER ONLY -- never fed back to agents).

Hybrid method:
  1. n-gram statistics over all utterances (n = 1..MAXN), with speaker, conversation and occurrence counts,
     first occurrence, novelty, nickname / label constructions, metalinguistic marking, and a penalty for
     plain factual repetition of world-provided wording. Only WELL-FORMED n-grams are counted
     (wording.WordClasses.reject: no person names outside nickname or label constructions, no function-word
     / reporting-frame edges, no bare relationship or role nouns, no numbers or spelled-out quantities, no
     clause fragments, no span across sentence punctuation or stage directions), and only n-grams that are
     UNITS: a span almost always followed by the same word is a truncation of a longer phrase
     ("hyperparameters on the neural" <- "... neural network"), and one almost always preceded by the same
     non-determiner function word is a stub of a collocation ("least you caught" <- "at least you ...").
     Three corpora then say what an expression is *not*:
       - system wording (wording.Infrastructure: relationship lines, routines, seed/ambient memories, memory
         frames, profile text, lexicon), now matched on a content-lemma bag as well, so a paraphrase or
         clipping of a routine does not escape;
       - world wording (event facts, verbatim or by lemma overlap: emergence.world_match / world_lemma_match),
         so the same incident retold in ten wordings is not ten expressions;
       - the actor model's own register (register.Background): a phrase several speakers also use in other,
         independent runs is how the model writes students, not what this campus coined.
     Phrases used mostly inside text recited from the speaker's own memories (wording.Recitation) are
     down-ranked. Ranking prefers COLLOCATION SURPRISE (how much more often the phrase occurs here than the
     product of its words' general-English rates) over word rarity, so a coinage made of ordinary words
     ("style points", "dying fish", "grader brain") outranks an unremarkable inflection ("memorizing"), and
     adds explicit boosts for metalinguistic marking (quoted, "we're calling it", "I'm stealing that") and
     for use across several conversations.
  2. grouping of lexical variants: lemma-identical forms and morphological variants merge unconditionally,
     everything else needs a shared head plus shared evidence (overlapping uses); the canonical form is the
     best-scoring variant that is not system wording, and the group's wording flags come from the share of
     its USES that match, so junk cannot hide inside a clean group and a coinage cannot be renamed by one
     system-wording variant;
  3. `bucket` splits the ranked pool into what could be culture ("expression"), one person's tic
     ("personal"), ordinary / model-register language ("ordinary") and system or world wording ("wording"),
     and the pool is ordered by bucket first, so the top of the list the UI calls culture can only hold
     things that could be culture;
  4. an LLM classifier (and an LLM discovery pass) on the grouped candidates. Only a real (non-mock) judge's
     verdict counts: c["llm"]["is_convention"] is None for mock placeholders (judge.llm_block).
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from functools import cached_property
from backend.llm.client import llm_purpose
from backend.analysis.register import scaffolding
from backend.llm.embeddings import HashEmbedder

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


MAXN = 5                     # longest n-gram: a coined label can be four or five words ("the great coffee catastrophe")
TRUNCATE_SHARE = 0.75        # the same word follows this often -> the span is a truncation, not a unit
STUB_SHARE = 0.8             # the same non-determiner function word precedes this often -> a collocation stub
UNIT_MIN_OCC = 3             # fewer occurrences than this say nothing about the boundaries
SURPRISE_FULL_OCC = 6        # occurrences at which the collocation-surprise evidence counts in full
MERGE_OVERLAP = 0.5          # share of the bigger variant's uses two forms must share to be one expression
LEFT_DET = {"the", "a", "an", "my", "your", "his", "her", "its", "their", "our", "this", "that", "these", "those",
            "some", "any", "no", "another", "every", "each", "and", "or", "but", "so", "then", "just", "really"}
#: metalinguistic ratification: the speakers themselves say the phrase is a coined term. The strongest
#: available evidence of a convention, and independent of how often it is said.
META = ("is sending me", "i love that", "love that", "i'm stealing that", "im stealing that", "stealing that",
        "we're calling it", "what we're calling", "that's what we call", "calling it that", "invent the term",
        "invented the term", "the term", "that name", "good way to put it", "way to put it", "on the door",
        "coined", "officially", "unofficial", "we are absolutely keeping it", "keeping it", "obsessed with that",
        "is now a thing", "new word", "i'm using that", "using that one", "gonna steal that", "steal that")
#: buckets, in the order the pool is ranked: only "expression" can be culture
BUCKETS = ("expression", "personal", "ordinary", "wording")
BUCKET_RANK = {b: i for i, b in enumerate(BUCKETS)}


def bucket_of(*, system: bool, world: bool, ordinary: bool, personal: bool, planted: bool = False) -> tuple:
    """(bucket, reasons) of an expression record. The planted control is always shown as an expression."""
    reasons = []
    if planted:
        return "expression", []
    if system:
        reasons.append("system_wording")
    if world:
        reasons.append("world_wording")
    if reasons:
        return "wording", reasons
    if ordinary:
        return "ordinary", ["model_register"]
    if personal:
        return "personal", ["one_speaker_or_conversation"]
    return "expression", []


class CandidateExtractor:
    def __init__(self, rd, cfg: dict):
        from backend.analysis import wording as W
        from backend.analysis.register import Background
        self.rd = rd
        self.cfg = cfg
        self.wc = W.WordClasses(W.run_names(rd), W.run_places(rd), W.run_roles(rd), W.run_full_names(rd))
        self.names = self.wc.names
        self.locs = self.wc.places
        self.emb = HashEmbedder(256)
        self.bg = Background(getattr(rd, "dir", None) and rd.dir.name)
        # world wording as a token stream, so factual repetition respects word boundaries ("tray" is not in "portrayed")
        self.world_tokens = " " + " ".join(tokens(rd.world_text)) + " "
        self.rejected: Counter = Counter()          # n-gram occurrences dropped by the well-formedness rule, by reason
        self.totals: Counter = Counter()            # n -> n-gram positions in the run (the collocation denominator)

    @cached_property
    def meta_window(self) -> dict:
        """utterance id -> the lower-cased text of that turn plus the next turn of the same conversation
        (a coinage is usually ratified in the reply: "'feral grad student' is sending me")."""
        by_conv: dict = defaultdict(list)
        for u in self.rd.utterances:
            by_conv[u.get("conversation_id") or u["id"]].append(u)
        out = {}
        for us in by_conv.values():
            us = sorted(us, key=lambda u: (u["tick"], u.get("idx") or 0, u["id"]))
            for i, u in enumerate(us):
                nxt = us[i + 1]["text"] if i + 1 < len(us) else ""
                out[u["id"]] = f"{u['text']} {nxt}".lower()
        return out

    @cached_property
    def conv_last(self) -> dict:
        """conversation id -> the index of its last turn (register.scaffolding)."""
        out: dict = {}
        for u in self.rd.utterances:
            c = u.get("conversation_id")
            if c:
                out[c] = max(out.get(c, 0), u.get("idx") or 0)
        return out

    @cached_property
    def infra(self):
        """System wording of the run (wording.Infrastructure)."""
        from backend.analysis.wording import Infrastructure
        return Infrastructure(self.rd, self.wc)

    @cached_property
    def recite(self):
        from backend.analysis.wording import Recitation
        return Recitation(self.rd)

    def wellformed(self, g) -> str | None:
        """Why the phrase (tokens) is not a plausible reusable expression, or None (names taken as capitalized)."""
        g = [t for t in g if t]
        return self.wc.reject(g, raws=[t[:1].upper() + t[1:] for t in g])

    def explain(self, phrase: str) -> dict:
        """Diagnostics for one phrase: {"phrase", "wellformed", "reject_reason", "system_wording",
        "system_match", "background_runs", "stock"}."""
        g = tokens(phrase)
        why = self.wellformed(g)
        w = self.infra.wording(g)
        loc = self.bg.localness([" ".join(g)])
        return {"phrase": " ".join(g), "wellformed": why is None, "reject_reason": why,
                "system_wording": w["system"], "system_match": w["match"],
                "background_runs": loc["background_runs"], "stock": loc["stock"]}

    def ngrams(self):
        from backend.analysis.wording import segment
        stats = defaultdict(lambda: {"uses": [], "speakers": set(), "surface": defaultdict(int), "recited_ids": set(),
                                     "nickname": False, "conversations": set(), "occurrences": 0,
                                     "right": Counter(), "left": Counter()})
        self.rejected = Counter()
        self.totals = Counter()
        for u in self.rd.utterances:
            raw, toks, brk, stage = segment(u["text"])
            cls = self.wc.classes(toks, raw)
            rec = self.recite.mask(u, toks)
            seen = set()
            for n in range(1, MAXN + 1):
                self.totals[n] += max(0, len(toks) - n + 1)
                for i in range(len(toks) - n + 1):
                    if any(brk[i + 1:i + n]) or any(stage[i:i + n]):
                        continue                  # spans sentence punctuation, a quote mark or a *stage direction*
                    g = tuple(toks[i:i + n])
                    c = cls[i:i + n]
                    why = self.wc.reject(g, c)
                    if why:
                        self.rejected[why] += 1
                        continue
                    st = stats[g]
                    st["occurrences"] += 1
                    # the words this span is embedded in: a unit varies its neighbours (boundary entropy)
                    st["right"][toks[i + n] if i + n < len(toks) and not brk[i + n] else ""] += 1
                    st["left"][toks[i - 1] if i > 0 and not brk[i] else ""] += 1
                    if g in seen:
                        continue
                    seen.add(g)
                    st["uses"].append(u["id"])
                    st["speakers"].add(u["speaker"])
                    st["conversations"].add(u.get("conversation_id") or u["id"])
                    st["surface"][" ".join(raw[i:i + n])] += 1
                    if all(rec[i:i + n]):
                        st["recited_ids"].add(u["id"])
                    if "name" in c:
                        st["nickname"] = True
        return stats

    # ---------------------------------------------------------------- unit test (boundary entropy)
    def not_a_unit(self, g: tuple, st: dict, stats: dict | None = None) -> str | None:
        """Why this span is not a lexical unit, or None. A span that is almost always followed by the same
        word is a truncation of a longer phrase; one almost always preceded by the same non-determiner
        function word is a piece of that collocation. Both are published as expressions today
        ("coffee sounds" <- "coffee sounds great", "least you caught" <- "at least you caught")."""
        from backend.analysis.wording import AUX, FUNCTION, PRONOUN
        occ = st.get("occurrences") or len(st["uses"])
        if occ < UNIT_MIN_OCC:
            return None                            # too few occurrences to read the boundaries
        right = st.get("right") or Counter()
        if right:
            tok, k = right.most_common(1)[0]
            # only a word that could CONTINUE the phrase counts: a following preposition, copula or
            # conjunction starts the next constituent ("the tray return TO my table" is a unit)
            if tok and k >= TRUNCATE_SHARE * occ and tok not in FUNCTION and tok not in AUX \
                    and tok not in PRONOUN:
                return "truncated_unit"
        left = st.get("left") or Counter()
        if left:
            tok, k = left.most_common(1)[0]
            if tok and k >= STUB_SHARE * occ and (tok in FUNCTION or tok in STOP) and tok not in LEFT_DET:
                return "collocation_stub"
        return None

    # ---------------------------------------------------------------- scoring
    def surprise(self, g: tuple, occ: int) -> float:
        """log10 of (rate of this phrase in the run) / (rate expected if its words were independent).
        A rare *combination* of ordinary words -- what slang looks like -- scores high; an unremarkable
        inflection of one common word does not. Replaces the old mean-word-rarity prior, which gave
        technical single words a bonus and penalised coinages built from everyday words."""
        total = self.totals.get(len(g)) or max(1, len(self.rd.utterances) * 12)
        p_obs = max(occ, 1) / total
        log_exp = sum(min(zipf(t), 6.8) - 9 for t in g)
        return math.log10(p_obs) - log_exp

    def meta_uses(self, phrase: str, st: dict) -> int:
        """Uses whose turn (or the reply to it) talks about the wording itself: a ratification event."""
        n = 0
        for uid in st["uses"]:
            w = self.meta_window.get(uid) or ""
            if phrase in w and any(m in w for m in META):
                n += 1
        return n

    def score(self, g: tuple, st: dict) -> dict:
        from backend.analysis.wording import WordClasses, lemma_zipf, NOVEL_MAX_ZIPF
        phrase = " ".join(g)
        uses, spk = len(st["uses"]), len(st["speakers"])
        occ = st.get("occurrences") or uses
        convs = len(st.get("conversations") or ()) or 1
        cls = self.wc.classes(list(g), [t.capitalize() for t in g])
        # novelty on the LEMMA: "memorizing", "recopied", "napkins" are ordinary words, not coinages
        novel = any(lemma_zipf(self.wc.base(t)) < NOVEL_MAX_ZIPF and t not in self.names for t in g)
        nickname = bool(st.get("nickname")) or WordClasses.nickname(list(g), cls) or "nick" in cls
        label = WordClasses.label(list(g), cls)
        factual = f" {phrase} " in self.world_tokens
        quoted = sum(1 for uid in st["uses"] if re.search(r"[\"'“‘][^\"'”’]*" + re.escape(phrase), self.rd.utt_by_id[uid]["text"].lower()))
        wording = self.infra.wording(list(g))
        meta = self.meta_uses(phrase, st) if st.get("uses") else 0
        loc = self.bg.localness([phrase])
        scaff = scaffolding([self.rd.utt_by_id[u] for u in st["uses"] if u in self.rd.utt_by_id], self.conv_last)
        recited = len(set(st.get("recited_ids") or ()) & set(st["uses"]))
        rshare = recited / uses if uses else 0.0
        sup = self.surprise(g, occ)
        # shrink the collocation evidence towards "no information" when there are few occurrences:
        # two uses cannot tell a fixed collocation from a fresh one
        smult = min(1.8, max(0.35, (sup - 1.5) / 2.5))
        w = min(1.0, occ / SURPRISE_FULL_OCC)
        smult = 1.0 + w * (smult - 1.0)
        personal = spk < 2 or (convs < 2 and not (meta or quoted))
        s = math.log1p(uses) * (1 + math.log(spk)) * (1 + 0.15 * (len(g) - 1))
        s *= 1 + 0.5 * novel + 0.5 * (nickname or label) + 0.8 * min(1, meta) + 0.3 * (quoted > 0)
        s *= 1 + 0.35 * (convs >= 2 and spk >= 2)        # said again, elsewhere, by someone else
        s *= smult * loc["multiplier"]
        s *= 0.35 if factual else 1.0
        s *= 0.3 if wording["system"] else 1.0           # system wording (relationship lines, routines, frames, lexicon ...)
        s *= 1.0 - 0.7 * rshare                          # mostly recited from the speaker's own memory text
        s *= 0.3 if personal else 1.0                    # one speaker or one conversation: a personal tic
        s *= 0.5 if scaff else 1.0                       # only ever said opening or closing a conversation
        content = [t for t in g if t not in STOP] or list(g)
        mz = sum(zipf(t) for t in content) / len(content)
        return {"phrase": phrase, "uses": uses, "occurrences": occ, "conversations": convs, "mean_zipf": round(mz, 2),
                "speakers": spk, "novel": novel, "nickname": nickname, "label": label,
                "factual_repetition": factual, "quoted": quoted, "meta_marked": meta,
                "surprise": round(sup, 2), "background_runs": loc["background_runs"],
                "stock_phrase": loc["stock"], "ordinary_register": loc["ordinary"] or scaff,
                "scaffolding": scaff, "personal": personal,
                "system_wording": wording["system"],
                "system_match": wording["match"], "in_lexicon": wording["in_lexicon"], "recited": recited,
                "recited_share": round(rshare, 3), "score": round(s, 3)}

    def extract(self, max_candidates: int = 40) -> list[dict]:
        min_spk = int(self.cfg.get("min_speakers", 2))
        min_uses = int(self.cfg.get("min_uses", 3))
        stats = self.ngrams()
        rows = []
        for g, st in stats.items():
            if len(st["speakers"]) < min_spk or len(st["uses"]) < min_uses:
                continue
            why = self.not_a_unit(g, st, stats)
            if why:
                self.rejected[why] += st.get("occurrences") or len(st["uses"])
                continue
            rows.append((g, st, self.score(g, st)))
        # subsumption: drop n-grams whose uses are (almost) all covered by a longer kept n-gram. A novel or
        # meta-marked short form is kept: the reusable unit is the verb ("yeeted"), not the anecdote it came in.
        rows.sort(key=lambda r: (-len(r[0]), -r[2]["score"]))
        kept = []
        for g, st, sc in rows:
            us = set(st["uses"])
            standalone = sc["novel"] or sc["meta_marked"] or sc["nickname"] or sc["label"]
            covered = False
            for g2, st2, sc2 in kept:
                if len(g2) > len(g) and " ".join(g) in " ".join(g2) and len(us - set(st2["uses"])) <= max(1, 0.2 * len(us)):
                    if standalone and sc["score"] > sc2["score"]:
                        continue
                    covered = True
                    break
            if not covered:
                kept.append((g, st, sc))
        kept.sort(key=lambda r: -r[2]["score"])
        return self.group(kept[: max_candidates * 3])[:max_candidates]

    # ---------------------------------------------------------------- grouping
    @staticmethod
    def _key(phrase: str) -> tuple:
        from backend.analysis.wording import fold
        return tuple(fold(t) for t in phrase.split())

    @staticmethod
    def _head(phrase: str) -> str:
        """The folded last content word: two variants of one expression share a head ("frisbee analogy" /
        "the frisbee analogy"); "hike sounds" and "coffee sounds" share only a head, which is not enough."""
        from backend.analysis.wording import FUNCTION, fold
        for t in reversed(phrase.split()):
            if t not in STOP and t not in FUNCTION:
                return fold(t)
        return fold(phrase.split()[-1]) if phrase else ""

    @staticmethod
    def _content(phrase: str) -> set:
        from backend.analysis.wording import FUNCTION, fold
        return {fold(t) for t in phrase.split() if t not in STOP and t not in FUNCTION}

    def _merges(self, phrase: str, st: dict, grp: dict) -> bool:
        """Whether `phrase` belongs to the group. Lemma-identical and single-word morphological variants
        merge unconditionally (they are the same word); anything else needs a shared head, shared further
        evidence, and overlapping uses -- so a shared modifier or a world-provided noun cannot fuse two
        unrelated expressions into one 'emerged' group."""
        key, head, cont = self._key(phrase), self._head(phrase), self._content(phrase)
        if key in grp["_keys"]:
            return True
        solo = len(phrase.split()) == 1
        for v in grp["variants"]:
            if solo and len(v.split()) == 1:
                a, b = phrase, v
                if len(a) >= 5 and len(b) >= 5 and a[:5] == b[:5]:
                    return True                     # "rewrote" / "rewriting", "timestamp" / "timestamps"
                continue
            contain = f" {phrase} " in f" {v} " or f" {v} " in f" {phrase} "
            if not contain:
                if self._head(v) != head:
                    continue                    # "hike sounds" and "coffee sounds" share only a modifier
                vc = self._content(v)
                if not (len(cont & vc) >= 2 or (len(cont) == 1 and cont <= vc)):
                    continue
            us, vu = set(st["uses"]), grp["_vuses"].get(v) or set()
            # half the uses of the BIGGER of the two, and measured against that variant rather than the
            # whole group: a group must not get easier to join as it grows, or one chain of containments
            # swallows half the run ("robotics" -> "duct tape patch from robotics" -> "duct tape patch")
            if vu and len(us & vu) >= MERGE_OVERLAP * max(len(us), len(vu)):
                return True
        return False

    def group(self, rows) -> list[dict]:
        groups: list[dict] = []
        for g, st, sc in rows:
            phrase = sc["phrase"]
            target = None
            for grp in groups:
                if self._merges(phrase, st, grp):
                    target = grp
                    break
            if target is None:
                target = {"variants": [], "_uses": set(), "_speakers": set(), "_scores": [], "_surface": defaultdict(int),
                          "_recited": set(), "_keys": set(), "_vuses": {}}
                groups.append(target)
            target["variants"].append(phrase)
            target["_keys"].add(self._key(phrase))
            target["_vuses"][phrase] = set(st["uses"])
            target["_uses"] |= set(st["uses"])
            target["_speakers"] |= st["speakers"]
            target["_recited"] |= set(st.get("recited_ids") or ())
            target["_scores"].append(sc)
            for k, v in st["surface"].items():
                target["_surface"][k] += v
        out = []
        for i, grp in enumerate(groups):
            best = max(grp["_scores"], key=lambda s: s["score"])
            surface = max(grp["_surface"].items(), key=lambda kv: (kv[1], -len(kv[0])))[0]
            # the canonical form is the best-scoring variant that is NOT system wording: one routine-wording
            # variant must not be able to rename (and disqualify) a genuine group
            clean = [s for s in grp["_scores"] if not s["system_wording"]] or grp["_scores"]
            canon = max(clean, key=lambda s: (s["score"], s["uses"], len(s["phrase"].split())))["phrase"]
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
            rec = grp["_recited"] & grp["_uses"]
            for u in usages:
                u["recited"] = u["utterance_id"] in rec
            # the group's wording flags come from the SHARE OF USES that match, not from the canonical
            # string: junk cannot hide inside a clean group, nor one system variant sink a genuine one
            n_uses = len(grp["_uses"]) or 1
            sys_uses = {u for s in grp["_scores"] if s["system_wording"] for u in grp["_vuses"][s["phrase"]]}
            wording = self.infra.wording(tokens(canon))
            system = len(sys_uses) >= 0.5 * n_uses
            smatch = wording["match"] or next((s["system_match"] for s in grp["_scores"] if s["system_match"]), None)
            lex_uses = {u for s in grp["_scores"] if s["in_lexicon"] for u in grp["_vuses"][s["phrase"]]}
            cbest = next(s for s in grp["_scores"] if s["phrase"] == canon)
            # the register test is about the expression the observer publishes: a coinage that happens to
            # contain an ordinary word ("inbox apocalypse" / "inbox") is not itself the model's register
            ordinary = cbest["ordinary_register"] or all(s["ordinary_register"] for s in grp["_scores"])
            out.append({"id": f"m{i:02d}", "canonical_form": canon, "display_form": surface,
                        "variants": grp["variants"], "usage_count": len(usages),
                        "speakers": sorted(grp["_speakers"]),
                        "n_conversations": len({u["conversation_id"] for u in usages if u.get("conversation_id")}),
                        "first_occurrence": {k: first[k] for k in ("tick", "time", "speaker", "utterance_id")},
                        "contexts": [u["context"] for u in usages[:12]], "usages": usages,
                        "features": best, "wording": {"system": system, "system_match": smatch,
                                                      "in_lexicon": len(lex_uses) >= 0.5 * n_uses,
                                                      "canonical_system": wording["system"],
                                                      "system_use_share": round(len(sys_uses) / n_uses, 3)},
                        "register": {"ordinary": ordinary,
                                     "background_runs": max(s["background_runs"] for s in grp["_scores"]),
                                     "stock": any(s["stock_phrase"] for s in grp["_scores"]),
                                     "meta_marked": sum(s["meta_marked"] for s in grp["_scores"]),
                                     "quoted": sum(s["quoted"] for s in grp["_scores"]),
                                     "surprise": cbest["surprise"], "novel": cbest["novel"]},
                        "recited_share": round(len(rec) / len(usages), 3) if usages else 0.0,
                        # the group's rank comes from its BEST variant: productive variation is evidence a
                        # coinage is alive, and must not average it away
                        "score": round(best["score"] + 0.3 * math.log1p(len(usages)), 3)})
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
    c["llm"] = {is_convention, gloss, confidence, raw} (plus the judge's provenance, judge.llm_block: a mock
    judge's verdict is a placeholder, so is_convention/gloss are None there) and adds c["judgements"][judge_id] = Verdict.
    judge=None: the legacy classifier prompt (judge prompt v0) on `llm`, i.e. the pre-judge behaviour."""
    from backend.analysis.judge import LLMJudge, judge_input, llm_block
    judge = judge or LLMJudge.legacy(llm)
    for c in cands[:top]:
        expr, contexts = judge_input(c)
        w = c.get("wording") or {}
        extra = {"n_uses": c.get("usage_count"), "n_speakers": len(c.get("speakers") or []),
                 "system_wording": bool(w.get("system"))}
        v = judge.judge(expr, contexts, extra)
        c["llm"] = llm_block(v)
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
