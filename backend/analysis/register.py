"""The null hypothesis for "local culture": the actor model's own register (OBSERVER ONLY).

One LLM writes every agent, so its stock wording is used by several speakers in every run and looks like
a spreading convention. The test that separates it from culture is *cross-run recurrence*: a phrase that
turns up, with >= 2 speakers, in independent runs (other seeds, other conditions, other models) is how the
model writes students, not what this campus coined.

- `Background` is a table built from the archived real runs (`data/register_background.json`,
  rebuilt by `build_background`), in two layers: `phrases` -> the runs whose agents used it with >= 2
  speakers (shared usage elsewhere), and `common` -> the runs it occurs in at all, kept from 4 runs up
  (the model's plain turn of phrase). Lookups are **leave-one-out**: the run being analysed never counts
  towards its own background, so the table can include it without circularity. Unknown phrases have
  background 0 and are not penalised.
- `STOCK` is a small hand-written list of conversational openers, closers and politeness formulas
  ("mind if I sit", "fingers crossed", "thanks for checking"): greeting scaffolding, never culture.
- `scaffolding` marks an expression whose uses are all in the opening or closing turn of a conversation.

`localness()` returns the multiplier and the `ordinary` flag the extractor uses. That flag is a gate, not
just a penalty: an expression the model says everywhere can never be `emerged` (emergence.in_register) and
never reaches the convention tier (tiers.tier_of(ordinary=True)).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "register_background.json"
#: a phrase used with >= 2 speakers in this many OTHER runs is population-wide register, not local culture
ORDINARY_RUNS = 2
#: ... and so is one that merely OCCURS in this many other runs: the model's ordinary turn of phrase
#: ("sounds awesome", "email the professor", "figure out what happened")
ORDINARY_COMMON_RUNS = 3
#: runs a phrase must occur in to be stored in the `common` layer at all
COMMON_KEEP_RUNS = 4
#: stock conversational formulas (openers, closers, politeness): scaffolding of the dialogue, never culture
STOCK = (
    "mind if i sit", "mind if i join", "mind if i", "is this seat taken", "how's it going", "how are things",
    "i don't think we've officially met", "we've officially met", "nice to meet you", "good to see you",
    "see you around", "see you later", "catch you later", "take care", "have a good one", "good luck",
    "fingers crossed", "hope it goes well", "hoping for the best", "thanks for checking", "thanks for asking",
    "thanks for the heads up", "no worries", "no problem", "sounds good", "sounds great", "sounds perfect",
    "sounds intense", "sounds like a plan", "let me know", "keep me posted", "compare notes", "crossed paths",
    "you're a lifesaver", "lifesaver", "that's rough", "hang in there", "good talk", "i'll let you go",
)


@lru_cache(maxsize=4)
def _load(path: str) -> tuple:
    """(runs, multi-speaker table, any-speaker table) of the background, or an empty background."""
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return (), {}, {}
    runs = tuple(doc.get("built_from") or [])
    return (runs, {p: frozenset(v) for p, v in (doc.get("phrases") or {}).items()},
            {p: frozenset(v) for p, v in (doc.get("common") or {}).items()})


class Background:
    """Cross-run background frequency of expressions, looked up leave-one-out for `run_id`."""

    def __init__(self, run_id: str | None = None, path: str | Path = DATA):
        self.runs, self.phrases, self.common = _load(str(path))
        self.run_id = run_id
        self.self_index = self.runs.index(run_id) if run_id in self.runs else None

    def _count(self, table: dict, phrase: str) -> int:
        idx = table.get(phrase)
        if not idx:
            return 0
        return len(idx - {self.self_index}) if self.self_index is not None else len(idx)

    def runs_with(self, phrase: str) -> int:
        """How many OTHER archived runs used this phrase with >= 2 speakers."""
        return self._count(self.phrases, phrase)

    def common_runs(self, phrase: str) -> int:
        """How many OTHER archived runs contain this phrase at all."""
        return self._count(self.common, phrase)

    def best(self, forms) -> tuple[int, str | None]:
        """(highest background count over the forms, the form that reached it)."""
        best, who = 0, None
        for f in forms:
            n = self.runs_with(f)
            if n > best:
                best, who = n, f
        return best, who

    def localness(self, forms) -> dict:
        """{"background_runs", "background_form", "stock", "ordinary", "multiplier"} for an expression.

        multiplier: 1.0 when nothing else in the corpus says it, 0.55 when one other run does (weak
        evidence, e.g. two runs of the same world), 0.12 once >= ORDINARY_RUNS other runs do."""
        fs = [f for f in forms if f]
        n, who = self.best(fs)
        c = max((self.common_runs(f) for f in fs), default=0)
        stock = any(is_stock(f) for f in fs)
        ordinary = n >= ORDINARY_RUNS or c >= ORDINARY_COMMON_RUNS or stock
        mult = 0.12 if ordinary else (0.55 if n == 1 or c == 2 else 1.0)
        return {"background_runs": n, "background_form": who, "common_runs": c, "stock": stock,
                "ordinary": ordinary, "multiplier": mult}


def is_stock(phrase: str) -> bool:
    """A stock conversational formula (or a phrase wholly inside one)."""
    p = f" {phrase} "
    return any(p in f" {s} " or f" {s} " in p for s in STOCK)


#: a conversation shorter than this has no "middle", so where a phrase sits in it says nothing
SCAFFOLD_MIN_TURNS = 4


def scaffolding(usages: list[dict], conv_last: dict) -> bool:
    """Every use sits in the opening or the closing turn of its conversation: greeting scaffolding, not
    something the speakers talk WITH. Only conversations long enough to have a middle are looked at."""
    us = [u for u in usages if u.get("conversation_id")
          and (conv_last.get(u["conversation_id"]) or 0) >= SCAFFOLD_MIN_TURNS - 1]
    if len(us) < 2:
        return False
    return all((u.get("idx") or 0) in (0, conv_last.get(u["conversation_id"])) for u in us)


# ------------------------------------------------------------------------------------------------ builder
def build_background(run_dirs, out: str | Path = DATA, max_n: int = 5) -> dict:
    """Rebuild the table from run directories: every within-sentence n-gram, in two layers -- used by >= 2
    speakers and kept from 2 runs up, and occurring at all and kept from COMMON_KEEP_RUNS runs up (fewer
    can never reach the leave-one-out thresholds)."""
    from collections import defaultdict

    from backend.analysis.rundata import RunData
    from backend.analysis.wording import segment
    dirs = [Path(d) for d in run_dirs]
    runs = [d.name for d in dirs]
    table: dict[str, list[int]] = defaultdict(list)
    seen_table: dict[str, list[int]] = defaultdict(list)
    n_utt, n_tok = {}, {}
    for i, d in enumerate(dirs):
        rd = RunData(d)
        spk: dict[str, set] = defaultdict(set)
        toks_total = 0
        for u in rd.utterances:
            _raw, toks, brk, stage = segment(u["text"])
            toks_total += len(toks)
            for n in range(1, max_n + 1):
                for j in range(len(toks) - n + 1):
                    if any(brk[j + 1:j + n]) or any(stage[j:j + n]):
                        continue
                    spk[" ".join(toks[j:j + n])].add(u["speaker"])
        for p, s in spk.items():
            seen_table[p].append(i)
            if len(s) >= 2:
                table[p].append(i)
        n_utt[d.name], n_tok[d.name] = len(rd.utterances), toks_total
    doc = {"version": 1, "built_from": runs,
           "note": ("Multi-speaker n-grams (n<=%d, within a sentence) of each archived real run. A phrase listed "
                    "for >= %d OTHER runs is population-wide model register, not this run's local culture "
                    "(leave-one-out). Only phrases seen in >= 2 runs are stored." % (max_n, ORDINARY_RUNS)),
           "n_utterances": n_utt, "n_tokens": n_tok,
           "phrases": {p: sorted(v) for p, v in table.items() if len(v) >= 2},
           "common": {p: sorted(v) for p, v in seen_table.items() if len(v) >= COMMON_KEEP_RUNS}}
    Path(out).write_text(json.dumps(doc, separators=(",", ":"), sort_keys=True))
    _load.cache_clear()
    return doc
