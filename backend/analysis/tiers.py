"""Status and tier of an expression (OBSERVER ONLY), used alike by pipeline.py (analysis.json), outcomes.py
(outcomes.json), live.py and report.py.

`status` (one chip, first applicable) and `flags` (every applicable chip):
    planted > system_wording > world_wording > emerged > spreading > echo > new
- planted: matches controls.planted_phrase (the positive control);
- system_wording: a substring / template match of text the SYSTEM gave the agents (wording.Infrastructure:
  relationship lines, routines, seed and ambient memories, memory frames, profile text, lexicon ...);
- world_wording: event wording (facts, referent names, viewpoint renderings, co-op surfaces);
- emerged / spreading / echo / new: the exposure test (emergence.py).

`tier`, the three-step ladder from "said a lot" to "culture":
- "candidate": a well-formed recurring expression (frequency only);
- "spreading": >= 1 exposure-driven adopter who used it in a different conversation at a strictly later tick
  (a carried adopter), and it is neither system nor world wording;
- "convention": spreading, emerged under the exposure test (>= 2 carried adopters, more than independent
  users), AND a REAL judge (not the mock judge / mock backend: judge.is_real_verdict) said is_convention=true,
  and it is not the planted control (the planted phrase can reach the tier only as the control: `control`
  is set on it and it is never counted among conventions).
`tier_reasons` says why an expression stopped where it did (e.g. ["system_wording"], ["no_real_judge"]).

`ordinary` (register.Background) stops an expression at "candidate" the same way wording does: a phrase
several speakers also use in other, independent runs is the actor model's own register, so it is everyone's
language and cannot be this campus's convention -- the judge is not asked to be the backstop for that.
`bucket` (candidates.bucket_of) is the matching split of the ranked pool: expression > personal > ordinary
> wording, so the top of the list can only hold things that could be culture.
"""
from __future__ import annotations

from backend.analysis.candidates import tokens

STATUSES = ("planted", "system_wording", "world_wording", "emerged", "spreading", "echo", "new")
TIERS = ("candidate", "spreading", "convention")
NO_REAL_JUDGE = "no real judge has run"


def classify_status(em: dict, *, world: bool, planted: bool, system: bool = False,
                    ordinary: bool = False) -> tuple[str, list[str]]:
    """(status, flags). The dynamic chip: emerged (spread, not world/system wording and not the actor
    model's own register) > spreading (>= 1 carried adopter) > echo (adopters, none carried) > new.
    `ordinary` also adds the "ordinary" chip to `flags` (it is not a status of its own, so the UI's seven
    status chips are unchanged)."""
    if em.get("spread") and not world and not system and not ordinary:
        dyn = "emerged"
    elif (em.get("n_adopters_carried") or 0) >= 1:
        dyn = "spreading"
    elif (em.get("n_adopters") or 0) >= 1:
        dyn = "echo"
    else:
        dyn = "new"
    flags = (["planted"] if planted else []) + (["system_wording"] if system else []) + \
            (["world_wording"] if world else []) + [dyn] + (["ordinary"] if ordinary and not planted else [])
    return flags[0], flags


def tier_of(em: dict, *, system: bool, world: bool, planted: bool, verdict: dict | None = None,
            placeholder: dict | None = None, ordinary: bool = False) -> tuple[str, list[str]]:
    """(tier, reasons). `verdict`: the newest REAL judge verdict for the expression (or None); `placeholder`:
    a mock verdict, recorded only to say why there is no real one. `ordinary`: the expression is the actor
    model's own register or a stock formula (register.Background) -- everyone's language, so it stops at
    "candidate" whatever a judge says about it."""
    stop = []
    if system:
        stop.append("system_wording")
    if world:
        stop.append("world_wording")
    if ordinary and not planted:
        stop.append("model_register")
    if (em.get("n_adopters_carried") or 0) < 1:
        stop.append("no_carried_adopter")
    if stop:
        return "candidate", stop
    up = []
    if not em.get("spread"):
        up.append("not_emerged")
    if verdict is None:
        up.append("only_mock_verdict" if placeholder is not None else "no_real_judge")
    elif not verdict.get("is_convention"):
        up.append("judge_rejected")
    if up:
        return "spreading", up
    return "convention", (["planted_control"] if planted else [])


def forms(x: dict) -> set[str]:
    """Normalized surface forms (canonical / phrase + variants) of an expression or a verdict row."""
    out = set()
    for f in [x.get("phrase"), x.get("canonical_form"), *(x.get("variants") or [])]:
        if f:
            n = " ".join(tokens(str(f)))
            if n:
                out.add(n)
    return out


class VerdictIndex:
    """Judge verdict rows (judge.analysis_verdicts / judgement_verdicts) by normalized form, split into real and
    placeholder (mock) verdicts; `lookup` returns the newest of each for an expression."""

    def __init__(self, rows: list[dict]):
        from backend.analysis.judge import is_real_verdict
        self.rows = list(rows or [])
        self.by_form: dict[str, list[dict]] = {}
        for r in self.rows:
            r = dict(r, real=is_real_verdict(r.get("verdict") or {}))
            for f in forms(r):
                self.by_form.setdefault(f, []).append(r)
        self.n_real = sum(1 for r in self.rows if is_real_verdict(r.get("verdict") or {}))
        self.n_placeholder = len(self.rows) - self.n_real

    def lookup(self, fs: set[str], usable=None) -> tuple[dict | None, dict | None]:
        """(newest real row, newest placeholder row) among rows matching any form in fs (and usable(row))."""
        rows = {id(r): r for f in fs for r in self.by_form.get(f, []) if usable is None or usable(r)}
        real = [r for r in rows.values() if r["real"]]
        mock = [r for r in rows.values() if not r["real"]]
        key = lambda r: str(r.get("judged_at") or "")
        return (max(real, key=key) if real else None), (max(mock, key=key) if mock else None)

    def state(self) -> dict:
        """{"status": real | mock_only | none, "n_real", "n_placeholder", "note"}."""
        if self.n_real:
            return {"status": "real", "n_real": self.n_real, "n_placeholder": self.n_placeholder,
                    "note": f"{self.n_real} real judge verdicts available"}
        if self.n_placeholder:
            return {"status": "mock_only", "n_real": 0, "n_placeholder": self.n_placeholder,
                    "note": f"{NO_REAL_JUDGE}: only mock judge verdicts (placeholders), so no expression is a convention"}
        return {"status": "none", "n_real": 0, "n_placeholder": 0,
                "note": f"{NO_REAL_JUDGE}: no expression is a convention until one does"}


def verdict_summary(row: dict | None) -> dict | None:
    """The part of a verdict row an expression record carries."""
    if not row:
        return None
    v = row.get("verdict") or {}
    return {"judge_id": v.get("judge_id"), "provider": v.get("provider"), "model": v.get("model"),
            "prompt_version": v.get("prompt_version"), "is_convention": v.get("is_convention"),
            "gloss": v.get("gloss"), "confidence": v.get("confidence"), "real": bool(row.get("real")),
            "judged_at": row.get("judged_at"), "source": row.get("source")}


def tier_counts(records: list[dict]) -> dict:
    """{"candidate", "spreading", "convention"} over records carrying `tier`; the planted control is left out."""
    out = {t: 0 for t in TIERS}
    for r in records:
        if r.get("control") or r.get("planted"):
            continue
        t = r.get("tier")
        if t in out:
            out[t] += 1
    return out
