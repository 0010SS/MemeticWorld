"""Meme report for a run (OBSERVER ONLY; no LLM calls, never written into the run).

meme_report(run_dir, debug=False) merges, per expression:
- the live (LLM-free) analysis at the latest tick (live.py): status chips, first use, adopters timeline,
  transmission tree rooted at the originator, contexts, world-wording flags;
- analysis.json when present (pipeline: emergence, legacy card, classifier annotation, v2 probe answers,
  grounding in debug mode only);
- every judge verdict available (analysis_judgements/*.json and analysis.json `judgements`), newest first;
- meaning over time from v3 checkpoint probes (meaning.json / probes/C*/responses.jsonl) when the
  expression is a probe target (hidden-derived: marked {"hidden": true});
plus a run-level summary paragraph built from the numbers by template (no LLM) and a judge_slot section.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from backend.analysis import judge as JG
from backend.analysis.candidates import tokens
from backend.analysis.live import live_snapshot, snapshot_context, strip_hidden, transmission_tree

REPORT_VERSION = "report-v1"
MAX_CONTEXTS = 6
HIDDEN_FIELDS = ["cards[].meaning_over_time", "v3.first_attempt_accuracy", "v3.old_regime_response",
                 "v3.k1_vs_k2", "v3.regime"]


def _load(p: Path):
    try:
        return json.load(open(p))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _norm(s: str) -> str:
    return " ".join(tokens(s or ""))


def _forms(x: dict) -> set:
    return {_norm(f) for f in [x.get("phrase") or x.get("canonical_form"), *(x.get("variants") or [])] if f}


def _match(forms: set, phrase: str) -> bool:
    p = _norm(phrase)
    return bool(p) and p in forms


def _spread(items: list, k: int) -> list:
    if len(items) <= k:
        return list(items)
    step = (len(items) - 1) / (k - 1)
    return [items[round(i * step)] for i in range(k)]


def _n(k, word: str, plural: str | None = None) -> str:
    return f"{k} {word if k == 1 else (plural or word + 's')}"


# ------------------------------------------------------------------------------------------------ judgements
def collect_verdicts(run_dir: Path, analysis: dict | None) -> list[dict]:
    """Every verdict on this run: [{phrase, variants, expression_id, judged_at, source, verdict}]."""
    out = []
    for doc in JG.load_judgements(run_dir):
        for row in doc.get("verdicts") or []:
            if isinstance(row, dict) and isinstance(row.get("verdict"), dict):
                out.append({"phrase": row.get("phrase"), "variants": row.get("variants") or [],
                            "expression_id": row.get("expression_id"),
                            "judged_at": row.get("judged_at") or doc.get("generated_at"),
                            "source": doc.get("_file"), "verdict": row["verdict"]})
    for c in (analysis or {}).get("candidates") or []:
        for _jid, v in (c.get("judgements") or {}).items():
            if isinstance(v, dict):
                out.append({"phrase": c.get("canonical_form"), "variants": c.get("variants") or [],
                            "expression_id": c.get("id"), "judged_at": (analysis or {}).get("generated_at"),
                            "source": "analysis.json", "verdict": v})
    return out


def _card_verdicts(forms: set, verdicts: list[dict], debug: bool) -> list[dict]:
    rows = []
    for e in verdicts:
        if not ({_norm(e["phrase"] or "")} | {_norm(v) for v in e["variants"]}) & forms:
            continue
        v = e["verdict"]
        row = {k: v.get(k) for k in ("judge_id", "provider", "model", "prompt_version", "is_convention", "gloss",
                                     "function", "meaning_consistency", "confidence", "rationale")}
        row.update(judged_at=e["judged_at"], source=e["source"], expression_id=e["expression_id"])
        if debug:
            row["raw"] = v.get("raw")
        rows.append(row)
    rows.sort(key=lambda r: str(r.get("judged_at") or ""), reverse=True)
    return rows


def judge_slot(run_dir: Path, cfg: dict, analysis: dict | None) -> dict:
    ran = JG.available_judges(run_dir, cfg)
    runs = list(ran.get("ran") or [])
    if analysis:
        ids = {}
        for c in analysis.get("candidates") or []:
            for jid, v in (c.get("judgements") or {}).items():
                d = ids.setdefault(jid, {"judge_id": jid, "provider": v.get("provider"), "model": v.get("model"),
                                         "prompt_version": v.get("prompt_version"),
                                         "generated_at": analysis.get("generated_at"), "n_verdicts": 0,
                                         "n_conventions": 0, "source": "analysis.json", "file": "analysis.json"})
                d["n_verdicts"] += 1
                d["n_conventions"] += int(bool(v.get("is_convention")))
        runs += list(ids.values())
    runs.sort(key=lambda r: str(r.get("generated_at") or ""), reverse=True)
    return {"config_key": JG.CONFIG_KEY, "configured": JG.normalize_spec(cfg), "providers": ran["providers"],
            "prompt_versions": ran["prompt_versions"], "ran": runs,
            "note": "POST a judge spec {provider, model, prompt_version} to judge_run(); verdicts land in "
                    "analysis_judgements/<judge_id>.json and appear on every card, newest first."}


# ------------------------------------------------------------------------------------------------ meaning
def _meaning_over_time(run_dir: Path, rd, forms: set) -> dict | None:
    """v3 checkpoint probes for an expression that is a probe target (X / W / Y in probes/targets.json)."""
    if not (Path(run_dir) / "probes").exists() and not (Path(run_dir) / "meaning.json").exists():
        return None
    try:
        from backend.analysis.outcomes_v3 import _target_forms, load_targets
        tforms = _target_forms(load_targets(run_dir), rd.cfg or {})
    except Exception:   # noqa: BLE001 - optional v3 layer
        return None
    cue = next((k for k in ("X", "W", "Y") if tforms.get(k) and _norm(str(tforms[k])) in forms), None)
    if cue is None:
        return None
    mj = _load(Path(run_dir) / "meaning.json")
    if not mj or not mj.get("checkpoints"):
        try:
            from backend.analysis.meaning import analyze_meaning
            mj = analyze_meaning(run_dir, None, tforms, write=False)
        except Exception:   # noqa: BLE001
            mj = None
    cks = (mj or {}).get("checkpoints") or {}
    if not cks:
        return None
    rows = []
    for ck in sorted(cks, key=lambda k: int(str(k).lstrip("Cc").split("_")[0] or 0)):
        cell = cks[ck] or {}
        ext = (cell.get("ext") or {}).get(cue) or {}
        rows.append({"checkpoint": ck, "regime": cell.get("regime"), "sri": (cell.get("sri") or {}).get(cue),
                     "ext": ext.get("ext"), "breadth": ext.get("breadth"), "know": cell.get("know"),
                     "n_responses": cell.get("n_responses")})
    return {"hidden": True, "cue": cue, "target": tforms.get(cue), "checkpoints": rows}


# ------------------------------------------------------------------------------------------------ cards
def _analysis_block(c: dict, an: dict, debug: bool) -> dict:
    cid = c["id"]
    card = dict(c.get("card") or {})
    if not debug:
        card.pop("latent_alignment", None)       # pre-v2 alignment with hidden families
    out = {"id": cid, "legacy_status": c.get("status"), "card": card, "classifier": c.get("llm"),
           "emergence": (an.get("emergence") or {}).get(cid), "score": c.get("score"),
           "transmission_depth": ((an.get("transmission") or {}).get(cid) or {}).get("depth")}
    pr = (an.get("probes") or {}).get(cid)
    if pr:
        out["probes"] = [{"agent": a, "heard_before": v.get("heard_before"), "used": v.get("used"),
                          "meaning": v.get("meaning"), **({"match_family": v.get("match_family")} if debug else {})}
                         for a, v in sorted((pr.get("agents") or {}).items())]
    if debug:
        out["grounding"] = ((an.get("grounding") or {}).get("candidates") or {}).get(cid)
        out["evaluation"] = (an.get("evaluation") or {}).get(cid)
    return out


def _card(sn, r: dict, a: dict | None, an: dict | None, verdicts: list[dict], debug: bool, run_dir: Path) -> dict:
    rd = sn.rd
    names = rd.names
    forms = _forms(r) | (_forms(a) if a else set())
    usages = r["_usages"]
    contexts = [{"tick": u["tick"], "time_label": _label(sn, u["tick"]), "speaker": u["speaker"],
                 "utterance_id": u["utterance_id"],
                 "text": (rd.conversation_context(rd.utt_by_id[u["utterance_id"]]) if u["utterance_id"] in rd.utt_by_id
                          else u.get("context") or "").strip()} for u in _spread(usages, MAX_CONTEXTS)]
    js = _card_verdicts(forms, verdicts, debug)
    card = {
        "id": r["id"], "analysis_id": a["id"] if a else None, "phrase": r["phrase"], "display": r["display"],
        "variants": r["variants"], "status": r["status"], "flags": r["flags"],
        "first_use": r["first_use"], "uses": r["uses"], "n_speakers": r["n_speakers"],
        "speakers": [{"agent": s, "name": names.get(s, s)} for s in r["speakers"]],
        "last_use_tick": r["last_use_tick"], "trend": r["trend"], "adoption_curve": r["adoption_curve"],
        "adopters_timeline": [{"agent": x["agent"], "name": names.get(x["agent"], x["agent"]), "role": x["role"],
                               "first_exposure_tick": x["first_exposure_tick"], "first_use_tick": x["first_use_tick"],
                               "carried_tick": x["carried_tick"], "heard_from": (x["source"] or {}).get("agent")}
                              for x in r["_adopters"]],
        "transmission_tree": transmission_tree(r["_adopters"], names),
        "contexts": contexts,
        "world_wording": dict(r["world_wording"], planted=r["planted"]),
        "emergence": r["emergence"],
        "analysis": _analysis_block(a, an, debug) if a and an else None,
        "meaning_over_time": _meaning_over_time(run_dir, rd, forms),
        "judgements": js, "latest_verdict": js[0] if js else None,
    }
    if not debug and card["analysis"] and "grounding" in card["analysis"]:
        card["analysis"].pop("grounding")
    return card


def _label(sn, tick: int) -> str:
    from backend.analysis.live import _label as L
    return L(sn.man, tick)


# ------------------------------------------------------------------------------------------------ summary
def _summary(snap: dict, cards: list[dict], slot: dict, debug: bool) -> dict:
    c = snap["status_counts"]
    running = snap.get("run_status") != "finished"
    parts = [f"Run {snap['run_id']} ({snap.get('run_status') or 'unknown status'}) is analysed up to "
             f"{snap['time_label']} (tick {snap['tick']}{', still being written' if running else ''}).",
             f"Agents produced {_n(snap['n_utterances'], 'utterance')} in {_n(snap['n_conversations'], 'conversation')} "
             f"among {_n(snap['n_agents_active'], 'active agent')}, and the world released {_n(snap['n_events'], 'event')}."]
    n = snap["n_expressions"]
    if n:
        bits = [f"{c['emerged']} emerged", f"{c['spreading']} spreading after exposure",
                f"{c['echo']} echoed only inside a conversation", f"{c['world_wording']} world or lexicon wording",
                f"{c['new']} new"] + ([f"{c['planted']} planted"] if c.get("planted") else [])
        parts.append(f"The observer tracks {_n(n, 'recurring expression')}: " + ", ".join(bits) + ".")
        lead = next((x for s in ("emerged", "spreading") for x in cards if x["status"] == s or s in x["flags"]), None)
        if lead:
            fu = lead["first_use"]
            em = lead["emergence"]
            parts.append(f"The most established is \"{lead['display']}\", first said by {fu['speaker_name']} "
                         f"({fu['time_label']}); {_n(lead['n_speakers'], 'speaker')} used it {_n(lead['uses'], 'time')}, "
                         f"{em.get('n_adopters_carried') or 0} of them after hearing it in another conversation "
                         f"(versus {em.get('n_independent') or 0} independent).")
        else:
            parts.append("No expression has yet been carried by an exposed agent into a different conversation.")
    else:
        parts.append("No expression has recurred yet.")
    pl = snap.get("planted")
    if pl:
        parts.append(f"The planted control \"{pl['phrase']}\" ({pl['agent']}) "
                     + (f"has been used {_n(pl['uses'], 'time')} and is {pl['status']}." if pl.get("uses")
                        else "has not been used yet."))
    f = snap["funnel"]["cumulative"]
    parts.append(f"Mediators so far: {f['events_witnessed']} of {f['events_released']} events witnessed and "
                 f"{f['events_discussed']} discussed; {_n(f['cross_incident_links'], 'cross-incident memory link')} "
                 f"and {_n(f['reminding_linked'], 'linked reminding')}; {f['stuck_other']} wordings stuck verbatim "
                 f"from others ({f['stuck_self']} from own speech); {f['reuse_after_exposure']} reuses after exposure.")
    v3 = snap.get("v3")
    if v3:
        j, b, ro = v3["jobs"], v3["binder"], v3["roster"]
        s = (f"At the co-op, {_n(j['started'], 'job')} started and {j['delivered']} were delivered; members made "
             f"{_n(b['writes'], 'binder entry', 'binder entries')} and read the binder {_n(b['reads'], 'time')}")
        s += f" (last change {b['staleness_ticks']} ticks ago)." if b.get("staleness_ticks") is not None else "."
        if ro["departed"] or len(ro["cohorts"]) > len(ro["active"]):
            newc = sum(1 for a, k in ro["cohorts"].items() if k != "founder")
            s += f" {_n(len(ro['departed']), 'member')} left and {newc} joined."
        parts.append(s)
        if debug:
            acc = (v3.get("first_attempt_accuracy") or {}).get("all") or {}
            if acc.get("acc") is not None:
                parts.append(f"[debug] First-attempt accuracy on faulted jobs is {acc['acc']:.0%} (n={acc['n']}); "
                             f"current regime {v3['regime']['current']}.")
    ran = slot.get("ran") or []
    if ran:
        last = ran[0]
        parts.append(f"{_n(len(ran), 'judge verdict set')} attached; the latest ({last['judge_id']}) calls "
                     f"{last.get('n_conventions') or 0} of {last.get('n_verdicts') or 0} judged expressions conventions.")
    else:
        parts.append("No LLM judge has been run on this run yet.")
    numbers = {k: snap[k] for k in ("tick", "n_utterances", "n_conversations", "n_events", "n_agents_active",
                                    "n_expressions")}
    numbers.update(status_counts=c, funnel=f, n_judge_sets=len(ran))
    return {"text": " ".join(parts), "numbers": numbers}


# ------------------------------------------------------------------------------------------------ entry
def meme_report(run_dir, debug: bool = False, top: int = 30, tick: int | None = None) -> dict:
    """Per-expression cards + template summary + judge slot. debug=False leaves out grounding, probe
    family matches and raw judge text; hidden-derived blocks stay marked {"hidden": true} (strip them
    with live.strip_hidden for demo mode)."""
    run_dir = Path(run_dir)
    snap = live_snapshot(run_dir, tick, top)
    sn = snapshot_context(run_dir, tick, top)
    an = _load(run_dir / "analysis.json")
    verdicts = collect_verdicts(run_dir, an)
    pool = sn.pool
    shown = pool[:top] + [r for r in pool[top:] if r["planted"]][:1]
    acands = list((an or {}).get("candidates") or [])
    used, cards = set(), []
    for r in shown:
        f = _forms(r)
        a = next((c for c in acands if c["id"] not in used and (_forms(c) & f)), None)
        if a:
            used.add(a["id"])
        cards.append(_card(sn, r, a, an, verdicts, debug, run_dir))
    for c in acands[:top]:                      # pipeline candidates the live view did not surface
        if c["id"] in used:
            continue
        us = [u for u in c.get("usages") or [] if u.get("utterance_id") in sn.rd.utt_by_id]
        if not us:
            continue
        r = sn.record(dict(c, usages=us))
        r["id"] = f"a{c['id']}"
        cards.append(_card(sn, r, c, an, verdicts, debug, run_dir))
    slot = judge_slot(run_dir, sn.rd.cfg or {}, an)
    out = {
        "run_id": run_dir.name, "report_version": REPORT_VERSION,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"), "debug": bool(debug),
        "tick": snap["tick"], "latest_tick": snap["latest_tick"], "time_label": snap["time_label"],
        "run_status": snap.get("run_status"),
        "summary": _summary(snap, cards, slot, debug), "cards": cards, "judge_slot": slot,
        "headline": {k: snap[k] for k in ("n_utterances", "n_conversations", "n_events", "n_agents_active",
                                          "n_expressions", "status_counts")},
        "funnel": snap["funnel"], "transmission": snap["transmission"], "planted": snap.get("planted"),
        "sources": {"live": True, "analysis_json": an is not None,
                    "analysis_version": (an or {}).get("analysis_version"), "observer": (an or {}).get("observer"),
                    "judgements": sorted({v["source"] for v in verdicts if v.get("source")}),
                    "meaning_json": (run_dir / "meaning.json").exists()},
        "hidden_fields": list(HIDDEN_FIELDS),
    }
    if snap.get("v3"):
        out["v3"] = snap["v3"]
    return out


def demo_report(run_dir, top: int = 30) -> dict:
    """meme_report without debug detail and without hidden-derived blocks (what demo mode may show)."""
    return strip_hidden(meme_report(run_dir, debug=False, top=top))
