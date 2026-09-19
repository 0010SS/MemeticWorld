"""Candidate quality: what the observer may call an expression, system wording, and the three tiers.

* the old junk of a mock run ('remembers that ethan', 'maya chen are roommates', 'labmates they know each', ...)
  is excluded by the well-formedness rule or marked system_wording, and a mock run never has a convention
  (live view, pipeline, outcomes.json, report): with only mock verdicts, "no real judge has run";
  the same checks run on the demo mock run when MEMEWORLD_DEMO_MOCK points at it (or runs/demo_mock exists);
* nickname constructions ("the Leo thing", "pulling a Maya", "Leo-proof") are kept, bare names are not;
* on a synthetic run a genuine coinage carried across conversations reaches "spreading"; it becomes a
  "convention" only when it emerged AND a REAL judge said so; a mock judge (or a mock "yes") never makes one;
  system wording that spreads stays a "candidate" even with a real "yes"; the planted control never counts;
* phrases recited from the speaker's own memory text are flagged (recited_share) and down-ranked.
"""
import json
import os
import shutil
from pathlib import Path

import pytest
import yaml

from backend.analysis import judge as JG
from backend.analysis.candidates import CandidateExtractor, tokens
from backend.analysis.live import clear_cache, live_snapshot, snapshot_context
from backend.analysis.report import meme_report
from backend.analysis.rundata import RunData
from backend.analysis.tiers import tier_of
from backend.analysis.wording import FUNCTION, REPORT, WordClasses, segment
from tests.conftest import ROOT, run_sim

# what the pre-fix extractor (and a mock judge) served as "conventions" on the demo mock run
OLD_JUNK = ["remembers that ethan", "owen reed remembers", "maya chen are roommates", "thinking about how tess",
            "labmates they know each", "acquaintances", "dev patel is preparing", "morgan remembers that tess",
            "remembers that a student", "maya chen are classmates", "tess morgan are friends", "benji answered a pile",
            "thinking about how hana", "tess morgan is eating", "leo martinez are friends", "lunch at the dining"]
RUN_FILES = ("manifest.json", "config.resolved.yaml", "trace.jsonl", "events.jsonl", "world_script.jsonl",
             "memory_meta.json")
# the demo configuration (v2 mechanisms on; the mock LLM stitches memory text into its dialogue), shortened
DEMO_LIKE = {"seed": 42, "population_size": 16, "day_end": "14:30",
             "llm": {"backend": "mock", "max_workers": 6},
             "latent_events": {"event_rate": 0.35, "referents": {"enabled": True}},
             "conversation": {"catchup": {"enabled": True}}, "need": {"enabled": True, "min_importance": 3},
             "reminding": {"enabled": True}, "memory": {"verbatim": {"enabled": True}}, "priming": {"enabled": True}}


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_cache()
    yield
    clear_cache()


def _demo_dir() -> Path | None:
    for p in (os.environ.get("MEMEWORLD_DEMO_MOCK"), ROOT / "runs" / "demo_mock"):
        if p and (Path(p) / "trace.jsonl").exists():
            return Path(p)
    return None


@pytest.fixture(scope="module")
def mock_junk_run(tmp_path_factory):
    return run_sim(tmp_path_factory.mktemp("cq_mock"), DEMO_LIKE, name="demo_like").run_dir


def _copy(src: Path, dst: Path, extra=()) -> Path:
    dst.mkdir(parents=True)
    for f in RUN_FILES + tuple(extra):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    return dst


def _assert_no_junk(d: Path):
    s = live_snapshot(d, top=60)
    sn = snapshot_context(d, top=60)
    ex = sn.extractor
    pool = {f: r for r in sn.pool for f in [r["phrase"], *r["variants"]]}
    for junk in OLD_JUNK:
        e = ex.explain(junk)
        assert not e["wellformed"] or e["system_wording"], e
        r = pool.get(" ".join(tokens(junk)))
        assert r is None or r["status"] == "system_wording", (junk, r and r["status"])
    for r in sn.pool:
        if r["planted"]:
            continue
        toks = r["phrase"].split()
        raw = segment(r["display"])[0]
        cls = ex.wc.classes(toks, raw if len(raw) == len(toks) else None)
        assert ex.wc.reject(toks, cls) is None, r["phrase"]
        if "name" in cls:                                   # a person's name only inside a nickname construction
            assert ex.wc.nickname(toks, cls), r["phrase"]
        assert toks[0] not in FUNCTION | REPORT and toks[-1] not in FUNCTION | REPORT, r["phrase"]
        assert r["tier"] != "convention"
    assert s["tier_counts"]["convention"] == 0 and s["n_conventions"] == 0
    assert s["judge"]["status"] in ("none", "mock_only") and not s["judge"]["real_judge"]
    return s, sn


# ------------------------------------------------------------------------------------------------ mock runs
def test_mock_run_junk_is_excluded_and_no_convention(mock_junk_run, tmp_path):
    d = _copy(mock_junk_run, tmp_path / "run")
    s, sn = _assert_no_junk(d)
    # the stitched memory frames and relationship lines really were there to be picked up
    rd = RunData(d)
    texts = " ".join(" ".join(tokens(u["text"])) for u in rd.utterances)
    assert " remembers that " in texts and any(f" are {r} " in texts for r in ("roommates", "friends", "classmates",
                                                                             "labmates", "acquaintances"))
    assert sn.extractor.rejected["person_name"] > 0 and sn.extractor.rejected["function_word_edge"] > 0
    # a mock judge on top: placeholders only, never a convention; the report says so
    JG.judge_run(d, {"provider": "mock"}, top=30)
    clear_cache()
    s2 = live_snapshot(d, top=60)
    assert s2["judge"]["status"] == "mock_only" and s2["tier_counts"]["convention"] == 0
    assert all(e["tier"] != "convention" for e in s2["expressions"])
    doc = json.load(open(d / "analysis_judgements" / "mock-mock-v1.json"))
    assert doc["placeholder"] is True and doc["real_judge"] is False and doc["n_conventions"] == 0
    rep = meme_report(d)
    assert "no real judge has run" in rep["summary"]["text"] and rep["headline"]["n_conventions"] == 0
    assert all(c["tier"] != "convention" for c in rep["cards"])


def test_mock_pipeline_has_no_convention(mock_junk_run, tmp_path):
    from backend.analysis.pipeline import analyze
    d = _copy(mock_junk_run, tmp_path / "an")
    out = analyze(d, llm_backend="mock", probes=False, verbose=False)
    sm = out["summary"]
    assert sm["n_conventions"] == 0 and sm["tiers"]["convention"] == 0 and sm["n_llm_conventions"] == 0
    assert sm["judge_status"] == "mock_only" and "no real judge has run" in sm["judge_note"]
    for c in out["candidates"]:
        assert c["tier"] in ("candidate", "spreading") and c["status"] in (
            "planted", "system_wording", "world_wording", "emerged", "spreading", "echo", "new")
        assert c["card"]["is_convention"] is None                  # no real judge: the UI shows "—"
        assert c["canonical_form"] not in OLD_JUNK or c["status"] == "system_wording"
        if c.get("llm"):                                           # the mock verdict is kept only as a placeholder
            assert c["llm"]["is_convention"] is None and c["llm"]["real_judge"] is False
    oc = json.load(open(d / "outcomes.json"))
    assert oc["n_conventions"] == 0 and oc["conventions"] == [] and oc["judge"]["status"] == "mock_only"
    assert set(oc["tiers"]) == {"candidate", "spreading", "convention"}


@pytest.mark.skipif(_demo_dir() is None, reason="demo mock run not available (set MEMEWORLD_DEMO_MOCK)")
def test_demo_mock_run(tmp_path):
    src = _demo_dir()
    d = _copy(src, tmp_path / "demo", extra=("analysis.json", "analysis_llm_calls.jsonl"))
    _assert_no_junk(d)
    old = json.load(open(d / "analysis.json"))
    sn = snapshot_context(d, top=60)
    for c in old["candidates"]:                     # every old candidate is now rejected or system/world wording
        e = sn.extractor.explain(c["canonical_form"])
        now = next((r for r in sn.pool if r["phrase"] == e["phrase"] or e["phrase"] in r["variants"]), None)
        assert not e["wellformed"] or e["system_wording"] or (now and now["status"] in ("system_wording", "world_wording")), c["canonical_form"]
    # the old analysis' mock classifier "conventions" are placeholders now (legacy provenance: model "mock")
    rows = JG.analysis_verdicts(d, old)
    assert rows and all(not JG.is_real_verdict(r["verdict"]) for r in rows)
    assert "no real judge has run" in meme_report(d)["summary"]["text"]


# ------------------------------------------------------------------------------------------------ word classes
@pytest.mark.parametrize("phrase", ["the leo thing", "pulling a maya", "leo-proof", "classic priya", "going full dev",
                                    "maya-style", "leo'd", "glimmer toast", "forty-seven steps", "running on fumes"])
def test_nickname_constructions_and_coinages_are_kept(phrase):
    wc = WordClasses({"leo", "maya", "priya", "dev", "chen"}, {"dorm"}, set(), {("maya", "chen")})
    toks = phrase.split()
    assert wc.reject(toks, raws=[t.capitalize() for t in toks]) is None


@pytest.mark.parametrize("phrase,why", [
    ("leo said", "person_name"), ("maya chen", "person_name"), ("remembers that leo", "person_name"),
    ("thinking about how maya", "person_name"), ("leo is preparing", "person_name"), ("maya's backpack", "person_name"),
    ("the leo", "person_name"), ("are friends", "function_word_edge"), ("roommates", "bare_role_or_place"),
    ("labmates", "bare_role_or_place"), ("a student", "function_word_edge"), ("dorms", "bare_role_or_place"),
    ("remembers that kettle", "function_word_edge"), ("kettle remembers that", "function_word_edge"),
    ("said that kettle thing", "function_word_edge"), ("kettle said that", "function_word_edge"),
    ("kettle is plotting", "clause_fragment"), ("kettle are roommates", "clause_fragment"),
    ("9am lecture", "number_or_time"), ("room 216", "number_or_time"), ("nine-thirty lecture", "function_word_edge"),
    ("tuesday rush", "function_word_edge"), ("kettle again", "function_word_edge"), ("nooo", "function_word_edge"),
    ("dinner's", "common_unigram"), ("might've", "function_word_edge"), ("deadlines", "common_unigram"),
    ("kettle they know each", "function_word_edge")])
def test_malformed_phrases_are_rejected(phrase, why):
    wc = WordClasses({"leo", "maya", "chen"}, {"dorm", "room"}, set(), {("maya", "chen")})
    toks = [t.lower() for t in phrase.split()]
    assert wc.reject(toks, raws=[t.capitalize() for t in phrase.split()]) == why


def test_ambiguous_names_need_capitals_but_full_names_do_not():
    wc = WordClasses({"miles", "carter", "park", "benji"}, set(), set(), {("miles", "carter"), ("benji", "park")})
    assert wc.reject(["miles", "away"], raws=["miles", "away"]) is None           # the unit, lower case
    assert wc.reject(["miles", "away"], raws=["Miles", "away"]) == "person_name"  # the person
    assert wc.reject(["park", "bench"], raws=["park", "bench"]) is None
    assert wc.reject(["miles", "carter"], raws=["miles", "carter"]) == "person_name"
    assert wc.reject(["benji", "park"], raws=["benji", "park"]) == "person_name"


def test_tier_rules_and_placeholder_verdicts():
    em = {"n_adopters_carried": 2, "n_adopters": 2, "spread": True}
    yes_real = {"provider": "claude_cli", "model": "sonnet", "is_convention": True}
    yes_mock = {"provider": "mock", "model": "mock", "is_convention": True}
    assert tier_of(em, system=False, world=False, planted=False, verdict=yes_real) == ("convention", [])
    assert tier_of(em, system=False, world=False, planted=False, placeholder=yes_mock) == ("spreading", ["only_mock_verdict"])
    assert tier_of(em, system=False, world=False, planted=False) == ("spreading", ["no_real_judge"])
    assert tier_of(em, system=True, world=False, planted=False, verdict=yes_real) == ("candidate", ["system_wording"])
    assert tier_of(em, system=False, world=True, planted=False, verdict=yes_real) == ("candidate", ["world_wording"])
    assert tier_of({**em, "spread": False, "n_adopters_carried": 1}, system=False, world=False, planted=False,
                   verdict=yes_real) == ("spreading", ["not_emerged"])
    assert tier_of({"n_adopters_carried": 0}, system=False, world=False, planted=False, verdict=yes_real)[0] == "candidate"
    assert tier_of(em, system=False, world=False, planted=True, verdict=yes_real) == ("convention", ["planted_control"])
    assert JG.is_real_verdict(yes_real) and not JG.is_real_verdict(yes_mock)
    assert not JG.is_real_verdict({"provider": "claude_cli", "model": "mock"})
    blk = JG.llm_block(JG.MockJudge().verdict({"is_convention": True, "gloss": "x"}, "{}"))
    assert blk["is_convention"] is None and blk["placeholder"]["is_convention"] is True and not blk["real_judge"]


# ------------------------------------------------------------------------------------------------ synthetic run
NAMES = {"maya": "Maya Chen", "leo": "Leo Martinez", "dev": "Dev Patel", "hana": "Hana Okafor"}
PLANTED = {"agent": "maya", "habit": 'Maya has a habit of calling any mess "a full pickle".'}


def _time(t):
    return f"2026-09-14T{7 + (30 + 15 * t) // 60:02d}:{(30 + 15 * t) % 60:02d}:00"


def _conv(cid, tick, turns):
    us = [{"type": "utterance", "tick": tick, "time": _time(tick), "id": f"{cid}.u{i}", "conversation_id": cid, "idx": i,
           "speaker": s, "listeners": ls, "text": t, "retrieved": [],
           "context": f"{NAMES[s]} was studying when {NAMES[s]} saw {NAMES[ls[0]]} in the middle of reading.\n"
                      f"{NAMES[s]} and {NAMES[ls[0]]} are classmates; they know each other a little and get along fine."}
          for i, (s, ls, t) in enumerate(turns)]
    conv = {"type": "conversation", "tick": tick, "id": cid, "trigger": None,
            "participants": sorted({s for s, *_ in turns} | {l for _, ls, _ in turns for l in ls}),
            "transcript": [[NAMES[s], t] for s, _, t in turns]}
    return [conv, *us]


def _mem(agent, tick, text, source="conversation", n=0):
    return {"type": "memory_encoded", "tick": tick, "time": _time(tick), "id": f"m{tick}{agent}{n}", "agent": agent,
            "node_id": f"{agent}:m{tick}{n}", "kind": "event", "text": text, "source_type": source,
            "observation": text if source == "ambient" else None, "utterance_ids": []}


def _synthetic(d: Path) -> Path:
    d.mkdir(parents=True)
    agents = {a: {"id": a, "name": n, "background": "", "habits": [], "routine": [
        {"time": "12:00", "location": "Dining Hall", "activity": "eating lunch"}]} for a, n in NAMES.items()}
    man = {"run_id": d.name, "status": "finished", "ticks": 120, "ticks_per_day": 60, "tick_minutes": 15,
           "start": "2026-09-14T07:30:00", "agents": agents, "groups": {}, "config": {},
           "world": {"graph": {"Dorm": [], "Dining Hall": [], "Lounge": []}, "arenas": {"Dorm": ["Room 214"]}},
           "population_lexicon": {"tokens": ["dining", "hall", "lunch", "eating", "lounge", "dorm"]}}
    cfg = {"seed": 3, "llm": {"backend": "mock", "model": "m"}, "controls": {"planted_phrase": PLANTED}}
    trace = [
        _mem("maya", 0, "Maya Chen and Leo Martinez are roommates; they know each other very well and get along well.", "seed"),
        _mem("leo", 0, "Leo Martinez and Maya Chen are roommates; they know each other very well and get along well.", "seed"),
        _mem("dev", 2, "Hana Okafor is eating lunch at the Dining Hall.", "ambient"),
        # a genuine coinage carried across conversations by two exposed agents: emerged
        *_conv("c1", 5, [("maya", ["leo"], "Grab me a glimmer toast.")]),
        *_conv("c2", 10, [("leo", ["dev"], "I want a glimmer toast.")]),
        *_conv("c3", 15, [("dev", ["hana"], "One glimmer toast please.")]),
        # carried once: spreading, not emerged
        *_conv("c4", 8, [("maya", ["leo"], "That snorkel dance was wild.")]),
        *_conv("c5", 12, [("leo", ["dev"], "Dev did the snorkel dance.")]),
        # system wording (ambient sighting / routine) that spreads the same way
        *_conv("c6", 20, [("dev", ["hana"], "I was eating lunch at the dining hall.")]),
        *_conv("c7", 25, [("hana", ["maya"], "Eating lunch at the dining hall again.")]),
        *_conv("c8", 30, [("maya", ["leo"], "Eating lunch at the dining hall is nice.")]),
        # stitched relationship lines (what the mock LLM does): names and frames, never a candidate
        *_conv("c9", 22, [("maya", ["leo"], "Maya Chen and Leo Martinez are roommates.")]),
        *_conv("c10", 26, [("leo", ["dev"], "Maya Chen and Leo Martinez are roommates, they know each other.")]),
        *_conv("c11", 27, [("dev", ["hana"], "Maya Chen and Leo Martinez are roommates.")]),
        # a nickname construction
        *_conv("c12", 40, [("maya", ["leo"], "That was the Leo thing again.")]),
        *_conv("c13", 45, [("leo", ["dev"], "Oh no, the Leo thing.")]),
        *_conv("c14", 50, [("dev", ["hana"], "Honestly the Leo thing is back.")]),
        # recited from memory (leo) vs said fresh (dev, hana)
        _mem("leo", 31, "Leo Martinez remembers that Maya said the purple kettle incident ruined the whole morning at the lounge."),
        *_conv("c15", 35, [("leo", ["dev"], "Maya said the purple kettle incident ruined the whole morning at the lounge.")]),
        *_conv("c16", 38, [("dev", ["hana"], "The purple kettle incident was funny.")]),
        # the planted control, carried twice
        *_conv("c17", 60, [("maya", ["leo"], "What a full pickle.")]),
        *_conv("c18", 65, [("leo", ["dev"], "Total full pickle here.")]),
        *_conv("c19", 70, [("dev", ["hana"], "Another full pickle today.")]),
    ]
    trace.sort(key=lambda r: r["tick"])
    json.dump(man, open(d / "manifest.json", "w"))
    yaml.safe_dump(cfg, open(d / "config.resolved.yaml", "w"))
    (d / "events.jsonl").write_text("")
    (d / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in trace))
    return d


def _find(items, phrase):
    return next((e for e in items if e["phrase"] == phrase or phrase in e["variants"]), None)


def _verdict_file(d: Path, provider: str, model: str, yes: list[str], tick: int):
    rows = []
    for p in yes:
        v = {"judge_id": JG.make_judge_id(provider, model, "v1"), "provider": provider, "model": model,
             "prompt_version": "v1", "is_convention": True, "gloss": f"local name for {p}", "function": "label",
             "meaning_consistency": 0.9, "confidence": 0.9, "rationale": "test", "raw": "{}"}
        assert JG.validate_verdict(v) == []
        rows.append({"expression_id": None, "phrase": p, "variants": [p], "judged_at": "2026-09-19T12:00:00", "verdict": v})
    doc = {"judge": {k: rows[0]["verdict"][k] for k in ("judge_id", "provider", "model", "prompt_version")},
           "run_id": d.name, "generated_at": "2026-09-19T12:00:00", "source": "live", "tick": tick, "top": 30,
           "n_verdicts": len(rows), "n_conventions": len(rows), "verdicts": rows}
    (d / "analysis_judgements").mkdir(exist_ok=True)
    json.dump(doc, open(d / "analysis_judgements" / f"{doc['judge']['judge_id']}.json", "w"))


def test_coinage_spreads_and_only_a_real_judge_makes_a_convention(tmp_path):
    d = _synthetic(tmp_path / "syn")
    s = live_snapshot(d, top=40)
    phrases = [e["phrase"] for e in s["expressions"]]
    glim, snork, nick = _find(s["expressions"], "glimmer toast"), _find(s["expressions"], "snorkel dance"), \
        _find(s["expressions"], "the leo thing")
    assert glim["status"] == "emerged" and glim["tier"] == "spreading" and glim["tier_reasons"] == ["no_real_judge"]
    assert snork["status"] == "spreading" and snork["tier"] == "spreading" and "not_emerged" in snork["tier_reasons"]
    assert nick and nick["tier"] == "spreading" and nick["emergence"]["carried_adopters"] == ["dev", "leo"]
    # system wording spreads too, but is a candidate forever; names and relationship frames never surface
    sysw = [e for e in s["expressions"] if e["status"] == "system_wording"]
    assert any("lunch" in e["phrase"] for e in sysw) and all(e["tier"] == "candidate" for e in sysw)
    lunch = next(e for e in sysw if "lunch" in e["phrase"])
    assert lunch["emergence"]["n_adopters_carried"] >= 2 and "system_wording" in lunch["tier_reasons"]
    assert lunch["world_wording"]["system_match"]["kind"] in ("ambient", "routine", "context", "lexicon")
    for p in phrases:
        assert not {"chen", "martinez", "patel", "okafor", "roommates"} & set(p.split()), p
    assert s["judge"]["status"] == "none" and s["tier_counts"]["convention"] == 0
    # recitation: leo's use copies his own earlier memory verbatim, dev's is fresh
    kettle = next(e for e in s["expressions"] if "kettle incident" in e["phrase"] or "kettle incident" in " ".join(e["variants"]))
    assert kettle["recited_share"] == 0.5

    # a mock judge that says "yes" to everything: still no convention, and the report says no real judge ran
    _verdict_file(d, "mock", "mock", ["glimmer toast", "snorkel dance", "lunch at the dining", "full pickle"], tick=70)
    clear_cache()
    s = live_snapshot(d, top=40)
    glim = _find(s["expressions"], "glimmer toast")
    assert glim["tier"] == "spreading" and glim["tier_reasons"] == ["only_mock_verdict"]
    assert s["judge"]["status"] == "mock_only" and s["n_conventions"] == 0
    rep = meme_report(d)
    assert "no real judge has run" in rep["summary"]["text"] and "0 conventions" in rep["summary"]["text"]
    assert all(c["tier"] != "convention" for c in rep["cards"])

    # a real judge: the emerged coinage becomes a convention; the rest stay where the exposure test puts them
    _verdict_file(d, "claude_cli", "sonnet", ["glimmer toast", "snorkel dance", "lunch at the dining", "full pickle"], tick=70)
    clear_cache()
    s = live_snapshot(d, top=40)
    glim, snork = _find(s["expressions"], "glimmer toast"), _find(s["expressions"], "snorkel dance")
    lunch = _find(s["expressions"], lunch["phrase"])
    pick = next(e for e in s["expressions"] if e["planted"])
    assert glim["tier"] == "convention" and glim["verdict"]["real"] and glim["verdict"]["provider"] == "claude_cli"
    assert snork["tier"] == "spreading" and snork["tier_reasons"] == ["not_emerged"]
    assert lunch["tier"] == "candidate" and lunch["status"] == "system_wording"
    assert pick["tier"] == "convention" and pick["tier_reasons"] == ["planted_control"] and pick["control"]
    assert s["judge"]["status"] == "real" and s["n_conventions"] == 1 and s["tier_counts"]["convention"] == 1
    rep = meme_report(d)
    assert "1 convention" in rep["summary"]["text"] and "no real judge" not in rep["summary"]["text"]
    card = _find(rep["cards"], "glimmer toast")
    assert card["tier"] == "convention" and card["latest_real_verdict"]["real"]
    # a verdict judged at tick 70 is not used by a snapshot of tick 20 (it would leak the future)
    early = _find(live_snapshot(d, tick=20, top=40)["expressions"], "glimmer toast")
    assert early["tier"] == "spreading" and early["tier_reasons"] == ["no_real_judge"]


def test_recited_phrases_are_down_ranked(tmp_path):
    d = _synthetic(tmp_path / "syn")
    rd = RunData(d)
    ex = CandidateExtractor(rd, {"min_speakers": 1, "min_uses": 2})
    stats = ex.ngrams()
    g = ("kettle", "incident")
    st = stats[g]
    assert len(st["uses"]) == 2 and st["recited_ids"] == {"c15.u0"}
    fresh = dict(st, recited_ids=set())
    assert ex.score(g, st)["recited_share"] == 0.5 and ex.score(g, st)["score"] < ex.score(g, fresh)["score"]


def test_pipeline_tiers_and_outcomes(tmp_path):
    """analysis.json / outcomes.json use the same tiers: with only mock verdicts nothing is a convention; with a
    real verdict the emerged coinage is, system wording is not, and the planted control is reported apart."""
    from backend.analysis.pipeline import analyze
    d = _synthetic(tmp_path / "syn")
    _verdict_file(d, "mock", "mock", ["glimmer toast", "lunch at the dining", "full pickle"], tick=70)
    out = analyze(d, llm_backend="mock", probes=False, verbose=False)
    assert out["summary"]["n_conventions"] == 0 and out["summary"]["judge_status"] == "mock_only"
    assert json.load(open(d / "outcomes.json"))["n_conventions"] == 0
    _verdict_file(d, "claude_cli", "sonnet", ["glimmer toast", "lunch at the dining", "full pickle"], tick=70)
    out = analyze(d, llm_backend="mock", probes=False, verbose=False)
    by = {c["canonical_form"]: c for c in out["candidates"]}
    assert by["glimmer toast"]["tier"] == "convention" and by["glimmer toast"]["card"]["is_convention"] is True
    lunch = next(c for f, c in by.items() if "lunch" in f)
    assert lunch["tier"] == "candidate" and lunch["status"] == "system_wording" and lunch["card"]["is_convention"] is False
    assert not out["emergence"][lunch["id"]]["emerged"] and out["emergence"][lunch["id"]]["in_system_text"]
    assert by["the leo thing"]["tier"] == "spreading" and by["the leo thing"]["features"]["nickname"]
    assert by["full pickle"]["planted"] and by["full pickle"]["tier_reasons"] == ["planted_control"]
    assert out["summary"]["n_conventions"] == 1 and out["summary"]["judge_status"] == "real"
    oc = json.load(open(d / "outcomes.json"))
    assert oc["n_conventions"] == 1 and [c["phrase"] for c in oc["conventions"]] == ["glimmer toast"]
    assert oc["planted_control"]["tier"] == "convention" and oc["judge"]["status"] == "real"
    assert oc["n_spread_system"] == 1 and oc["tiers"] == {"candidate": 1, "spreading": 1, "convention": 1}
