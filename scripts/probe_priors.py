"""Prior check for the meme registry (PRE-RUN VALIDITY TOOL; never imported by the simulation).

Why this exists. An injected phrase is only evidence of *transmission* if the model does not already
produce it, and a boundary measured at the end of a run only means something against the boundary the
model started with. Both facts are properties of the model, not of the world, so they have to be measured
outside the simulation, before it runs, and re-measured whenever a phrase is swapped.

It replaces the hand probing done for the retired naming study (Haiku had no preference between "FFC" and
"Hopkins Cafe", did not know "FFC" 6/6, and produced neither unprompted 8/8) with an artifact aimed at the
registry: it reads `memes.registry` from a config and tests every entry in it, so swapping a phrase
re-runs its own check.

Three questions per registry entry, N samples each (default 8):

  (a) arbitrary   does the phrase already mean something to a fresh model?
  (b) coinage     THE GO/NO-GO. The meme's origin situation is described WITHOUT the phrase's distinctive
                  words and the model is asked what people would call it. If the model reaches for the
                  phrase by itself, in-run spread cannot be attributed to transmission rather than
                  independent rediscovery (ontology R2), and THE MEME MUST BE REPLACED.
  (c) boundary    the four gradient situations and the foils, cold: the model's PRIOR applicability
                  boundary. EXT/breadth at the end of a run is read as a shift from this, not from zero,
                  and the foil fit rate here is the pre-run false-positive floor for "broadening".

Non-obvious operational points:

  * The CLI runs from a FRESH EMPTY DIRECTORY with `--setting-sources ""`. Inside the repo it picks up
    project context and answers as a coding assistant; that produced a 50% refusal rate and answers
    mentioning the project by name when this was probed by hand. Refusals are counted as their own
    category -- a refusal is not evidence of absence -- and answers that mention the project or the
    simulation are counted separately as `context_leak`, which invalidates the sample.
  * Question (b)'s prompt is machine-checked against the phrase's distinctive words before it is sent.
    A contaminated origin text would hand the model the answer, so the check hard-blocks (b) for that
    meme instead of reporting a meaningless pass.
  * The answering model defaults to the AGENTS' model (Haiku), because the prior that matters is the
    prior of the model that will run the simulation (ONTOLOGY_V3 §5.2).
  * Sampling is the model's; everything this script controls (which situations, in which order) is seeded
    from the meme id and the sample index, so a rerun sends byte-identical prompts.

Usage:
    .venv/bin/python scripts/probe_priors.py configs/<config with memes.registry>.yaml \
        [--n 8] [--model haiku] [--workers 8] [--only <meme id>] [--boundary batched|per_item] \
        [--out <path.json>] [--keep-prompts] [--dry-run]

Measured on the draft registry of docs/EXPERIMENT_PLAN_MEMES.md: 4 memes at n=8 is 96 calls, 31 s at 8
workers, $0.0778, 0 refusals, 0 errors.

Exit status: 0 every meme passed (or --dry-run, which measures nothing), 2 at least one meme must be
replaced or reviewed, 1 the probe itself failed (no registry, every call failed, a contaminated prompt
with nothing left to measure).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config import load_config  # noqa: E402

VERSION = "probe_priors/1"

# A neutral survey voice. It must not mention campuses, simulations, slang or coining: question (b) asks
# for a nickname on its own and the system prompt may not pre-load that frame for (a) and (c).
SYSTEM = "You answer short questions about English words and phrases. Reply with JSON and nothing else."

# Words that carry no identity of their own, so they are not what makes a phrase guessable.
STOP = {"a", "an", "and", "at", "be", "by", "for", "from", "in", "is", "it", "its", "of", "on", "or",
        "that", "the", "this", "to", "with", "was", "were"}
_WORD = re.compile(r"[a-z]+")

# A refusal is not an answer and not evidence of absence, so it is counted apart from yes/no. The matched
# cue is recorded with the response so the classification can be audited rather than trusted.
_REFUSAL = [
    re.compile(r"\bi (?:can'?t|cannot|won'?t|am unable|'m unable)\b", re.I),
    re.compile(r"\bi'?m not able to\b", re.I),
    re.compile(r"\b(?:i'?m sorry|sorry),? (?:but )?i\b", re.I),
    re.compile(r"\bas an ai\b", re.I),
    re.compile(r"\bi (?:don'?t|do not) (?:have (?:enough|any) (?:context|information)|feel comfortable)\b", re.I),
]
# The answer came from a model that had picked up this repository's context; the sample is void.
_LEAK = re.compile(r"memeworld|this (?:repo|repository|codebase)|your (?:project|repo)|claude code"
                   r"|the simulation|agent[- ]based", re.I)

FIT_SCORE = {"fits": 1.0, "not sure": 0.5, "doesn't fit": 0.0}
GRADIENT_KEYS = ("literal", "near", "mid", "far")


# --- registry ----------------------------------------------------------------------------------------

def read_registry(config_path: str) -> tuple[list[dict], str]:
    """The registry is config, never code: adding, removing or swapping a meme is a config edit (the
    reason this script takes a config path rather than a hard-coded list of phrases).

    The full merge over `configs/default.yaml` is tried first, so `extends:` and the shipped defaults
    apply. A registry is self-contained data, though, and a prior check must stay usable while the rest
    of the config is mid-edit or while a phrase is being swapped in a scratch file, so a failed merge
    falls back to reading this file alone -- loudly, never silently."""
    source = "merged"
    try:
        cfg = load_config(config_path)
    except Exception as exc:  # noqa: BLE001 - the fallback is the point; the reason is printed
        sys.stderr.write(f"probe_priors: {config_path} would not merge over configs/default.yaml "
                         f"({type(exc).__name__}: {exc}); reading the registry from this file alone\n")
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        source = "raw"
    memes = cfg.get("memes") or {}
    registry = memes.get("registry") or []
    if not isinstance(registry, list):
        raise ValueError(f"{config_path}: memes.registry must be a list of meme entries")
    for i, entry in enumerate(registry):
        if not isinstance(entry, dict) or not entry.get("id") or not entry.get("phrase"):
            raise ValueError(f"{config_path}: memes.registry[{i}] needs at least an id and a phrase")
    return registry, source


def distinctive(entry: dict) -> list[str]:
    """The words that make this phrase guessable from its origin, i.e. the words R2 forbids the world to
    emit.

    One definition, shared with the observer's own registry module, because the R2 argument only holds if
    the word list this probe avoids is the word list the world is audited against. A per-entry
    `distinctive:` list may WIDEN it (a compound can be guessable through a word it does not contain), and
    is deliberately unable to narrow it: narrowing here would quietly weaken the coinage test."""
    words = {w for w in _WORD.findall(str(entry["phrase"]).lower()) if w not in STOP and len(w) > 2}
    try:                                      # the shared derivation, when the observer module is present
        from backend.analysis.battery.registry import parse_entry
        words |= {str(w).lower() for w in parse_entry(entry).distinctive_words}
    except Exception:  # noqa: BLE001 - a standalone probe must still run without the analysis layer
        pass
    words |= {str(w).lower() for w in (entry.get("distinctive") or [])}
    return sorted(words)


def situations(entry: dict) -> list[tuple[str, str]]:
    """(key, text) for the gradient and the foils. Foil keys are `foil0`, `foil1`, ... so the scoring can
    separate the false-positive floor from the gradient without another lookup."""
    grad = entry.get("probe_gradient") or {}
    out = [(k, str(grad[k])) for k in GRADIENT_KEYS if grad.get(k)]
    out += [(f"foil{i}", str(t)) for i, t in enumerate(entry.get("foils") or [])]
    return out


def origin_text(entry: dict) -> str:
    """What question (b) describes. `origin_probe` when the registry supplies one written for this
    purpose; otherwise the gradient's `literal` situation, which is by definition the origin and is
    already written without the phrase in it."""
    return str(entry.get("origin_probe") or (entry.get("probe_gradient") or {}).get("literal") or "").strip()


def contamination(entry: dict) -> dict[str, list[str]]:
    """R2, machine-checked: no text this script shows the model may contain the phrase's distinctive
    words. In question (b) that would hand over the answer and turn a coinage test into a copying test."""
    words = distinctive(entry)
    found: dict[str, list[str]] = {}
    texts = [("origin", origin_text(entry))] + [(k, t) for k, t in situations(entry)]
    for field, text in texts:
        tokens = set(_WORD.findall(text.lower()))
        hits = [w for w in words if w in tokens]
        if hits:
            found[field] = hits
    return found


# --- prompts -----------------------------------------------------------------------------------------

def q_arbitrary(entry: dict) -> str:
    return (f'Someone says the phrase "{entry["phrase"]}".\n\n'
            "Do those words already mean something specific to you, beyond what the individual words mean?\n\n"
            'Reply with JSON only: {"known": "yes" or "no" or "unsure", '
            '"meaning": "if yes, the established meaning in under 20 words; otherwise an empty string"}')


def q_coinage(entry: dict) -> str:
    """Deliberately generous to the model: three candidates, most likely first. A test that gives the
    model three chances to produce the phrase is conservative in the direction that protects the study."""
    return ("Here is something that happened at a university this term:\n\n"
            f"{origin_text(entry)}\n\n"
            "Suppose students there started using a short nickname for this kind of thing. "
            "What are the three most likely nicknames? Most likely first.\n\n"
            'Reply with JSON only: {"candidates": ["...", "...", "..."]}')


def q_boundary(entry: dict, order: list[tuple[str, str]]) -> str:
    lines = "\n".join(f"{i + 1}. {text}" for i, (_, text) in enumerate(order))
    keys = " ".join(f'"{i + 1}": "fits" or "doesn\'t fit" or "not sure",' for i in range(len(order)))
    return (f'At one university people use the phrase "{entry["phrase"]}".\n\n'
            "For each situation below, would someone who uses that phrase say it fits what happened?\n\n"
            f"{lines}\n\n"
            "Reply with JSON only: {" + keys.rstrip(",") + "}")


def build_calls(entry: dict, n: int, blocked: bool, boundary_mode: str) -> list[dict]:
    """One call per (question, sample). The boundary question's situation order is re-drawn per sample
    from a seed fixed by the meme id and the sample index, so order cannot be an artifact and a rerun
    still sends byte-identical prompts.

    `boundary_mode` decides whether the situations are shown together (`batched`, the shape of the v3
    applicability probe, §5.3 format A) or one per call (`per_item`). They are not interchangeable, and
    batched is the default because it is the one that measures anything: shown together, the situations
    are a contrast set the model can discriminate within; shown alone, it abstains. Measured on the
    draft registry (haiku, n=8, the same texts, 192 item-responses each): batched gave 107 fits / 72
    doesn't fit / 13 not sure with the foils at 0 fits, per_item gave 8 / 54 / 130 with 8 stray fits on
    the foils. `per_item` is kept because the batched set can also give the category away, and the gap
    between the two is worth being able to measure again after a registry change."""
    mid = entry["id"]
    calls = [{"meme": mid, "kind": "arbitrary", "sample": i, "prompt": q_arbitrary(entry)} for i in range(n)]
    if not blocked:
        calls += [{"meme": mid, "kind": "coinage", "sample": i, "prompt": q_coinage(entry)} for i in range(n)]
    if not situations(entry):          # nothing to ask: an entry with no probe_gradient and no foils
        return calls
    for i in range(n):
        if boundary_mode == "per_item":
            for key, text in situations(entry):
                calls.append({"meme": mid, "kind": "boundary", "sample": i,
                              "prompt": q_boundary(entry, [(key, text)]), "order": [key]})
        else:
            order = situations(entry)
            random.Random(f"{mid}|boundary|{i}").shuffle(order)
            calls.append({"meme": mid, "kind": "boundary", "sample": i, "prompt": q_boundary(entry, order),
                          "order": [k for k, _ in order]})
    return calls


# --- the CLI -----------------------------------------------------------------------------------------

def cli_command(model: str) -> list[str]:
    """The project's own invocation (backend/llm/client.py::ClaudeCLIBackend), minus the simulation.
    `--setting-sources ""` plus an empty working directory is what keeps the answering model fresh."""
    return ["claude", "-p", "--model", model, "--output-format", "json",
            "--tools", "", "--no-session-persistence", "--setting-sources", "",
            "--disable-slash-commands", "--strict-mcp-config",
            # extended thinking is on by default in the CLI; it multiplies latency ~3-10x
            "--settings", '{"alwaysThinkingEnabled": false}',
            "--system-prompt", SYSTEM]


def ask(prompt: str, model: str, cwd: str, timeout: int) -> dict:
    cmd = cli_command(model)
    last = None
    for attempt in range(3):
        try:
            p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                               timeout=timeout, cwd=cwd)
            data = json.loads(p.stdout)
            if data.get("is_error"):
                raise RuntimeError(str(data.get("result"))[:300])
            return {"text": data["result"], "cost_usd": data.get("total_cost_usd"),
                    "duration_ms": data.get("duration_ms")}
        except Exception as exc:  # noqa: BLE001 - retried, then reported as an error sample
            last = exc
    return {"text": None, "error": f"{type(last).__name__}: {last}"[:300]}


# --- scoring -----------------------------------------------------------------------------------------

def extract_json(text: str):
    """The models fence or preface their JSON often enough that a strict parse would report refusals that
    are really formatting. Take the outermost balanced object and parse that."""
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except ValueError:
                    return None
    return None


def classify(result: dict) -> dict:
    """Every sample lands in exactly one of: error, refusal, context_leak, unparsed, answered."""
    text = result.get("text")
    if text is None:
        return {"status": "error", "detail": result.get("error")}
    leak = _LEAK.search(text)
    if leak:
        return {"status": "context_leak", "detail": leak.group(0)}
    parsed = extract_json(text)
    if parsed is not None:
        return {"status": "answered", "parsed": parsed}
    for rx in _REFUSAL:
        m = rx.search(text)
        if m:
            return {"status": "refusal", "detail": m.group(0)}
    return {"status": "unparsed", "detail": text[:200]}


def norm(text: str) -> str:
    """Casing, hyphens and plurals must not hide a hit: "the Blue Trays" is the phrase."""
    t = re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in t.split())


def hit(candidate: str, phrase: str, words: list[str]) -> dict:
    c = norm(candidate)
    return {"exact": norm(phrase) in c,
            "all_words": bool(words) and all(norm(w) in c for w in words)}


def score_arbitrary(samples: list[dict]) -> dict:
    counts = {"yes": 0, "no": 0, "unsure": 0}
    for s in samples:
        if s["status"] != "answered":
            continue
        known = str((s.get("parsed") or {}).get("known", "")).strip().lower()
        counts[known if known in counts else "unsure"] += 1
    answered = sum(counts.values())
    return {"counts": counts, "answered": answered,
            "known_rate": round(counts["yes"] / answered, 3) if answered else None,
            "meanings": [str((s.get("parsed") or {}).get("meaning", "")).strip()
                         for s in samples if s["status"] == "answered"
                         and str((s.get("parsed") or {}).get("known", "")).lower() == "yes"]}


def score_coinage(samples: list[dict], phrase: str, words: list[str]) -> dict:
    any_hits, exact_hits, first_hits, answered, all_candidates = 0, 0, 0, 0, []
    for s in samples:
        if s["status"] != "answered":
            continue
        cands = [str(c) for c in ((s.get("parsed") or {}).get("candidates") or [])]
        all_candidates.append(cands)
        answered += 1
        marks = [hit(c, phrase, words) for c in cands]
        if any(m["exact"] or m["all_words"] for m in marks):
            any_hits += 1
        if any(m["exact"] for m in marks):
            exact_hits += 1
        if marks and (marks[0]["exact"] or marks[0]["all_words"]):
            first_hits += 1
    # hits_exact and hits_any are reported apart because `all_words` fires on the distinctive words alone
    # ("the blue plates" counts for "blue-tray"): deliberately generous, so a near miss is still visible,
    # and the candidates are kept verbatim so a generous hit can be inspected rather than trusted.
    return {"answered": answered, "hits_any": any_hits, "hits_exact": exact_hits, "hits_first": first_hits,
            "hit_rate": round(any_hits / answered, 3) if answered else None,
            "candidates": all_candidates}


def score_boundary(samples: list[dict], calls: list[dict], keys: list[str], mode: str) -> dict:
    """"not sure" is counted as its own answer as well as scored at 0.5 (ONTOLOGY_V3 §5.5 reports HEDGE
    apart from the three-way share). The two presentation formats differ almost entirely in hedging --
    192 item-responses each, batched 107/72/13 against per_item 8/54/130 -- so a mean fit alone would
    read an abstention as half a fit and call the two formats similar."""
    per: dict[str, list[float]] = {k: [] for k in keys}
    hedge: dict[str, int] = {k: 0 for k in keys}
    yes: dict[str, int] = {k: 0 for k in keys}
    answered = 0
    for s, call in zip(samples, calls):
        if s["status"] != "answered":
            continue
        answered += 1
        parsed = s.get("parsed") or {}
        for pos, key in enumerate(call["order"], start=1):
            raw = str(parsed.get(str(pos), parsed.get(pos, ""))).strip().lower().replace("’", "'")
            raw = {"does not fit": "doesn't fit", "no": "doesn't fit", "yes": "fits",
                   "unsure": "not sure"}.get(raw, raw)
            if raw in FIT_SCORE:
                per[key].append(FIT_SCORE[raw])
                hedge[key] += raw == "not sure"
                yes[key] += raw == "fits"
    mean = {k: (round(sum(v) / len(v), 3) if v else None) for k, v in per.items()}
    n_all = sum(len(v) for v in per.values())
    grad = [k for k in keys if not k.startswith("foil")]
    foils = [k for k in keys if k.startswith("foil")]
    foil_means = [mean[k] for k in foils if mean[k] is not None]
    foil_yes = [yes[k] / len(per[k]) for k in foils if per[k]]
    return {"answered": answered, "mode": mode, "mean_fit": mean, "hedge": hedge,
            "fit_share": {k: (round(yes[k] / len(per[k]), 3) if per[k] else None) for k in keys},
            # the false-positive floor the run is read against: how often a fresh model says a foil FITS,
            # counting only real acceptances, never hedges (ONTOLOGY_V3 §5.11.3).
            "foil_accept_rate": round(sum(foil_yes) / len(foil_yes), 3) if foil_yes else None,
            "hedge_rate": round(sum(hedge.values()) / n_all, 3) if n_all else None,
            "n_responses": {k: len(v) for k, v in per.items()},
            # breadth counts an item in only on a real majority of "fits": at the 0.5 threshold a column
            # of pure "not sure" would otherwise be counted as inside the boundary.
            "prior_breadth": sum(1 for k in grad if (mean[k] or 0) >= 0.5
                                 and hedge[k] < len(per[k]) / 2),
            "gradient_items": len(grad),
            "foil_fit_rate": round(sum(foil_means) / len(foil_means), 3) if foil_means else None}


def verdict(meme: dict, n: int, coinage_max: int) -> dict:
    """Go / no-go. REPLACE is mechanical (the model produced the phrase, or the prompt was contaminated);
    REVIEW means the meme is measurable only with a caveat the plan has to state."""
    reasons, level = [], "PASS"

    def escalate(new, why):
        nonlocal level
        order = {"PASS": 0, "REVIEW": 1, "REPLACE": 2}
        reasons.append(why)
        if order[new] > order[level]:
            level = new

    if meme.get("blocked"):
        escalate("REPLACE", f"origin text contains the phrase's own words {meme['contamination']['origin']}"
                            " (R2): question (b) cannot be asked")
    coin = meme.get("coinage") or {}
    if not meme.get("blocked") and coin.get("answered", 0) < (n + 1) // 2:
        escalate("REVIEW", f"coinage answered on only {coin.get('answered', 0)}/{n} samples; a refusal is "
                           "not evidence of absence")
    if (coin.get("hits_any") or 0) > coinage_max:
        escalate("REPLACE", f"the model coined the phrase itself on {coin['hits_any']}/{coin['answered']} "
                            "samples: spread could not be attributed to transmission")
    ar = meme.get("arbitrary") or {}
    if (ar.get("known_rate") or 0) >= 0.5:
        escalate("REVIEW", f"the phrase already means something to the model on "
                           f"{ar['counts']['yes']}/{ar['answered']} samples")
    bo = meme.get("boundary") or {}
    if (bo.get("hedge_rate") or 0) >= 0.5:
        escalate("REVIEW", f"the model answered 'not sure' on {bo['hedge_rate']:.0%} of the situations: "
                           "this format does not elicit a boundary to compare the run against")
    elif (bo.get("foil_accept_rate") or 0) >= 0.25:
        escalate("REVIEW", f"a fresh model already accepts the foils {bo['foil_accept_rate']:.0%} of the "
                           "time: that is the false-positive floor a 'broadening' result has to clear")
    if bo.get("gradient_items") and bo.get("prior_breadth") == bo["gradient_items"]:
        escalate("REVIEW", "the phrase already fits every gradient situation before the run: probe "
                           "breadth has no headroom, so broadening has to be read from in-run use")
    for field, words in (meme.get("contamination") or {}).items():
        if field != "origin":
            escalate("REVIEW", f"probe text '{field}' contains {words} (R2)")
    return {"level": level, "reasons": reasons}


# --- main --------------------------------------------------------------------------------------------

def probe(registry: list[dict], args) -> dict:
    work = tempfile.mkdtemp(prefix="probe_priors_")  # empty and outside the repo: no project context
    calls, plan = [], {}
    for entry in registry:
        cont = contamination(entry)
        # No origin text is as fatal to question (b) as a contaminated one: there is nothing to describe,
        # so the meme's go/no-go cannot be answered and the entry is reported as blocked, never as a pass.
        if not origin_text(entry):
            cont = {**cont, "origin": ["<missing: no origin_probe and no probe_gradient.literal>"]}
        blocked = "origin" in cont
        entry_calls = build_calls(entry, args.n, blocked, args.boundary)
        for c in entry_calls:
            c["i"] = len(calls)
            calls.append(c)
        plan[entry["id"]] = {"entry": entry, "contamination": cont, "blocked": blocked, "calls": entry_calls}

    if args.dry_run:
        results = [{"text": None, "error": "dry-run"} for _ in calls]
    else:
        results = [None] * len(calls)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(ask, c["prompt"], args.model, work, args.timeout): c["i"] for c in calls}
            for done, fut in enumerate(as_completed(futures), start=1):
                results[futures[fut]] = fut.result()
                print(f"\r  {done}/{len(calls)} calls", end="", file=sys.stderr, flush=True)
        print("", file=sys.stderr)

    memes, cost, errors = {}, 0.0, 0
    for mid, p in plan.items():
        entry = p["entry"]
        samples = {"arbitrary": [], "coinage": [], "boundary": []}
        boundary_calls = []
        for c in p["calls"]:
            r = results[c["i"]]
            s = dict(classify(r))
            s["prompt"] = c["prompt"] if args.keep_prompts else None
            s["response"] = r.get("text")
            samples[c["kind"]].append(s)
            if c["kind"] == "boundary":
                boundary_calls.append(c)
            cost += float(r.get("cost_usd") or 0.0)
            if r.get("text") is None and not args.dry_run:
                errors += 1
        keys = [k for k, _ in situations(entry)]
        out = {"phrase": entry["phrase"], "grounding": entry.get("grounding"),
               "breadth": entry.get("breadth"), "distinctive": distinctive(entry),
               "contamination": p["contamination"], "blocked": p["blocked"],
               "origin_probe": origin_text(entry),
               "arbitrary": score_arbitrary(samples["arbitrary"]),
               "coinage": None if p["blocked"] else score_coinage(samples["coinage"], entry["phrase"],
                                                                  distinctive(entry)),
               "boundary": score_boundary(samples["boundary"], boundary_calls, keys, args.boundary),
               "status_counts": {k: _status_counts(v) for k, v in samples.items()},
               "samples": {k: v for k, v in samples.items()} if args.keep_responses else None}
        out["verdict"] = verdict(out, args.n, args.coinage_max)
        memes[mid] = out

    return {"version": VERSION, "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "config": str(args.config), "registry_sha256": _sha(registry), "model": args.model,
            "n": args.n, "coinage_max": args.coinage_max, "boundary_mode": args.boundary,
            "dry_run": args.dry_run,
            "cli": {"command": cli_command(args.model), "cwd": work, "system": SYSTEM},
            "calls": len(calls), "errors": errors, "cost_usd": round(cost, 4),
            "memes": memes,
            "summary": {"replace": sorted(m for m, v in memes.items() if v["verdict"]["level"] == "REPLACE"),
                        "review": sorted(m for m, v in memes.items() if v["verdict"]["level"] == "REVIEW"),
                        "pass": sorted(m for m, v in memes.items() if v["verdict"]["level"] == "PASS")}}


def _status_counts(samples: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for s in samples:
        counts[s["status"]] = counts.get(s["status"], 0) + 1
    return counts


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def render(report: dict) -> str:
    rows = ["", f"prior check  model={report['model']}  n={report['n']}  calls={report['calls']}  "
                f"boundary={report['boundary_mode']}  errors={report['errors']}  "
                f"cost=${report['cost_usd']}", ""]
    head = (f"{'meme':<16}{'phrase':<18}{'cell':<20}{'coined':<9}{'known':<8}{'breadth':<9}"
            f"{'foil+':<7}{'hedge':<7}verdict")
    rows += [head, "-" * len(head)]
    for mid, m in report["memes"].items():
        co = m["coinage"] or {}
        ar = m["arbitrary"]
        bo = m["boundary"]
        cell = f"{m.get('grounding') or '?'}/{m.get('breadth') or '?'}"
        coined = "blocked" if m["blocked"] else f"{co.get('hits_any', 0)}/{co.get('answered', 0)}"
        known = f"{ar['counts']['yes']}/{ar['answered']}"
        breadth = f"{bo['prior_breadth']}/{bo['gradient_items']}"
        foils = "-" if bo["foil_accept_rate"] is None else f"{bo['foil_accept_rate']:.2f}"
        hedged = "-" if bo["hedge_rate"] is None else f"{bo['hedge_rate']:.2f}"
        rows.append(f"{mid:<16}{m['phrase'][:17]:<18}{cell:<20}{coined:<9}{known:<8}{breadth:<9}"
                    f"{foils:<7}{hedged:<7}{m['verdict']['level']}")
    rows.append("")
    for mid, m in report["memes"].items():
        for why in m["verdict"]["reasons"]:
            rows.append(f"  {mid}: {why}")
    # What the model reached for instead is the readable half of the coinage result: it shows the check
    # ran on a model that understood the situation, rather than one that said nothing useful.
    rows.append("")
    for mid, m in report["memes"].items():
        firsts = [c[0] for c in ((m["coinage"] or {}).get("candidates") or []) if c]
        if firsts:
            top = sorted({f: firsts.count(f) for f in firsts}.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
            rows.append(f"  {mid} coined instead: " + ", ".join(f'"{f}" x{n}' for f, n in top))
    return "\n".join(rows) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prior check for every meme in a config's memes.registry.")
    ap.add_argument("config", type=Path, help="config carrying memes.registry (merged over default.yaml)")
    ap.add_argument("--n", type=int, default=8, help="samples per question (default 8)")
    ap.add_argument("--model", default="haiku", help="answering model; default haiku, the agents' model")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--only", action="append", help="probe only these meme ids (repeatable)")
    ap.add_argument("--boundary", choices=("batched", "per_item"), default="batched",
                    help="show the gradient and foils together (batched, the shape of the v3 "
                         "applicability probe) or one situation per call (per_item, |situations| times "
                         "the calls, but the item set cannot give the category away)")
    ap.add_argument("--coinage-max", type=int, default=0,
                    help="samples on which the model may coin the phrase itself before the meme is "
                         "rejected (default 0)")
    ap.add_argument("--out", type=Path, help="write the JSON report here "
                                             "(default runs/dev_priors/<config>_<model>_n<N>.json)")
    ap.add_argument("--keep-responses", action="store_true", default=True)
    ap.add_argument("--no-keep-responses", dest="keep_responses", action="store_false")
    ap.add_argument("--keep-prompts", action="store_true", help="store every rendered prompt in the report")
    ap.add_argument("--dry-run", action="store_true", help="build and store the prompts, call nothing")
    args = ap.parse_args(argv)

    try:
        registry, source = read_registry(str(args.config))
    except (OSError, ValueError) as exc:
        return _fail(f"cannot read a registry from {args.config}: {exc}")
    if args.only:
        registry = [e for e in registry if e["id"] in set(args.only)]
    if not registry:
        return _fail(f"{args.config} has no memes.registry entries to probe "
                     "(the registry is config: add the entries there, not in code)")

    report = probe(registry, args)
    report["registry_source"] = source
    out = args.out or (ROOT / "runs" / "dev_priors" /
                       f"{args.config.stem}_{args.model}_n{args.n}_{args.boundary}"
                       f"{'_dry' if args.dry_run else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    sys.stdout.write(render(report))
    sys.stdout.write(f"report: {out}\n")
    if args.dry_run:
        return 0
    if report["errors"] == report["calls"]:
        # Every call failed: that is a broken CLI, not a finding about any meme, and must not be read as
        # "no meme produced the phrase".
        return _fail("every call failed; check that the `claude` CLI is installed and authenticated")
    return 2 if (report["summary"]["replace"] or report["summary"]["review"]) else 0


def _fail(msg: str) -> int:
    sys.stderr.write(f"probe_priors: {msg}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
