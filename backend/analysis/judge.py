"""Pluggable LLM-as-judge for meme candidates (OBSERVER ONLY; never imported by simulation code).

A judge reads an expression and some of its uses and returns a Verdict:

    {judge_id, provider, model, prompt_version, is_convention: bool, gloss: str,
     function: description|warning|joke|instruction|label|greeting|other,
     meaning_consistency: 0..1, confidence: 0..1, rationale: str, raw: str}

Implementations:
- LLMJudge: any `backend.llm.client` backend (claude_cli, anthropic, mock, auto) with a versioned prompt
  (`backend/prompts/judge_convention_<version>.txt`; "v0" is the legacy pipeline classifier prompt,
  candidates.CLASSIFY, kept so old analyses can be reproduced). Its calls are cached in the run's own
  `judge_llm_calls.jsonl` (hits are served, not re-appended), never in a simulation cache.
- MockJudge: deterministic (a hash of the phrase), for tests and offline UI work.
- a registry: `get_judge(spec)`; `register_judge(provider, factory)` lets the team plug in another judge.

Config (read with defaults; the key is NOT in configs/default.yaml yet):
    analysis:
      judge: {provider: claude_cli, model: sonnet, prompt_version: v1}   # provider: claude_cli | anthropic | mock

`judge_run(run_dir, spec, top)` writes `analysis_judgements/<judge_id>.json` and nothing else (plus
`judge_llm_calls.jsonl` for LLM judges). The analysis pipeline routes its classifier through the default
judge (`candidates.llm_classify(..., judge=...)`), so analysis.json keeps `llm` and adds `judgements`.

Provenance: a verdict from the mock judge or the mock backend (provider mock/replay, or model "mock") is a
PLACEHOLDER (`is_real_verdict` is False). It is recorded, but it never makes an expression a convention
(tiers.py): `llm_block` sets c["llm"]["is_convention"] = None for it (the mock answer is kept under
c["llm"]["placeholder"]), judge_run files mark it `placeholder: true` with n_conventions = 0, and reports say
"no real judge has run". Old analysis.json files carry only the legacy `llm` block; `analysis_verdicts` turns
it into a prompt-v0 verdict whose provenance comes from the run's analysis_llm_calls.jsonl / config.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Callable

from backend.llm.client import AnthropicBackend, LLMClient, llm_purpose, llm_scope, make_backend

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
FUNCTIONS = ("description", "warning", "joke", "instruction", "label", "greeting", "other")
VERDICT_KEYS = ("judge_id", "provider", "model", "prompt_version", "is_convention", "gloss", "function",
                "meaning_consistency", "confidence", "rationale", "raw")
DEFAULT_SPEC = {"provider": "claude_cli", "model": "sonnet", "prompt_version": "v1"}
CONFIG_KEY = "analysis.judge"
OUT_DIR = "analysis_judgements"
CACHE_FILE = "judge_llm_calls.jsonl"
MAX_CONTEXTS = 8
# provider "anthropic" takes API model ids; the short aliases the CLI accepts are mapped to current ids
ANTHROPIC_ALIASES = {"sonnet": "claude-sonnet-5", "haiku": "claude-haiku-4-5", "opus": "claude-opus-5"}
_MARKER = "<commentblockmarker>###</commentblockmarker>"
MOCK_PROVIDERS = ("mock", "replay")
NO_REAL_JUDGE = "no real judge has run"


# ------------------------------------------------------------------------------------------------ verdicts
def _unit(x, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if v != v:                       # NaN
        return default
    return round(min(1.0, max(0.0, v)), 3)


def _bool(x) -> bool:
    if isinstance(x, str):
        return x.strip().lower() in ("true", "yes", "y", "1")
    return bool(x)


def validate_verdict(v) -> list[str]:
    """Schema problems of a verdict (an empty list means it is valid)."""
    if not isinstance(v, dict):
        return ["verdict is not an object"]
    errs = [f"missing {k}" for k in VERDICT_KEYS if k not in v]
    errs += [f"unexpected {k}" for k in v if k not in VERDICT_KEYS]
    if errs:
        return errs
    for k in ("judge_id", "provider", "model", "prompt_version", "gloss", "rationale", "raw"):
        if not isinstance(v[k], str):
            errs.append(f"{k} is not a string")
    if not isinstance(v["is_convention"], bool):
        errs.append("is_convention is not a bool")
    if v["function"] not in FUNCTIONS:
        errs.append(f"function {v['function']!r} not in {FUNCTIONS}")
    for k in ("meaning_consistency", "confidence"):
        if isinstance(v[k], bool) or not isinstance(v[k], (int, float)) or not 0.0 <= float(v[k]) <= 1.0:
            errs.append(f"{k} not a number in [0, 1]")
    return errs


def is_real_verdict(v) -> bool:
    """True for a verdict (or judge description) from a real LLM judge; False for the mock judge, the mock
    backend or a replay stand-in (placeholders)."""
    if not isinstance(v, dict):
        return False
    prov = str(v.get("provider") or "").strip().lower()
    model = str(v.get("model") or "").strip().lower()
    return bool(prov) and prov not in MOCK_PROVIDERS and not model.startswith("mock")


def llm_block(v: dict) -> dict:
    """analysis.json's c["llm"] for a verdict: the legacy {is_convention, gloss, confidence, raw} plus the judge's
    provenance. A placeholder (mock) verdict leaves is_convention / gloss None and confidence 0, and keeps the
    mock answer under "placeholder"."""
    real = is_real_verdict(v)
    out = {"is_convention": bool(v.get("is_convention")) if real else None,
           "gloss": (v.get("gloss") or None) if real else None,
           "confidence": float(v.get("confidence") or 0) if real else 0.0,
           "raw": str(v.get("raw") or "")[:500],
           "judge_id": v.get("judge_id"), "provider": v.get("provider"), "model": v.get("model"),
           "prompt_version": v.get("prompt_version"), "real_judge": real}
    if not real:
        out["placeholder"] = {"is_convention": bool(v.get("is_convention")), "gloss": v.get("gloss") or None,
                              "confidence": v.get("confidence"), "note": "mock judge: placeholder, never a convention"}
    return out


def _as_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) else None


# ------------------------------------------------------------------------------------------------ judges
class Judge:
    """Base class (protocol): `id`, `provider`, `model`, `prompt_version`, `judge(...) -> Verdict`."""
    provider = "base"

    def __init__(self, model: str, prompt_version: str = "v1", judge_id: str | None = None):
        self.model = str(model)
        self.prompt_version = str(prompt_version)
        self._id = judge_id

    @property
    def id(self) -> str:
        return self._id or make_judge_id(self.provider, self.model, self.prompt_version)

    def describe(self) -> dict:
        return {"judge_id": self.id, "provider": self.provider, "model": self.model,
                "prompt_version": self.prompt_version}

    def judge(self, expression: dict, contexts: list[str], extra: dict | None = None) -> dict:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def verdict(self, d: dict | None, raw: str = "") -> dict:
        """A schema-valid Verdict from a (possibly partial or malformed) parsed answer."""
        parsed = d is not None
        d = d or {}
        fn = str(d.get("function") or "other").strip().lower()
        return {**self.describe(), "is_convention": _bool(d.get("is_convention", False)),
                "gloss": str(d.get("gloss") or ""), "function": fn if fn in FUNCTIONS else "other",
                "meaning_consistency": _unit(d.get("meaning_consistency")),
                "confidence": _unit(d.get("confidence")),
                "rationale": str(d.get("rationale") or ("" if parsed else "unparseable judge response")),
                "raw": str(raw or "")[:2000]}


def make_judge_id(provider: str, model: str, prompt_version: str) -> str:
    """File-system-safe id, e.g. claude_cli-sonnet-v1 (used as analysis_judgements/<id>.json)."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{provider}-{model}-{prompt_version}")


class MockJudge(Judge):
    """Deterministic stand-in: the verdict is a pure function of the phrase (not of the contexts, so a
    phrase keeps its verdict as a run grows). World or system wording (extra["world_wording"] /
    extra["system_wording"]) is never a convention. Its verdicts are placeholders (is_real_verdict is False)."""
    provider = "mock"

    def __init__(self, model: str = "mock", prompt_version: str = "v1", judge_id: str | None = None):
        super().__init__(model, prompt_version, judge_id)

    def judge(self, expression: dict, contexts: list[str], extra: dict | None = None) -> dict:
        phrase = str(expression.get("phrase") or expression.get("canonical_form") or "")
        h = int(hashlib.md5(f"{self.model}|{phrase}".encode()).hexdigest()[:8], 16)
        conv = h % 3 == 0 and not (extra or {}).get("world_wording") and not (extra or {}).get("system_wording")
        d = {"is_convention": conv, "function": FUNCTIONS[h % len(FUNCTIONS)],
             "gloss": f"(mock placeholder) a locally used expression: {phrase}" if conv else f"(mock placeholder) ordinary wording: {phrase}",
             "meaning_consistency": ((h >> 4) % 101) / 100, "confidence": 0.5 + ((h >> 12) % 51) / 100,
             "rationale": f"mock judge; {len(contexts)} uses shown"}
        return self.verdict(d, json.dumps(d))


class _AnthropicNoSampling(AnthropicBackend):
    """The agents' AnthropicBackend always sends `temperature`, which current Claude models (Sonnet 5,
    Opus 5) reject; the judge omits it and leaves room for adaptive thinking in max_tokens."""

    def generate(self, prompt, system, max_tokens, temperature):
        kw = dict(model=self.model, max_tokens=max(int(max_tokens), 4096),
                  messages=[{"role": "user", "content": prompt}])
        if system:
            kw["system"] = system
        msg = self.client.messages.create(**kw)
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def render_prompt(version: str, expression: dict, contexts: list[str], extra: dict | None = None) -> tuple[str, int]:
    """(prompt, max_tokens) for a prompt version. v0 = the legacy classifier (candidates.CLASSIFY), byte-for-byte."""
    uses = "\n".join(f"{i + 1}. " + str(c).replace("\n", " / ") for i, c in enumerate(list(contexts)[:MAX_CONTEXTS]))
    phrase = str(expression.get("phrase") or expression.get("canonical_form") or "")
    if version == "v0":
        from backend.analysis.candidates import CLASSIFY
        return CLASSIFY.format(expr=phrase, uses=uses), 200
    path = PROMPTS / f"judge_convention_{version}.txt"
    if not path.exists():
        raise ValueError(f"no judge prompt for version {version!r} ({path})")
    body = path.read_text().split(_MARKER, 1)[-1].lstrip("\n")
    others = [v for v in expression.get("variants") or [] if v != phrase]
    n_uses = expression.get("n_uses", expression.get("uses", expression.get("usage_count")))
    n_spk = expression.get("n_speakers")
    if n_spk is None and isinstance(expression.get("speakers"), (list, tuple, set)):
        n_spk = len(expression["speakers"])
    counts = f"It was used {n_uses if n_uses is not None else len(contexts)} times by {n_spk if n_spk is not None else 'several'} different speakers."
    inputs = [phrase, ", ".join(f'"{v}"' for v in others) or "none", uses or "(no uses recorded)", counts,
              " | ".join(FUNCTIONS)]
    for i, x in enumerate(inputs):
        body = body.replace(f"!<INPUT {i}>!", str(x))
    return body, 400


class LLMJudge(Judge):
    """A judge backed by any backend.llm.client backend.

    spec: {provider: claude_cli|anthropic|auto|mock, model, prompt_version, [id], [mock_seed]}. With `run_dir`
    the judge owns an LLMClient on <run_dir>/judge_llm_calls.jsonl (re-running is idempotent: cached answers
    are served and not re-appended). `client=` injects a client instead (used by `LLMJudge.legacy`)."""

    def __init__(self, spec: dict | None = None, run_dir: Path | None = None, client: LLMClient | None = None,
                 cache_path: Path | None = None):
        spec = normalize_spec(spec)
        self.provider = spec["provider"]
        model = spec["model"]
        if self.provider == "anthropic":
            model = ANTHROPIC_ALIASES.get(str(model), model)
        super().__init__(model, spec["prompt_version"], spec.get("id"))
        self.spec = spec
        self._client = client
        self._own = client is None
        self.cache_path = Path(cache_path) if cache_path else (Path(run_dir) / CACHE_FILE if run_dir else None)
        if self._own and self.cache_path is None:
            raise ValueError("LLMJudge needs run_dir (or cache_path) for its own judge_llm_calls.jsonl, or client=")

    @classmethod
    def legacy(cls, client: LLMClient) -> "LLMJudge":
        """The pre-judge pipeline classifier as a judge: prompt v0 on the caller's client and scope."""
        b = client.backend
        return cls({"provider": getattr(b, "name", "llm"), "model": getattr(b, "model", None) or getattr(b, "name", "llm"),
                    "prompt_version": "v0"}, client=client)

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            if self.provider == "anthropic":
                backend = _AnthropicNoSampling(self.model)
            else:
                backend = make_backend({"backend": self.provider, "model": self.model,
                                        "mock_seed": self.spec.get("mock_seed", 0)})
            p = self.cache_path
            p.parent.mkdir(parents=True, exist_ok=True)
            self._client = LLMClient(backend, p, mode="record", replay_path=p if p.exists() else None,
                                     record_cached=False)
        return self._client

    def judge(self, expression: dict, contexts: list[str], extra: dict | None = None) -> dict:
        prompt, max_tokens = render_prompt(self.prompt_version, expression, contexts, extra)
        with llm_purpose("analysis_classifier"):   # the mock backend answers this purpose with a verdict-like JSON
            if self._own:
                with llm_scope(f"judge:{self.id}"):
                    raw = self.client.complete(prompt, max_tokens=max_tokens, temperature=0)
            else:
                raw = self.client.complete(prompt, max_tokens=max_tokens, temperature=0)
        return self.verdict(_as_json(raw), raw)

    def close(self) -> None:
        if self._own and self._client is not None:
            self._client.close()
            self._client = None


# ------------------------------------------------------------------------------------------------ registry
def _llm_factory(spec, run_dir):
    return LLMJudge(spec, run_dir=run_dir)


def _mock_factory(spec, run_dir):
    return MockJudge(spec.get("model") or "mock", spec["prompt_version"], spec.get("id"))


PROVIDERS: dict[str, Callable] = {"claude_cli": _llm_factory, "anthropic": _llm_factory, "auto": _llm_factory,
                                  "mock": _mock_factory, "replay": _mock_factory}


def register_judge(provider: str, factory: Callable[[dict, Path | None], Judge]) -> None:
    """Plug in another judge: factory(normalized_spec, run_dir) -> Judge."""
    PROVIDERS[str(provider)] = factory


def normalize_spec(spec=None) -> dict:
    """A judge spec from: None (defaults), "provider[:model[:prompt_version]]", a spec dict
    ({provider|backend, model, prompt_version, [id]}), or a whole run config (reads analysis.judge)."""
    if spec is None:
        spec = {}
    elif isinstance(spec, str):
        parts = spec.split(":")
        spec = dict(zip(("provider", "model", "prompt_version"), parts))
    elif isinstance(spec, dict) and ("analysis" in spec or "llm" in spec):
        seed = (spec.get("llm") or {}).get("mock_seed")
        spec = dict(((spec.get("analysis") or {}).get("judge")) or {})
        if seed is not None:
            spec.setdefault("mock_seed", seed)
    spec = dict(spec)
    if "provider" not in spec and spec.get("backend"):
        spec["provider"] = spec.pop("backend")
    spec.pop("backend", None)
    out = {**DEFAULT_SPEC, **{k: v for k, v in spec.items() if v is not None}}
    if out["provider"] in ("mock", "replay") and "model" not in spec:
        out["model"] = "mock"
    out["provider"], out["model"], out["prompt_version"] = str(out["provider"]), str(out["model"]), str(out["prompt_version"])
    return out


def get_judge(cfg_or_spec=None, run_dir: Path | None = None) -> Judge:
    """The judge for a spec (see normalize_spec). A Judge instance is returned unchanged."""
    if isinstance(cfg_or_spec, Judge):
        return cfg_or_spec
    spec = normalize_spec(cfg_or_spec)
    factory = PROVIDERS.get(spec["provider"])
    if factory is None:
        raise ValueError(f"unknown judge provider {spec['provider']!r}; known: {sorted(PROVIDERS)}")
    return factory(spec, Path(run_dir) if run_dir else None)


def pipeline_spec(cfg: dict, observer: dict, explicit_override: bool = False) -> dict:
    """The judge the analysis pipeline uses: analysis.judge when configured, else the observer's
    backend/model with prompt v1. An explicit analyze(llm_backend=/llm_model=) override also overrides the
    judge's provider/model, so a mock re-analysis never calls a paid model."""
    js = dict(((cfg.get("analysis") or {}).get("judge")) or {})
    obs = {"provider": observer.get("backend"), "model": observer.get("model")}
    obs = {k: v for k, v in obs.items() if v}
    if js and not explicit_override:
        spec = js
    else:
        spec = {**js, **obs}
        if "prompt_version" not in spec:
            spec["prompt_version"] = DEFAULT_SPEC["prompt_version"]
    seed = (cfg.get("llm") or {}).get("mock_seed")
    if seed is not None:
        spec.setdefault("mock_seed", seed)
    return normalize_spec(spec)


# ------------------------------------------------------------------------------------------------ runs
def judge_input(c: dict) -> tuple[dict, list[str]]:
    """(expression, contexts) for an analysis.json candidate or a live expression record."""
    phrase = c.get("canonical_form") or c.get("phrase") or ""
    speakers = c.get("speakers")
    n_spk = c.get("n_speakers", len(speakers) if isinstance(speakers, (list, tuple, set)) else None)
    uses = c.get("usage_count")
    if uses is None:
        uses = c.get("uses") if isinstance(c.get("uses"), int) else None
    expr = {"id": c.get("id"), "phrase": phrase, "variants": list(c.get("variants") or [phrase]),
            "n_uses": uses, "n_speakers": n_spk}
    usages = c.get("usages") or c.get("_usages") or []
    ctx = [u.get("context") for u in usages if u.get("context")] or list(c.get("contexts") or [])
    return expr, ctx[:MAX_CONTEXTS]


def _expressions_for(run_dir: Path, top: int) -> tuple[str, int | None, list[dict]]:
    """(source, tick, candidates): analysis.json's candidates when present, else the live snapshot's."""
    p = run_dir / "analysis.json"
    if p.exists():
        try:
            a = json.load(open(p))
            return "analysis.json", None, list(a.get("candidates") or [])[:top]
        except (json.JSONDecodeError, OSError):
            pass
    from backend.analysis.live import expression_pool
    tick, pool = expression_pool(run_dir, top=top)
    return "live", tick, pool[:top]


def _atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def judge_run(run_dir, spec=None, top: int = 30) -> dict:
    """Judge the run's top expressions with one judge; write analysis_judgements/<judge_id>.json.

    spec: anything `get_judge` accepts (None -> the run config's analysis.judge with defaults). Reads
    analysis.json candidates when present, else the live (LLM-free) expression list. Never writes a
    simulation file: the output goes to analysis_judgements/, and LLM judges cache their calls in
    judge_llm_calls.jsonl."""
    run_dir = Path(run_dir)
    if spec is None:
        import yaml
        cp = run_dir / "config.resolved.yaml"
        spec = yaml.safe_load(open(cp)) if cp.exists() else None
    judge = get_judge(spec, run_dir=run_dir)
    source, tick, cands = _expressions_for(run_dir, top)
    rows = []
    try:
        for c in cands:
            expr, ctx = judge_input(c)
            ww = c.get("world_wording") or {}
            extra = {"status": c.get("status"),
                     "world_wording": bool(ww.get("any")) if source == "live" else None,
                     "system_wording": bool(ww.get("system")) if source == "live" else bool((c.get("wording") or {}).get("system"))}
            v = judge.judge(expr, ctx, extra)
            rows.append({"expression_id": c.get("id"), "phrase": expr["phrase"], "variants": expr["variants"],
                         "judged_at": dt.datetime.now().isoformat(timespec="seconds"), "verdict": v})
    finally:
        judge.close()
    real = is_real_verdict(judge.describe())
    yes = sum(1 for r in rows if r["verdict"]["is_convention"])
    doc = {"judge": judge.describe(), "run_id": run_dir.name,
           "generated_at": dt.datetime.now().isoformat(timespec="seconds"), "source": source, "tick": tick,
           "top": top, "real_judge": real, "placeholder": not real, "n_verdicts": len(rows),
           "n_conventions": yes if real else 0, "n_placeholder_yes": 0 if real else yes,
           "verdicts": rows}
    _atomic_json(run_dir / OUT_DIR / f"{judge.id}.json", doc)
    return doc


def load_judgements(run_dir) -> list[dict]:
    """Every analysis_judgements/*.json of a run (newest first by generated_at)."""
    d = Path(run_dir) / OUT_DIR
    out = []
    if d.exists():
        for p in sorted(d.glob("*.json")):
            try:
                doc = json.load(open(p))
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(doc, dict) and isinstance(doc.get("verdicts"), list):
                doc["_file"] = str(p.relative_to(Path(run_dir)))
                out.append(doc)
    out.sort(key=lambda d: str(d.get("generated_at") or ""), reverse=True)
    return out


def available_judges(run_dir=None, cfg: dict | None = None) -> dict:
    """For /api/judges: registered providers, the configured default spec, prompt versions on disk and
    (with run_dir) the judges that already ran on that run."""
    versions = sorted({p.stem.rsplit("_", 1)[-1] for p in PROMPTS.glob("judge_convention_v*.txt")} | {"v0"})
    out = {"config_key": CONFIG_KEY, "providers": sorted(PROVIDERS), "prompt_versions": versions,
           "default": normalize_spec(cfg) if cfg is not None else dict(DEFAULT_SPEC),
           "functions": list(FUNCTIONS), "verdict_keys": list(VERDICT_KEYS)}
    if run_dir is not None:
        out["ran"] = []
        for d in load_judgements(run_dir):
            real = is_real_verdict(d.get("judge") or {})
            yes = sum(1 for r in d.get("verdicts") or [] if (r.get("verdict") or {}).get("is_convention"))
            out["ran"].append({**{k: d.get("judge", {}).get(k) for k in ("judge_id", "provider", "model", "prompt_version")},
                               "generated_at": d.get("generated_at"), "n_verdicts": d.get("n_verdicts"),
                               "n_conventions": yes if real else 0, "n_placeholder_yes": 0 if real else yes,
                               "real_judge": real, "source": d.get("source"), "tick": d.get("tick"),
                               "file": d.get("_file")})
    return out


# ------------------------------------------------------------------------------------------------ verdict rows
def legacy_provenance(run_dir, analysis: dict | None) -> dict:
    """{provider, model} of an old analysis.json's legacy classifier (`llm` blocks without `judgements`): the
    model of the first analysis_classifier call in analysis_llm_calls.jsonl ("mock" for the mock backend), else
    the recorded observer, else the run config's llm block."""
    run_dir = Path(run_dir)
    an = analysis or {}
    obs = an.get("observer") or {}
    model = None
    p = run_dir / "analysis_llm_calls.jsonl"
    if p.exists():
        try:
            with open(p) as f:
                for i, line in enumerate(f):
                    if i > 20000:
                        break
                    if '"analysis_classifier"' not in line:
                        continue
                    try:
                        model = json.loads(line).get("model")
                    except json.JSONDecodeError:
                        continue
                    break
        except OSError:
            pass
    cfg = {}
    cp = run_dir / "config.resolved.yaml"
    if cp.exists():
        try:
            import yaml
            cfg = yaml.safe_load(open(cp)) or {}
        except Exception:   # noqa: BLE001
            cfg = {}
    llm = cfg.get("llm") or {}
    provider = obs.get("backend") or llm.get("backend") or "unknown"
    model = model or an.get("analysis_model") or obs.get("model") or llm.get("model") or "unknown"
    if str(model).lower() == "mock" or str(provider).lower() in MOCK_PROVIDERS:
        provider, model = "mock", "mock"
    return {"provider": str(provider), "model": str(model)}


def legacy_verdict(llm: dict, prov: dict) -> dict:
    """A schema-valid prompt-v0 Verdict from a legacy c["llm"] block."""
    return {"judge_id": make_judge_id(f"legacy_{prov['provider']}", prov["model"], "v0"),
            "provider": prov["provider"], "model": prov["model"], "prompt_version": "v0",
            "is_convention": bool(llm.get("is_convention")), "gloss": str(llm.get("gloss") or ""), "function": "other",
            "meaning_consistency": 0.0, "confidence": _unit(llm.get("confidence")),
            "rationale": "legacy pipeline classifier annotation (analysis.json llm block)",
            "raw": str(llm.get("raw") or "")[:2000]}


def analysis_verdicts(run_dir, analysis: dict | None) -> list[dict]:
    """Verdict rows of analysis.json: every c["judgements"] entry, or (older analyses) the legacy `llm` block as a
    prompt-v0 verdict. Rows: {phrase, variants, expression_id, judged_at, source, tick, verdict}; tick None = judged
    on the whole run."""
    out, prov = [], None
    for c in (analysis or {}).get("candidates") or []:
        vs = [v for v in (c.get("judgements") or {}).values() if isinstance(v, dict)]
        if not vs and isinstance(c.get("llm"), dict) and "real_judge" not in c["llm"]:
            prov = prov or legacy_provenance(run_dir, analysis)
            vs = [legacy_verdict(c["llm"], prov)]
        for v in vs:
            out.append({"phrase": c.get("canonical_form"), "variants": c.get("variants") or [],
                        "expression_id": c.get("id"), "judged_at": (analysis or {}).get("generated_at"),
                        "source": "analysis.json", "tick": None, "verdict": v})
    return out


def judgement_verdicts(run_dir) -> list[dict]:
    """Verdict rows of analysis_judgements/*.json (same row shape as analysis_verdicts; tick = the live tick judged,
    None when the judge read analysis.json)."""
    out = []
    for doc in load_judgements(run_dir):
        for row in doc.get("verdicts") or []:
            if isinstance(row, dict) and isinstance(row.get("verdict"), dict):
                out.append({"phrase": row.get("phrase"), "variants": row.get("variants") or [],
                            "expression_id": row.get("expression_id"),
                            "judged_at": row.get("judged_at") or doc.get("generated_at"),
                            "source": doc.get("_file"), "tick": doc.get("tick") if doc.get("source") == "live" else None,
                            "verdict": row["verdict"]})
    return out
