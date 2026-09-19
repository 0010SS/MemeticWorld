"""Experiment configuration: YAML files deep-merged over configs/default.yaml."""
from __future__ import annotations

import copy
import hashlib
import warnings
from functools import lru_cache
from pathlib import Path

import yaml

from backend.ga_compat import REPO_ROOT

DEFAULT = REPO_ROOT / "configs" / "default.yaml"


@lru_cache(maxsize=128)
def _yaml_cached(path: str, modified: int, size: int):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def read_config_yaml(path):
    p = Path(path).resolve()
    stat = p.stat()
    return copy.deepcopy(_yaml_cached(str(p), stat.st_mtime_ns, stat.st_size))

# keys dropped in ontology v2 -> what to use instead (shown in the unknown-key warning)
REMOVED = {
    "latent_events.holdout_from_day": "use latent_events.holdout_frac (held-out skins interleaved from day 1)",
    "latent_events.generator": "the LLM surface generator was dropped in v2; events come from the world script",
}


class ConfigKeyWarning(UserWarning):
    """An overlay sets a key that configs/default.yaml does not define (usually a typo or a v1 key)."""


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def unknown_keys(over: dict, ref: dict, prefix: str = "") -> list[str]:
    """Dotted keys in `over` that `ref` does not define. Open maps (an empty dict or null in `ref`, e.g.
    retrieval.source_weights) accept anything; keys starting with '_' are private (e.g. `_condition`)."""
    out = []
    for k, v in (over or {}).items():
        key = f"{prefix}{k}"
        if str(k).startswith("_") or (not prefix and k == "extends"):
            continue
        if k not in ref:
            out.append(key)
        elif isinstance(v, dict) and isinstance(ref[k], dict) and ref[k]:
            out += unknown_keys(v, ref[k], key + ".")
    return out


def check_keys(over: dict, source: str) -> list[str]:
    """Warn (never fail) about overlay keys that default.yaml does not know: a silently ignored typo
    would make two conditions identical while their labels say they differ."""
    bad = unknown_keys(over, read_config_yaml(DEFAULT))
    for key in bad:
        hint = REMOVED.get(key, "not in configs/default.yaml; the engine will ignore it")
        warnings.warn(f"{source}: unknown config key '{key}' ({hint})", ConfigKeyWarning, stacklevel=3)
    return bad


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = read_config_yaml(DEFAULT)
    if path:
        p = Path(path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        user = read_config_yaml(p)
        base_name = user.pop("extends", None)
        if base_name:
            cfg = load_config(base_name)
        check_keys(user, str(path))
        cfg = deep_merge(cfg, user)
    if overrides:
        check_keys(overrides, "overrides")
        cfg = deep_merge(cfg, overrides)
    resolve_background(cfg)
    from backend.llm.environment import resolve_model
    if cfg['llm']['backend'] == 'openai' or str(cfg['llm'].get('model', '')).startswith('$OPENAI_'):
        cfg['llm']['model'] = resolve_model(cfg['llm'].get('model'))
    observer = cfg.get('analysis', {}).get('observer') or {}
    if observer.get('backend') == 'openai' or str(observer.get('model', '')).startswith('$OPENAI_'):
        observer['model'] = resolve_model(observer.get('model'), observer=True)
    return cfg


def resolve_background(cfg: dict, *, frozen=False):
    """Read an introduction before execution; a resolved recording is self-contained for replay."""
    spec = cfg.setdefault("shared_background", {"file": None, "markdown": None, "sha256": None})
    if not isinstance(spec, dict):
        raise ValueError("shared_background must be a mapping")
    if spec.get("file") and not frozen:
        path = Path(spec["file"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        spec["markdown"] = path.read_text(encoding="utf-8-sig")
    text = spec.get("markdown")
    if text is not None and (not isinstance(text, str) or not text.strip()):
        raise ValueError("Shared background must contain nonempty Markdown text")
    actual = hashlib.sha256(text.encode("utf-8")).hexdigest() if text is not None else None
    if frozen and spec.get("sha256") not in (None, actual):
        raise ValueError("Shared background hash does not match the recorded Markdown")
    spec["sha256"] = actual
    return cfg


def input_fingerprints(cfg):
    """Record mutable file inputs so edits cannot silently satisfy or resume an old study."""
    result = {}
    for key in ("population", "initial_memories_file"):
        if cfg.get(key):
            path = Path(cfg[key])
            if not path.is_absolute():
                path = REPO_ROOT / path
            result[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def observer_spec(cfg: dict, backend: str | None = None, model: str | None = None) -> tuple[str | None, str | None]:
    """Observer LLM for `analyze`: explicit arguments, else `analysis.observer`, else None (= the run's
    own llm settings). One fixed observer across conditions keeps observer differences out of effects (D41)."""
    obs = (cfg.get("analysis") or {}).get("observer") or {}
    return backend or obs.get("backend"), model or obs.get("model")


def parse_overrides(pairs: list[str]) -> dict:
    """['memory.encoding_noise=0.5', 'llm.backend=mock'] -> nested dict."""
    out: dict = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        cur = out
        parts = k.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = yaml.safe_load(v)
    return out
