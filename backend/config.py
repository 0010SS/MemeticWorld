"""Experiment configuration: YAML files deep-merged over configs/default.yaml."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from backend.ga_compat import REPO_ROOT

DEFAULT = REPO_ROOT / "configs" / "default.yaml"


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = yaml.safe_load(open(DEFAULT))
    if path:
        p = Path(path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        user = yaml.safe_load(open(p)) or {}
        base_name = user.pop("extends", None)
        if base_name:
            cfg = load_config(base_name)
        cfg = deep_merge(cfg, user)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


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
