"""LB1 item loading (ONTOLOGY_V3 §5.3). OBSERVER ONLY."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def _pack_surfaces(pack: str) -> dict:
    """World surfaces per class from the content pack (step 2), if it is importable: {class: [text, ...]}."""
    try:
        import importlib
        mod = importlib.import_module(f"backend.simulation.content.{pack}")
    except Exception:   # noqa: BLE001 - pack not built yet: the YAML fallbacks are used
        return {}
    for attr in ("SURFACES", "surfaces", "CONTENT"):
        v = getattr(mod, attr, None)
        if isinstance(v, dict):
            v = v.get("surfaces", v)
            if isinstance(v, dict) and any(isinstance(x, (list, tuple)) for x in v.values()):
                return {k: list(x) for k, x in v.items() if isinstance(x, (list, tuple))}
    return {}


def _strip_code(text: str) -> str:
    """Twins carry no panel-code sentence (no item shows it)."""
    parts = [s for s in text.replace("\n", " ").split(". ") if "panel" not in s.lower()]
    return ". ".join(parts).strip()


@lru_cache(maxsize=8)
def _load(pack: str, mapping: str) -> tuple:
    p = HERE / f"items_lb1_{pack}_{mapping}.yaml"
    if not p.exists():
        raise NotImplementedError(f"battery items for {pack}/{mapping} not written (laser_beta is stubbed)")
    doc = yaml.safe_load(open(p))
    surf = _pack_surfaces(pack)
    by_id = {}
    items = []
    for raw in doc["items"]:
        it = dict(raw)
        tw = it.get("twin")
        if tw:
            pool = surf.get(tw["class"]) or []
            if len(pool) > int(tw["surface"]):
                it["text"] = _strip_code(str(pool[int(tw["surface"])]))
                it["text_source"] = "pack"
            else:
                it["text_source"] = "fallback"
        if it["type"] == "CUE":
            it["text"] = by_id[it["base"]]["text"] + " " + doc["cue_extras"][it["extra"]]
        by_id[it["id"]] = it
        items.append(it)
    return tuple(items)


def load_items(pack: str = "laser_alpha", mapping: str = "M1") -> list[dict]:
    return [dict(i) for i in _load(pack, mapping)]


def heldout_texts(pack: str = "laser_alpha", mapping: str | None = None) -> list[str]:
    """Held-out battery wording (for the world's lexical-hygiene test: no content bigram shared with surfaces)."""
    maps = [mapping] if mapping else ["M1", "M2"]
    out = []
    for m in maps:
        out += [i["text"] for i in load_items(pack, m) if i.get("wording") == "heldout"]
    return sorted(set(out))


def select(items: list[dict], spec: str) -> list[dict]:
    """Item subsets by the §5.3 shorthand: 'all', 'K1c', 'K1c_h' (held-out only), 'K2_h', 'K1c2' (first two K1c
    held-out), 'A13' (the 13 applicability situations)."""
    h = [i for i in items if i.get("wording") == "heldout"]
    if spec == "all":
        return list(items)
    if spec == "A13":
        return ([i for i in h if i["type"] == "K1c"] + [i for i in items if i["type"] in ("K1a", "K3")]
                + [i for i in h if i["type"] == "K2"] + [i for i in items if i["type"] == "K0"])
    if spec.endswith("_h"):
        return [i for i in h if i["type"] == spec[:-2]]
    if spec == "K1c2":
        return [i for i in h if i["type"] == "K1c"][:2]
    return [i for i in items if i["type"] == spec]
