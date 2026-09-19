"""Surface skins of the hidden families (ontology v2 §1.2). SIMULATOR-ONLY.

Each module e1.py .. e4.py defines `SKINS: list[dict]` (11 skins: one per train domain and one per
holdout domain). They are imported lazily and tolerantly: a missing or broken module is recorded
in `load_errors()` instead of breaking every import of the simulator; the world-script generator
raises a clear error only if a family it actually needs has no skins.

`SKINS` (module attribute) is the aggregate list, loaded on first access.
"""
from __future__ import annotations

import importlib

FAMILY_MODULES = ("e1", "e2", "e3", "e4")
_ERRORS: dict[str, str] = {}


def load_skins(strict: bool = False) -> list[dict]:
    """All skins of all family modules that import cleanly, sorted by key (import-order independent)."""
    out = []
    _ERRORS.clear()
    for m in FAMILY_MODULES:
        try:
            mod = importlib.import_module(f"{__name__}.{m}")
            out += list(getattr(mod, "SKINS"))
        except Exception as e:  # noqa: BLE001  (module missing or mid-edit)
            if strict:
                raise
            _ERRORS[m] = f"{type(e).__name__}: {e}"
    return sorted(out, key=lambda s: str(s.get("key")))


def load_errors() -> dict[str, str]:
    """module -> error of the last load_skins() call (empty when every family module loaded)."""
    return dict(_ERRORS)


def missing_modules() -> list[str]:
    load_skins()
    return sorted(_ERRORS)


def __getattr__(name: str):
    if name == "SKINS":
        return load_skins()
    raise AttributeError(name)
