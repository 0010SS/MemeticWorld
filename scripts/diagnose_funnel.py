"""Meme-pipeline funnel diagnostics (OBSERVER ONLY): where does the path
event -> talk -> recurrence -> linking -> naming -> verbatim memory -> reuse -> spread break?

Thin wrapper over backend/analysis/funnel.py (the same metrics `analyze` puts in analysis.json and
outcomes.json), plus the per-mechanism manipulation checks and validity.

Usage: .venv/bin/python scripts/diagnose_funnel.py <run> [<run> ...] [--json=out.json] [--checks]
(<run> is a run directory or a directory name under runs/).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.analysis.funnel import diagnose, table  # noqa: E402
from backend.analysis.rundata import RunData  # noqa: E402


def _dir(run: str) -> Path:
    p = Path(run)
    return p if (p / "manifest.json").exists() else ROOT / "runs" / run


def main(argv: list[str]) -> None:
    runs = [a for a in argv if not a.startswith("--")]
    if not runs:
        sys.exit(__doc__)
    rows = [{"run": r, **diagnose(RunData(_dir(r)))} for r in runs]
    print(table(rows))
    if "--checks" in argv:
        for r in rows:
            print(f"\n== {r['run']}\nvalidity: {json.dumps(r['validity'])}\nmanipulation: {json.dumps(r['manipulation'], indent=1)}")
    out = [a.split("=", 1)[1] for a in argv if a.startswith("--json=")]
    if out:
        json.dump(rows, open(out[0], "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
