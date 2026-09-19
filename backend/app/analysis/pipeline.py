"""One call to analyze a run: load, pair with its control run, detect, label referents."""
from __future__ import annotations

from app.analysis import meme_detector, semantics
from app.analysis.meme_detector import Corpus
from app.db import database as db

AUTO = -1


def analyze_run(conn, run_id: int, control_id: int | None = AUTO) -> tuple[Corpus, list[dict], Corpus | None]:
    """control_id=AUTO pairs a full run with the latest same-seed control run; None disables it."""
    corpus = meme_detector.load_corpus(conn, run_id)
    if control_id == AUTO:
        control_id = meme_detector.find_control(conn, corpus.run)
    control = meme_detector.load_corpus(conn, control_id) if control_id and control_id != run_id else None
    memes = meme_detector.detect_memes(corpus, control)
    semantics.assign_referents(corpus, memes)
    return corpus, memes, control


def last_event_id(conn, run_id: int | None) -> int:
    if not run_id:
        return 0
    return conn.execute("SELECT COALESCE(MAX(id), 0) FROM events WHERE run_id = ?", (run_id,)).fetchone()[0]


def control_for(conn, run_id: int) -> int | None:
    run = db.get_run(conn, run_id, with_config=False)
    return meme_detector.find_control(conn, run) if run else None
