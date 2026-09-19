"""Record-only reader (ONTOLOGY_V3 §5.6). OBSERVER ONLY.

The persona-free responder ("a student who volunteers at a campus makerspace co-op") with no memories, given
only the binder view at the checkpoint, answers the P-sit items and N. It measures what the text alone
conveys: if newcomers beat it, talk and experience contributed something. Scope
`probe:C{d}:record_only:{item}:{form}:{cue}:{framing}:{order}` (same observer cache as the checkpoint's probes).
"""
from __future__ import annotations

from backend.analysis.battery.runner import PERSONA_FREE, PERSONA_FREE_ISS, Responder, Session

RID = "record_only"


def run_record_only(sess: Session, items: list[dict], cues: dict, cue_keys: list[str], framings: list[str]) -> list[dict]:
    """P-sit items (form "R") and N (form "RN") answered from the binder view alone."""
    if not sess.binder_view:
        return []
    r = Responder(RID, PERSONA_FREE_ISS, PERSONA_FREE)
    out = [sess.act(r, it, "R", 0, True) for it in items]
    for ck in cue_keys:
        if ck != "NONE" and not cues.get(ck):
            continue
        for fr in framings:
            out.append(sess.note(r, ck, cues.get(ck), fr, situated=True, form="RN"))
    return out
