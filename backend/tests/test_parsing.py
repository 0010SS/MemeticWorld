import pytest
from pydantic import ValidationError

from app.llm.client import extract_json
from app.simulation.agent import Decision, Reply


def test_extract_json_tolerates_fences_and_chatter():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": {"b": 2}} Hope that helps.') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_decision_normalizes_and_validates():
    d = Decision.model_validate({"action": "talk", "target": "Maya", "utterance": '"hey"', "importance": "12"})
    assert d.action == "TALK" and d.utterance == "hey" and d.importance == 10
    assert Decision.model_validate({"action": "continue", "target": "null"}).action == "CONTINUE_ACTIVITY"
    with pytest.raises(ValidationError):
        Decision.model_validate({"action": "TALK", "target": "Maya"})  # no utterance
    with pytest.raises(ValidationError):
        Decision.model_validate({"action": "MOVE"})  # no target
    with pytest.raises(ValidationError):
        Decision.model_validate({"action": "DANCE"})


def test_reply_parses_string_booleans():
    r = Reply.model_validate({"utterance": "ok", "end_conversation": "true"})
    assert r.end_conversation is True
    assert Reply.model_validate({"utterance": None}).utterance is None
